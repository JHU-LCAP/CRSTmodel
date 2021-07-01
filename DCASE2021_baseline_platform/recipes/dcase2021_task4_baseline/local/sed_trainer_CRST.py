import os
import random
from copy import deepcopy
from pathlib import Path

import local.config as cfg
import pandas as pd
import pytorch_lightning as pl
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from desed_task.data_augm import mixup, frame_shift, add_noise, temporal_reverse
from desed_task.utils.scaler import TorchScaler
import numpy as np

from .utils import (
    batched_decode_preds,
    log_sedeval_metrics,
    JSD,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
)

class SEDTask4_2021(pl.LightningModule):
    """ Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionnary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed_student: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: asteroid.engine.schedulers.BaseScheduler subclass object, the scheduler to be used. This is
            used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False
    ):
        super(SEDTask4_2021, self).__init__()
        self.hparams = hparams

        # manual optimization
        self.automatic_optimization = False

        self.encoder = encoder
        self.sed_student1 = sed_student[0]
        self.sed_teacher1 = deepcopy(sed_student[0])
        self.sed_student2 = sed_student[1]
        self.sed_teacher2 = deepcopy(sed_student[1])
        self.opt1 = opt[0]
        self.opt2 = opt[1]
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler1 = scheduler[0]
        self.scheduler2 = scheduler[1]
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]


        # add class_label
        self.softmax = torch.nn.Softmax(dim=2)
        self.jsd = JSD()
        self.class_label = torch.tensor(cfg.class_label).cuda()

        feat_params = self.hparams["feats"]
        #self.lin_spec = LinearSpectrogram(nCh=128, n_fft=2048, hop_length=256, win_fn = torch.hamming_window)
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )

        for param in self.sed_teacher1.parameters():
            param.detach_()

        for param in self.sed_teacher2.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = pl.metrics.classification.F1(
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        self.get_weak_teacher_f1_seg_macro = pl.metrics.classification.F1(
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        self.scaler = self._init_scaler()

        # buffer for event based scores which we compute using sed-eval
        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student1 = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher1 = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_student2 = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher2 = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_student  = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher  = {k: pd.DataFrame() for k in test_thresholds}

        self.decoded_student1_05_buffer = pd.DataFrame()
        self.decoded_teacher1_05_buffer = pd.DataFrame()
        self.decoded_student2_05_buffer = pd.DataFrame()
        self.decoded_teacher2_05_buffer = pd.DataFrame()
        self.decoded_student_05_buffer  = pd.DataFrame()
        self.decoded_teacher_05_buffer  = pd.DataFrame()


    def update_ema(self, alpha, global_step, model, ema_model):
        """ Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        """Scaler inizialization

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler("instance", "minmax", self.hparams["scaler"]["dims"])

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        """ Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code


    def training_step(self, batch, batch_indx, optimizer_idx):
        """ Applying the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        audio, labels, padded_indxs = batch
        indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        features = self.mel_spec(audio)

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1

        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()

        mixup_type = self.hparams["training"].get("mixup")
        if mixup_type is not None and 0.5 > random.random():
            features[weak_mask], labels_weak = mixup(
                features[weak_mask], labels_weak, mixup_label_type=mixup_type
            )
            features[strong_mask], labels[strong_mask] = mixup(
                features[strong_mask], labels[strong_mask], mixup_label_type=mixup_type
            )

        # perturbation
        ori_features = self.scaler(self.take_log(features))
        ema_features = ori_features.clone().detach()
        ema_labels   = labels.clone().detach()
        ema_features, ema_labels = frame_shift(ema_features, ema_labels)
        ema_labels_weak = (torch.sum(ema_labels[weak_mask], -1) > 0).float()

        # sed students forward
        strong_preds_student1, weak_preds_student1 = self.sed_student1(ori_features)
        strong_preds_student2, weak_preds_student2 = self.sed_student2(ema_features)

        # supervised loss on strong labels
        loss_strong1 = self.supervised_loss(
            strong_preds_student1[strong_mask], labels[strong_mask]
        )
        loss_strong2 = self.supervised_loss(
            strong_preds_student2[strong_mask], ema_labels[strong_mask]
        )

        # supervised loss on weakly labelled
        loss_weak1 = self.supervised_loss(weak_preds_student1[weak_mask], labels_weak)
        loss_weak2 = self.supervised_loss(weak_preds_student2[weak_mask], ema_labels_weak)

        # total supervised loss
        tot_loss_supervised1 = loss_strong1 + loss_weak1
        tot_loss_supervised2 = loss_strong2 + loss_weak2

        with torch.no_grad():
            strong_preds_teacher1, weak_preds_teacher1 = self.sed_teacher1(ema_features)
            strong_preds_teacher2, weak_preds_teacher2 = self.sed_teacher2(ori_features)

            nClass = self.hparams['net']['nclass']

            sp1   = torch.clamp(strong_preds_teacher1, 1.0e-4, 1-1.0e-4)
            p1_h1 = torch.log(sp1.permute(0,2,1))
            p1_h0 = torch.log(1-sp1.permute(0,2,1))

            sp2   = torch.clamp(strong_preds_teacher2, 1.0e-4, 1-1.0e-4)
            p2_h1 = torch.log(sp2.permute(0,2,1))
            p2_h0 = torch.log(1-sp2.permute(0,2,1))

            p_h0  = torch.cat((p1_h0, p2_h0), 1)
            p_h1  = torch.cat((p1_h1, p2_h1), 1)

            # K = 0
            P0 = p_h0.sum(2)
            
            # K = 1
            P1 = P0[:,:,None] + p_h1 - p_h0
            #P  = torch.cat([P0.reshape(157,1), P1], 1)

            # K = 2
            P2 = []
            for cter in range(1,nClass):
                P2.append(P1[:,:,:-cter]+P1[:,:,cter:])
            P2 = torch.cat(P2, 2)
            P2 = P2 - P0[:,:,None]
            #P = torch.cat([P0.reshape(156*2,1), P1, P2], 1)
            
            # K: up to 3
            P3 = []
            for cter1 in range(1,nClass):
                for cter2 in range(1,nClass-cter1):
                    P3.append(P1[:,:,:-(cter1+cter2)]+P1[:,:,cter1:-cter2]+P1[:,:,(cter1+cter2):])
            P3 = torch.cat(P3,2)
            P3 = P3 - 2*P0[:,:,None]
            P  = torch.cat([P0.reshape(batch_num,156*2,1), P1, P2, P3], 2)

            P = self.softmax(P)
            prob_v, prob_i = torch.sort(P, dim=2, descending=True)

            # 5 best potential labels
            norm_p = prob_v[:,:,:].sum(2)
            prob_v = prob_v[:,:,:]/norm_p[:,:,None]

            cl = self.class_label[prob_i[:,:,:].tolist(),:]
            # picking up the best label
            cl = torch.mul(cl, prob_v[:,:,:,None]).sum(2)
                
            est_strong_target1 = torch.squeeze(cl[:,:156,:]).float()
            est_strong_target2 = torch.squeeze(cl[:,156:,:]).float()

            est_strong_target1 = est_strong_target1.permute((0,2,1))	# for ema_feature
            est_strong_target2 = est_strong_target2.permute((0,2,1))	# for ori_feature

            est_weak_target1 = est_strong_target1.mean(2)
            est_weak_target2 = est_strong_target2.mean(2)

            loss_strong_teacher1 = self.supervised_loss(
                strong_preds_teacher1[strong_mask], ema_labels[strong_mask]
            )
            loss_strong_teacher2 = self.supervised_loss(
                strong_preds_teacher2[strong_mask], labels[strong_mask]
            )

            loss_weak_teacher1 = self.supervised_loss(
                weak_preds_teacher1[weak_mask], ema_labels_weak
            )
            loss_weak_teacher2 = self.supervised_loss(
                weak_preds_teacher2[weak_mask], labels_weak
            )

        # we apply consistency between the predictions, use the scheduler for learning rate (to be changed ?)
        weight1 = (
            self.hparams["training"]["const_max"]
            * self.scheduler1["scheduler"]._get_scaling_factor()
        )
        weight2 = (
            self.hparams["training"]["const_max"]
            * self.scheduler2["scheduler"]._get_scaling_factor()
        )

        strong_reliability1 = weight1*(1-self.jsd(est_strong_target1[strong_mask], ema_labels[strong_mask]))
        strong_reliability2 = weight2*(1-self.jsd(est_strong_target2[strong_mask], labels[strong_mask]))
        weak_reliability1   = weight1*(1-self.jsd(est_weak_target1[weak_mask], ema_labels_weak))
        weak_reliability2   = weight2*(1-self.jsd(est_weak_target2[weak_mask], labels_weak))

        strong_self_sup_loss1 = self.selfsup_loss(
            strong_preds_student1[24:], est_strong_target2[24:]		# for ori_feature
        )
        strong_self_sup_loss2 = self.selfsup_loss(
            strong_preds_student2[24:], est_strong_target1[24:]		# for ema_feature
        )

        weak_self_sup_loss1 = self.selfsup_loss(
            weak_preds_student1[weak_mask], est_weak_target2[weak_mask]
        )
        weak_self_sup_loss2 = self.selfsup_loss(
            weak_preds_student2[weak_mask], est_weak_target1[weak_mask]
        )

        tot_self_loss1 = strong_reliability2*strong_self_sup_loss1 + weak_reliability2*weak_self_sup_loss1
        tot_self_loss2 = strong_reliability1*strong_self_sup_loss2 + weak_reliability1*weak_self_sup_loss2

        tot_loss1 = tot_loss_supervised1 + tot_self_loss1
        tot_loss2 = tot_loss_supervised2 + tot_self_loss2

        #self.log("train/student/loss_strong1", loss_strong1)
        #self.log("train/student/loss_weak1", loss_weak1)
        #self.log("train/student/loss_strong2", loss_strong2)
        #self.log("train/student/loss_weak2", loss_weak2)
        #self.log("train/teacher/loss_strong1", loss_strong_teacher1)
        #self.log("train/teacher/loss_weak1", loss_weak_teacher1)
        #self.log("train/teacher/loss_strong2", loss_strong_teacher2)
        #self.log("train/teacher/loss_weak2", loss_weak_teacher2)
        self.log("train/step1", self.scheduler1["scheduler"].step_num, prog_bar=True)
        self.log("train/step2", self.scheduler2["scheduler"].step_num, prog_bar=True)
        self.log("train/student/tot_loss1", tot_loss1, prog_bar=True)
        self.log("train/student/tot_loss2", tot_loss2, prog_bar=True)
        self.log("train/strong_reliability1", strong_reliability1, prog_bar=True)
        self.log("train/strong_reliability2", strong_reliability2, prog_bar=True)
        #self.log("train/student/tot_self_loss1", tot_self_loss1, prog_bar=True)
        #self.log("train/student/weak_self_sup_loss1", weak_self_sup_loss1)
        #self.log("train/student/strong_self_sup_loss1", strong_self_sup_loss1)
        #self.log("train/student/tot_self_loss2", tot_self_loss2, prog_bar=True)
        #self.log("train/student/weak_self_sup_loss2", weak_self_sup_loss2)
        #self.log("train/student/strong_self_sup_loss2", strong_self_sup_loss2)
        self.log("train/lr1", self.opt1.param_groups[-1]["lr"], prog_bar=True)
        self.log("train/lr2", self.opt2.param_groups[-1]["lr"], prog_bar=True)


        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler1["scheduler"].step_num,
            self.sed_student1,
            self.sed_teacher1,
        )

        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler2["scheduler"].step_num,
            self.sed_student2,
            self.sed_teacher2,
        )

        # training Model I
        self.opt1.zero_grad()
        self.manual_backward(tot_loss1, self.opt1)
        self.opt1.step()

        # training Model II
        self.opt2.zero_grad()
        self.manual_backward(tot_loss2, self.opt2)
        self.opt2.step()

        return {'tot_loss1': tot_loss1, 'tot_loss2': tot_loss2}



    def validation_step(self, batch, batch_indx):
        """ Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames = batch
        features = self.mel_spec(audio)
        #features2 = self.lin_spec(audio)
        #features  = torch.cat([features1, features2], 1)

        logmels = self.scaler(self.take_log(features))

        # prediction for strudent
        strong_preds_student1, weak_preds_student1 = self.sed_student1(logmels)
        strong_preds_student2, weak_preds_student2 = self.sed_student2(logmels)
        strong_preds_student = (strong_preds_student1 + strong_preds_student2)/2
        weak_preds_student   = (weak_preds_student1   + weak_preds_student2)/2
        # prediction for teacher
        strong_preds_teacher1, weak_preds_teacher1 = self.sed_teacher1(logmels)
        strong_preds_teacher2, weak_preds_teacher2 = self.sed_teacher2(logmels)
        strong_preds_teacher = (strong_preds_teacher1 + strong_preds_teacher2)/2
        weak_preds_teacher   = (weak_preds_teacher1   + weak_preds_teacher2)/2

        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )
        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["synth_val_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()
            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )
            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[mask_weak], labels_weak
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["synth_val_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_student_synth.keys()),
            )

            for th in self.val_buffer_student_synth.keys():
                self.val_buffer_student_synth[th] = self.val_buffer_student_synth[
                    th
                ].append(decoded_student_strong[th], ignore_index=True)

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_teacher_synth.keys()),
            )
            for th in self.val_buffer_teacher_synth.keys():
                self.val_buffer_teacher_synth[th] = self.val_buffer_teacher_synth[
                    th
                ].append(decoded_teacher_strong[th], ignore_index=True)

        return

    def validation_epoch_end(self, outputs):
        """ Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """

        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()

        # synth dataset
        intersection_f1_macro_student = compute_per_intersection_macro_f1(
            self.val_buffer_student_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["synth_val_dur"],
        )

        synth_student_event_macro = log_sedeval_metrics(
            self.val_buffer_student_synth[0.5], self.hparams["data"]["synth_val_tsv"],
        )[0]

        intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
            self.val_buffer_teacher_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["synth_val_dur"],
        )

        synth_teacher_event_macro = log_sedeval_metrics(
            self.val_buffer_teacher_synth[0.5], self.hparams["data"]["synth_val_tsv"],
        )[0]

        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = intersection_f1_macro_student
        elif obj_metric_synth_type == "event":
            synth_metric = synth_student_event_macro
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_student
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        obj_metric = torch.tensor(weak_student_f1_macro.item() + synth_metric)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/macro_F1", weak_student_f1_macro)
        self.log("val/weak/teacher/macro_F1", weak_teacher_f1_macro)
        self.log(
            "val/synth/student/intersection_f1_macro", intersection_f1_macro_student
        )
        self.log(
            "val/synth/teacher/intersection_f1_macro", intersection_f1_macro_teacher
        )
        self.log("val/synth/student/event_f1_macro", synth_student_event_macro)
        self.log("val/synth/teacher/event_f1_macro", synth_teacher_event_macro)

        # free the buffers
        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student1"] = self.sed_student1.state_dict()
        checkpoint["sed_teacher1"] = self.sed_teacher1.state_dict()
        checkpoint["sed_student2"] = self.sed_student2.state_dict()
        checkpoint["sed_teacher2"] = self.sed_teacher2.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """ Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames = batch
        features = self.mel_spec(audio)
        #features2 = self.lin_spec(audio)
        #features  = torch.cat([features1, features2], 1)

        # prediction for student
        logmels = self.scaler(self.take_log(features))
        strong_preds_student1, weak_preds_student1 = self.sed_student1(logmels)
        strong_preds_student2, weak_preds_student2 = self.sed_student2(logmels)
        strong_preds_student = (strong_preds_student1 + strong_preds_student2)/2
        weak_preds_student   = (weak_preds_student1 + weak_preds_student2)/2

        # prediction for teacher
        strong_preds_teacher1, weak_preds_teacher1 = self.sed_teacher1(logmels)
        strong_preds_teacher2, weak_preds_teacher2 = self.sed_teacher2(logmels)
        strong_preds_teacher = (strong_preds_teacher1 + strong_preds_teacher2)/2
        weak_preds_teacher   = (weak_preds_teacher1 + weak_preds_teacher2)/2

        
        bsz = len(filenames)
        for bter in range(bsz):
            path, filename = os.path.split(filenames[bter])

            pred_student = strong_preds_student[bter].cpu().numpy()
            pred_teacher = strong_preds_teacher[bter].cpu().numpy()
            np.save('./Posterior/student/{}.npy'.format(filename), pred_student)
            np.save('./Posterior/teacher/{}.npy'.format(filename), pred_teacher)
        

        if not self.evaluation:
            loss_strong_student1 = self.supervised_loss(strong_preds_student1, labels)
            loss_strong_student2 = self.supervised_loss(strong_preds_student2, labels)
            loss_strong_student  = self.supervised_loss(strong_preds_student,  labels)
            loss_strong_teacher1 = self.supervised_loss(strong_preds_teacher1, labels)
            loss_strong_teacher2 = self.supervised_loss(strong_preds_teacher2, labels)
            loss_strong_teacher  = self.supervised_loss(strong_preds_teacher,  labels)

#            self.log("test/student1/loss_strong", loss_strong_student1)
#            self.log("test/student2/loss_strong", loss_strong_student2)
            self.log("test/student/loss_strong",  loss_strong_student)
#            self.log("test/teacher1/loss_strong", loss_strong_teacher1)
#            self.log("test/teacher2/loss_strong", loss_strong_teacher2)
            self.log("test/teacher/loss_strong",  loss_strong_teacher)

        # compute psds
        decoded_student1_strong = batched_decode_preds(
            strong_preds_student1,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student1.keys()),
        )
        for th in self.test_psds_buffer_student1.keys():
            self.test_psds_buffer_student1[th] = self.test_psds_buffer_student1[
                th
            ].append(decoded_student1_strong[th], ignore_index=True)


        decoded_student2_strong = batched_decode_preds(
            strong_preds_student2,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student2.keys()),
        )
        for th in self.test_psds_buffer_student2.keys():
            self.test_psds_buffer_student2[th] = self.test_psds_buffer_student2[
                th
            ].append(decoded_student2_strong[th], ignore_index=True)

        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()),
        )
        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = self.test_psds_buffer_student[
                th
            ].append(decoded_student_strong[th], ignore_index=True)


        decoded_teacher1_strong = batched_decode_preds(
            strong_preds_teacher1,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher1.keys()),
        )
        for th in self.test_psds_buffer_teacher1.keys():
            self.test_psds_buffer_teacher1[th] = self.test_psds_buffer_teacher1[
                th
            ].append(decoded_teacher1_strong[th], ignore_index=True)


        decoded_teacher2_strong = batched_decode_preds(
            strong_preds_teacher2,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher2.keys()),
        )
        for th in self.test_psds_buffer_teacher2.keys():
            self.test_psds_buffer_teacher2[th] = self.test_psds_buffer_teacher2[
                th
            ].append(decoded_teacher2_strong[th], ignore_index=True)

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()),
        )
        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = self.test_psds_buffer_teacher[
                th
            ].append(decoded_teacher_strong[th], ignore_index=True)


        # compute f1 score
        decoded_student1_strong = batched_decode_preds(
            strong_preds_student1,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_student1_05_buffer = self.decoded_student1_05_buffer.append(
            decoded_student1_strong[0.5]
        )

        decoded_student2_strong = batched_decode_preds(
            strong_preds_student2,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_student2_05_buffer = self.decoded_student2_05_buffer.append(
            decoded_student2_strong[0.5]
        )

        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_student_05_buffer = self.decoded_student_05_buffer.append(
            decoded_student_strong[0.5]
        )


        decoded_teacher1_strong = batched_decode_preds(
            strong_preds_teacher1,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_teacher1_05_buffer = self.decoded_teacher1_05_buffer.append(
            decoded_teacher1_strong[0.5]
        )

        decoded_teacher2_strong = batched_decode_preds(
            strong_preds_teacher2,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_teacher2_05_buffer = self.decoded_teacher2_05_buffer.append(
            decoded_teacher2_strong[0.5]
        )

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_teacher_05_buffer = self.decoded_teacher_05_buffer.append(
            decoded_teacher_strong[0.5]
        )



    def on_test_epoch_end(self):
        # pub eval dataset
        try:
            log_dir = self.logger.log_dir
        except Exception as e:
            log_dir = self.hparams["log_dir"]
        save_dir = os.path.join(log_dir, "metrics_test")

        if self.evaluation:
            # only save the predictions
            save_dir_student = os.path.join(save_dir, "student")
            os.makedirs(save_dir_student, exist_ok=True)
            self.decoded_student_05_buffer.to_csv(
                os.path.join(save_dir_student, f"predictions_05_student.tsv"),
                sep="\t",
                index=False
            )
            for k in self.test_psds_buffer_student.keys():
                self.test_psds_buffer_student[k].to_csv(
                    os.path.join(save_dir_student, f"predictions_th_{k:.2f}.tsv"),
                    sep="\t",
                    index=False,
                )
            print(f"\nPredictions for student saved in: {save_dir_student}")
            
            save_dir_teacher = os.path.join(save_dir, "teacher")
            os.makedirs(save_dir_teacher, exist_ok=True)
           
            self.decoded_teacher_05_buffer.to_csv(
                os.path.join(save_dir_teacher, f"predictions_05_teacher.tsv"),
                sep="\t",
                index=False
            )
            for k in self.test_psds_buffer_student.keys():
                self.test_psds_buffer_student[k].to_csv(
                    os.path.join(save_dir_teacher, f"predictions_th_{k:.2f}.tsv"),
                    sep="\t",
                    index=False,
                )
            print(f"\nPredictions for teacher saved in: {save_dir_teacher}")

        else:
            # calculate the metrics
            psds_score_student1_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_student1,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student1", "scenario1"),
            )

            psds_score_student1_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_student1,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student1", "scenario2"),
            )

            psds_score_student2_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_student2,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student2", "scenario1"),
            )

            psds_score_student2_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_student2,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student2", "scenario2"),
            )

            psds_score_student_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )

            psds_score_student_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )


            psds_score_teacher1_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher1,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher1", "scenario1"),
            )

            psds_score_teacher1_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher1,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher1", "scenario2"),
            )

            psds_score_teacher2_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher2,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher2", "scenario1"),
            )

            psds_score_teacher2_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher2,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher2", "scenario2"),
            )

            psds_score_teacher_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )

            psds_score_teacher_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )



            event_macro_student1 = log_sedeval_metrics(
                self.decoded_student1_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student1"),
            )[0]

            event_macro_student2 = log_sedeval_metrics(
                self.decoded_student2_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student2"),
            )[0]

            event_macro_student = log_sedeval_metrics(
                self.decoded_student_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]


            event_macro_teacher1 = log_sedeval_metrics(
                self.decoded_teacher1_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher1"),
            )[0]

            event_macro_teacher2 = log_sedeval_metrics(
                self.decoded_teacher2_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher2"),
            )[0]

            event_macro_teacher = log_sedeval_metrics(
                self.decoded_teacher_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]


            # synth dataset
            intersection_f1_macro_student1 = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student1_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher1 = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher1_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_student2 = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student2_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher2 = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher2_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_student = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )


            best_test_result1 = torch.tensor(max(psds_score_student1_scenario1, psds_score_student1_scenario2))
            best_test_result2 = torch.tensor(max(psds_score_student2_scenario1, psds_score_student2_scenario2))
            best_test_result  = torch.tensor(max(psds_score_student_scenario1,  psds_score_student_scenario2))

            results = {
                "hp_metric": best_test_result,
                "test/student/psds_score_scenario1": psds_score_student_scenario1,
                "test/student/psds_score_scenario2": psds_score_student_scenario2,
                "test/teacher/psds_score_scenario1": psds_score_teacher_scenario1,
                "test/teacher/psds_score_scenario2": psds_score_teacher_scenario2,
                "test/student/event_f1_macro": event_macro_student,
                "test/student/intersection_f1_macro": intersection_f1_macro_student,
                "test/teacher/event_f1_macro": event_macro_teacher,
                "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
                #"hp_metric_I": best_test_result1,
                #"test/student1/psds_score_scenario1": psds_score_student1_scenario1,
                #"test/student1/psds_score_scenario2": psds_score_student1_scenario2,
                #"test/teacher1/psds_score_scenario1": psds_score_teacher1_scenario1,
                #"test/teacher1/psds_score_scenario2": psds_score_teacher1_scenario2,
                #"test/student1/event_f1_macro": event_macro_student1,
                #"test/student1/intersection_f1_macro": intersection_f1_macro_student1,
                #"test/teacher1/event_f1_macro": event_macro_teacher1,
                #"test/teacher1/intersection_f1_macro": intersection_f1_macro_teacher1,
                #"hp_metric_II": best_test_result2,
                #"test/student2/psds_score_scenario1": psds_score_student2_scenario1,
                #"test/student2/psds_score_scenario2": psds_score_student2_scenario2,
                #"test/teacher2/psds_score_scenario1": psds_score_teacher2_scenario1,
                #"test/teacher2/psds_score_scenario2": psds_score_teacher2_scenario2,
                #"test/student2/event_f1_macro": event_macro_student2,
                #"test/student2/intersection_f1_macro": intersection_f1_macro_student2,
                #"test/teacher2/event_f1_macro": event_macro_teacher2,
                #"test/teacher2/intersection_f1_macro": intersection_f1_macro_teacher2,
            }
            if self.logger is not None:
                self.logger.log_metrics(results)
                self.logger.log_hyperparams(self.hparams, results)

            for key in results.keys():
                self.log(key, results[key], prog_bar=True, logger=False)

    def configure_optimizers(self):
        return [self.opt1, self.opt2], [self.scheduler1, self.scheduler2]

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader
