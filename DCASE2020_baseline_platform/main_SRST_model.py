# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_model
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df, median_smoothing
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms


def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    #LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    #We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, adjust_lr=False):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """

    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss(reduction='none')
    mse_criterion = nn.MSELoss(reduction='none')
    reliability_criterion = nn.CrossEntropyLoss(reduction='none')
    softmax = nn.Softmax(dim=1)

    class_label = torch.tensor(cfg.class_label).cuda()
    class_criterion, mse_criterion, softmax = to_cuda_if_available(class_criterion, mse_criterion, softmax)

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()

    #plabel = []
    for i, ((batch_input, batch_input_ema), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup2*len(train_loader))

        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        batch_input, batch_input_ema, target = to_cuda_if_available(batch_input, batch_input_ema, target)

        # Outputs
        strong_pred, weak_pred = model(batch_input)
        strong_predict1, weak_predict1 = ema_model(batch_input)
        strong_predict2, weak_predict2 = ema_model(batch_input_ema)

        strong_predict = (strong_predict1 + strong_predict2)/2
        weak_predict   = (weak_predict1 + weak_predict2)/2
        strong_predict = strong_predict.detach()
        weak_predict   = weak_predict.detach()

     # core for Interpolation Consistency Training (ICT) 
        n_unlabeled = int(3*cfg.batch_size/4)	# mask for unlabeled and weakly labeled data
        unlabeled_data1 = batch_input[:n_unlabeled]
        unlabeled_data2 = batch_input_ema[:n_unlabeled]

        strong_prediction1, weak_prediction1 = ema_model(unlabeled_data1)
        strong_prediction2, weak_prediction2 = ema_model(unlabeled_data2)

        lambda_ = torch.rand(1).cuda()
        mixed_unlabeled_data = lambda_*unlabeled_data1 + (1.0-lambda_)*unlabeled_data2
        mixed_strong_plabel = lambda_*strong_prediction1 + (1.0-lambda_)*strong_prediction2
        mixed_weak_plabel = lambda_*weak_prediction1 + (1.0-lambda_)*weak_prediction2

        strong_prediction_mixed, weak_prediction_mixed = model(mixed_unlabeled_data)

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            temp = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])
            weak_class_loss = temp.mean()

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_weak[mask_weak]} \n "
                          f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")
            meters.update('weak_class_loss', weak_class_loss.item())
            #meters.update('Weak EMA loss', ema_class_loss.mean().item())

        # Strong BCE loss
        if mask_strong is not None:
            temp = class_criterion(strong_pred[mask_strong], target[mask_strong])
            strong_class_loss = temp.mean()
            meters.update('Strong loss', strong_class_loss.item())

        # Teacher-student consistency cost
        if ema_model is not None:
            rampup_weight = cfg.max_rampup_weight * rampup_value
            meters.update('Rampup weight', rampup_weight)

            # Take consistency about strong predictions (all data)
            consistency_loss_strong = rampup_weight * mse_criterion(strong_prediction_mixed, mixed_strong_plabel).mean()
            meters.update('Consistency strong', consistency_loss_strong.item())
            #if loss is not None:
            #    loss += consistency_loss_strong
            #else:
            #    loss = consistency_loss_strong
            #meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = rampup_weight * mse_criterion(weak_prediction_mixed, mixed_weak_plabel).mean()
            meters.update('Consistency weak', consistency_loss_weak.item())
            #if loss is not None:
            #    loss += consistency_loss_weak
            #else:
            #    loss = consistency_loss_weak

        # Self-labeling
        est_strong_target  = torch.zeros(cfg.batch_size,157,cfg.nClass).cuda()
        for bter in range(cfg.batch_size):
            sp = strong_predict[bter]
            sp = torch.clamp(sp, 0.0001, 0.9999)
            p_h1 = torch.log(sp)
            p_h0 = torch.log(1-sp)

            # K = 0
            P0 = p_h0.sum(1)

            # K = 1
            P1 = P0[:,None] + p_h1 - p_h0
            #P  = torch.cat([P0.reshape(157,1), P1], 1)

            # K = 2
            P2 = []
            for cter in range(1,cfg.nClass):
                P2.append(P1[:,:-cter]+P1[:,cter:])
            P2 = torch.cat(P2, 1)
            P2 = P2 - P0[:,None]
            P = torch.cat([P0.reshape(157,1), P1, P2], 1)
            
            # K: up to 3
            #P3 = []
            #for cter1 in range(1,cfg.nClass):
            #    for cter2 in range(1, cfg.nClass-cter1):
            #        P3.append(P1[:,:-(cter1+cter2)]+P1[:,cter1:-cter2]+P1[:,(cter1+cter2):])
            #P3 = torch.cat(P3,1)
            #P3 = P3 - 2*P0[:,None]
            #P  = torch.cat([P0.reshape(157,1), P1, P2, P3], 1)

            P = softmax(P)
            prob_v, prob_i = torch.sort(P, dim=1, descending=True)

            norm_p = prob_v.sum(1)
            prob_v = prob_v/norm_p[:,None]

            cl = class_label[prob_i.tolist(),:]
            cl = torch.mul(cl, prob_v[:,:,None]).sum(1)
            
            est_strong_target[bter,:,:] = torch.squeeze(cl)

        est_weak_target = est_strong_target.mean(1)

        reliability = rampup_weight/class_criterion(est_strong_target[mask_strong], target[mask_strong]).mean()
        reliability = torch.clamp(reliability, 0, 2*rampup_weight)
        meters.update('Reliability of pseudo label', reliability.item())

        # classification error with pseudo label
        pred_strong_loss = mse_criterion(strong_pred[:n_unlabeled], est_strong_target[:n_unlabeled]).mean([1,2])
        pred_weak_loss  = mse_criterion(weak_pred[:n_unlabeled], est_weak_target[:n_unlabeled]).mean(1)

        pred_loss = pred_strong_loss + pred_weak_loss
        expect_loss = reliability * pred_loss.mean()
        meters.update('Expectation of predict loss', expect_loss.item())

        loss = weak_class_loss + strong_class_loss + consistency_loss_strong + consistency_loss_weak + expect_loss
        meters.update('Loss', loss.item())

        if (np.isnan(loss.item()) or loss.item() > 1e5):
            print(loss)
        else:
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if ema_model is not None:
                update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss

def get_dfs(desed_dataset, nb_files=None, separated_sources=False):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                       nb_files=nb_files, download=False)
    log.debug(f"synthetic: {synthetic_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                }

    return data_dfs


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic

    store_dir = os.path.join("stored_data", "MeanTeacher_with_ICT_plabel")
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")

    if os.path.exists(store_dir):
        if os.path.exists(saved_model_dir):
            load_flag = True
        else:
            load_flag = False
            os.makedirs(saved_model_dir, exist_ok=True)
            os.makedirs(saved_pred_dir, exist_ok=True)
    else:
        load_flag = False
        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(saved_model_dir, exist_ok=True)
        os.makedirs(saved_pred_dir, exist_ok=True)

    n_channel = 1
    add_axis_conv = 0

    # Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
    n_layers = 7
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
    pooling_time_ratio = 4  # 2 * 2

    out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    logger.debug(f"median_window: {median_window}")
    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)
    dfs = get_dfs(dataset, reduced_number_of_data)

    # Meta path for psds
    durations_synth = get_durations_df(cfg.synthetic)
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df

    # Normalisation per audio or on the full dataset
    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
        unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms)
        train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms)
        scaler_args = []
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data, train_synth_data]))
        logger.debug(f"scaler mean: {scaler.mean_}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)

    weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms, in_memory=cfg.in_memory_unlab)
    train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms, in_memory=cfg.in_memory)
    valid_synth_data = DataLoadDf(dfs["valid_synthetic"], encod_func, transforms_valid,
                                  return_indexes=True, in_memory=cfg.in_memory)
    logger.debug(f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}")

    if not no_synthetic:
        list_dataset = [weak_data, unlabel_data, train_synth_data]
        batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
        strong_mask = slice((3*cfg.batch_size)//4, cfg.batch_size)
    else:
        list_dataset = [weak_data, unlabel_data]
        batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
        strong_mask = None
    weak_mask = slice(batch_sizes[0])  # Assume weak data is always the first one

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)
    training_loader = DataLoader(concat_dataset, batch_sampler=sampler)
    valid_synth_loader = DataLoader(valid_synth_data, batch_size=cfg.batch_size)

    # ##############
    # Model
    # ##############
    if load_flag:
        mlist = os.listdir(saved_model_dir)
        modelName = mlist[-1]
        n_epoch = np.int(modelName.split('_')[-1]) + 1
        model_fname = os.path.join(saved_model_dir, modelName)
        state = torch.load(model_fname)
        crnn = _load_model(state, 'crnn')
        logger.info(f"training model: {model_fname}, epoch: {state['epoch']}")

        crnn_ema = _load_model(state, 'crnn')
        for param in crnn_ema.parameters():
            param.detach()

        optim_kwargs = state['optimizer']["kwargs"]
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    else:
        n_epoch = 0
        crnn = CRNN(**crnn_kwargs)
        pytorch_total_params = sum(p.numel() for p in crnn.parameters() if p.requires_grad)
        logger.info(crnn)
        logger.info("number of parameters in the model: {}".format(pytorch_total_params))
        crnn.apply(weights_init)

        crnn_ema = CRNN(**crnn_kwargs)
        crnn_ema.apply(weights_init)
        for param in crnn_ema.parameters():
            param.detach_()

        optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)

        state = {
            'model': {"name": crnn.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn.state_dict()},
            'model_ema': {"name": crnn_ema.__class__.__name__,
                          'args': '',
                          "kwargs": crnn_kwargs,
                          'state_dict': crnn_ema.state_dict()},
            'optimizer': {"name": optim.__class__.__name__,
                          'args': '',
                          "kwargs": optim_kwargs,
                          'state_dict': optim.state_dict()},
            "pooling_time_ratio": pooling_time_ratio,
            "scaler": {
                "type": type(scaler).__name__,
                "args": scaler_args,
                "state_dict": scaler.state_dict()},
            "many_hot_encoder": many_hot_encoder.state_dict(),
            "median_window": median_window,
            "desed": dataset.state_dict()
        }

    save_best_cb = SaveBest("sup")
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    # ##############
    # Train
    # ##############
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(n_epoch, n_epoch+cfg.n_epoch):
        crnn.train()
        crnn_ema.train()
        crnn, crnn_ema = to_cuda_if_available(crnn, crnn_ema)

        loss_value = train(training_loader, crnn, optim, epoch,
                           ema_model=crnn_ema, mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=cfg.adjust_lr)

        # Validation
        crnn = crnn.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_loader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      median_window=median_window, save_predictions=None)
        # Validation with synthetic data (dropping feature_filename for psds)
        valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        valid_synth_f1, psds_m_f1 = compute_metrics(predictions, valid_synth, durations_synth)

        # Update state
        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optim.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_synth_f1
        state['valid_f1_psds'] = psds_m_f1

        # Callbacks
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)
            results.loc[epoch, "global_valid"] = valid_synth_f1
        results.loc[epoch, "loss"] = loss_value.item()
        results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

        if cfg.early_stopping:
            if early_stopping_call.apply(valid_synth_f1):
                logger.warn("EARLY STOPPING")
                break

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        crnn = _load_model(state, 'crnn')
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info("testing model of last epoch: {}".format(cfg.n_epoch))
    results_df = pd.DataFrame(results).to_csv(os.path.join(saved_pred_dir, "results.tsv"),
                                              sep="\t", index=False, float_format="%.4f")
    # ##############
    # Validation
    # ##############
    crnn.eval()
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

    validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
    validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
    # Preds with only one value
    valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                        pooling_time_ratio, median_window=median_window,
                                        save_predictions=predicitons_fname)
    compute_metrics(valid_predictions, validation_labels_df, durations_validation)

    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    pred_ss_thresh = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                     pooling_time_ratio, thresholds=list_thresholds, median_window=median_window,
                                     save_predictions=predicitons_fname)
    psds = compute_psds_from_operating_points(pred_ss_thresh, validation_labels_df, durations_validation)
    psds_score(psds, filename_roc_curves=os.path.join(saved_pred_dir, "figures/psds_roc.png"))
