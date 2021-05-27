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
from TestModel_dual import _load_model_v2
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, JSD, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df, median_smoothing
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms, get_transforms_v2


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


def train(train_loader, model1, model2, optimizer1, optimizer2, c_epoch, ema_model1=None, ema_model2=None, mask_weak=None, mask_strong=None, adjust_lr=False):
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
    jsd = JSD()

    softmax = nn.Softmax(dim=1)

    class_label = torch.tensor(cfg.class_label).cuda()
    class_criterion, mse_criterion, softmax = to_cuda_if_available(class_criterion, mse_criterion, softmax)

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()

    #plabel = []
    #for i, (((batch_input, batch_input_ema), target2), target) in enumerate(train_loader):
    for i, (((batch_input, _), _), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup2*len(train_loader))

        if adjust_lr:
            adjust_learning_rate(optimizer1, rampup_value, rampdown_value=1.0)
            adjust_learning_rate(optimizer2, rampup_value, rampdown_value=0.9)
        meters.update('lr', optimizer1.param_groups[0]['lr'])
        #target2 = target2.type(torch.FloatTensor)

        #batch_input, batch_input_ema, target, target2 = to_cuda_if_available(batch_input, batch_input_ema, target, target2)
        batch_input, target = to_cuda_if_available(batch_input, target)

        strong_labeled_data = batch_input[mask_strong]
        strong_label        = target[mask_strong]
        weak_labeled_data   = batch_input[mask_weak]
        weak_label          = target[mask_weak]
        unlabeled_data      = batch_input[6:18,:]
        unlabel             = target[6:18,:]

        # mixup
        c = np.random.beta(0.2,0.2)
        perm = torch.randperm(6)
        strong_mixed_data = c*strong_labeled_data + (1-c)*strong_labeled_data[perm,:]
        strong_mixed_label= torch.clamp(
            strong_label + strong_label[perm,:], min=0, max=1)
        weak_mixed_data   = c*weak_labeled_data   + (1-c)*weak_labeled_data[perm,:]
        weak_mixed_label  = torch.clamp(
            weak_label + weak_label[perm,:], min=0, max=1)

        perm = torch.randperm(12)
        unlabeled_mixed_data = c*unlabeled_data   + (1-c)*unlabeled_data[perm,:]
        
        batch_input_ema = torch.cat(
            (weak_mixed_data, unlabeled_mixed_data, strong_mixed_data), 0)
        target2 = torch.cat(
            (weak_mixed_label, unlabel, strong_mixed_label ),0)

        # Outputs
        strong_pred1, weak_pred1 = model1(batch_input)
        strong_predict1, weak_predict1 = ema_model1(batch_input_ema)
        strong_predict1 = strong_predict1.detach()
        weak_predict1   = weak_predict1.detach()

        # data augmentation    
        strong_pred2, weak_pred2 = model2(batch_input_ema)
        strong_predict2, weak_predict2 = ema_model2(batch_input)
        strong_predict2 = strong_predict2.detach()
        weak_predict2   = weak_predict2.detach()


        # Weak BCE Loss
        target_weak  = target.max(-2)[0]  # Take the max in the time axis
        target2_weak = target2.max(-2)[0]
        if mask_weak is not None:
            weak_class_loss1 = class_criterion(weak_pred1[mask_weak], target_weak[mask_weak]).mean()
            weak_class_loss2 = class_criterion(weak_pred2[mask_weak], target2_weak[mask_weak]).mean()

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_weak[mask_weak]} \n "
                          f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss1: {weak_class_loss1} \t rampup_value: {rampup_value}"
                          f"weak loss2: {weak_class_loss2} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")
            meters.update('weak_class_loss1', weak_class_loss1.item())
            meters.update('weak_class_loss2', weak_class_loss2.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss1 = class_criterion(strong_pred1[mask_strong], target[mask_strong]).mean()
            strong_class_loss2 = class_criterion(strong_pred2[mask_strong], target2[mask_strong]).mean()
            meters.update('Strong loss1', strong_class_loss1.item())
            meters.update('Strong loss2', strong_class_loss2.item())

        # Teacher-student consistency cost
        if ema_model1 is not None:
            rampup_weight = cfg.max_rampup_weight * rampup_value
            meters.update('Rampup weight', rampup_weight)


        # Self-labeling
        n_unlabeled = int(3*cfg.batch_size/4)
        est_strong_target1 = torch.zeros(cfg.batch_size,157,cfg.nClass).cuda()
        est_strong_target2 = torch.zeros(cfg.batch_size,157,cfg.nClass).cuda()
        for bter in range(cfg.batch_size):
            sp1 = strong_predict1[bter]
            sp1 = torch.clamp(sp1, 1.0e-4, 1-1.0e-4)
            p1_h1 = torch.log(sp1)
            p1_h0 = torch.log(1-sp1)

            sp2 = strong_predict2[bter]
            sp2 = torch.clamp(sp2, 1.0e-4, 1-1.0e-4)
            p2_h1 = torch.log(sp2)
            p2_h0 = torch.log(1-sp2)

            p_h0 = torch.cat((p1_h0, p2_h0), 0)
            p_h1 = torch.cat((p1_h1, p2_h1), 0)
            
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
            P = torch.cat([P0.reshape(157*2,1), P1, P2], 1)
            
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
            
            est_strong_target1[bter,:,:] = torch.squeeze(cl[:157,:])
            est_strong_target2[bter,:,:] = torch.squeeze(cl[157:,:])

        est_weak_target1 = est_strong_target1.mean(1)
        est_weak_target2 = est_strong_target2.mean(1)

        strong_reliability1 = rampup_weight*(1-jsd.apply(est_strong_target1[mask_strong], target2[mask_strong]).mean())
        strong_reliability2 = rampup_weight*(1-jsd.apply(est_strong_target2[mask_strong], target[mask_strong]).mean())
        weak_reliability1   = rampup_weight*(1-jsd.apply(est_weak_target1[mask_weak], target2_weak[mask_weak]).mean())
        weak_reliability2   = rampup_weight*(1-jsd.apply(est_weak_target2[mask_weak], target_weak[mask_weak]).mean())

        meters.update('Reliability of pseudo label1', strong_reliability1.item())
        meters.update('Reliability of pseudo label2', strong_reliability2.item())

        # classification error with pseudo label
        pred_strong_loss1 = mse_criterion(strong_pred1[6:n_unlabeled], est_strong_target2[6:n_unlabeled]).mean([1,2])
        pred_weak_loss1   = mse_criterion(strong_pred1[mask_weak], est_strong_target2[mask_weak]).mean([1,2])
        pred_strong_loss2 = mse_criterion(strong_pred2[6:n_unlabeled], est_strong_target1[6:n_unlabeled]).mean([1,2])
        pred_weak_loss2   = mse_criterion(strong_pred2[mask_weak], est_strong_target1[mask_weak]).mean([1,2])

        expect_loss1 = strong_reliability2*pred_strong_loss1.mean() + weak_reliability2*pred_weak_loss1.mean()
        expect_loss2 = strong_reliability1*pred_strong_loss2.mean() + weak_reliability1*pred_weak_loss2.mean()
        meters.update('Expectation of predict loss1', expect_loss1.item())
        meters.update('Expectation of predict loss2', expect_loss2.item())

        loss1 = weak_class_loss1 + strong_class_loss1 + expect_loss1
        loss2 = weak_class_loss2 + strong_class_loss2 + expect_loss2
        meters.update('Loss1', loss1.item())
        meters.update('Loss2', loss2.item())

        if (np.isnan(loss1.item()) or loss1.item() > 1e5):
            print(loss1)
            print(loss2)
        else:
            # compute gradient and do optimizer step
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            global_step += 1
            if ema_model1 is not None:
                update_ema_variables(model1, ema_model1, 0.999, global_step)
            if ema_model2 is not None:
                update_ema_variables(model2, ema_model2, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss1, loss2

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

    store_dir = os.path.join("stored_data", "MeanTeacher_with_dual_v3_mixup6")
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

#    transforms = get_transforms_v2(cfg.max_frames, scaler, add_axis_conv,
#                                shift_dict_params={"net_pooling": 4})
    transforms = get_transforms_v2(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params={"mean":0, "snr": cfg.noise_snr})


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
        crnn1 = _load_model_v2(state, 1, 'crnn')
        crnn2 = _load_model_v2(state, 2, 'crnn')
        logger.info(f"training model: {model_fname}, epoch: {state['epoch']}")

        crnn1_ema = _load_model_v2(state, 1, 'crnn')
        for param in crnn1_ema.parameters():
            param.detach()
        crnn2_ema = _load_model_v2(state, 2, 'crnn')
        for param in crnn2_ema.parameters():
            param.detach()


        optim_kwargs = state['optimizer']["kwargs"]
        optim1 = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn1.parameters()), **optim_kwargs)
        optim2 = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn2.parameters()), **optim_kwargs)
    else:
        n_epoch = 0
        crnn1 = CRNN(**crnn_kwargs)
        crnn2 = CRNN(**crnn_kwargs)
        pytorch_total_params = sum(p.numel() for p in crnn1.parameters() if p.requires_grad)
        logger.info(crnn1)
        logger.info("number of parameters in the model: {}".format(pytorch_total_params))
        crnn1.apply(weights_init)
        crnn2.apply(weights_init)

        crnn1_ema = CRNN(**crnn_kwargs)
        crnn2_ema = CRNN(**crnn_kwargs)
        crnn1_ema.apply(weights_init)
        crnn2_ema.apply(weights_init)
        for param in crnn1_ema.parameters():
            param.detach_()
        for param in crnn2_ema.parameters():
            param.detach_()

        optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
        optim1 = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn1.parameters()), **optim_kwargs)
        optim2 = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn2.parameters()), **optim_kwargs)
        state = {
            'model': {"name1": crnn1.__class__.__name__,
                      "name2": crnn2.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict1': crnn1.state_dict(),
                      'state_dict2': crnn2.state_dict()},
            'model_ema': {"name1": crnn1_ema.__class__.__name__,
                          "name2": crnn2_ema.__class__.__name__,
                          'args': '',
                          "kwargs": crnn_kwargs,
                          'state_dict1': crnn1_ema.state_dict(),
                          'state_dict2': crnn2_ema.state_dict()},
            'optimizer': {"name1": optim1.__class__.__name__,
                          "name2": optim2.__class__.__name__,
                          'args': '',
                          "kwargs": optim_kwargs,
                          'state_dict1': optim1.state_dict(),
                          'state_dict2': optim2.state_dict()},
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
#    save_best_cb = SaveBest("inf")

    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    # ##############
    # Train
    # ##############
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(n_epoch, n_epoch+cfg.n_epoch):
        crnn1.train()
        crnn2.train()
        crnn1_ema.train()
        crnn2_ema.train()
        crnn1, crnn2, crnn1_ema, crnn2_ema = to_cuda_if_available(crnn1, crnn2, crnn1_ema, crnn2_ema)

        loss_value, loss_value2 = train(training_loader, crnn1, crnn2, optim1, optim2, epoch,
                           ema_model1=crnn1_ema, ema_model2=crnn2_ema, mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=cfg.adjust_lr)

        # Validation
        crnn1 = crnn1.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn1, valid_synth_loader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      median_window=median_window, save_predictions=None)
        # Validation with synthetic data (dropping feature_filename for psds)
        valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        valid_synth_f1, psds_m_f1 = compute_metrics(predictions, valid_synth, durations_synth)

        # Update state
        state['model']['state_dict1'] = crnn1.state_dict()
        state['model']['state_dict2'] = crnn2.state_dict()
        state['model_ema']['state_dict1'] = crnn1_ema.state_dict()
        state['model_ema']['state_dict2'] = crnn2_ema.state_dict()
        state['optimizer']['state_dict1'] = optim1.state_dict()
        state['optimizer']['state_dict2'] = optim2.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_synth_f1
        state['valid_f1_psds'] = psds_m_f1

        # Callbacks
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            #stop_criterior = (loss_value.item()+loss_value2.item())/2 + np.abs(loss_value.item()-loss_value2.item())
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

                #crnn1.eval()
                #transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)
                #predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

                #validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
                #validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
                #validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
                #durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
                # Preds with only one value
                #valid_predictions = get_predictions(crnn1, validation_dataloader, many_hot_encoder.decode_strong,
                #                                    pooling_time_ratio, median_window=median_window,
                #                                    save_predictions=predicitons_fname)
                #compute_metrics(valid_predictions, validation_labels_df, durations_validation)

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
        crnn = _load_model_v2(state, 1, 'crnn')
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
#    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
#    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
#    pred_ss_thresh = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
#                                     pooling_time_ratio, thresholds=list_thresholds, median_window=median_window,
#                                     save_predictions=predicitons_fname)
#    psds = compute_psds_from_operating_points(pred_ss_thresh, validation_labels_df, durations_validation)
#    psds_score(psds, filename_roc_curves=os.path.join(saved_pred_dir, "figures/psds_roc.png"))
