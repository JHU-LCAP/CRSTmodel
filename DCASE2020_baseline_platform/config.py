import logging
import math
import os
import pandas as pd
import numpy as np

dataspace = "/home/Databases/DESED/"
workspace = ".."
# DESED Paths
weak = os.path.join(dataspace, 'dcase2019/dataset/metadata/train/weak.tsv')
unlabel = os.path.join(dataspace, 'dcase2019/dataset/metadata/train/unlabel_in_domain.tsv')
synthetic = os.path.join(dataspace, 'dataset/metadata/train/synthetic20/soundscapes.tsv')
validation = os.path.join(dataspace, 'dcase2019/dataset/metadata/validation/validation.tsv')
test2018 = os.path.join(dataspace, 'dataset/metadata/validation/test_dcase2018.tsv')
eval2018 = os.path.join(dataspace, 'dataset/metadata/validation/eval_dcase2018.tsv')
eval_desed = os.path.join(dataspace, "dataset/metadata/eval/public.tsv")
# Useful because does not correspond to the tsv file path (metadata replace by audio), (due to subsets test/eval2018)
audio_validation_dir = os.path.join(dataspace, 'dcase2019/dataset/audio/validation')
# Separated data
weak_ss = os.path.join(dataspace, 'weaklabel_speech')
unlabel_ss = os.path.join(dataspace, 'unlabel_speech')
synthetic_ss = os.path.join(dataspace, 'dataset/audio/train/synthetic20/separated_sources')
validation_ss = os.path.join(dataspace, 'dataset/audio/validation_ss/separated_sources')
eval_desed_ss = os.path.join(dataspace, "dataset/audio/eval/public_ss/separated_sources")

# Scaling data
scaler_type = "dataset"

# Data preparation
ref_db = -55
sample_rate = 16000
max_len_seconds =10.
# features
n_window = 2048 #1024
hop_size = 255  #323
n_mels = 128    #64
max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)
mel_f_min = 0.
mel_f_max = 8000.

# Model
max_consistency_cost = 2.0
max_rampup_weight    = 3.0 # 1.0

# Training
in_memory = True
in_memory_unlab = False
num_workers = 12
batch_size = 24
noise_snr = 30

n_epoch = 300
n_epoch_rampup = 50
n_epoch_rampup2 = 100

checkpoint_epochs = 1
save_best = True
early_stopping = None
es_init_wait = 50  # es for early stopping
adjust_lr = True
max_learning_rate = 0.001  # Used if adjust_lr is True
default_learning_rate = 0.001  # Used if adjust_lr is False

# Post processing
median_window_s = 0.45

# Classes
file_path = os.path.abspath(os.path.dirname(__file__))
classes = pd.read_csv(os.path.join(file_path, validation), sep="\t").event_label.dropna().sort_values().unique()
nClass = len(classes)
# Logger
terminal_level = logging.INFO

# Make class label
tlab = np.diag(np.ones(nClass),-1)[:,:-1]
bag = [tlab]
for iter in range(1,nClass):
    temp = np.diag(np.ones(nClass)) + np.diag(np.ones(nClass),iter)[:nClass,:nClass]
    bag.append(temp[:nClass-iter,:])
for iter in range(1,nClass):
    for jter in range(1,nClass-iter):
        temp = np.diag(np.ones(nClass)) + np.diag(np.ones(nClass),iter)[:nClass, :nClass] + np.diag(np.ones(nClass),iter+jter)[:nClass,:nClass]
        bag.append(temp[:nClass-(iter+jter),:])
class_label = np.concatenate(bag,0)
nComs = class_label.shape[0]

#temp = []
#for iter in range(157):
#    temp.append(np.reshape(class_label,(1,nComs,nClass)))
#class_label_ext = np.concatenate(temp,0)


