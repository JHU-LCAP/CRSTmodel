# Cross-Referencing Self-Training (CRST) model for Sound Event Detection
This is the implementation of cross-referencing self-training approach described in a paper "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures" submitted to IEEE Transactions on Multimedia.
This approach was implemented based on a baseline for the task4 Sound Event Detection (SED) in DCASE challenge 2020.
Original implementation can be found in https://github.com/turpaultn/dcase20_task4.

# Dependencies
Python >= 3.6, pytorch >= 1.0, cudatoolkit>=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2, scaper >= 1.3.5, librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, dcase_util >= 0.2.5, sed-eval >= 0.2.1, psds-eval >= 0.1.0, desed >= 1.1.7

# Dataset
In order to train a network, the challenge dataset for SED task was used.
The challenge dataset is categorized into three groups: strong labeled data, weakly labeled data, and unlabeled data and they are available in https://project.inria.fr/desed/.

- training data
 1) weakly labeled data (DESED, real:weakly labeled): 1,578
 2) unlabeled data (DESED, real:unlabeled): 14,412
 3) strong labeled data (DESED, synthetic:training): 2,595
 
- test data
 1) validation data (DESED, real:validation): 1,168

# Usage
Same with the challenge baseline (DCASE2020).

Step 1. download the dataase

Step 2. modify data path in "config.py"

Step 3. run "main_CRST_model.py"

# Description for different way to train CRNN network
The CRNN network can be trained by:
 1) Mean Teacher (MT) approach [1] in "main_MT_model.py". (This is the DCASE challenge baseline for task 4 in 2020 [2].)
 2) Interpolation Consistency Training (ICT) approach [3] in "main_ICT_model.py".
 3) Self-Referencing Self-Training (SRST) approach [4] in "main_SRST_model.py".

# Reference
[1] 

# Citation
S. Park, D. K. Han, and M. Elhilali, "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures," IEEE Transactions on Multimedia (under review).
