# Cross-Referencing Self-Training (CRST) model for Sound Event Detection
This is an implementation of cross-referencing self-training approach described in a paper "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures" submitted to IEEE Transactions on Multimedia.
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

# Description of each main file
A Convolutional Recurrent Neural Network (CRNN) is trained with:
 1) Mean Teacher (MT) approach [1] in "main_MT_model.py". (This is the DCASE challenge baseline for task 4 in 2020 [2].)
 2) Interpolation Consistency Training (ICT) approach [3] in "main_ICT_model.py".
 3) Self-Referencing Self-Training (SRST) approach [4] in "main_SRST_model.py".
 4) Cross-Referencing Self-Training (CRST) approach [5] in "main_CRST_model.py".

There are two different version of perturbation for CRST model. For a fair comparision to the challenge baseline, adding Gaussian noise is applied in "main_CRST_model.py".
In version 2 ("main_CRST_model_v2.py"), frame-shifting is applied to data perturbation in CRST model.

# Reference
[1] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Advances in Neural Information Processing Systems, vol. 2017-Decem, no. Nips, 2017, pp. 1196–1205.

[2] N. Turpault and R. Serizel, “Training Sound Event Detection On A Heterogeneous Dataset,” in DCASE workshop, 2020.

[3] V. Verma, A. Lamb, J. Kannala, Y. Bengio, and D. Lopez-Paz, “Interpolation Consistency Training for Semi-supervised Learning,” in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, vol. 2019-Augus. California: International Joint Conferences on Artificial Intelligence Organization, 8 2019, pp. 3635–3641.

[4] S. Park, A. Bellur, D. K. Han, and M. Elhilali, “Self-training for Sound Event Detection in Audio Mixtures,” in proc. of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 341–345.

[5] S. Park, D. K. Han, and M. Elhilali, "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures," arXiv:2105.13392(Online available: https://arxiv.org/abs/2105.13392).

# Citation
S. Park, D. K. Han, and M. Elhilali, "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures," arXiv:2105.13392.
