# Cross-Referencing Self-Training (CRST) model for Sound Event Detection
This is an implementation of cross-referencing self-training approach for a submission to DCASE2021 task 4. sound event detection and separation in domestic environments. 
This approach was implemented based on a baseline for the task4 Sound Event Detection (SED) in DCASE challenge 2021.
Original implementation can be found in https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2021_task4_baseline

# Dependencies
To be updated...
(please refer to original implemtation)

# Dataset
In order to train a network, the challenge (DCASE2021) dataset for SED task was used.
The challenge dataset is categorized into three groups: strong labeled data, weakly labeled data, and unlabeled data and they are available in https://project.inria.fr/desed/.

- training data
 1) weakly labeled data (DESED, real:weakly labeled): 1,578
 2) unlabeled data (DESED, real:unlabeled): 14,412
 3) strong labeled data (DESED, synthetic:training): 10,000
 
- test data
 1) validation data (DESED, real:validation): 1,168
 2) public evaluation data (DESED, real:public evaluation): 692

# Description of each main file
A Convolutional Recurrent Neural Network (CRNN) is trained with:
 1) Mean Teacher (MT) approach [1] in "sed_trainer.py". (This is the DCASE challenge baseline for task 4 in 2021 [2].)
 2) Self-Referencing Self-Training (SRST) approach [3] in "sed_trainer_SRST.py".
 3) Cross-Referencing Self-Training (CRST) approach [4] in "sed_trainer_CRST.py".

# Reference
[1] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Advances in Neural Information Processing Systems, vol. 2017-Decem, no. Nips, 2017, pp. 1196–1205.

[2] https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2021_task4_baseline

[3] S. Park, A. Bellur, D. K. Han, and M. Elhilali, “Self-training for Sound Event Detection in Audio Mixtures,” in proc. of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 341–345.

[4] S. Park, D. K. Han, and M. Elhilali, "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures," arXiv:2105.13392(Online available: https://arxiv.org/abs/2105.13392).

# Citation
S. Park, D. K. Han, and M. Elhilali, "Cross-Referencing Self-Training Network for Sound Event Detection in Audio Mixtures," arXiv:2105.13392.
S. Park, W. Choi, and M. Elhilali, "Sound Event Detection with Cross-Referencing Self-Training," technical report, DCASE2021.
