import warnings

import numpy as np
import torch.nn as nn
import torch

class SDA(nn.Module):
    def __init__(self, featDim, keyDim, valDim, dropout=0.5):
        super(SDA, self).__init__()
        self.featDim = featDim
        self.keyDim = keyDim
        self.valDim = valDim

        self.Q_dense = nn.Linear(featDim, keyDim)
        self.K_dense = nn.Linear(featDim, keyDim)
        self.V_dense = nn.Linear(featDim, valDim)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

    def load_state_dict(self, state_dict, strict=True):
        self.Q_dense.load_state_dict(state_dict['Q_dense'])
        self.K_dense.load_state_dict(state_dict['K_dense'])
        self.V_dense.load_state_dict(state_dict['V_dense'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {'Q_dense': self.Q_dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'K_dense': self.K_dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'V_dense': self.V_dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'Q_dense': self.Q_dense.state_dict(), 'K_dense': self.K_dense.state_dict(), 'V_dense': self.V_dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_frames, n_feat)
        bs, nfrms, nfeat = x.size()
        Q = self.Q_dense(x)
        K = self.K_dense(x)
        V = self.V_dense(x)

        wlayer = torch.matmul(Q, K.transpose(2,1))/np.sqrt(nfeat)
        wlayer = self.softmax(wlayer)

        out = torch.matmul(wlayer, V)
        out = self.dropout(out)
        return out

if __name__ == '__main__':
    SDA(featDim=128, keyDm=256, valDim=256)
