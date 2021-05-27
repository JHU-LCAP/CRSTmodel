import warnings

import torch.nn as nn
import torch

from models.SDA import SDA

class MultiHead(nn.Module):
    def __init__(self, nHead, outDim, featDim, keyDim, valDim, dropout_sda, dropout=0.5):
        super(MultiHead, self).__init__()
        self.nHead = nHead
        self.featDim = featDim
        self.keyDim = keyDim
        self.valDim = valDim

        self.sda = nn.ModuleList()
        for iter in range(nHead):
            self.sda.append(SDA(featDim, keyDim, valDim, dropout_sda))
        self.dense = nn.Linear(valDim*nHead, outDim)
        self.dropout = nn.Dropout(dropout)

    def load_state_dict(self, state_dict, strict=True):
        for iter in range(self.nHead):
            self.sda[iter].load_state_dict(state_dict['sda{0}'.format(iter)])
        self.dense.load_state_dict(state_dict['dense'])


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {}
        for iter in range(self.nHead):
            state_dict.update({'sda{0}'.format(iter): self.sda[iter].state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)})

        state_dict.update({'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)})
        return state_dict

    def save(self, filename):
        parameters = {}
        for iter in range(self.nHead):
            parameters.update({'sda{0}'.format(iter): self.sda[iter].state_dict()})

        parameters.update({'dense': self.dense.state_dict()})
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_frames, n_feat)
        concat_layer = None
        for iter in range(self.nHead):
            if concat_layer is None:
                concat_layer = self.sda[iter](x)
            else:
                concat_layer = torch.cat((concat_layer, self.sda[iter](x)), dim=-1)

        bs, nfrm, ndim = concat_layer.size()
        out = self.dense(concat_layer)
        out = self.dropout(out)
        return out

if __name__ == '__main__':
    MultiHead(nHead=16, outDim=10, featDim=128, keyDim=256, valDim=256, dropout_sda=0.5)

