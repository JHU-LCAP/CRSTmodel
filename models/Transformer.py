import warnings

import torch.nn as nn
import torch

from models.CNN import CNN
from models.MultiHead import MultiHead


class Transformer(nn.Module):
    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, nHead=8, outDim=256, featDim=128, keyDim=128, valDim=128, dropout_sda=0.5, **kwargs):
        super(Transformer, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        n_in_cnn = n_in_channel
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn

        # MultiHead
        self.nHead = nHead
        self.featDim = self.cnn.nb_filters[-1]
        self.multihead = MultiHead(nHead, outDim, self.featDim, keyDim, valDim, dropout_sda, dropout)

        self.norm1   = nn.LayerNorm([124,self.featDim])	# hard coding~the number of frames
        self.dense1  = nn.Linear(outDim, outDim)
        self.relu    = nn.ReLU()
        
        self.norm2   = nn.LayerNorm([124,self.featDim])	# hard coding~the number of frames
        self.dense2  = nn.Linear(outDim, nclass)
        self.sigmoid = nn.Sigmoid()
        if attention:
            self.dense_softmax = nn.Linear(outDim, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.multihead.load_state_dict(state_dict["multihead"])
        self.dense1.load_state_dict(state_dict["dense1"])
        self.dense2.load_state_dict(state_dict["dense2"])
        self.dense_softmax.load_state_dict(state_dict["softmax"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "multihead": self.multihead.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense1': self.dense1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense2': self.dense2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'softmax': self.dense_softmax.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(),
                      'multihead': self.multihead.state_dict(),
                      'dense1': self.dense1.state_dict(),
                      'dense2': self.dense2.state_dict(),
                      'softmax': self.dense_softmax.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()

        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        m = self.multihead(x)
        n1 = self.norm1(m+x)

        n2 = self.dense1(n1)
        n2 = self.relu(n2+n1)
        n2 = self.norm2(n2)

        strong = self.dense2(n2)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong, weak


if __name__ == '__main__':
    Transformer(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64], pooling=[(1, 4), (1, 4), (1, 4)])
