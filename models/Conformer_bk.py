import warnings

import torch.nn as nn
import torch

from models.CNN import CNN, GLU
from models.MultiHead import MultiHead

class FeedForwardModule(nn.Module):
    def __init__(self, featSize, expDim):
        super(FeedForwardModule, self).__init__()
        self.norm    = nn.LayerNorm(featSize)
        self.dense1  = nn.Linear(featSize[-1], expDim)
        self.sigmoid = nn.Sigmoid()
        self.dense2  = nn.Linear(expDim, featSize[-1])

    def load_state_dict(self, state_dict, strict=True):
        self.dense1.load_state_dict(state_dict["dense1"])
        self.dense2.load_state_dict(state_dict["dense2"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"dense1": self.dense1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense2": self.dense2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'dense1': self.dense1.state_dict(),
                      'dense2': self.dense2.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        x1 = self.norm(x)
        x2 = torch.mul(self.dense1(x1), 0.5)
        x3 = torch.mul(x2, self.sigmoid(x2))
        x4 = torch.mul(self.dense2(x3), 0.5)
        
        return x4

class ConvolutionModule(nn.Module):
    def __init__(self, featSize, pconv_size, dconv_size, pstride, dstride):
        super(ConvolutionModule, self).__init__()
        self.lnorm = nn.LayerNorm(featSize)
        self.pconv1= nn.Conv1d(featSize[-1], featSize[-1], kernel_size=pconv_size, stride=pstride, padding=3)
        self.dconv = nn.Conv1d(featSize[-2], featSize[-2], kernel_size=dconv_size, stride=dstride, padding=3)
        self.bnorm = nn.BatchNorm1d(featSize[-2])
        self.sigmoid = nn.Sigmoid()
        self.pconv2= nn.Conv1d(featSize[-1], featSize[-1], kernel_size=pconv_size, stride=pstride, padding=3)

    def load_state_dict(self, state_dict, strict=True):
        self.pconv1.load_state_dict(state_dict["pconv1"])
        self.pconv2.load_state_dict(state_dict["pconv2"])
        self.dconv.load_state_dict(state_dict["dconv"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"pconv1": self.pconv1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "pconv2": self.pconv2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dconv": self.dconv.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'pconv1': self.pconv1.state_dict(),
                      'pconv2': self.pconv2.state_dict(),
                      'dconv': self.dconv.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        x1 = self.lnorm(x)
        x2 = self.pconv1(x1.permute(0,2,1))
        x3 = self.dconv(x2.permute(0,2,1))
        x4 = self.bnorm(x3)
        x5 = torch.mul(x4, self.sigmoid(x4))
        x6 = self.pconv2(x5.permute(0,2,1))

        return x6.permute(0,2,1)

class ConformerModule(nn.Module):
    def __init__(self, nfrms, nHead=8, outDim=256, featDim=128, keyDim=128, valDim=128, dropout_sda=0.1, dropout=0.5, ks_p=7, ks_d=7, step_p=1, step_d=1, **kwargs):
        super(ConformerModule, self).__init__()
        # feedforward
        self.feed1 = FeedForwardModule([nfrms, featDim], featDim*4)
        # MultiHead
        self.multihead = MultiHead(nHead, outDim, featDim, keyDim, valDim, dropout_sda, dropout)
        # feedforward
        self.feed2 = FeedForwardModule([nfrms, featDim], featDim*4)
        # convolution
        self.conv = ConvolutionModule([nfrms, featDim], ks_p, ks_d, step_p, step_d)
        # layernorm
        self.norm = nn.LayerNorm([nfrms, featDim])

    def load_state_dict(self, state_dict, strict=True):
        self.feed1.load_state_dict(state_dict["feed1"])
        self.multihead.load_state_dict(state_dict["multihead"])
        self.feed2.load_state_dict(state_dict["feed2"])
        self.conv.load_state_dict(state_dict["conv"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"feed1": self.feed1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "multihead": self.multihead.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "feed2": self.feed2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "conv": self.conv.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'feed1': self.feed1.state_dict(),
                      'multihead': self.multihead.state_dict(),
                      'feed2': self.feed2.state_dict(),
                      'conv': self.conv.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # 1st feedforward
        f1 = self.feed1(x)
        f1 = torch.add(x, f1)
        
        # rnn features
        m = self.multihead(f1)
        m = torch.add(f1, m)
        
        # 2nd feedforward
        f2 = self.feed2(m)
        f2 = torch.add(m, f2)

        # convolution
        c = self.conv(f2)
        c = torch.add(f2, c)

        # layer normalization
        n = self.norm(c)

        return n

class Conformer(nn.Module):
    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, nConfs=4, nHead=8, outDim=256, featDim=128, keyDim=128, valDim=128, dropout_sda=0.5, **kwargs):
        super(Conformer, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        n_in_cnn = n_in_channel

        # encoding by CNN
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn

        # conformer
        self.featDim = self.cnn.nb_filters[-1]
        self.nConfs  = nConfs
        self.nFrms   = 124
        self.conformers = nn.ModuleList()
        for iter in range(nConfs):
            self.conformers.append(
                ConformerModule(nfrms=self.nFrms, nHead=nHead, outDim=outDim, featDim=self.featDim, keyDim=keyDim, valDim=valDim, dropout_sda=dropout_sda, dropout=dropout))

        # output layer
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
        self.dense2.load_state_dict(state_dict['dense2'])
        self.dense_softmax.load_state_dict(state_dict['softmax'])
        for iter in range(self.nConfs):
            self.conformers[iter].load_state_dict(state_dict['conformer{0}'.format(iter)])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {}
        for iter in range(self.nConfs):
            state_dict.update({'conformer{0}'.format(iter): self.conformers[iter].state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)})

        state_dict.update({"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)})
        state_dict.update({"dense2": self.dense2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)})
        state_dict.update({"softmax": self.dense_softmax.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)})
        return state_dict

    def save(self, filename):
        parameters = {}
        for iter in range(self.nConfs):
            parameters.update({'conformer{0}'.format(iter): self.conformers[iter].state_dict()})

        parameters.update({'cnn': self.cnn.state_dict()})
        parameters.update({'dense2': self.dense2.state_dict()})
        parameters.update({'softmax': self.dense_softmax.state_dict()})
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

        for iter in range(self.nConfs):
            x = self.conformers[iter](x)

        # output
        strong = self.dense2(x)  # [bs, frames, nclass]
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
