import numpy as np
import scipy.signal as sp
import wave, struct
import torch
import torch.nn as nn
from scipy.io import wavfile, loadmat
from torchaudio.functional import lfilter
from torchaudio.transforms import Spectrogram


class LinearSpectrogram(nn.Module):
	def __init__(self, nCh=128, n_fft=2048, hop_length=256, win_fn=torch.hamming_window):
		super(LinearSpectrogram, self).__init__()
		self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, window_fn=win_fn)
		self.nCh  = nCh

		nbin = n_fft//2
		fbin = nbin // nCh
		linfilt = torch.zeros([nbin, nCh]).cuda()
		for ch in range(nCh):
			stridx = ch*fbin
			endidx = ch*fbin+fbin
			linfilt[stridx:endidx,ch] = 1.0
		self.lfilter = linfilt

	def forward(self, wavData):
		specData = self.spec(wavData)
		bs, nfrq, nfrm = specData.size(0), specData.size(1), specData.size(2)
		specData = specData[:,1:,:]

		out = torch.matmul(specData.permute(0,2,1),self.lfilter)
		return out.permute(0,2,1)


class AuditorySpectrogram(nn.Module):
	def __init__(self, frmRate=16, tc=8, fac=1, shft=0):
		super(AuditorySpectrogram, self).__init__()
		self.frmRate = frmRate
		self.tc = tc
		self.fac = fac
		self.shft = shft
		self.haircell_tc = 0.5

		cochlear = loadmat('./aud24.mat')
		cochba   = torch.from_numpy(cochlear['COCHBA']).cuda()

		L, M     = cochba.shape
		self.L   = L
		self.M   = M

		A = []
		B = []
		for ch in range(M-1,-1,-1):
			p = torch.real(cochba[0, ch]).to(torch.long)
			B.append(torch.real(cochba[1:p+2, ch]).to(torch.float))
			A.append(torch.imag(cochba[1:p+2, ch]).to(torch.float))

		self.A   = A
		self.B   = B
		self.nCh = len(A)

		alpha = torch.exp(torch.tensor(-1/(tc*2**(4+shft)))).cuda()
		beta  = torch.exp(torch.tensor(-1/(self.haircell_tc*2**(4+shft)))).cuda()
		self.alpha = alpha
		self.L_frm = torch.tensor(frmRate*2**(4+shft)).cuda()

		# hair-cell membrane
		self.hair_a = torch.tensor([1, -beta]).cuda().to(torch.float)
		self.hair_b = torch.tensor([1, 0]).cuda().to(torch.float)

		# temporal integration
		self.temp_a = torch.tensor([1, -alpha]).cuda().to(torch.float)
		self.temp_b = torch.tensor([1,0]).cuda().to(torch.float)


	def forward(self, wavData):
		bs, wavLeng = wavData.size(0), wavData.size(1)

		y1 = lfilter(wavData, self.A[0], self.B[0])
		y2 = torch.sigmoid(y1*self.fac)
	
		# hair cell membrane (low-pass <= 4kHz)
		if not self.fac == -2:
			y2 = lfilter(y2, self.hair_a, self.hair_b)
	
		y2_h       = y2
		y3_h       = 0
		
		#####################################################
		# All other channels
		#####################################################
		audData = []
		for ch in range(self.nCh):
			y1         = lfilter(wavData, self.A[ch], self.B[ch])

			########################################
			# TRANSDUCTION: hair cells
			########################################
			# Fluid cillia coupling (preemphasis) (ignored)
			# ionic channels (sigmoid function)
			y2         = torch.sigmoid(y1*self.fac)

			# hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
			if not self.fac == -2:
				y2 = lfilter(y2, self.hair_a, self.hair_b)
		
			########################################
			# REDUCTION: lateral inhibitory network
			########################################
			# masked by higher (frequency) spatial response
			y3   = y2 - y2_h
			y2_h = y2
		
			# half-wave rectifier ---> y4
			y4 = torch.maximum(torch.tensor(0).cuda(), y3)
		
			# temporal integration window ---> y5
			if self.alpha:	# leaky integration
				y5 = lfilter(y4, self.temp_a, self.temp_b)
				audData.append(y5[:,0:-1:self.L_frm])
			else:		# short-term average
				if L_frm == 1:
					audData.append(y4)
				else:
					audData.append(torch.mean(torch.reshape(y4, [self.L_frm, self.N]), 0))
		audData = torch.stack(audData,2)
		return audData.permute(0,2,1)


def audioread(audioPath):
	FS, wavData = wavfile.read(audioPath)
	maxV        = np.amax(abs(wavData))
	wavData     = wavData/maxV
	return wavData, FS

def wav2aud(batchWave, frmLeng, tc, fac, shft):
	nbatch = batchWave.shape[0]

	# define parameters and load cochlear filter
	cochlear   = loadmat('./aud24.mat')
	COCHBA     = torch.from_numpy(cochlear['COCHBA']).cuda()
	L, M       = COCHBA.shape
	haircell_tc= 0.5
	
	alpha      = torch.exp(torch.tensor(-1/(tc*2**(4+shft)))).cuda()
	beta       = torch.exp(torch.tensor(-1/(haircell_tc*2**(4+shft)))).cuda()
	L_frm      = torch.tensor(frmLeng*2**(4+shft)).cuda()

	batchAud = []
	for bter in range(nbatch):
		wavData = batchWave[bter]

		L_x        = len(wavData)
		N          = torch.ceil(L_x/L_frm).to(torch.long).cuda()
		buff       = torch.zeros([N*L_frm]).cuda()
		buff[:L_x] = wavData
		wavData    = buff
	
		# initialize output
		audData    = torch.zeros([N, M-1]).cuda()
	
		#####################################################
		# Last channel (highest frequency)
		#####################################################
		p          = torch.real(COCHBA[0, M-1]).to(torch.long)
		B          = torch.real(COCHBA[1:p+2, M-1]).to(torch.float)
		A          = torch.imag(COCHBA[1:p+2, M-1]).to(torch.float)
		y1         = lfilter(wavData, A, B)
		y2         = torch.sigmoid(y1*fac)
	
		# hair cell membrane (low-pass <= 4kHz)
		if not fac == -2:
			b  = torch.tensor([1, 0]).cuda().to(torch.float)
			a  = torch.tensor([1, -beta]).cuda().to(torch.float)
			y2 = lfilter(y2, a, b)
	
		y2_h       = y2
		y3_h       = 0
		
		#####################################################
		# All other channels
		#####################################################
		for ch in range(M-2,-1,-1):
			########################################
			# ANALYSIS: cochlear filterbank
			########################################
			# (IIR) filter bank convolution ---> y1
			p          = torch.real(COCHBA[0, ch]).to(torch.long)
			B          = torch.real(COCHBA[1:p+2, ch]).to(torch.float)
			A          = torch.imag(COCHBA[1:p+2, ch]).to(torch.float)
			y1         = lfilter(wavData, A, B)

			########################################
			# TRANSDUCTION: hair cells
			########################################
			# Fluid cillia coupling (preemphasis) (ignored)
			# ionic channels (sigmoid function)
			y2         = torch.sigmoid(y1*fac)

			# hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
			if not fac == -2:
				b  = torch.tensor([1, 0]).cuda().to(torch.float)
				a  = torch.tensor([1, -beta]).cuda().to(torch.float)
				y2 = lfilter(y2, a, b)
		
			########################################
			# REDUCTION: lateral inhibitory network
			########################################
			# masked by higher (frequency) spatial response
			y3   = y2 - y2_h
			y2_h = y2
		
			# half-wave rectifier ---> y4
			y4 = torch.maximum(torch.tensor(0).cuda(), y3)
		
			# temporal integration window ---> y5
			if alpha:	# leaky integration
				b = torch.tensor([1, 0]).cuda().to(torch.float)
				a = torch.tensor([1, -alpha]).cuda().to(torch.float)

				y5 = lfilter(y4, a, b)
				audData[:, ch] = y5[0:-1:L_frm]
			else:		# short-term average
				if L_frm == 1:
					audData[:, ch] = y4
				else:
					audData[:, ch] = torch.mean(torch.reshape(y4, [L_frm, N]), 0)

			batchAud.append(audData)

	batchAud = torch.cat(batchAud, 0).permute(0,2,1)

	return batchAud
		
def sigmoid(x, a):
	x = np.exp(-x/a)
	return 1/(1+x)


def DataNormalization(target, meanV=None, stdV=None):
	nData, nDim = target.shape[0], target.shape[1]
	
	output = np.zeros(shape=[nData, nDim], dtype=float)
	if meanV is None:
		meanV = np.mean(target, axis=0)
		stdV = np.std(target, axis=0, ddof=1)
		for dter in range(nData):
			output[dter,:nDim] = (target[dter,:nDim]-meanV) / stdV
	else:
		for dter in range(nData):
			output[dter,:nDim] = (target[dter,:nDim]-meanV) / stdV
	
	return output, meanV, stdV


def DataRegularization(target):
	nData, nSeq = target.shape[0], target.shape[1]
	for dter in range(nData):
		for ster in range(nSeq):
			temp = target[dter, ster]
			maxV = np.amax(temp)
			minV = np.amin(temp)
			reg_temp = 2*(temp-minV)/(maxV-minV)
			target[dter, ster] = reg_temp - np.mean(reg_temp)

	return target


def weights_init(m):
    """ Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def calc_error(samples, labels):
	batch_size, nSnaps, nDim = list(samples.size())
	_, _, nClass = list(labels.size())
	samples = samples.permute(1,0,2)
	labels  = labels.permute(1,0,2).cpu().numpy()
	cidx    = np.where(labels[0])[1]

	idx = np.arange(nSnaps)
	idx = np.delete(idx, 0)
	v0  = samples[0]
	v1  = samples[idx]
	v   = v1 - v0

	nVec, batch_size, nDim = list(v.size())
	error = None
	for iter in range(nVec):
		idx = np.arange(nVec)
		idx = np.roll(idx, iter)

		v1_norm = torch.norm(v[idx[1]], dim=1)**2
		v2_norm = torch.norm(v[idx[2]], dim=1)**2
		v01_dot = torch.mul(v[idx[0]], v[idx[1]]).sum(1)
		v02_dot = torch.mul(v[idx[0]], v[idx[2]]).sum(1)
		alpha   = torch.div(v01_dot, v1_norm)
		beta    = torch.div(v02_dot, v2_norm)
		n_vec   = v[idx[0]] - torch.mul(alpha[:,None],v[idx[1]]) - torch.mul(beta[:,None],v[idx[2]])
		n_vec_norm = torch.norm(n_vec, dim=1).mean()

		orthogonality = 0
		for cter in range(nClass):
			tidx  = np.where(cidx==cter)[0]
			ntidx = np.arange(batch_size)
			ntidx = np.delete(ntidx, tidx)

			vecs  = v[idx[0]]
			nvec  = torch.norm(vecs, dim=1)
			vecs  = torch.div(vecs, nvec[:,None])

			tvec  = vecs[tidx]
			ntvec = vecs[ntidx].permute(1,0)

			inners = torch.matmul(tvec, ntvec)**2
			orthogonality += inners.mean()

		if error is None:
			error = (n_vec_norm + orthogonality/nClass)
		else:
			error += (n_vec_norm + orthogonality/nClass)

	return error


