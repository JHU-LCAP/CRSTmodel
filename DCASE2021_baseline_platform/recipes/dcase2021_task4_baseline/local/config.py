import logging
import math
import os
import pandas as pd
import numpy as np

nClass = 10

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


