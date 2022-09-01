#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.metrics import DetCurveDisplay
from numpy import linalg as la
import matplotlib.pyplot as plt

path3 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\3\train'
path4 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\4\train'
path6 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\6\train'
path7 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\7\train'
pathz = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\z\train'
path31 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\3\dev'
path41 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\4\dev'
path61 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\6\dev'
path71 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\7\dev'
pathz1 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\z\dev'
paths = [path3, path4, path6, path7, pathz, path31, path41, path61, path71, pathz1]

train3 = []
train4 = []
train6 =[]
train7 =[]
trainz =[]
dev3 =[]
dev4 =[]
dev6 =[]
dev7 =[]
devz =[]
i =0

data = [train3, train4, train6, train7, trainz, dev3, dev4, dev6, dev7, devz]

def read_files(file_path, dataf):
    with open(file_path, 'r') as file:
        mtrx = pd.read_csv(file, sep = ' ', skiprows = 1)
        mtrx.dropna(axis = 1, how = 'any', inplace = True)
        mtrx = np.asmatrix(mtrx)
        dataf.append(mtrx)
for path in paths:
    os.chdir(path)
    for file in os.listdir():
        if file.endswith('.mfcc'):
            file_path =f"{path}/{file}"
            read_files(file_path, data[i])
    i+=1


# In[2]:


def DTW(train, dev):
    rtr, ctr = train.shape
    rdv, cdv = dev.shape
    cost = np.zeros([rtr+1, rdv+1])
    for j in range(rtr+1):
        for k in range(rdv+1):
            cost[j,k] = np.inf
    cost[0,0] = 0
    for i in range(1, rtr+1):
        for j in range(1, rdv+1):
            diff = la.norm(train[i-1,:] - dev[j-1,:])
            min_cost = np.min([cost[i-1, j], cost[i, j-1], cost[i-1, j-1]])
            cost[i, j] = diff + min_cost
    return cost[rtr,rdv]


# In[3]:


train = [train3, train4, train6, train7, trainz]
dev = [dev3, dev4, dev6, dev7, devz]
cnf = np.zeros([60,5])
count = 0
for devdata in dev:
    for i in range(12):
        err = np.zeros([195,2])
        k = 0
        l = 0
        for traindata in train:            
            for j in range(39):
                err[k,0] = DTW(traindata[j], devdata[i])
                err[k,1] = l
                k+=1
            l+=1
        err = err[err[:,0].argsort()]
        for m in range(39):
            x = int(err[m,1])
            cnf[count, x] += 1
        count += 1


# In[48]:


cfmtrx = np.zeros([5,5])
i = 1

for j in range(5):
    for k in range(12):
        rows = cnf[j*12+k,:]
        ind = np.where(rows==max(rows))
        cfmtrx[ind,j] += 1


# In[82]:


roc = np.zeros([39, 2])
det = np.zeros([39, 2])

for rc in range(39):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    count = 0
    for i in range(12):
        if cnf[count,0] > rc:
            TP +=1
        else:
            FN +=1
        
        if cnf[count,1] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,2] > rc:
            FP +=1         
        else:
            TN+=1      
        
        if cnf[count,3] > rc:
            FP +=1
        else:
            TN+=1      
        
        if cnf[count,4] > rc:
            FP +=1
        else:
            TN+=1
        
        count +=1
    
    for i in range(12):
        
        if cnf[count,0] > rc:
            FP +=1
        else:
            TN +=1
        
        if cnf[count,1] > rc:
            TP +=1
        else:
            FN+=1
        
        if cnf[count,2] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,3] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,4] > rc:
            FP +=1
        else:
            TN+=1
        
        count +=1
    
    for i in range(12):
        if cnf[count,0] > rc:
            FP +=1
        else:
            TN +=1
        
        if cnf[count,1] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,2] > rc:
            TP +=1
        else:
            FN+=1
        
        if cnf[count,3] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,4] > rc:
            FP +=1
        else:
            TN+=1
        
        count +=1
    
    for i in range(12):
        if cnf[count,0] > rc:
            FP +=1
        else:
            TN +=1
        
        if cnf[count,1] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,2] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,3] > rc:
            TP +=1
        else:
            FN+=1
        
        if cnf[count,4] > rc:
            FP +=1
        else:
            TN+=1
        
        count +=1
    
    for i in range(12):
        if cnf[count,0] > rc:
            FP +=1
        else:
            TN +=1
        
        if cnf[count,1] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,2] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,3] > rc:
            FP +=1
        else:
            TN+=1
        
        if cnf[count,4] > rc:
            TP +=1
        else:
            FN+=1
        
        count +=1
 
    FPR = FP/(FP + TN)
    TPR = TP/(TP + FN)
    FNR = 1 - TPR
    roc[rc] = ([FPR,TPR])
    det[rc] = ([norm.ppf(FPR),norm.ppf(FNR)])
x = roc[:,0]
y = roc[:,1]
x1 = det[:,0]
y1 = det[:,1]
plt.figure(figsize = (10,10))
plt.plot(x, y, linewidth = 2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve')
plt.legend({'Spoken digits'})
plt.figure(figsize = (10,10))
plt.plot(x1, y1, linewidth = 2)
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('Detection Error Tradeoff Curve')
plt.legend({'Spoken Digits'})


# In[78]:


import seaborn as sns
plt.figure(figsize = (10,10))
catg = ['3','4','6','7','z']
sns.heatmap(cfmtrx, annot = True, xticklabels = catg, yticklabels = catg, cmap = 'Blues')


# In[ ]:




