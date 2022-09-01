#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.metrics import DetCurveDisplay
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm


patha = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\a\train'
pathba = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\bA\train'
pathda = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\dA\train'
pathla = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\lA\train'
pathta = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\tA\train'
patha1 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\a\dev'
pathba1 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\bA\dev'
pathda1 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\dA\dev'
pathla1 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\lA\dev'
pathta1 = r'C:\Users\visha\Downloads\Acads and Intern\Sem 8\PRML\Assignment 3\tA\dev'
paths = [patha, pathba, pathda, pathla, pathta, patha1, pathba1, pathda1, pathla1, pathta1]

traina = []
trainba = []
trainda =[]
trainla =[]
trainta =[]
deva =[]
devba =[]
devda =[]
devla =[]
devta =[]
i =0

data = [traina, trainba, trainda, trainla, trainta, deva, devba, devda, devla, devta]

def read_files(file_path, dataf):
    with open(file_path, 'r') as file:
        mtrx = pd.read_csv(file, sep = ' ', header = None)
        mtrx = mtrx.to_numpy()
        n = int(mtrx[0,0])
        matrix = np.zeros([n,2])
        for i in range(n):
            matrix[i,0] = mtrx[0,1+2*i]
            matrix[i,1] = mtrx[0,2+2*i]   
        dataf.append(matrix)
for path in paths:
    os.chdir(path)
    for file in os.listdir():
        if file.endswith('.txt'):
            file_path =f"{path}/{file}"
            read_files(file_path, data[i])
    i+=1



# In[3]:


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


# In[4]:


train = [traina, trainba, trainda, trainla, trainta]
dev = [deva, devba, devda, devla, devta]
cnf = np.zeros([100,5])
count = 0
for devdata in dev:
    for i in range(20):
        err = np.zeros([342,2])
        k = 0
        l = 0
        for traindata in train:            
            for j in range(len(traindata)):
                err[k,0] = DTW(traindata[j], devdata[i])
                err[k,1] = l
                k+=1
            l+=1
        err = err[err[:,0].argsort()]
        for m in range(67):
            x = int(err[m,1])
            cnf[count, x] += 1
        count += 1


# In[5]:


cfmtrx = np.zeros([5,5])
i = 1

for j in range(5):
    for k in range(20):
        rows = cnf[j*20+k,:]
        ind = np.where(rows==max(rows))
        cfmtrx[ind,j] += 1

import seaborn as sns
plt.figure(figsize = (10,10))
catg = ['a','bA','dA','lA','tA']
sns.heatmap(cfmtrx, annot = True, xticklabels = catg, yticklabels = catg, cmap = 'Blues')


# In[11]:


roc = np.zeros([67, 2])
det = np.zeros([67, 2])

for rc in range(67):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    count = 0
    for i in range(20):
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
    
    for i in range(20):
        
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
    
    for i in range(20):
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
    
    for i in range(20):
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
    
    for i in range(20):
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
plt.legend({'Handwritten Telugu Characters'})
plt.figure(figsize = (10,10))
plt.plot(x1, y1, linewidth = 2)
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('Detection Error Tradeoff Curve')
plt.legend({'Handwritten Telugu Characters - Unscaled'})

