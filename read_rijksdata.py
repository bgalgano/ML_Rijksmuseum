#!/usr/bin/env python
# coding: utf-8

# In[25]:


import scipy.io
import h5py
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf

#from google.colab import drive
#drive.mount('/content/drive')os.chdir('/content/drive/MyDrive/Colab Notebooks/rijkmuseum')
# In[27]:

def progess_bar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 40, fill = '█', printEnd = "\r"):

    percent = ("{:}/{:}").format(iteration,total)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent} {suffix}', end = printEnd)

def load_data(img_folder='out_img',labels_file='labels.txt', names_file='names.txt'):

    #Read in labels
    labels = pd.read_csv('labels.txt',delimiter = ',',header=None,engine='python')
    names = pd.read_csv('names.txt',delimiter = '/t',header=None,engine='python')
    labels = labels.to_numpy(dtype = 'int')
    labels = np.squeeze(labels)
    
    ##Load in Art
    numart = labels.shape[0] 

    #Create dummy array
    artlog = np.zeros((numart,56,56,3),dtype = 'int')

    #Get addresses
    img_filenames = os.listdir(img_folder)
    art_addr = []
    for ifn in img_filenames:
        art_addr.append(img_folder + '/' + ifn)

    #Load in pieces one at a time
    for x in range(numart):
        newart = cv2.imread(art_addr[x])
        progess_bar(iteration=x, total=numart)

        if len(newart.shape) == 2:
            newart = np.expand_dims(newart,axis=2)
            newart = np.concatenate((newart,newart,newart),axis=2)
            artlog[x,:,:,:] = newart


    #Create Shuffle
    mixer = np.arange(numart)
    np.random.shuffle(mixer)

    #Apply Shuffle
    artlog = artlog[mixer,:,:,:]
    labels = labels[mixer]


    ## Remove Anonymous/Unknown Author Art (If you wish. Would probably help performance but reduces data by 15%)
    named_art = (labels <= (names.shape[0] - 2))
    artlog = artlog[named_art,:,:,:]
    labels = labels[named_art]
    numart = labels.shape[0]

    ## Split Data

    """
    
    """
    #Declare Split Fractions
    testfrac = 0.1
    valfrac = 0.1

    #Divide into Splits
    test_train_split = round((1-(testfrac+valfrac)) * artlog.shape[0])
    test_val_split = round((1-(valfrac)) * artlog.shape[0])

    artlog_train = artlog[:test_train_split,:,:,:]
    artlog_test = artlog[test_train_split:test_val_split,:,:,:]
    artlog_val = artlog[test_val_split:,:,:,:]

    labels_train = labels[:test_train_split]
    labels_test = labels[test_train_split:test_val_split]
    labels_val = labels[test_val_split:]

    ## Direct Name Labels (If you want them)
    names = np.array(names)
    namelabels_train = names[labels_train]
    namelabels_test = names[labels_test]
    namelabels_val = names[labels_val]

    # set proper variable names and return
    X_train = artlog_train
    Y_train = labels_train
    label_train = namelabels_train
    
    X_val = artlog_val
    Y_val = labels_val
    label_val = namelabels_val
    return X_train, Y_train, label_train, X_val, Y_val, label_val

