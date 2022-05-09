#!/usr/bin/env python
# coding: utf-8

import scipy.io
import h5py
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf

def progess_bar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 40, fill = '█', printEnd = "\r"):

    percent = ("{:}/{:}").format(iteration,total)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent} {suffix}', end = printEnd)

def load_data(MIN_NUM_ARTWORK=10,img_folder ='/Users/erebor/Downloads/out_img',labels_file='labels.txt',names_file='names.txt'):
    
    #Read in labels
    labels = pd.read_csv(labels_file,delimiter = ',',header=None, engine='python')
    names = pd.read_csv(names_file,delimiter = '/t',header=None, engine='python')
    names = np.array(names)

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
    print("Loading {} artworks...".format(numart))
    for x in range(numart):
        newart = cv2.imread(art_addr[x])
        progess_bar(iteration=x, total=numart)
        if len(newart.shape) == 2:
            newart = np.expand_dims(newart,axis=2)
            newart = np.concatenate((newart,newart,newart),axis=2)
            artlog[x,:,:,:] = newart

    # Shuffle

    # Create Shuffle
    #mixer = np.arange(numart)
    #np.random.shuffle(mixer)

    # Apply Shuffle
    #artlog = artlog[mixer,:,:,:]
    #labels = labels[mixer]

    # Remove Anonymous/Unknown Author Art (If you wish. Would probably help performance but reduces data by 15%)
    print("\n\nThrowing away artwork with no artist...".format(MIN_NUM_ARTWORK))

    named_art = (labels <= (names.shape[0] - 2))
    artlog = artlog[named_art,:,:,:]
    labels = labels[named_art]
    numart = labels.shape[0]

    ## Remove Art from Artists with fewer than 10 pieces
    test_passers = (labels < -2)

    print("\nFiltering for artists with at least {} artworks...".format(MIN_NUM_ARTWORK))
    for x in np.arange(1,names.shape[0]+1):
        artistnum = np.count_nonzero(labels == x)
        progess_bar(iteration=x, total=names.shape[0])

        if artistnum >= MIN_NUM_ARTWORK:
            test_passers = np.logical_or(test_passers,(labels == x))

    artlog = artlog[test_passers,:,:,:]
    labels = labels[test_passers]
    numart = labels.shape[0]
    names_ = names[labels]

    # package and return data
    images = artlog
    depth  = len(list(set(labels)))
    labels_onehot = tf.one_hot(indices=labels,depth=depth)
    names = names_
    print('\n\nDataset loaded!')
    print("images shape:",images.shape)
    print("labels shape:",labels.shape)
    print("labels (one-hot):",labels_onehot.shape)
    print("names shape:",names.shape)
    return images, labels_onehot, labels, names

