#!/usr/bin/env python

import scipy.io
import h5py
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pylab as plt
import matplotlib as mpl
def progess_bar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 40, fill = '█', printEnd = "\r"):

    percent = ("{:}/{:}").format(iteration,total)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} [{bar}] {percent} {suffix}', end = printEnd)

def load_data(MIN_NUM_ARTWORK=10,MAX_NUM_IMG=1000,IMG_SIZE=56,img_folder ='/Users/erebor/Downloads/out_img',labels_file='labels.txt',names_file='names.txt'):

    #Read in labels
    labels = pd.read_csv(labels_file,delimiter = ',',header=None,engine='python')
    names = pd.read_csv(names_file,delimiter = '/t',header=None,engine='python')
    names = np.array(names)
    labels = labels.to_numpy(dtype = 'int')
    labels = np.squeeze(labels)

    ##Load in Art
    numart = labels.shape[0] 

    #Create dummy array
    artlog = np.zeros((numart,IMG_SIZE,IMG_SIZE,3),dtype = 'int')

    #Get addresses
    img_filenames = os.listdir(img_folder)
    img_filenames.sort()
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


    # Remove Art from Artists with fewer than 10 pieces
    test_passers = (labels < -2)

    for x in np.arange(1,names.shape[0]+1):
        artistnum = np.count_nonzero(labels == x)
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
    print("\nimages shape:\t ",images.shape)
    print("labels shape:\t ",labels.shape)
    print("labels (one-hot):",labels_onehot.shape)
    print("names shape:\t ",names.shape)
    
    classes = len(list(set(labels)))
    print('\n              # of unique artists:',classes)

    counts = pd.DataFrame(labels).value_counts()
    print('Min # of artworks for all artists:',min(counts))
    print('      Min # of artworks specified:',MIN_NUM_ARTWORK)
    #labels = labels.reshape(labels.shape[0],1)
    return images, labels_onehot, labels, names

def plot_artwork(images,labels,names,artist_label,n=3):
    # plot a selection of n^2 (nxn) artwork from single artist
    pos_idx = np.where(labels == artist_label)[0]
    artistname = names[artist_label]
    artistimages = images[pos_idx,:,:,:]
    fig, axes = plt.subplots(figsize=(10,10),nrows=n,ncols=n)
    fig.patch.set_facecolor('white')
    i = 0 
    for ax in axes.reshape(-1): 
        ax.imshow(artistimages[i,:,:,:])
        ax.set_xticks([]),ax.set_yticks([])
        i +=1
    plt.suptitle('Artist: {}'.format(artistname),fontsize=15,y=0.95)
    fig.subplots_adjust(top=0.9,wspace=0.1,hspace=0.1)
    
    plt.savefig('figs/samples/artist_{}.png'.format(artistname[0].replace(',','-').replace(' ','-')),dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def get_aggregrate_vectors(vectors,labels,names):
    # Create aggregate vectors
    # Count how many pieces each artist has
    total_bc = np.bincount(labels) # get count of artists
    artcounts = total_bc[np.unique(labels)] # get count of artworks for each unique artist
    artistnames = names[np.unique(labels)] # get the name for each unique artist

    aggregate_vectors = []
    artistnums = []
    for i in range(len(artcounts)):
        progess_bar(iteration=i, total=len(artcounts))

        artistnum = np.unique(labels)[i] #Gets the number that represents this artist from labels
        artistname = artistnames[i]
        artcount = artcounts[i] #Gets number of art pieces by this artist

        artistnums.append(artistnum)
        pos_idx = np.where(labels == artistnum)
        artist_vector = np.mean(vectors[pos_idx],axis=0)

        aggregate_vectors.append(artist_vector)

    aggregate_vectors = np.array(aggregate_vectors)
    
    print('\nDone! Aggregrate vectors for {} artist(s) generated.'.format(aggregate_vectors.shape[0]))
    print('\naggregrate_vectors shape:',aggregate_vectors.shape)
    fname = 'aggregrate_vectors.csv'
    print('saved to:',fname)
    np.savetxt(X=vectors,fname=fname,delimiter=',')
    return aggregate_vectors, artistnames, artistnums

def plot_aggregate_vectors(aggregate_vectors,artistnames,n=3):
    idx = list(range(aggregate_vectors.shape[0]))
    idxs = np.random.choice(a=idx,size=n*n)
    vectors = aggregate_vectors[idxs]
    artists = artistnames[idxs]
    
    fig, axes = plt.subplots(nrows=n,ncols=n,figsize=(n*3,n+2))
    for ax_idx, ax in enumerate(fig.axes):
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax.imshow(np.atleast_2d(vectors[ax_idx]), aspect=n*4, cmap='rainbow', interpolation=None,norm=norm)
        ax.set_xticks([]),ax.set_yticks([])
        ax.set_xlabel(artists[ax_idx][0])
    plt.savefig('figs/aggregrates_vector_sample.png',dpi=200,tight_layout=True)
    plt.suptitle('Artist Aggregrate Vectors',fontsize=n*4,y=0.95)
    plt.show()
    plt.close()
    
def load_vectors(vectors,aggregate_vectors,artistnums,names,labels,train_val_split,dev_split):


    total_bc = np.bincount(labels) # get count of artists
    artcounts = total_bc[np.unique(labels)] # get count of artworks for each unique artist
    artistnames = names[np.unique(labels)] # get the name for each unique artist

    # Generate pairs
    total_pairs = np.zeros(shape=(len(labels)*2,2))
    total_labels = np.zeros(shape=(len(labels)*2,1))

    j = 0
    for i in range(len(artcounts)):
        artistnum = np.unique(labels)[i]
        artcount = artcounts[i]

        #Retreiving negative and positive indices
        pos_idx = np.where(labels == artistnum)[0]
        neg_idx = np.where(labels != artistnum)[0]

        #Adding Positive Pairs for a given artist
        for idx in pos_idx:
            #print('j:',j)
            #print('total pairs:', total_pairs)
            #print('idx:',idx)
            #print('artistnum:',artistnum)
            total_pairs[j,:] = [idx,artistnum]
            total_labels[j] = 1
            j = j + 1

        #Adding Negative Pairs for a given artist
        neg_selec = np.random.choice(neg_idx,artcount,replace=False)
        for idx in neg_selec:
            total_pairs[j,:] = [idx,artistnum]
            total_labels[j] = 0
            j = j + 1

    # Create and Test Split Order
    goodbalance = False
    spltest_labels = labels

    while goodbalance == False:

        #Shuffle data
        mixer = np.arange(len(labels))
        np.random.shuffle(mixer)
        total_pairs = total_pairs[mixer,:]
        total_labels = total_labels[mixer]

        #Make Cutoffs
        train_cutoff = int(len(total_pairs) * train_val_split)
        dev_cutoff = int(len(total_pairs) * (train_val_split + dev_split))

        #Test Splits for Balance
        spltest_labels = spltest_labels[mixer]
        train_spltest = spltest_labels[:train_cutoff]
        dev_spltest = spltest_labels[train_cutoff:dev_cutoff]
        val_spltest = spltest_labels[dev_cutoff:]

        train_bc = np.bincount(train_spltest)
        dev_bc = np.bincount(dev_spltest)
        val_bc = np.bincount(val_spltest)

        check_bool = np.array([],dtype=bool)

        for i in np.unique(labels):
            if (len(np.unique(spltest_labels)) == len(np.unique(train_spltest))) and \
                (len(np.unique(spltest_labels)) == len(np.unique(dev_spltest))) and \
                (len(np.unique(spltest_labels)) == len(np.unique(train_spltest))):

                train_check = abs(total_bc[i]*0.8 - train_bc[i]) >= total_bc[i]*0.2 
                dev_check = abs(total_bc[i]*0.1 - dev_bc[i]) >= total_bc[i]*0.08
                val_check = abs(total_bc[i]*0.1 - val_bc[i]) >= total_bc[i]*0.08 
                check_bool = np.append(check_bool,(train_check or dev_check or val_check))

            elif total_bc[i] != 0:
                check_bool = np.append(check_bool,False)

        if sum(check_bool) <= 0:
            goodbalance = True

    # Turn Pairs of Indices into Pairs of Vectors
    final_pairs = []
    total_pairs = total_pairs.astype(int)
    for total_pair in total_pairs:
        # get query vector index
        vector_idx = int(total_pair[0])

        # get aggregrate index  
        aggregrate_idx = np.where(artistnums==total_pair[-1])[0]
        # get vectors
        aggregrate_vector = aggregate_vectors[aggregrate_idx]
        query_vector = vectors[vector_idx]
        final_pair = np.vstack([aggregrate_vector,query_vector]).T
        final_pairs.append(final_pair)

    final_pairs = np.array(final_pairs)
    # set aggregrate and query image pairs and labels into different
    train_pairs = final_pairs[:train_cutoff,:,:]
    train_labels = total_labels[:train_cutoff]

    dev_pairs = final_pairs[train_cutoff:dev_cutoff,:,:]
    dev_labels = total_labels[train_cutoff:dev_cutoff]

    val_pairs = final_pairs[dev_cutoff:,:,:]
    val_labels = total_labels[dev_cutoff:]
    return train_pairs, train_labels, dev_pairs, dev_labels, val_pairs, val_labels