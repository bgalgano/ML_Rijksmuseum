import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa 

import matplotlib.pylab as plt
import matplotlib as mpl

def load_encoder(input_shape=(56,56,3),vector_length=50,activation='sigmoid'):

    # LOAD PRE-TRAINED ENCODER
    base_kwargs = {'include_top':False,
                   'weights':'imagenet',
                   'input_shape':input_shape,
                   'pooling':None,
                   'classes':vector_length}
    #enet_base = tf.keras.applications.efficientnet.EfficientNetB7(**enet_kwargs)
    base = tf.keras.applications.vgg19.VGG19(**base_kwargs)

    # set that the encoder DOES NOT train on the images
    base.trainable = False

    # set pre-trained model as base
    encoder = tf.keras.models.Sequential()
    encoder.add(base)

    # add two final top layers
    encoder.add(tf.keras.layers.GlobalMaxPooling2D())
    encoder.add(tf.keras.layers.Dense(vector_length, activation=activation)) # last (top) layer of network
    
    return encoder

