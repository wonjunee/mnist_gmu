import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.datasets
import sklearn.svm

from sklearn.svm import SVC 

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from functions.training_functions_local_svd import *
from functions.test_functions_local_svd import *
from helper.helper_functions import *


def relu(x):
    return tf.nn.relu(x)

def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def dense(x, output_shape, mu=0, sigma=0.1):
    x_shape = x.get_shape().as_list()
    
    fc_W = tf.Variable(tf.truncated_normal(shape=(x_shape[1], output_shape), mean = mu, stddev = sigma))
    fc_b = tf.Variable(tf.zeros(output_shape))
    fc   = tf.matmul(x, fc_W) + fc_b        
    return fc

def conv2D(x, kernel_size, W_input, W_output, mu=0, sigma=0.1):
    conv_W = tf.Variable(tf.truncated_normal(shape=(kernel_size, kernel_size, W_input, W_output), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(W_output))
    conv   = tf.nn.conv2d(x, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
    return conv

def pipeline(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x32.
    model = conv2D(x, kernel_size=5, W_input=1, W_output=6, mu=mu, sigma=sigma)

    # Activation.
    model = tf.nn.relu(model)
        
    # Pooling. Input = 28x28x32. Output = 14x14x32.
    model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 14x14x64.
    model = conv2D(model, kernel_size=5, W_input=6, W_output=16, mu=mu, sigma=sigma)
    
    # Activation.
    model = tf.nn.relu(model)
    
    # Pooling. Input = 14x14x64. Output = 7x7x64.
    model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 7x7x64. Output = 7*7*64.
    model = flatten(model)
    
    # Fully Connected.
    fc6 = dense(model, 1024, mu=mu, sigma=sigma)
    
    # Activation.
    model = tf.nn.relu(model)

    # Fully Connected.
    fc4 = dense(model, 84, mu=mu, sigma=sigma)
    
    # Activation.
    model = tf.nn.relu(model)

    # Fully Connected. Input = 120. Output = 84.
    model = dense(model, 10, mu=mu, sigma=sigma)
    
    return model, fc4, fc6



print("Function is ready")