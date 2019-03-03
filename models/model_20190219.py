import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.datasets
import sklearn.svm

from sklearn.svm import SVC 

import tensorflow as tf

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

def pipeline(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO:
    # use sigmoid activation different activation functions
    # linear functions
    
    # Fully Connected 1. Input = 8192. Output = 4096  
    model = dense(x, 256, mu=mu, sigma=sigma)
    model = leaky_relu(model,0.5)
    model = tf.nn.dropout(model, 0.5)    
    
    # Fully Connected 1. Input = 8192. Output = 4096  
    fc64 = dense(model, 128, mu=mu, sigma=sigma)
    model = leaky_relu(fc64,0.5)     
    
    # Fully Connected 1. Input = 8192. Output = 4096  
    model = dense(model, 64, mu=mu, sigma=sigma)
    model = leaky_relu(model,0.5)
    model = tf.nn.dropout(model, 0.5)    
    
    # Fully Connected 1. Input = 8192. Output = 4096  
    fc16 = dense(model, 32, mu=mu, sigma=sigma)
    model = leaky_relu(fc16,0.5)    
    
     # Fully Connected 1. Inut = 8192. Output = 4096  
    fc6 = dense(model, 16, mu=mu, sigma=sigma)
    model = leaky_relu(fc6, 0.5)
    model = tf.nn.dropout(model, 0.5)    

     # Fully Connected 1. Input = 8192. Output = 4096  
    model = dense(model, 8, mu=mu, sigma=sigma)
    model = leaky_relu(model, 0.5)   
    
    # Fully Connected 1. Input = 8192. Output = 4096  
    fc4 = dense(model, 4, mu=mu, sigma=sigma)
    model = leaky_relu(fc4, 0.5) 
    model = tf.nn.dropout(model, 0.5)    
    
    model = dense(model, 2, mu=mu, sigma=sigma)
    
    return model, fc4, fc6, fc16, fc64



print("Function is ready")