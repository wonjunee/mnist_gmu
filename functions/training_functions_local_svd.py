import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from helper.helper_functions import *

def _make_matrixA(layer_out, k_pre, k_post, num_sub, N):
    A = np.zeros((k_pre*k_post*num_sub, N))
    for i1 in range(k_pre):
        for i2 in range(k_post):
            for i3 in range(N):
                A[(i1*k_post+i2)*num_sub:(i1*k_post+i2+1)*num_sub, i3] = np.copy(layer_out[i1,i2,i3,:].ravel())
    return A    
    
def deep_learning_pca(data, param, mean_subtract=True, display=True, indices=True, pool='None', pool_size=2, \
    labels=[], method='none', random_sampling=0, matrixA='single', decomp=['svd','svd','svd']):
    """
    data shape must be (total number of images, image height, image width, image depth)
    for example (10000, 32, 32, 3)
    parameters = [[subsize, stride, k], [subsize, stride, k], ...]
    return matrix A

    random_sampling: the square root of the number of subimages taken from each image.
                     ex) if random_sampling is 3 then 9 random images will be taken
    """    
    layer_size = len(param)

    # turning data shape into (image depth, total number of images, image height, image width)
    # for example (3, 10000, 32, 32)
    layer_in   = np.zeros((data.shape[3], data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[3]):
        layer_in[i,:,:,:] = data[:,:,:,i]
    
    U_stack     = []
    means_stack = []
    
    for layer_i in range(layer_size):
        ### input shape: k_pre x N x 100 x 100
        ### output shape: k_pre x k_post x N x num_sub1
        # setup parameters
        subsize, stride, k_post = param[layer_i]
        k_pre    = layer_in.shape[0]
        N        = layer_in.shape[1]
        img_size = layer_in.shape[2]

        if random_sampling == 0:
            row_sub  = int((img_size - subsize) / stride + 1)
        else:
            row_sub = random_sampling

        num_sub  = row_sub*row_sub

        if display:
            print('\n----------- Layer {} -----------'.format(layer_i))        
            print('image size:    {:3d}'.format(img_size))
            print('subimage size: {:3d}'.format(subsize))
            print('stride:        {:3d}'.format(stride))
            print('k:             {:3d}'.format(k_post))
            print('number of subimages per row:', row_sub)
            print('\nlayer {} in: {}'.format(layer_i, layer_in.shape))

        layer_out = np.zeros((k_pre, k_post, N, num_sub))

        # if indices is true then add 2 more rows
        indices_row = indices*2

        U = np.zeros((k_pre, subsize*subsize + indices_row, k_post))
        means = np.zeros((k_pre))


        for i1 in range(k_pre):
            ### input: N, size, size
            ### output: N, k, num_sub

            # img_sub_combined: subsize*subsize+indices_row,num_sub*N
            #                   841, 16 * N
            img_sub_combined = take_subimages_combined(layer_in[i1,:,:,:], subsize, stride, indices=indices, random_sampling=random_sampling)
            img_sub_combined = img_sub_combined.reshape((subsize*subsize+indices_row,num_sub*N))

            # img_sub_stack   : N,subsize*subsize+indices_row,num_sub
            #                   N, 841, 16
            img_sub_stack = take_subimages_stack(layer_in[i1,:,:,:], subsize, stride, indices=indices, random_sampling=random_sampling)
            
            # Subtract Mean
            if mean_subtract:
                img_mean = np.mean(img_sub_combined)
            else:
                img_mean = 0

            # new matrix with means subtracted
            img_sub_combined_mean_subtract = img_sub_combined - img_mean

            # append mean to means array
            means[i1] = img_mean

            # calculate U matrix from combined
            U[i1, :, :], _, _ = svd(img_sub_combined_mean_subtract, k_post, labels=labels, method=method)
            
            # initialize output
            Ut_subimages = np.zeros((N, k_post, num_sub + indices_row))

            # U transpose x subimage matrix
            for data_i in range(N):
                Ut_subimages[data_i,:,:] = np.matmul(U[i1, :, :].transpose(), (img_sub_stack[data_i,:,:] - img_mean))
            
            # swap axes - after swap: (k_in, k, N, num_sub)
            layer_out[i1,:,:,:] = np.swapaxes(np.copy(Ut_subimages), 0, 1)

        if mean_subtract:
            means_stack.append(means)

        # Append U to U_stack
        U_stack.append(U)

        if display:
            print('layer {} out: {}'.format(layer_i, layer_out.shape))
            print('U shape:', U.shape)

        ### change for the next layer
        # calculate the next image size
        img_size = row_sub

        # k_pre  = layer_out.shape[0]
        # k_post = layer_out.shape[1]    

        layer_in = np.copy(layer_out.reshape((k_pre*k_post, N, img_size, img_size)))

        if random_sampling == 0:
            if pool.lower() != 'none':
                img_size = math.ceil(img_size/pool_size)
                layer_in_max_pool = np.zeros((k_pre*k_post, N, img_size, img_size))
                for i in range(layer_in.shape[0]):
                    for j in range(layer_in.shape[1]):
                        layer_in_max_pool[i,j,:,:] = pool_2d(layer_in[i,j,:,:], pool=pool, pool_size=pool_size)
                layer_in = np.copy(layer_in_max_pool)

        if matrixA == 'combine':
            if layer_i == 0:
                A  = _make_matrixA(layer_out, k_pre, k_post, num_sub, N)
            else:
                A_ = _make_matrixA(layer_out, k_pre, k_post, num_sub, N)
                A = np.append(A, A_, axis=0)
        # take random sampling (if not 0) only at the first layer
        random_sampling = 0        
    
    num_sub = layer_out.shape[3]
    
    # construct matrix A from the last output
    if matrixA != 'combine':
        A = _make_matrixA(layer_out, k_pre, k_post, num_sub, N)

    if display:
        print('\nA shape:', A.shape)
    
    return A, U_stack, means_stack

####################### SVD #######################
def calculate_x_svd(A, k, labels):
    # Calculate usv matrices from matrix A
    u, s, v = svd(A, k) ##<<<<<< delete
    # Add a bias to v matrix
    v = np.append(v, np.ones((1,70)), axis = 0)
    # Solve v x = B where B is true values
    x = backslash(v.transpose(), np.matrix(labels))
    # Return the matrix x
    return x, u, s

####################### SVM #######################
def train_svm(features, labels, kernel="linear"):
    # Use a linear SVC 
    svc = SVC(kernel=kernel)
    t=time.time() # Check the training time for the SVC
    svc.fit(features.transpose(), labels)
    print(round(time.time()-t, 2), 'Seconds to train SVC...')
    return svc


print('------ TRAINING FUNCTIONS IMPORTED ------')