import math
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
    
def predict(test_data, param, U_stack, means_stack=[], display=True, indices=True, pool='none', pool_size=2, random_sampling=0, matrixA='single'):
    """
    predict the output based on U matrices computed above
    """

    if display:
        print('\nPrediction Starts')

    # check the parameters
    layer_size = len(param)
    
    # turning data shape into (image depth, total number of images, image height, image width)
    # for example (3, 10000, 32, 32)
    layer_in   = np.zeros((test_data.shape[3], test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    for i in range(test_data.shape[3]):
        layer_in[i,:,:,:] = test_data[:,:,:,i]
    
    for layer_i in range(layer_size):
        U = U_stack[layer_i]
        if len(means_stack)>0:
            means = means_stack[layer_i]
        else:
            means = []

        if display:
            print('\n----------- Layer {} -----------'.format(layer_i))        
            print('layer {} in shape: {}'.format(layer_i, layer_in.shape))

        """
        takes 4D matrix
        input: k_in, N, size, size
        output: k_in, k, N, num_sub
        """
        subsize, stride, k_post = param[layer_i]

        k_pre    = layer_in.shape[0]
        N        = layer_in.shape[1]
        img_size = layer_in.shape[2]

        if random_sampling == 0:
            row_sub  = int((img_size - subsize) / stride + 1)
        else:
            row_sub = random_sampling

        num_sub  = row_sub*row_sub
        
        layer_out = np.zeros((k_pre, k_post, N, num_sub))

        for i1 in range(k_pre):
            # input: N, size, size
            # output: N, k, num_sub
            if len(means) > 0:
                means_i1 = means[i1]
            else:
                means_i1 = 0

            """
            takes 3D matrix
            input: N, size, size
            output: N, k, num_sub
            """

            layer_in_i1 = layer_in[i1,:,:,:]
            # if indices is true then add 2 more rows
            indices_row = indices*2
            # initialize stack
            img_sub_stack = np.zeros((N,subsize*subsize+indices_row,num_sub))

            for data_i in range(N):
                # take a single image
                img = layer_in_i1[data_i]
                # convert to subimage space
                # take random sampling (if not 0) only at the first layer
                img_sub = take_subimages(img, subsize, stride, indices=indices, random_sampling=random_sampling)
                # assigne matrices
                img_sub_stack[data_i,:,:] = np.copy(img_sub)
            
            # initialize preoutput
            out_pre = np.zeros((N, k_post, num_sub))
            for data_i in range(N):
                out_pre[data_i,:,:] = np.matmul(U[i1,:,:].transpose(), (img_sub_stack[data_i,:,:] - means_i1)) # subtract means

            # swap axes
            layer_out[i1,:,:,:] = np.swapaxes(np.copy(out_pre), 0, 1)

        if display:
            print('layer {} out shape: {}'.format(layer_i, layer_out.shape))
            print('U shape:', U.shape)

        img_size = row_sub

        layer_in = np.copy(layer_out).reshape((k_pre*k_post, N, img_size, img_size))

        if random_sampling == 0:
            if pool.lower() != 'none':
                layer_in_max_pool = np.zeros((k_pre*k_post, N, math.ceil(img_size/pool_size), math.ceil(img_size/pool_size)))
                for i in range(layer_in.shape[0]):
                    for j in range(layer_in.shape[1]):
                        layer_in_max_pool[i,j,:,:] = pool_2d(layer_in[i,j,:,:], pool=pool, pool_size=pool_size)
                layer_in = np.copy(layer_in_max_pool)

        if matrixA == 'combine':
            if layer_i == 0:
                A = _make_matrixA(layer_out, k_pre, k_post, num_sub, N)
            else:
                A_ = _make_matrixA(layer_out, k_pre, k_post, num_sub, N)
                A = np.append(A, A_, axis=0)

        # take random sampling (if not 0) only at the first layer
        random_sampling = 0
        
    num_sub = layer_out.shape[3]
    
    # construct matrix A from the last output
    if matrixA != 'combine':
        A = _make_matrixA(layer_out, k_pre, k_post, num_sub, N)
                
    print('\nA shape:', A.shape)
    
    return A

####################### SVD #######################
def calculate_Ax_svd(A, x, u, s, test_label, display=False, error=False):
    v = np.matmul(np.matmul(np.linalg.inv(np.diag(s)), u.transpose()), A)
    
    # multiply with x matrix
    v = np.append(v, np.ones((1,len(test_label))), axis = 0)
    prediction = np.matmul(v.transpose(), x)
    
    if display:
        print('prediction shape:', prediction.shape)
        print('v shape:', v.shape)
    
    if error:
        for test_ind in range(A.shape[1]):
            if display:
                print('\ntest_ind {}'.format(test_ind))
                print("Prediction:", prediction[test_ind])
                print("Actual:", test_label[test_ind])
                plt.figure(figsize = (4,4))
                plt.plot(prediction[test_ind,0], prediction[test_ind,1], 'ro', label='test')
                plt.plot(test_label[test_ind][0], test_label[test_ind][1], 'yo', label='actual')
                plt.ylim(0,100)
                plt.xlim(0,100)
                plt.legend(bbox_to_anchor=(1.5, 1))
                plt.show()
            error.append(dist(prediction[test_ind].reshape((2,1)), test_label[test_ind].reshape((2,1))))
        
        # return the distance between the prediction and the actual
        return error
    else:
        return prediction

####################### SVM #######################
def predict_svm(test_A, svc, labels, u=[], s=[]):
    if len(u)>0:
        test_A = np.matmul(np.matmul(np.linalg.inv(np.diag(s)), u.transpose()), test_A)
    test_A = test_A.transpose()
    prediction = []
    # iterate through A matrix and predict the value using the model
    for i in range(test_A.shape[0]):
        test_A_i = test_A[i,:].reshape((1,test_A.shape[1]))
        prediction.append(svc.predict(test_A_i))
    return prediction

def predict_svm_multiplesvd(test_A, svc, labels, u_list=[], s_list=[]):
    for i in reversed(range(len(u_list))):
        print('i',i)
        u = u_list[i]
        s = s_list[i]
        test_A = np.matmul(np.matmul(np.linalg.inv(np.diag(s)), u.transpose()), test_A)
    test_A = test_A.transpose()
    # initialize a prediction array
    prediction = []
    # iterate through A matrix and predict the value using the model
    for i in range(test_A.shape[0]):
        test_A_i = test_A[i,:].reshape((1,test_A.shape[1]))
        prediction.append(svc.predict(test_A_i))
    return prediction

print('------ TEST     FUNCTIONS IMPORTED ------')