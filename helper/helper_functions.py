import math
import numpy as np
import scipy.linalg as sl
import numpy.matlib as nl
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC

def vec2image(vec, row, col):
    return vec.reshape((row, col))

### put original_ in order to use svd at moresvd version
def original_svd(mat, k, labels=[], method='none'):
    """ 
    Apply SVD to mat and calculus u, s, and v matrices.
    returns k number of features
    """
    u,s,v = np.linalg.svd(mat, full_matrices=0)

    # if it is not forward propagation
    if len(labels) == 0:
        return u[:,:k], s[:k], v[:k,:]

    # find a matrix x for weights
    v_size = v.shape[1]
    labels_size = labels.shape[0]
    num_sub = int(v_size/labels_size)

    # extend labels according to the mat
    labels_ext = np.zeros((labels.shape[0]*num_sub, labels.shape[1]), dtype=np.float64)

    print('labels:',labels.shape, 'v:',v.shape) # (10000, 10), (subsize^2, num_sub * 10000)

    for i in range(labels_size):
        for j in range(num_sub):
            labels_ext[i*num_sub + j, :] = labels[i, :] # labels_ext: (num_sub * 10000, 10)

    """
    calculate weight using correlation <<< labels must be not one hot encoded
    """
    # initialize x
    x = np.zeros((v.shape[0]), dtype=np.float64)

    # v transpose: (num_sub*10000, subsize^2) labels_ext: (num_sub*10000, 1)
    # x : (subsize^2)
    if labels_ext.shape[1] != 1:
        print('\n===========================================')
        print(".      DON'T USE ONE HOT ENCODED LABELS.     ")
        print('===========================================\n')

    print('v shape:', v[0, :].ravel().shape, 'labels:', labels_ext.ravel().shape)
    # calculate correlation per singular vector
    for i in range(v.shape[0]):
        x[i] = np.correlate(v[i, :].ravel(), labels_ext.ravel())[0]

    print('x:', x)
    sum_x_sort = np.argsort(np.abs(x))

    """ 
    calculate weight using backslash <<< labels must be one hot encoded
    """
    # x = backslash(v.transpose(), labels_ext) # shape: (subsize^2, 10)

    # sum by rows and sort from the smallest to the largest
    # sum_x_sort = np.argsort(np.sum(x, axis=1))

    # sum_x_sort = np.argsort(np.sum(np.abs(x), axis=1))
    
    # sum_x_sort = np.argsort(np.max(np.abs(x), axis=1))

    print('\nsum: {}\n'.format(sum_x_sort))

    u_return = np.zeros_like(u[:,:k])
    s_return = np.zeros_like(s[:k])
    v_return = np.zeros_like(v[:k,:])

    for i in range(k):
        u_return[:, i] = u[:, sum_x_sort[-(i+1)]]
        s_return[i]    = s[sum_x_sort[-(i+1)]]
        v_return[i, :] = v[sum_x_sort[-(i+1)], :]

    return u_return, s_return, v_return

### in moresvd version, no need to extend labels since v matrix is 10,000
def svd(mat, k, labels=[], method='none'):
    """ 
    Apply SVD to mat and calculus u, s, and v matrices.
    returns k number of features
    """
    u,s,v = np.linalg.svd(mat, full_matrices=0)

    # if it is not forward propagation
    if len(labels) == 0:
        return u[:,:k], s[:k], v[:k,:]

    # find a matrix x for weights
    v_size = v.shape[1]
    labels_size = labels.shape[0]
    num_sub = int(v_size/labels_size)

    # print('labels:',labels.shape, 'v:',v.shape) # (10000, 10), (subsize^2, 10000)

    """

    calculate weight using correlation <<< labels must not be one hot encoded

    """

    # initialize x
    x = np.zeros((v.shape[0]), dtype=np.float64)

    # v transpose: (10000, subsize^2) labels_ext: (10000, 1) x : (subsize^2)

    if labels.shape[1] != 1:
        print('\n===========================================')
        print(".      DON'T USE ONE HOT ENCODED LABELS.     ")
        print('===========================================\n')

    # print('v shape:', v[0, :].ravel().shape, 'labels:', labels_ext.ravel().shape)
    # calculate correlation per singular vector
    for i in range(v.shape[0]):
        x[i] = np.correlate(v[i, :].ravel(), labels.ravel())[0]

    # print('x:', x)
    sum_x_sort = np.argsort(np.abs(x))

    print("sum_x_sort:",sum_x_sort)
 
    """ 

    calculate weight using backslash <<< labels must be one hot encoded

    """

    # x = backslash(v.transpose(), labels) # shape: (subsize^2, 10)

    # # sum by rows and sort from the smallest to the largest
    # if method == 'abs':
    #     sum_x_sort = np.argsort(np.abs(x))
    # elif method == 'sum':
    #     sum_x_sort = np.argsort(np.sum(x, axis=1))
    # elif method == 'sumabs':
    #     sum_x_sort = np.argsort(np.sum(np.abs(x), axis=1))
    # elif method == 'maxabs':
    #     sum_x_sort = np.argsort(np.max(np.abs(x), axis=1))

    """ 

    calculate weight using svm <<< labels must not be one hot encoded
    IT TAKES TOO LONG

    """

    # svc = SVC(kernel='linear')    
    # svc.fit(v.transpose(), labels)

    # x = svc.support_vectors_ # (10000, 25)

    # # sum by rows and sort from the smallest to the largest
    # if method == 'sum':
    #     sum_x_sort = np.argsort(np.sum(x, axis=0))
    # elif method == 'sumabs':
    #     sum_x_sort = np.argsort(np.sum(np.abs(x), axis=0))
    # elif method == 'maxabs':
    #     sum_x_sort = np.argsort(np.max(np.abs(x), axis=0))

    ######################################################################
    # below is the same for all methods

    u_return = np.zeros_like(u[:,:k])
    s_return = np.zeros_like(s[:k])
    v_return = np.zeros_like(v[:k,:])

    for i in range(k):
        print("u shape:",u[:, sum_x_sort[-(i+1)]].shape)
        print("u_return:", u_return.shape)
        u_return[:, i] = u[:, sum_x_sort[-(i+1)]].ravel()
        s_return[i]    = s[sum_x_sort[-(i+1)]].ravel()
        v_return[i, :] = v[sum_x_sort[-(i+1)], :].ravel()

    return u_return, s_return, v_return

def take_subimages_new(A, subimSize, stride, indices=False, random_sampling=0):
    """ 
    Take sub images from a 2D image
    and convert them to a 2D matrix
    """
    # this will take the subimages
    n=subimSize
    N=A.shape[0]

    nsubs = N-n+1
    subinds_x = []
    subinds_y = []    
    for j in range(nsubs):
        for i in range(nsubs):
            [X,Y]=np.meshgrid([k+i for k in range(n)],[k+j for k in range(n)])
            subinds_x.append(X.ravel())
            subinds_y.append(Y.ravel())

    subims = A.T[subinds_x, subinds_y]
    return subims.T

def take_subimages_stack(A, subimSize, stride, indices=False, random_sampling=0):
    """ 
    Take sub images from a 2D image
    and convert them to a 2D matrix
    """
    # this will take the subimages
    n=subimSize
    N=A.shape[1]

    nsubs = N-n+1
    subinds_x = []
    subinds_y = []    
    for j in range(nsubs):
        for i in range(nsubs):
            [X,Y]=np.meshgrid([k+i for k in range(n)],[k+j for k in range(n)])
            subinds_x.append(X.ravel())
            subinds_y.append(Y.ravel())

    subims = A[:, subinds_y, subinds_x]
    return np.swapaxes(subims, 1, 2)

def take_subimages_combined(A, subimSize, stride, indices=False, random_sampling=0):
    """ 
    Take sub images from a 2D image
    and convert them to a 2D matrix
    """
    # this will take the subimages
    n=subimSize
    dataN = A.shape[0]
    N=A.shape[1]

    nsubs = N-n+1
    subinds_x = []
    subinds_y = []    
    subinds_z = []
    for j in range(nsubs):
        for i in range(nsubs):
            [X,Y,Z]=np.meshgrid([k+i for k in range(n)],[k+j for k in range(n)],[k for k in range(dataN)])
            subinds_x.append(X.ravel())
            subinds_y.append(Y.ravel())
            subinds_z.append(Z.ravel())

    subims = A[subinds_z, subinds_y, subinds_x]
    # return np.swapaxes(subims, 1, 2)    
    return subims.T

def backslash(A, B):
    """ A*x = B, x = B\A """
    return np.matmul(np.linalg.pinv(A), B)

def dist(a, b):
    """ find the distance between two points """
    return np.sqrt(math.pow(a[0,0]-b[0,0],2)+math.pow(a[1,0]-b[1,0],2))

def calc_rmse(error):
    """ error must be an array """
    s = 0
    for i in range(len(error)):
        s += error[i] * error[i]
    s /= len(error)
    return np.sqrt(s)

def pool_2d(image, pool='max', pool_size=2):
    """
    image must be a numpy array
    """
    # check if a row and a column of the image are even numbers
    # if not then add paddings
    img_col = image.shape[0]
    img_row = image.shape[1]
    
    new_col = img_col
    new_row = img_row
    
    if img_col % pool_size != 0:
        new_col = new_col + pool_size - new_col % pool_size
    if img_row % pool_size != 0:
        new_row = new_row + pool_size - new_row % pool_size

    # initiate the new image matrix
    new_img = np.zeros((new_col, new_row), dtype=np.float64)
    new_img[:img_col, :img_row] = np.copy(image)
    result = np.zeros((int(new_col/pool_size), int(new_row/pool_size)), dtype=np.float64)
    
    for r in range(0,new_col,pool_size):
        for c in range(0,new_row,pool_size):
            if pool.lower() == 'max':
                result[int(c/pool_size), int(r/pool_size)] = np.max(new_img[c:c+pool_size,r:r+pool_size])
            elif pool.lower() == 'absmax':
                ## TODO FIX THIS!
                small_img = new_img[c:c+pool_size,r:r+pool_size]
                if np.max(small_img) > - np.min(small_img):
                    result[int(c/pool_size), int(r/pool_size)] = np.max(small_img)
                else:
                    result[int(c/pool_size), int(r/pool_size)] = np.min(small_img)
            elif pool.lower() == 'newabsmax':
                small_img = new_img[c:c+pool_size,r:r+pool_size]
                absmax = np.max(np.abs(small_img))
                if absmax in small_img:
                    result[int(c/pool_size), int(r/pool_size)] = absmax
                else:
                    result[int(c/pool_size), int(r/pool_size)] = - absmax
            elif pool.lower() == 'average':
                result[int(c/pool_size), int(r/pool_size)] = np.average(new_img[c:c+pool_size,r:r+pool_size])
            elif pool.lower() == 'min':
                small_img = new_img[c:c+pool_size,r:r+pool_size]
                absmin = np.min(np.abs(small_img))
                if absmin in small_img:
                    result[int(c/pool_size), int(r/pool_size)] = absmin
                else:
                    result[int(c/pool_size), int(r/pool_size)] = - absmin
    return result

def pool_3d(image, pool='max', pool_size=2):
    """
    image must be a numpy array
    """
    # check if a row and a column of the image are even numbers
    # if not then add paddings
    img_col = image.shape[0]
    img_row = image.shape[1]
    depth   = image.shape[2]
    
    new_col = np.copy(img_col)
    new_row = np.copy(img_row)
    
    if img_col % pool_size != 0:
        new_col = new_col + pool_size - new_col % pool_size
    if img_row % pool_size != 0:
        new_row = new_row + pool_size - new_row % pool_size

    # initiate the new image matrix
    new_img = np.zeros((new_col, new_row, depth), dtype=np.float64)
    new_img[:img_col, :img_row, :] = np.copy(image)
    result = np.zeros((int(new_col/pool_size), int(new_row/pool_size), depth), dtype=np.float64)
    
    for r in range(0,new_col,pool_size):
        for c in range(0,new_row,pool_size):

            if pool.lower() == 'max':
                result[int(c/pool_size), int(r/pool_size), :] = np.max(new_img[c:c+pool_size,r:r+pool_size, :])

            elif pool.lower() == 'absmax':
                result[int(c/pool_size), int(r/pool_size), :] = np.max(np.abs(new_img[c:c+pool_size,r:r+pool_size, :]))

            elif pool.lower() == 'newabsmax':
                small_img = new_img[c:c+pool_size, r:r+pool_size, :]
                absmax = np.max(np.abs(new_img[c:c+pool_size,r:r+pool_size, :]))
                if absmax in small_img:
                    result[int(c/pool_size), int(r/pool_size), :] = absmax
                else:
                    result[int(c/pool_size), int(r/pool_size), :] = - absmax

            elif pool.lower() == 'average':
                result[int(c/pool_size), int(r/pool_size), :] = np.average(new_img[c:c+pool_size,r:r+pool_size,:])

    return result

def LDA(data, k=None, labels=[]):
    """
    LDA    - Linear Discriminatnt Analysis projection of a labeled data set
    data   - n-by-N matrix representing a aset of N points in R^n
    labels - 1-by-N vector of categorical labels containing C unique values
    """

    # there must be labels when using LDA

    assert(len(labels)>0)

    # data shape. : (subsize * subsize, num_sub * 10000)
    # labels shape: (10000, 1)
    n = data.shape[0] # (subsize * subsize)
    C = len(np.unique(labels))

    # extend labels
    num_data = data.shape[1]
    num_labels = labels.shape[0] # 10000
    num_sub = int(data.shape[1] / num_labels)

    labels_ext = np.zeros((num_data, 1), dtype=np.float64)
    for i in range(num_labels):
        for j in range(num_sub):
            labels_ext[i*num_sub + j] = labels[i]

    categoryMeans = np.zeros((n,C), dtype=np.float64)

    for i in range(C):
        # Find all the indices with label i
        iinds = np.argwhere(labels_ext[:,0]==i)
        # Find the mean of the data points with the i-th label
        categoryMeans[:,i] = np.mean(data[:,iinds],axis=1).ravel()
        # Subtract the i-th mean from each data point with label i
        data[:,iinds] -= nl.repmat(categoryMeans[:,i].reshape((-1,1)),1,len(iinds)).reshape((data[:,iinds].shape))

    sigmab = np.cov(categoryMeans)  # Cov matrix of the means
    sigma  = np.cov(data)           # Cov matrix of the centered data
    L, U = sl.eig(sigmab, sigma)    # Solve the generalized eigenvlaue problem
                                    # A*u=lambda*B*u, u=eigenvector, lambda=eigenvalue
                                    # A = sigmab, B = sigma

    L = np.real(L)

    sinds = np.argsort(np.abs(L))[::-1] # Sort the eigenvalues
    U = U[:, sinds]
    L = L[sinds]

    return U[:,:k], L[:k]


print('------ HELPER   FUNCTIONS IMPORTED ------')
