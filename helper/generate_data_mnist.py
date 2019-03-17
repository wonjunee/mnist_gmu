import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data

def mnist_generate_data(reduced=False, train_size=1000, padding=True):
	'''
	If reduced is false then train_size and test_size won't matter. It will use the full MNIST data set.
	If reduced is true then it will reduce the size of train and test.
	'''

	print()
	mnist = input_data.read_data_sets("../MNIST_data/", reshape=False)

	X_train, y_train           = mnist.train.images, mnist.train.labels
	X_test, y_test             = mnist.test.images, mnist.test.labels

	assert(len(X_train) == len(y_train))
	assert(len(X_test) == len(y_test))

	if padding:
		# Pad images with 0s
		X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
		X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

		if not reduced:
			X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')

	# Reassign matrices and shuffle
	X_train, y_train = shuffle(X_train, y_train)

	if not reduced:
		X_validation, y_validation = mnist.validation.images, mnist.validation.labels
		assert(len(X_validation) == len(y_validation))

	if reduced:
		print("\nTaking {} per digit for train data".format(train_size))

		# Initialize
		X_train = X_train[:train_size*10]
		y_train = y_train[:train_size*10]

	y_train = y_train.reshape((y_train.shape[0], 1))
	y_test = y_test.reshape((y_test.shape[0], 1))

	y_train_onehot = np.zeros((len(y_train),10))
	y_test_onehot = np.zeros((len(y_test), 10))

	for i in range(len(y_train)):
		y_train_onehot[i, int(y_train[i])] = 1

	for i in range(len(y_test)):
		y_test_onehot[i, int(y_test[i])] = 1

	print('y train one hot encoding:', y_train_onehot.shape)
	print('y test one hot encoding:', y_test_onehot.shape)

	print()
	print("Training Set:  ",X_train.shape)

	if not reduced:
		print("Validation Set:",X_validation.shape)

	print("Test Set:      ",X_test.shape)
	print("Done!")

	return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot