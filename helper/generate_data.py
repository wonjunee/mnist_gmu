import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Set the number of samples
n_samples = 100

# Create an empty numpy array for the data and labels
data = np.zeros((n_samples,100,100,1))
labels = []

# radius of a circle
r = 40 

for i in range(n_samples):
    theta = 2 * math.pi / n_samples * i
    x = int(r*math.cos(theta)) + 50  # x coordination
    y = int(-r*math.sin(theta)) + 50 # y coordination
    
    # draw a circle per image from data
    data[i, y:y+2, x:x+2,0] = 1
    # append label values
    labels.append([x, y])
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Split data
# Get randomized datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.30,
    random_state=832289)

print('')
print("Train data shape   :", X_train.shape)
print("Train labels shape :", y_train.shape)
print("Test data shape    :", X_test.shape)
print("Test labels shape  :", y_test.shape)

print("\nX_train, X_tes, y_train, y_test created")