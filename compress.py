# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:13:21 2020

@author: Solomon
"""


import os
import sys
sys.path.insert(1,'./functions/')
import functions
from matplotlib import pyplot
import matplotlib as mpl

#Set Parameters
K = 3
max_iters = 16

# Load image (.png)
A = mpl.image.imread(os.path.join('images', 'image.png'))
# ==========================================================

# Divide by 255 to map values between 0 and 1
img = A / 255

# Reshape the image into an Nx3 matrix where N = number of pixels.
X = img.reshape(-1, 3)

# Run K-Means
centroids, idx = functions.runkMeans(X, K,
                                 functions.findClosestCentroids,
                                 functions.computeCentroids,
                                 max_iters)

#Map each pixel to centroid value and reshape
X_recovered = centroids[idx, :].reshape(A.shape)

# Display the original and compressed images
fig, ax = pyplot.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(A)
ax[0].set_title('Original',fontsize=22)
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed, with %d colors' % K,fontsize=22)
ax[1].grid(False)