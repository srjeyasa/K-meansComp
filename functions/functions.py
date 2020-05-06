# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:17:25 2020

@author: Solomon
"""
import numpy as np
from matplotlib import pyplot

def findClosestCentroids(X, centroids):
    """
    X: m x n matrix. m = number of examples, n = dimension
    centroids: K x n matrix. K = number of clusters, n = dimension
    returns idx: (m,) vector; m = number of examples. 
        
    Computes the closest centroid for every example.
    Returns vector idx where index of closest centroid to X[i,:] is 
    stored in idx[i]
    
    """
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    prev_dist = 0

    for i in range(X.shape[0]):
        #initialize prev_dist to distance of first centroid
        #idx already initialized to first centroid (0), no need to update
        prev_dist = sum(np.square(X[i,:]-centroids[0,:]))
        for j in range(0,K):
            dist = sum(np.square(X[i,:]-centroids[j,:]))
            if dist < prev_dist:
                prev_dist = dist
                idx[i] = j
    return idx

def computeCentroids(X, idx, K):
    """
    Compute new centroids as the mean of all data points assigned to each 
    centroid.
    
    X: m x n matrix. m = number of examples, n = dimension
    K: number of clusters
    idx: (m,) vector; m = number of examples. 

    returns: centroids: K x n matrix. K = number of clusters, n = dimension
    
    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i,:]=np.sum(X[(idx==i),:],axis=0)/np.sum(idx==i)
    
    return centroids

def runkMeans(X, K, findClosestCentroids, computeCentroids,
              max_iters=10):
    """
    Randomly initializes centroids and runs the K-means algorithm.

    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    #random permutation of m examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    idx = None

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

    return centroids, idx

def displayData(X):
    """
    Displays 2D data stored in X in a grid.
    
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=(10,10))
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')