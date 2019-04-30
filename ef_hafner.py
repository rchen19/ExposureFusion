"""
CS6476 2018Spring
Final Project
Exposure Fusion: Hafner method
@rchen
email: ranchen@gatech.edu
Ref:
D. Hafner and J. Weickert, 
"Variational Exposure Fusion with Optimal Local Contrast" 
in Scale Space and Variational Methods in Computer Vision, 2015, pp. 425-436.
"""
import numpy as np
import cv2
import os
from itertools import product
from util import *
#import pandas as pd
#from numba import jit
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

# I/O directories
input_dir = "images"
output_dir = "output"


def project_simplex(weights):
    """
    Parameters
    ----------
    weights: list<np.float64>
        
    Returns
    -------
    new_weights: list<np.float64>
    """
    S = sorted(weights, reverse=True)
    #s = weights
    #s.sort(reverse=True)
    m = max([j for j in range(len(S)) if S[j]-(1./(j+1)) * (sum(S[:j+1]) -1 ) > 0])
    theta = (1./(m+1)) * (sum(S[:m+1]) -1)
    return [max(w-theta, 0) for w in weights ]

def project_simplex_stack1(weights):
    """
    Parameters
    ----------
    numpy.ndarray(dtype=np.float64)
        3D array: num_images X image_shape[0] X image_shape[1]
        
    Returns
    -------
    new_weights: list<np.float64>
    """
    num_images = weights.shape[0]
    h,w = weights.shape[1:]
    S = np.sort(weights, axis=0)[::-1]
    #s = weights
    #s.sort(reverse=True)
    J = np.array([1./(j+1) for j in range(num_images)])
    J = np.repeat(J[:, np.newaxis], h, axis=1)
    J = np.repeat(J[:, :, np.newaxis], w, axis=2)
    M = S - J * np.cumsum(S, axis=0)
    M = (M>0).astype(int)
    M_reverse = M[::-1]
    M_reverse = np.argmax(M_reverse, axis=0)
    M = num_images -1 - M_reverse
    M_2 = np.zeros((num_images,h,w), dtype=np.float64)
    for i, j in product(range(h), range(w)):
        M_2[:M[i,j]+1, i, j] = 1.0
    #m = max([j for j in range(num_images) if S[j]-(1./(j+1)) * (sum(S[:j+1]) -1 ) > 0])
    #theta = (1./(m+1)) * (sum(S[:m+1]) -1)
    theta = (1./(M+1)) * ((S*M_2).sum(axis=0)-1)
    new_weights = weights - theta
    new_weights[new_weights<0] = 0
    return new_weights

def initialize_weights(num_images, image_shape):
    """
    Parameters
    ----------
    num_images: int, number of images

    image_shape: 2D dimension of the original images

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        3D array: num_images X image_shape[0] X image_shape[1]
    """
    assert len(image_shape) == 2, "wrong image shape"
    W = np.ones((num_images, image_shape[0], image_shape[1]), dtype=np.float64)
    W_sum = W.sum(axis=0)
    W_sum[W_sum==0] = 1e10
    W = W / W_sum
    return W

def converge_check():
    return True

def minimize_energy(image_stack, t=0.1, alpha=1.0, beta=1.0, gamma=0.25, lamd=0.1, mju=None, sigma=None):
    num_images, h, w, _ = image_stack.shape
    if not mju: mju = image_stack.mean()
    if not sigma: sigma = np.sqrt(h**2 + w**2) * 0.1
    image_stack = image_stack.astype(np.float64) / 255.0
    fB = image_stack[:,:,:,0]
    fG = image_stack[:,:,:,1]
    fR = image_stack[:,:,:,2]
    images = list(image_stack)
    fY = 0.299*fR + 0.587*fG+0.114*fB
    fCr = (fR - fY) * 0.713 - 0.5
    fCb = (fB - fY) * 0.564 - 0.5
    
    fY_mean = fY.mean(axis=0)
    weights = initialize_weights(num_images, (h,w))
    k = 0
    while k <= 1000:
        uY = (fY * weights).sum(axis=0)
        uCb = (fCb * weights).sum(axis=0)
        uCr = (fCr * weights).sum(axis=0)
        phi = np.sqrt(lamb**2 + )
        for i in range(num_images):
            weights[i] = weights[i]

    return weights

if __name__ == "__main__":

    W = initialize_weights(2, (500,500))
    print W

    print project_simplex_stack1(W)