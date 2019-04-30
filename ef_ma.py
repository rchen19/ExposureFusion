"""
CS6476 2018Spring
Final Project
Exposure Fusion: Hafner method
@rchen
email: ranchen@gatech.edu
Ref:
K. Ma and Z. Wang, 
"Multi-exposure image fusion: A patch-wise approach,"
in 2015 IEEE International Conference on Image Processing (ICIP), 2015, pp. 1717-1721.
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

patch_size = 11
stride = int(patch_size/5.)
p = 4
sigma_g = 0.2
sigma_l = 0.5

"""
for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
"""

def extract_patch_vector_stack(image_stack, i, j, patch_size=11):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D array
    i, j: int
        top left index of a patch
    patch_size: int
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2D array
    """
    num_images, h, w, _ = image_stack.shape
    patch_stack = image_stack[:, i:i+patch_size, j:j+patch_size, :].reshape(num_images, -1)
    return patch_stack / 255.

def mean_value_stack(image_stack):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D array
        
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        1D array
    """
    num_images, h, w, _ = image_stack.shape
    return image_stack.reshape(num_images, -1).mean(axis=1) / 255.

def optimize_patch(patch_stack, global_mean, patch_size=11, p=4, sigma_g=0.2, sigma_l=0.5):
    """
    Parameters
    ----------
    patch_stack: numpy.ndarray(dtype=np.float64)
        2D array
    global_mean: numpy.ndarray(dtype=np.float64)
        1D array
    patch_size: int
    p: positive number
    sigma_g: float
    sigma_l: float
        
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        3D array
    """
    patch_mean = patch_stack.mean(axis=1) #mean intensity, l
    #print patch_mean.mean()
    c = np.linalg.norm(patch_stack - patch_mean[:,np.newaxis], ord=2, axis=1)
    c_2 = c[:]
    c_2[c_2==0] = 1.0
    s = (patch_stack - patch_mean[:,np.newaxis]) / c_2[:,np.newaxis]
    #c1 = np.linalg.norm(patch_stack - patch_mean.reshape(-1,1), ord=2, axis=1)
    #s1 = (patch_stack - patch_mean.reshape(-1,1)) / c.reshape(-1,1)
    c_optimal = c.max()
    #print c_optimal
    s_optimal = ((c**p)[:,np.newaxis] * s).sum(axis=0) / (c**p).sum()
    s_optimal = s_optimal / np.linalg.norm(s_optimal, ord=2)
    #print s_optimal.shape
    l_weight = np.exp(\
                        - ((global_mean-0.5)**2/(2*sigma_g**2)) \
                        - ((patch_mean-0.5)**2/(2*sigma_l**2)) \
                        )
    #print l_weight.sum()
    l_optimal = (l_weight * patch_mean).sum() / l_weight.sum()

    return (c_optimal * s_optimal + l_optimal).reshape(patch_size,patch_size,3)


def fuse(image_stack, patch_size=11, stride=stride, p=4, sigma_g=0.2, sigma_l=0.5):
    """
    Parameters
    ----------
    patch_stack: numpy.ndarray(dtype=np.float64)
        2D array
    patch_size: int
    p: positive number
    sigma_g: float
    sigma_l: float
        
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        3D array
    """
    patch_one = np.ones((patch_size,patch_size, 1), dtype=float)
    num_images, h, w, _ = image_stack.shape
    count = np.zeros((h,w, 1), dtype=float)
    out = np.zeros((h,w, 3), dtype=float)
    global_mean = mean_value_stack(image_stack)
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            patch_stack = extract_patch_vector_stack(image_stack, i, j, patch_size)
            patch_optimal = optimize_patch(patch_stack, global_mean, patch_size, p, sigma_g, sigma_l)
            #if i==300 and j==400:
            #    cv2.imshow('patch_optimal', patch_optimal)
            #    cv2.waitKey(0)
            #    cv2.destroyAllWindows()
            out[i:i+patch_size, j:j+patch_size, :] = out[i:i+patch_size, j:j+patch_size, :] + patch_optimal
            count[i:i+patch_size, j:j+patch_size, :] = count[i:i+patch_size, j:j+patch_size, :] + patch_one
            #if i==h-patch_size: print "yes"
    #print "count", (count==0).sum()
    count[count==0] = 1.0
    return np.uint8(normalize_0_255(out / count))



if __name__ == "__main__":
    print "run from main.py"
