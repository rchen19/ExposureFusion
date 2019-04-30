"""
CS6476 2018Spring
Final Project
Exposure Fusion: Mertens method
@rchen
email: ranchen@gatech.edu

"""
import numpy as np
import cv2
import os

#minmax normalization between [0,1] or [0,255]
def normalize_0_1(img):#same dtype as input
    return cv2.normalize(img, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX)

def normalize_0_255(img):#same dtype as input
    return cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def make_stack(images):
    """
    Parameters
    ----------
    images: list<numpy.ndarray(dtype=np.uint8)>
        list of 3D arrays
    
    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        4D numpy array of dimension N * H * W * 3, where N is the number of images
    """
    assert len(images[0].shape)==3, "wrong input shape"
    return np.stack(images, axis=0)

def undo_stack(image_stack):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D numpy array of dimension N * H * W * 3, where N is the number of images
    
    Returns
    -------
    list<numpy.ndarray(dtype=np.uint8)>
    """
    return list(image_stack)