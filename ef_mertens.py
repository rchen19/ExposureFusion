"""
CS6476 2018Spring
Final Project
Exposure Fusion: Mertens method
@rchen
email: ranchen@gatech.edu
Ref:
Mertens T., Kautz J., and Van Reeth F., 
"Exposure Fusion: A Simple and Practical Alternative to High Dynamic Range Photography"
Computer Graphics Forum, vol. 28, no. 1, pp. 161-171, Sep. 2008.
"""
import numpy as np
import cv2
import os
from util import *
#import pandas as pd
#from numba import jit
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

# I/O directories
input_dir = "images"
output_dir = "output"



def contrast(image):
    """
    Parameters
    ----------
    image: numpy.ndarray(dtype=np.uint8)
        3D array
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2D array
    """

    if len(image.shape) == 3: image_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: image_tmp = image.copy()
    #image_tmp = normalize_0_1(image_tmp.astype(np.float64))
    image_tmp = image_tmp.astype(np.float64) / 255.
    C = np.abs(cv2.Laplacian(image_tmp, ddepth=cv2.CV_64F, ksize=3))
    #print C.max()
    return C

def contrast_stack(image_stack):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D numpy array of dimension N * H * W * 3, where N is the number of images
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        3D array
    """
    image_tmp = image_stack.astype(np.float64) / 255.
    if len(image_stack.shape) == 4: 
        image_tmp = 0.299*image_stack[:,:,:,2] + 0.587*image_stack[:,:,:,1] + 0.114*image_stack[:,:,:,0]
    
    #image_tmp = normalize_0_1(image_tmp.astype(np.float64))
    C = [np.abs(cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=3)) for img in image_tmp]
    #print C.max()
    return np.stack(C, axis=0)

def saturation(image):
    """
    Parameters
    ----------
    image: numpy.ndarray(dtype=np.uint8)
        3D array
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2D array
    """
    assert len(image.shape) == 3, "wrong input dimension"
    #image_tmp = normalize_0_1(np.float64(image))
    image_tmp = image.astype(np.float64) / 255.
    S = image_tmp.std(axis=2)
    #print S.max()
    return S

def saturation_stack(image_stack):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D array
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        3D array
    """
    assert len(image_stack.shape) == 4, "wrong input dimension"
    #image_tmp = normalize_0_1(np.float64(image))
    image_tmp = image_stack.astype(np.float64) / 255.
    S = image_tmp.std(axis=3)
    #print S.max()
    return S

def ExpWellness(image, sigma=0.2):
    """
    Parameters
    ----------
    image: numpy.ndarray(dtype=np.uint8)
        3D array
    sigma: float
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2D array
    """
    assert len(image.shape) == 3, "wrong input dimension"
    #image_tmp = normalize_0_1(np.float64(image))
    image_tmp = image.astype(np.float64) / 255.
    #print image_tmp.max()
    E = (np.exp(-((image_tmp-0.5)**2.0)/(2.0*sigma**2.0))).prod(axis=2)
    #E = (np.exp(-((image-127.5)**2.0)/(2.0*sigma**2.0))).prod(axis=2)
    #print E.max()
    return E

def ExpWellness_stack(image_stack, sigma=0.2):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D array
    sigma: float
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        3D array
    """
    assert len(image_stack.shape) == 4, "wrong input dimension"
    #image_tmp = normalize_0_1(np.float64(image))
    image_tmp = image_stack.astype(np.float64) / 255.
    #print image_tmp.max()
    E = (np.exp(-((image_tmp-0.5)**2.0)/(2.0*sigma**2.0))).prod(axis=3)
    #E = (np.exp(-((image-127.5)**2.0)/(2.0*sigma**2.0))).prod(axis=2)
    #print E.max()
    return E

def weight_map(image, wc=1./3., ws=1./3., we=1./3., sigma=0.2):
    """
    Parameters
    ----------
    image: numpy.ndarray(dtype=np.uint8)
        3D array
    wc, ws, we: float

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2D array
    """
    assert len(image.shape) == 3, "wrong input dimension"
    W = contrast(image)**wc * saturation(image)**ws * ExpWellness(image, sigma=sigma)**we
    #print W.max()
    return W

def weight_map_stack(image_stack, wc=1./3., ws=1./3., we=1./3.,sigma=0.2):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D array
    wc, ws, we: float

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        4D array
    """
    assert len(image_stack.shape) == 4, "wrong input dimension"
    W = contrast_stack(image_stack)**wc * saturation_stack(image_stack)**ws * ExpWellness_stack(image_stack, sigma=sigma)**we
    #print W.max()
    W_sum = W.sum(axis=0)
    W_sum[W_sum==0] = 1e10
    W = W / W_sum #normalize over N images
    return np.repeat(W[:, :, :, np.newaxis], 3, axis=3)



def weight_stack(image_stack, wc=1./3., ws=1./3., we=1./3., sigma=0.2):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D numpy array of dimension N * H * W * 3, where N is the number of images
    
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        4D array of dimension N * H * W * 3, where N is the number of images, 
        3 is the number of channels, and all channels are the same
    """
    assert len(image_stack.shape) == 4, "wrong input dimension"
    weight_map_stack = np.stack([weight_map(image, wc, ws, we, sigma) for image in image_stack], axis=0)
    #print weight_map_stack.max()
    weight_map_sum = weight_map_stack.sum(axis=0)
    weight_map_sum[weight_map_sum==0] = 1e10
    weight_map_stack = weight_map_stack / weight_map_sum #normalize over N images
    #weight_map_stack[np.isnan(weight_map_stack)] = 0.
    #return weight_map_stack #single channels
    #below: expand to 3 channels by repeat
    #return np.repeat(np.expand_dims(weight_map_stack, axis=3), 3, axis=3)
    return np.repeat(weight_map_stack[:, :, :, np.newaxis], 3, axis=3)

def fuse_naive(image_stack, weight_stack):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D numpy array of dimension N * H * W * 3, where N is the number of images
    weight_stack: numpy.ndarray(dtype=np.float64)
        3D array of dimension N * H * W, where N is the number of images
    
    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        3D array of dimension H * W * 3
    """
    #naive fusion
    #image_fused = (np.float64(image_stack) * np.expand_dims(weight_stack, axis=3)).sum(axis=0) #if weight_stack is N * H * W
    image_fused = (np.float64(image_stack) * weight_stack).sum(axis=0) #if weight_stack is N * H * W * 3
    image_fused = normalize_0_255(image_fused)
    return np.uint8(image_fused)

def gaussPyramid(image, levels):
    """
    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """
    img_in = np.float64(image)
    #cv2.imshow('dst_rt5', normalize_0_255(img_in))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    pyrd = [img_in]
    for i in range(levels):
        img_out = cv2.pyrDown(img_in)
        #note that pyrDown and pyrUp support multichannel images
        #so the pyramid contains multichannel images at each level
        pyrd.append(img_out)
        img_in = img_out
    #cv2.imshow('dst_rt5', normalize_0_255(img_in))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return pyrd


def laplPyramid(gaussPyr):
    """
    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.

    """

    l_pyr = []
    for i in range(0, len(gaussPyr)-1):
        g_pyr_expanded = cv2.pyrUp(gaussPyr[i+1])
        ####
        #deals with differences in size after expansion
        if g_pyr_expanded.shape[0]-gaussPyr[i].shape[0] ==1:
            g_pyr_expanded = np.delete(g_pyr_expanded, -1, axis = 0)
            #print "lap cropped"
        if g_pyr_expanded.shape[1]-gaussPyr[i].shape[1] ==1:
            g_pyr_expanded = np.delete(g_pyr_expanded, -1, axis = 1) 
            #print "lap cropped" 
        ### 
        l_pyr.append(gaussPyr[i] - g_pyr_expanded)
        #cv2.imshow('dst_rt5', gaussPyr[i] - g_pyr_expanded)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    l_pyr.append(gaussPyr[-1])
    #cv2.imshow('dst_rt5', normalize_0_255(l_pyr[0]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return l_pyr


def blendPyrd(laplPyrds, weightPyrds):
    """
     Parameters
    ----------
    laplPyrds : list<list<numpy.ndarray(dtype=np.float)>>
        A list of laplacian pyramids, each pyramid is a list of numpy arrays.
        Each pyradmid corresponds to one image from the original image stack

    weightPyrds : list<list<numpy.ndarray(dtype=np.float)>>
        A list of pyramids, each pyramid corresponding the Gaussian Pyramid of a weight map.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list, or a pyramid, containing the blended layers of the multiple laplacian pyramids

    """
    assert len(laplPyrds) == len(weightPyrds), "number of image pyramids and the number of weight pyramids are not the same"
    assert len(laplPyrds[0]) == len(weightPyrds[0]), "number of levels in the image pyramids and in the weight pyramids are not the same"
    for i in range(len(laplPyrds)):
        assert laplPyrds[i][0].shape == laplPyrds[0][0].shape, "wrong image dimension for image pyramid {}".format(i)
        assert weightPyrds[i][0].shape == weightPyrds[0][0].shape, "wrong image dimension for weight pyramid {}".format(i)
        assert len(laplPyrds[0]) == len(laplPyrds[i]), "wrong number of levels for image pyramid {}".format(i)
        assert len(weightPyrds[0]) == len(weightPyrds[i]), "wrong number of levels for weight pyramid {}".format(i)
    blended_pyramid = []
    for i in range(len(laplPyrds[0])):
        blended_image = np.zeros(laplPyrds[0][i].shape, dtype=np.float64)
        for j in range(len(laplPyrds)):
            blended_image += laplPyrds[j][i] * weightPyrds[j][i]
        blended_pyramid.append(blended_image)
    #print len(blended_pyramid)
    #print blended_pyramid[-1].shape
    #cv2.imshow('dst_rt5', normalize_0_255((blended_pyramid[-1]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return blended_pyramid

def collapse(pyramid):
    """
    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.

    """

    pyramid_copy = pyramid[:]
    for i in range(len(pyramid_copy)-1, 0, -1):
        pyramid_current_layer_expanded = cv2.pyrUp(pyramid_copy[i])
        pyramid_next_layer = pyramid_copy[i-1]
        if pyramid_current_layer_expanded.shape[0]-pyramid_next_layer.shape[0] ==1:
            pyramid_current_layer_expanded = np.delete(pyramid_current_layer_expanded, -1, axis = 0)
        if pyramid_current_layer_expanded.shape[1]-pyramid_next_layer.shape[1] ==1:
            pyramid_current_layer_expanded = np.delete(pyramid_current_layer_expanded, -1, axis = 1)
        pyramid_copy[i-1] = pyramid_next_layer + pyramid_current_layer_expanded
    return pyramid_copy[0]

def fuse_pyr(image_stack, weight_stack, levels):
    """
    Parameters
    ----------
    image_stack: numpy.ndarray(dtype=np.uint8)
        4D numpy array of dimension N * H * W * 3, where N is the number of images
    weight_stack: numpy.ndarray(dtype=np.float64)
        3D array of dimension N * H * W * 3, where N is the number of images
    levels: int
        levels of pyramid to use
    
    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        3D array of dimension H * W * 3
    """
    #create a list of pyramids
    laplPyrds = []
    for image in image_stack:
        #cv2.imshow('dst_rt5', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        gaussPyr = gaussPyramid(image, levels)
        laplPyr = laplPyramid(gaussPyr)
        laplPyrds.append(laplPyr)

    weightPyrds = []
    for weight in weight_stack:
        #cv2.imshow('dst_rt5', weight)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        gaussPyr = gaussPyramid(weight, levels)
        weightPyrds.append(gaussPyr)

    blended_pyramid = blendPyrd(laplPyrds, weightPyrds)

    final_blend = collapse(blended_pyramid)

    return np.uint8(normalize_0_255(final_blend))






if __name__ == "__main__":
    print "run from main.py"
















