"""
CS6476 2018Spring
Final Project
Exposure Fusion
@rchen
email: ranchen@gatech.edu
"""
import numpy as np
import cv2
import os
import errno
from util import *
#import pandas as pd
#from numba import jit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ef_mertens as ef_mertens
import ef_ma as ef_ma
#import ef_hafner as ef_hafner
import hdr as hdr

# I/O directories
input_dir = "images"
output_dir = "output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
min_depth = 5

def create_stack(img_dir):
    """
    dir: directory to the images
    return 4D numpy array of dimension N * H * W * 3, where N is the number of images
    """
    image_files = sorted([os.path.join(img_dir, name) for name in os.listdir(img_dir)\
                            if (not name.startswith(".")) and \
                            (os.path.splitext(name)[-1][1:].lower() in EXTENSIONS)])
    stack = [cv2.imread(name) for name in image_files]
    if any([im is None for im in stack]):
        raise RuntimeError("One or more input files failed to load.")
    return np.stack(stack, axis=0)

def read_exposure_times(img_dir):
    assert os.path.isfile(os.path.join(img_dir, "exposure_time.txt")), "exposure time file missing" 
    with open(os.path.join(img_dir, "exposure_time.txt"),'r') as f:
        lines = f.readlines()
        EXPOSURE_TIMES = [eval(line) for line in lines]
    return EXPOSURE_TIMES

def print_files_exposure_time(img_dir):
    image_files = sorted([os.path.join(img_dir, name) for name in os.listdir(img_dir)\
                            if (not name.startswith(".")) and \
                            (os.path.splitext(name)[-1][1:].lower() in EXTENSIONS)])
    exposure_times = read_exposure_times(img_dir)
    assert len(image_files) == len(exposure_times), "number of images is not the same as number of exposure times"
    print "{:^30} {:>15}".format("Filename", "Exposure Time")
    print "\n".join(["{:>30} {:^15.4f}".format(*v)
                     for v in zip(image_files, exposure_times)])

def run_hdr_reinhard(dir_name="memorial", tonemapping="reinhard", save=False):
    
    file_dir = os.path.join(input_dir, dir_name)
    print_files_exposure_time(file_dir)
    img_stack = list(create_stack(file_dir))
    exposure_times = read_exposure_times(file_dir)
    assert len(img_stack) == len(exposure_times), "number of images is not the same as number of exposure times"
    log_exposure_times = np.log(exposure_times)
    hdr_image = hdr.computeHDR(images=img_stack, log_exposure_times=log_exposure_times, tonemapping=tonemapping)
    
    output_subfolder = os.path.join(output_dir, dir_name)
    
    if save:
        try:
            os.makedirs(output_subfolder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        cv2.imwrite(os.path.join(output_subfolder, "output_hdr_reinhard.png"), hdr_image)

    else:
        cv2.imshow('hrd_'+tonemapping, hdr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_ef_mertens(dir_name="memorial", save=False):
    file_dir = os.path.join(input_dir, dir_name)
    img_stack = create_stack(file_dir)
    #print img_stack.shape

    level = int(np.log2(min(img_stack.shape[1:3]))) - min_depth
    print level# = 5

    output_subfolder = os.path.join(output_dir, dir_name)
    if save:
        try:
            os.makedirs(output_subfolder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    for c,s,e in [(33.3,33.3,33.3),]:# (10,20,70), (20,30,50), (30,30,40), (40,50,10), (20,70,10), (70,20,10),]:
        print c,s,e
        #c,s,e = (20, 60, 20)
        weight = ef_mertens.weight_stack(img_stack, wc=c/100., ws=s/100., we=e/100., sigma=0.2)
        #fused_img = fuse_naive(img_stack, weight)
        fused_img = ef_mertens.fuse_pyr(img_stack, weight, level)
        #cv2.imshow('exp_fuse_mertens', fused_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if save:
            cv2.imwrite(os.path.join(output_subfolder, "output_em_mertens.png"), fused_img)
        else:
            cv2.imshow('exp_fuse_mertens', fused_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def run_ef_hafner(dir_name="memorial", save=False): # ignore this part
    file_dir = os.path.join(input_dir, dir_name)
    img_stack = create_stack(file_dir)

def run_ef_ma(dir_name="memorial", save=False): 
    output_subfolder = os.path.join(output_dir, dir_name)
    if save:
        try:
            os.makedirs(output_subfolder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    file_dir = os.path.join(input_dir, dir_name)
    img_stack = create_stack(file_dir)
    num_images, h, w, _ = img_stack.shape
    
    fused_img = ef_ma.fuse(img_stack, patch_size=11, stride=int(11/5.), p=4, sigma_g=0.2, sigma_l=0.5)
    if save:
            cv2.imwrite(os.path.join(output_subfolder, "output_em_ma_fail.png"), fused_img)
    else:
        cv2.imshow('exp_fuse_ma', fused_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    save = False
    dir_names =  [name for name in os.listdir(input_dir)\
                            if (not name.startswith("."))]
    #dir_names = ["sample"]
    #print dir_names
    for dir_name in dir_names:
        print "running Mertens' exposure fusion on {}".format(dir_name)
        run_ef_mertens(dir_name, save)
        print "running Ma's exposure fusion on {}".format(dir_name)
        run_ef_ma(dir_name, save)
        print "running HDR with Reinhard tonemapping on {}".format(dir_name)
        run_hdr_reinhard(dir_name, "reinhard", save)

        


    
