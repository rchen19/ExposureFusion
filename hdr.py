"""
CS6476 2018Spring
Final Project
HDR included as comparison
@rchen
email: ranchen@gatech.edu
Ref:
E. Reinhard, M. Stark, P. Shirley, and J. Ferwerda, 
"Photographic Tone Reproduction for Digital Images,"
in Proceedings of the 29th Annual Conference on Computer Graphics and Interactive Techniques, 
New York, NY, USA, 2002, pp. 267-276.
"""

import numpy as np
import scipy as sp
import cv2
import random


def linearWeight(pixel_value):
    
    z_min, z_max = 0., 255.
    z_mean = np.mean([z_min, z_max])
    # WRITE YOUR CODE HERE.
    pixel_value = np.float64(pixel_value)
    if pixel_value <= z_mean: weight = pixel_value - z_min
    else: weight = z_max - pixel_value

    return np.float64(weight)
    #raise NotImplementedError


def sampleIntensities(images):
    
    
    num_intensities = 256
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    
    mid_img = images[num_images // 2]  # using integer division is arbitrary
    for i in range(0, num_intensities, 1):
        for j in range(0, num_images, 1):
            indices = np.array(np.where(mid_img==i))
            if indices.shape[1] == 0: 
                continue
            else:
                #print indices
                sample = np.random.randint(low=0, high=indices.shape[1])
                intensity_values[i,j] = images[j][tuple(indices[:, sample])]

    return intensity_values



def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [Zmax - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    for i in range(0, intensity_samples.shape[0], 1):
        for j in range(0, intensity_samples.shape[1], 1):
            row_num = i * intensity_samples.shape[1] + j
            #print row_num, i, j
            weight = weighting_function(intensity_samples[i,j])
            mat_A[row_num, intensity_samples[i,j]] = weight
            #mat_A[row_num, num_samples + i] = -weight
            mat_A[row_num, intensity_range +1 + i] = -weight
            mat_b[row_num, 0] = weight * log_exposures[j]


    for k in range(1, 255, 1):
        row_num = num_images * num_samples -1 + k
        weight = weighting_function(k)
        mat_A[row_num, k - 1] = weight * smoothing_lambda
        mat_A[row_num, k] = -2.0 * weight * smoothing_lambda
        mat_A[row_num, k + 1] = weight * smoothing_lambda

    mat_A[-1, (255 - 0) // 2] = 1.

    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)
    g = x[0:intensity_range + 1]

    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):

    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    weighting_verctorized = np.vectorize(weighting_function)
    def g(z):
        return response_curve[z]
    g_vectorized = np.vectorize(g)
    Z = np.array(images)
    W = weighting_verctorized(Z)
    SumW = W.sum(axis=0)

    weighted_average_radiance = (W * (g_vectorized(Z) - log_exposure_times.reshape((-1,1,1)))).sum(axis=0) / SumW
    mid_image_radiance = g_vectorized(images[len(images) // 2]) - log_exposure_times[len(images) // 2]

    #print SumW.shape
    #print img_rad_map.shape
    #print weighted_average_radiance.shape
    #print mid_image_radiance.shape

    img_rad_map[np.where(SumW>0)] = weighted_average_radiance[np.where(SumW>0)]
    img_rad_map[np.where(SumW<=0)] = mid_image_radiance[np.where(SumW<=0)]


    return img_rad_map
    #raise NotImplementedError


def computeHDR(images, log_exposure_times, smoothing_lambda=100., tonemapping="linear"):

    images = map(np.atleast_3d, images)
    num_pixels = images[0].shape[0] * images[0].shape[1]
    num_channels = images[0].shape[2]

    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):

        # Collect the current layer of each input image from
        # the exposure stack
        layer_stack = [img[:, :, channel] for img in images]

        # Sample image intensities
        print "sample image intensities for channel {}".format(channel)
        intensity_samples = sampleIntensities(layer_stack)

        # Compute Response Curve
        print "compute response curve for channel {}".format(channel)
        response_curve = computeResponseCurve(intensity_samples,
                                                  log_exposure_times,
                                                  smoothing_lambda,
                                                  linearWeight)

        # Build radiance map
        print "compute radiance for channel {}".format(channel)
        img_rad_map = computeRadianceMap(layer_stack,
                                             log_exposure_times,
                                             response_curve,
                                             linearWeight)

        if tonemapping == "reinhard":
            print "applying Reinhard tonemapping on channel {}".format(channel)
            alpha = 2.#0.72
            lwhite = 1.0#0.5 #smallest luminance that will be mapped to pure white
            log_mean_radiance = np.exp(img_rad_map.sum() / num_pixels)
            scaled_radiance = alpha * (img_rad_map) / log_mean_radiance
            
            #final_radiance = scaled_radiance / (1 + scaled_radiance) #eq3 in Reinhard paper
            final_radiance = scaled_radiance * (1 + (scaled_radiance / lwhite**2)) / (1 + scaled_radiance) #eq4 in Reinhard paper

            hdr_image[..., channel] = cv2.normalize(final_radiance, alpha=0, beta=255,
                                                    norm_type=cv2.NORM_MINMAX)

        elif tonemapping == "linear":
            print "applying linear tone mapping on channel {}".format(channel)
            hdr_image[..., channel] = cv2.normalize(img_rad_map, alpha=0, beta=255,
                                                    norm_type=cv2.NORM_MINMAX)
        else: print "wrong tone mapping method"

    return np.uint8(hdr_image)


if __name__ == "__main__":
    print "run from main.py"











    
