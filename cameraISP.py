import numpy as np
import cv2
from scipy.interpolate import LinearNDInterpolator

from Camera import *


def compute_fill_factor(imgs):
    '''
    Given K frames, compute time-averaged fill factor fitness.
    Input: imgs: KxWxH array : either RoI or entire Image
    Returns: fill_factor_cost
    '''
    # For one frame
    num_filled = np.asarray([np.count_nonzero(img) for img in imgs])
    fill_factor = np.asarray([(nf/(imgs.shape[1]*imgs.shape[2])) for nf in num_filled])
    fill_factor_cost = np.mean(np.asarray([-np.log(f) for f in fill_factor]))
    return fill_factor_cost


def compute_mean(imgs):
    return mean

def compute_variance(imgs):

    return variance

def compute_energy(imgs):
    '''
    Custom energy function
    '''
    return energy


def compute_fitness(depth, roi = None):
    '''
    Function to compute map fitness across 'k' contiguous frames
    Input:
    depth - (KxMxN) array of K contiguous frames
    roi - Region of interest specified by array indices ()

    '''
    # The fitness function needs to be able to provide a quality metric
    if roi == None:
        roi_depth = depth
    else:
        roi_depth = depth[roi]
    # fitness component 1: Number of holes - Make this differentiable??
    fitness_fill_factor = compute_fill_factor(roi_depth)
    # fitness component 2: Depth map variance
    fitness_map_variance = compute_variance(roi_depth)
    # fitness componene 3: given the variance and fill factor, is there a way to 
    # model the image quality in relation to metrology?
    fitness_metrology = compute_energy(roi_depth)

    return fitness


camera = Camera()
camera.start_camera()
camera.load_preset('HighAccuracyPreset.json')
# Get depth data for 'k' frames
k = 1
try:
    while True:
        depth_frameset = []
        count = 0
        while count < k:
            # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            frames = camera.wait_for_frames()
            depth = np.asarray(frames.get_depth_frame().get_data())
            depth_frameset.append(depth)
            # Get current parameter set
            count = count + 1
        depth_frameset = np.asarray(depth_frameset)
        fill_factor_cost = compute_fill_factor(depth_frameset)
    
finally:
    camera.stop()
