import numpy as np

from Camera import *


def compute_fill_factor(imgs):
    return fill_factor


def compute_mean(imgs):
    return mean

def compute_variance(imgs):
    return variance

def compute_energy(imgs):
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
# Get depth data for 'k' frames
k = 10
count = 0
depth_frameset = []
while count < k:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = camera.wait_for_frames()
        print(frames.get_timestamp())
        depth = frames.get_depth_frame()
        depth_frameset.append(depth)
        # Get current parameter set
        print(camera.get_depth_sensor_params())
        count = count + 1
        # Compute depth loss
        loss = compute_loss(depth)

        if not depth: continue

camera.stop()
exit(0)