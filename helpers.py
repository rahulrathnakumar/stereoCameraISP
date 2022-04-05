import numpy as np
import torch


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

def compute_fill_factor(imgs):
    '''
    Given K frames, compute time-averaged fill factor fitness.
    Input: imgs: KxWxH array : either RoI or entire Image
    Returns: fill_factor_cost
    '''
    # For one frame
    num_filled = torch.from_numpy(np.asarray([np.count_nonzero(img) for img in imgs]))
    fill_factor = num_filled/(imgs.shape[1]*imgs.shape[2])
    fill_factor_cost = torch.mean(torch.tensor([-torch.log(f) for f in fill_factor]))
    # fill_factor_cost = np.mean(np.asarray([-np.log(f) for f in fill_factor]))
    return fill_factor_cost



# Function that normalizes paramters
def p_normalize(p, p_ave, p_diff):
    p_norm = 2.0*(p-p_ave)/p_diff
    return p_norm

# Function that un-normalizes parameters
def p_un_normalize(p, p_ave, p_diff):
    p_un_norm = p*p_diff/2.0 + p_ave
    return p_un_norm


def get_frameset(camera, k = 1):
    depth_frameset = []
    color_frameset = []
    count = 0
    while count < k:
        frames = camera.wait_for_frames()
        depth = np.asarray(frames.get_depth_frame().get_data())
        color = np.asarray(frames.get_color_frame().get_data())
        color_frameset.append(color)
        depth_frameset.append(depth)
        # Get current parameter set
        count = count + 1
    depth_frameset = np.asarray(depth_frameset)
    color_frameset = np.asarray(color_frameset)
    return depth_frameset, color_frameset


