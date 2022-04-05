from turtle import color
import numpy as np
import cv2
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from Camera import *
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz


np.seterr(all='raise')

def compute_fill_factor(imgs):
    '''
    Given K frames, compute time-averaged fill factor fitness.
    Input: imgs: KxWxH array : either RoI or entire Image
    Returns: fill_factor_cost
    '''
    # For one frame
    num_filled = np.asarray([np.count_nonzero(img) for img in imgs])
    fill_factor = np.asarray([(nf/(imgs.shape[1]*imgs.shape[2])) for nf in num_filled])
    fill_factor_cost = np.mean(np.asarray([-np.log(f) if f != 0 else 100.0 for f in fill_factor]))
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

def ES_step(p_n,i,cES_now,amplitude):
    # ES step for each parameter
    p_next = np.zeros(nES)
    
    # Loop through each parameter
    for j in np.arange(nES):
        p_next[j] = p_n[j] + amplitude*dtES*np.cos(dtES*i*wES[j]+kES*cES_now)*(aES[j]*wES[j])**0.5
    
        # For each new ES value, check that we stay within min/max constraints
        if p_next[j] < -1.0:
            p_next[j] = -1.0
        if p_next[j] > 1.0:
            p_next[j] = 1.0
            
    # Return the next value
    return p_next

# Function that normalizes paramters
def p_normalize(p):
    p_norm = 2.0*(p-p_ave)/p_diff
    return p_norm

# Function that un-normalizes parameters
def p_un_normalize(p):
    p_un_norm = p*p_diff/2.0 + p_ave
    return p_un_norm


def get_frameset(k = 1):
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



camera = Camera()
camera.start_camera()
camera.load_preset('HighAccuracyPreset.json')
camera.depth_sensor.set_option(rs.option.exposure, 1.)
camera.depth_sensor.set_option(rs.option.gain, 16.0)
camera.depth_sensor.set_option(rs.option.laser_power,0.0)
# Get depth data for 'k' frames
k = 5
ES_steps = 300
noise = np.random.randn(ES_steps)

depth_imgs = []
color_imgs = []
try:
    while True:
        depth_frameset, color_frameset = get_frameset(k = k)
        '''
        Extremum seeking controller parameterization:
        Alexander Scheinker, Auralee Edelen, Dorian Bohler, Claudio Emma, and Alberto Lutman.
        "Demonstration of model-independent control of the longitudinal phase space of electron beams 
        in the linac-coherent light source with femtosecond resolution." 
        Physical review letters 121.4 (2018): 044801.
        https://github.com/alexscheinker/ES_adaptive_optimization
        '''
        p_min, p_max = camera.query_sensor_param_bounds()

        nES = len(p_min)
        p_ave = (p_max + p_min)/2.0
        p_diff = p_max - p_min
        pES = np.zeros([ES_steps,nES])
        pES[0] = camera.get_depth_sensor_params()
        pES_n = np.zeros([ES_steps,nES])
        pES_n[0] = p_normalize(pES[0])
        cES = np.zeros(ES_steps)
        cES[0] = compute_fill_factor(depth_frameset)
        print("Fill Factor Cost:", cES[0])
        wES = np.linspace(1.0,1.75,nES)
        dtES = 2*np.pi/(10*np.max(wES))
        oscillation_size = .35
        aES = wES*(oscillation_size)**2
        kES = 0.1
        decay_rate = 0.99
        amplitude = 1.0
        for i in np.arange(ES_steps-1):
            if i % 100 == 0: 
                winsound.Beep(freq, duration)
                
            # Normalize previous parameter values
            pES_n[i] = p_normalize(pES[i])
            start_time_es = time.time()
            # Take one ES step based on previous cost value
            pES_n[i+1] = ES_step(pES_n[i],i,cES[i],amplitude)
            # Un-normalize to physical parameter values
            pES[i+1] = p_un_normalize(pES_n[i+1])
            print(camera.set_depth_sensor_params(params = pES[i+1]))
            depth_frameset, color_frameset = get_frameset(k=k)
            # Calculate new cost function values based on new settings
            start_time_cost = time.time()
            cES[i+1] = compute_fill_factor(depth_frameset)
            print("Fill Factor Cost: ", cES[i + 1])
            # Decay the amplitude
            amplitude = amplitude*decay_rate
            depth_imgs.append(depth_frameset[k-1])
            color_imgs.append(color_frameset[k-1])
        depth_imgs = [Image.fromarray(img) for img in depth_imgs]
        color_imgs = [Image.fromarray(img) for img in color_imgs]
        depth_imgs[0].save("depth_badInitialization_torch.gif", save_all=True, append_images=depth_imgs[1:], duration=10, loop=0)
        color_imgs[0].save("color_badInitialization_torch.gif", save_all=True, append_images=color_imgs[1:], duration=10, loop=0)

        # Plot some results
        plt.figure()
        plt.subplot(3,1,1)
        plt.title(f'$k_{{ES}}$={kES}, $a_{{ES}}$={aES}')
        plt.plot(cES)
        plt.ylabel('ES cost')
        plt.xticks([])
        plt.subplot(3,1,2)
        for i in range(3):
            plt.plot(pES_n[:,i],label='$p_{}$'.format(i))
        # plt.plot(p_opt[:,0],'k--',label='$p_{ES,1}$ opt')
        plt.legend(frameon=False)
        plt.ylabel('ES parameters')
        plt.xticks([])
        plt.tight_layout()
        plt.subplot(3,1,3)
        plt.imshow(depth_frameset[k-1])
        plt.subplot(4,1,4)
        plt.imshow(color_frameset[k-1])

finally:
    camera.stop()
