# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:25:43 2020

@author: Ibrahim Alperen Tunc
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import io, filters
from scipy.stats import reciprocal #log normal distribution
import random
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit
from scipy.ndimage.filters import correlate1d
from scipy.signal import savgol_filter
from PIL import Image

#Script for functions used in the saccadic suppression project

#Plot style: General figure parameters:
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 15,
           'ytick.labelsize' : 15,
           'legend.fontsize' : 15,
           'figure.titlesize' : 25,
           'image.cmap' : 'gray'}
plt.style.use(figdict)

rt = 1000 #sampling rate in Hz


def extract_saccade_data(root, angthres, flthres):
    """
    Extract the eye position saccade data from the .txt files
    
    Parameters
    -----------
    root: string
        The string of datapath which contains all data files.
    angthres: float
        The threshold for the saccade amplitude in degrees. Data is discarded if both eyes have a saccade magnitude smaller than threshold.  
    flucthres: float
        The threshold to discard data based on noise fluctuations. If the difference between 2 datapoints is bigger than this value times maximum
        value difference, then the data is discarded. Note that this value is between 0 and 1.
    Returns
    -------
    saccadedata: 3-D array (npos x 2 x ntrial)
        The data array containing the eye positions. First dimension is the eye position angular value over time, second dimension is the eye 
        position (0 is left eye, 1 is righr eye) and the last dimension is for trial index
    datlens: 1-D array
        The time length of each trial. Note that the length is not corrected by the sampling rate
    nmnposidx: 1-D array
        The index array of trials which deviate with their time length 
    saccadedataout: 3-D array (npos x 2 x ntrialout)
        The discarded data array containing the eye positions. Dimensions same as in saccadedata. This is kept just to check if taken out data is 
        indeed non-saccade
    saccadedataout: 3-D array (npos x 2 x ntrialout)
        The noisy (flthres determined) data array containing the eye positions. Dimensions same as in saccadedata. This is kept just to check if 
        noise threshold works well
    """
    #Import the saccade data txt files and combine them together for further analysis
    
    #Insert the external harddrive data path
    sys.path.insert(0, root)
    
    saccadefiles = [] #the list of saccade data (eye position) txt files    
    saccadedatal = [] #extract the data from the data files to the list (hence 'l' in the end)
    saccadefilesout = [] #the list of discarded saccade data (eye position) txt files
    saccadedatalout = [] #extract the data from the data files to the list (hence 'l' in the end, discarded data)
    saccadefilesnoise = [] #the list of noisy saccade data (eye position) txt files
    saccadedatalnoise = [] #extract the data from the data files to the list (hence 'l' in the end, discarded data)
    
    
    #extract the eye position txt files
    for path, subdirs, files in os.walk(root):
        for name in files:
            filename, file_extension = os.path.splitext(name)
            #print(filename, file_extension)
            if file_extension == '.txt' and filename[0:6] == 'eyepos':
                saccadefiles.append(os.path.join(path, name))
        
    for filename in saccadefiles:
        data = pd.read_csv(filename, sep='\t', header=None)
        #find local maxima of the data showing fluctuations bigger than flthres. Discard the first and last few seconds as some recordings show
        #not problematic fluctuations at the onset
        peaksleft, proml = find_peaks(np.array(data[4])[15:-15], prominence=flthres*(np.max(np.array(data[4]))-np.min(np.array(data[4]))))
        peaksright, promr = find_peaks(np.array(data[5])[15:-15], prominence=flthres*(np.max(np.array(data[5]))-np.min(np.array(data[5]))))

        
        #discard the data if the sign of the average difference between angle position of last 20 ms and first 20 ms does not match for both eyes 
        #(first 2 lines) or if this average difference is smaller than angthreshold for BOTH eyes (saccade too small, last 2 lines)
        if (np.sign(np.mean(np.array(data[4])[-20:]-np.array(data[4])[:20])) != \
                    np.sign(np.mean(np.array(data[5])[-20:]-np.array(data[5])[:20]))) or \
                    np.mean(np.abs(np.array(data[4])[:20]-np.array(data[4])[-20:])) < angthres and\
                    np.mean(np.abs(np.array(data[5])[:20]-np.array(data[5])[-20:])) < angthres:
                        
            saccadefilesout.append(os.path.join(path, name))
            saccadedatalout.append((np.array(data[4]),np.array(data[5]))) #order is LE RE
        
        #add the data to noisy list if any 2 timepoint shows fluctuations more than flthres * (max(data) - min(data)). This for any of the eye
        elif len(peaksleft)!=0 or len(peaksright)!=0:
           saccadefilesnoise.append(os.path.join(path, name))
           saccadedatalnoise.append((np.array(data[4]),np.array(data[5]))) #order is LE RE

        
        #add the data as saccade if none of the above holds.
        else:        
            saccadedatal.append((np.array(data[4]),np.array(data[5]))) #order is LE RE
    
    #!npos (number of datapoints for each saccade trial) is not equal for all trials
    datlens = [] #length of the trials
    datlensout = [] #length of the discarded trials
    datlensnoise = [] #length of the noisy trials

    
    #save lengths of all trials.
    for dat in saccadedatal:
        datlens.append(len(dat[0]))
    
    for dat in saccadedatalout:
        datlensout.append(len(dat[0]))
    
    for dat in saccadedatalnoise:
        datlensnoise.append(len(dat[0]))
    
    nmnposidx = np.squeeze(np.where(datlens!=np.median(datlens))) #the indices of the trials which have different length than most of the trials
    
    #adjust the data structure accordingly : datamatrix = npos x 2 x ntrials, 2 is for LE RE
    #datamatrix is adjusted in such a way, that the first dimension has the length of the maximum trial.        
    saccadedata = np.empty((np.max(datlens) ,len(saccadedatal[0]), len(saccadedatal)))
    saccadedata[:] = np.nan
    saccadedataout = np.empty((np.max(datlensout) ,len(saccadedatalout[0]), len(saccadedatalout)))
    saccadedataout[:] = np.nan
    saccadedatanoise = np.empty((np.max(datlensnoise) ,len(saccadedatalnoise[0]), len(saccadedatalnoise)))
    saccadedatanoise[:] = np.nan    
    
    #fill in the values for saccadedata. This will be the same loop as above (loop over saccadedatal), but for general usability of the code, I guess
    #it is better to first estimate the maximum length of the trial and then adjust accordingly the data matrix.
    for idx, dat in enumerate(saccadedatal):
        saccadedata[:datlens[idx],:,idx] = np.array(dat).T
    
    #same as above but for discarded data
    for idx, dat in enumerate(saccadedatalout):
        saccadedataout[:datlensout[idx],:,idx] = np.array(dat).T
    
    #same as above but for noisy data
    for idx, dat in enumerate(saccadedatalnoise):
        saccadedatanoise[:datlensnoise[idx],:,idx] = np.array(dat).T

    return saccadedata, datlens, nmnposidx, saccadedataout, saccadedatanoise


def running_median(array,n):
    """
    Running median function
    
    Parameters
    ----------
    array: 1-D array
        The array to be smoothed by the running median
    n: integer
        Size of the running median window
        
    Returns
    -------
    rmarray: 1-D array
        The running median-smoothed version of the array
    """
    idx = np.empty((array.shape[0], n)) #indices for all running medians
    idx[:array.shape[0]-4,:] = np.arange(-(n-1)/2,((n-1)/2)+1) + np.arange(len(array)-n+1)[:,None]
    idx[array.shape[0]-4:,:] = np.arange(-(n-1)/2,((n-1)/2)+1) + np.arange(len(array)-n+1,len(array))[:,None]
    idx[idx<0] = None   
    idx[idx>=len(array)] = None
    rmarray = np.zeros(len(array)) #running median array preallocated
    for i, window in enumerate(idx):
        arr = window[~np.isnan(window)].astype(int)
        rmarray[i] = np.median(array[arr])
    
    return rmarray
    

def detect_saccade(saccade, negativity):
    """
    Detect the saccade onset based on a threshold value
    
    Parameters
    ----------
    saccade: 1-D array
        The array containing the angular eye position during saccade
    negativity: integer
        The integer value determining the saccade threshold. 1 means after the saccade the eye angle value is bigger, and -1 is for the opposite.
        
    Returns
    -------
    thrsac: 1-D array
        The thresholded version of the saccade array.
    saconset: int
        The index of the saccade onset
    sacoffset: int
        The index of the saccade offset
    """
    meanonset = np.median(saccade[:10]) 
    sacthres =  0.4 + 10*np.std(saccade[:15])
    thres = False
    saconset = 10
    
    if negativity < 0:
        while thres == False:
            if np.median(saccade[saconset:saconset+20]) <= meanonset - sacthres:
                thres = True
            else:
                #print(i, np.median(saccade[i-5:i+5]))
                saconset+=1
    
    elif negativity > 0:
        while thres == False:
            if np.median(saccade[saconset:saconset+10]) >= meanonset + sacthres:
                thres = True 
            else:
                #print(i, np.median(saccade[i-5:i+5]))
                saconset+=1
    
    #onset works well with current implementation, improve the offset detection.
        
    meanoffset = np.mean(saccade[-200:]) 
    offthres = np.std(saccade[-150:])
    off = False
    sacoffset = saconset
    
    while off == False:
        #print(sacoffset, np.abs(saccade[sacoffset] - saccade[sacoffset-150]), np.mean(saccade[sacoffset-150:sacoffset]), meanoffset-offthres, meanoffset+offthres)
        if np.abs(saccade[sacoffset] - saccade[sacoffset-150]) <= 0.2*offthres \
          and np.mean(saccade[sacoffset-150:sacoffset]) <= meanoffset + 4*offthres\
          and np.mean(saccade[sacoffset-150:sacoffset]) >= meanoffset - 4*offthres:  
            
            print('Found')
            off = True
        
        elif sacoffset == len(saccade)-1:
            print('Last timepoint taken')
            off = True
            sacoffset = -1
                
        else:
            sacoffset += 1
        
    return saconset, sacoffset
    #-2 to get the right index. np.diff shifts the index by one (i.e. the difference between idx 0 and 1 has the np.diff
    #index of 0), and secondly saconset returns the index where thrsac already changed, and we want the index right before
    #that (i.e. the index right before the last thrsac transition)
    

def gabor_image_filter(img, gwl, theta, sigma_x, sigma_y, offset, n_stds, mode='reflect', returnimg=True):
    """
    Filter an image with a 2-D Gabor kernel.
    
    Parameters
    ----------
    img: n x m array
        The array of the image with the size n times m
   gwl: float
        The wavelength of the Gabor filter in pixels. The inverse of this is the spatial frequency.
        Due to Nyquist frequency this value cannot be smaller than 2
    theta: float
        The angle of the Gabor filter in radians
    sigma_x: float
        The standard deviation along x-direction, determining the extent of the filter along that direction. Note that
        rotation with theta occurs beforehand. If theta is pi, then sigma_x determines the extent in vertical 
        direction
    sigma_y: float
        The standard deviation along y-direction, determining the extent of the filter along that direction. Note that
        rotation with theta occurs beforehand. If theta is pi, then sigma_y determines the extent in horizontal 
        direction
    offset: float
        The Gabor filter angle offset in radians
    n_stds: float
        The Gaussian standard deviation cutoff value for the Gabor filter
    mode: string, {‘constant’, ‘nearest’, ‘reflect’, ‘mirror’, ‘wrap’}
        The mode of convolution. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
        for how each mode works
    returnimg: boolean
        If True, the whole image is convolved and returned. If false, only the kernel is returned
   
    Returns
    -------
    kerngab: k x l array
        The complex Gabor kernel. Real part has the cosine component (symmetric), imaginary the sine component
        (antisymmetric)
    gaborr: n x m array
        The real part of the convolved image with the Gabor filter
    gabori: n x m array
        The imaginary part of the convolved image with the Gabor filter
    """
    kerngab = filters.gabor_kernel(frequency=1/gwl, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, offset=offset,
                                   n_stds=n_stds)
    if returnimg == False:
        return kerngab
    
    else:
        gaborr, gabori = filters.gabor(img, mode=mode,frequency=1/gwl, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, 
                                       offset=offset, n_stds=n_stds)
        return kerngab, gaborr, gabori


def gaussian_image_filter(img, sigma, mode='reflect', truncate=3):
    """
    Filter an image with a 2-D Gaussian kernel
    
    Parameters
    ----------
    img: n x m array
        The array of the image with the size n times m       
    sigma: float
        The standard deviation of the kernel
    mode: string, {‘constant’, ‘nearest’, ‘reflect’, ‘mirror’, ‘wrap’}
        The mode of convolution. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
        for how each mode works  
    truncate: float
        The Gaussian standard deviation cutoff value
        
    Returns
    -------
    gaussimg: n x m array
        The filtered image
    """
    gaussimg = filters.gaussian(img, sigma=sigma, mode='reflect', truncate=3)
    return gaussimg


@jit(nopython=True)
def leaky_integrate_and_fire(stimulus, deltat=0.00005, v_zero=0.0, threshold=1.0, v_base=0.0,
             v_offset=-10.0, mem_tau=0.015, noise_strength=0.05,
             input_scaling=60.0, ref_period=0.001):
    """
    1-D leaky integrate and fire neuron.
    
    Parameters
    ----------
    stimulus: 1-D array
        The stimulus over time driving the neuron
    deltat: float
        Integration time step in seconds (Euler method to solve the differential equation numerically)
    v_zero: float
        The initial value of the membrane voltage
    threshold: float
        The threshold value for membrane voltage to record a spike and to reset to baseline voltage
    v_base: float
        The baseline membrane voltage
    v_offset: float
        The offset of the baseline membrane voltage. Without any stimulus perturbation, the membrane voltage
        decays to this value in steady state
    mem_tau: float
        The membrane voltage time constant in seconds
    noise_strength: float
        The strenght of the Gaussian noise used to jitter the stimulus
    input_scaling: float
        The input scaling which determines the stimulus sensitivity of the neuron
    ref_period: float
        The refractory period length in seconds
        
    Returns
    --------
    spike_times: 1-D array
        The array containing the time stamps of the spikes occured.
    """
    v_mem = v_zero

    # prepare noise:    
    noise = np.random.randn(len(stimulus))
    noise *= noise_strength / np.sqrt(deltat) # scale white noise with square root of time step, coz else they are 
                                              # dependent, this makes it time step invariant.
    # integrate:
    spike_times = []
    for i in range(len(stimulus)):
        v_mem += (v_base - v_mem + v_offset + 
                  stimulus[i] * input_scaling + noise[i]) / mem_tau * deltat #membrane voltage (integrate & fire)

        # refractory period:
        if len(spike_times) > 0 and (deltat * i) - spike_times[-1] < ref_period + deltat/2:
            v_mem = v_base

        # threshold crossing:
        if v_mem > threshold:
            v_mem = v_base
            spike_times.append(i * deltat)

    return np.array(spike_times)


def img_move(x0, y0, angle, speed):
    """
    Calculate the image location for the next frame
    
    Parameters
    -----------
    x0: integer
        The index of the initial pixel value in x direction. Initial pixel is the center of the image patch
    y0: integer
        The index of the initial pixel value in y direction. Initial pixel is the center of the image patch
    angle: float
        The movement direction angle in radians
    speed: float
        The movement speed in pixel/ms
    
    Returns
    -------
    xnew: float
        The new pixel location x value
    ynew: float
        The new pixel location y value
    """
    xnew = np.cos(angle)*speed + x0    
    ynew = np.sin(angle)*speed + y0
    return xnew, ynew


def generate_grating(rsl, imgsf):
    """
    Generate a grating. For now vertical, maybe in the future different angles can be added with an angle variable.
    
    Parameters
    ----------
    rsl: float
        The resolution of the image in pixels per degrees
    imgsf: float
        The spatial frequency of the image in degrees
    
    Returns
    --------
    img: 2-D array
        The grating image array
    """
    #initial values
    maxsf = rsl / 2 #maximum spatial frequency achievable with the resolution, 1/°
    imgsize = [180, 360] #the screen size in degrees, first is elevation and second is azimuth
    #Give error if image spatial frequency is bigger than maximum possible
    if imgsf > maxsf:    
        raise(ValueError('Image spatial frequency is too big to be adequately resolved. Maximum possible value is %.1f' %(maxsf)))
    
    else:
        pass

    #create the single grating
    #first generate the grating as vertical regardless of future grating angles
    grwl = rsl/imgsf #the wavelength of the image in pixels    
    grsin = (np.sin(2*np.pi*1/grwl*np.arange(0, 5*grwl))+1)/2 #create a 5-cycle long grating
    """
    Repeat the 5 cycle long grating to fill the complete azimuth. I chose this apprach as else the grating widths 
    were changing too radical in the later cycles.
    """
    grating = np.tile(np.round(grsin), np.ceil(imgsize[1] / grsin.shape[0]).astype(int)*rsl) 
    img = np.tile(grating[:360*rsl], imgsize[0]*rsl).reshape(np.array(imgsize)*rsl)
    
    return img


def crop_gabor_filter(rsl, sigma, sf, flts, ph, circlecrop = True, normalize=True):
    """
    Crop a Gabor filter with a given size and spatial frequency
    
    Parameters
    ----------
    rsl: float
        The resolution of the image in pixel per degrees.
    sigma: float
        The standard deviation of the Gabor filter. This value is kept the same for both x and y directions, hence 
        the resulting filter has a circular shape.
    sf: float
        The spatial frequency of the filter in degrees
    flts: float
        The size of the filter in degrees (i.e. diameter)
    ph: float
        The phase of the filter in degrees
    circlecrop: boolean
        If true, circle crop is returned, else the filter is returned in a quadratic array
    normalize: boolean
         If True, cropped Gabor is normalized by its max value.
        
    Returns
    --------
    newgabr: 2-D array
        The circular cropped real part of the Gabor filter (symmetric, cosine)
    newgabi: 2-D array
        The circular cropped imaginary part of the Gabor filter (antisymmetric, sine)
    """
    maxsf = rsl / 2 #maximum spatial frequency achievable with the resolution, 1/°
    if sf > maxsf:    
        raise(ValueError('Image spatial frequency is too big to be adequately resolved. Maximum possible value is %.1f' %(maxsf)))
    
    #set the gabor phase so that the filter starts at the sin(0) value
    
    gkernparams = {'gwl': rsl/sf,
                   'theta': np.deg2rad(0),
                   'sigma_x' : sigma,
                   'sigma_y' : sigma,
                   'offset' : np.deg2rad(ph),
                   'n_stds': 3}
    kerngab = gabor_image_filter(None, **gkernparams, returnimg=False) #everything of Gabor is now in pixels
    flts *= rsl
    
    if kerngab.shape[0] < flts: #return error if the initial gabor filter is smaller than expected filter size
        raise(ValueError("The size of the initial Gabor filter is too small, Please increase sigma."))
        
    fltc = np.array(kerngab.shape[0])/2-1 #the center index of the initital big filter
    flte = [np.round(fltc-flts/2).astype(int),np.round(fltc+flts/2).astype(int)] #extent of the filter. indices to be 
                                                                                #taken from kerngab to generate the 
                                                                                #filter
    
    newgabi = kerngab.imag.copy()[flte[0]:flte[1], flte[0]:flte[1]] #imaginary part of the filter cropped rectangular
    xlocs, ylocs = np.meshgrid(np.linspace(-newgabi.shape[0]/2, newgabi.shape[0]/2, newgabi.shape[0]),
                           np.linspace(-newgabi.shape[0]/2, newgabi.shape[0]/2, newgabi.shape[0]))
    
    newgabr = kerngab.real.copy()[flte[0]:flte[1], flte[0]:flte[1]] #real part of the filter cropped rectangular
    xlocs, ylocs = np.meshgrid(np.linspace(-newgabi.shape[0]/2, newgabi.shape[0]/2, newgabi.shape[0]),
                           np.linspace(-newgabi.shape[0]/2, newgabi.shape[0]/2, newgabi.shape[0]))

    if circlecrop == False:
        return newgabr, newgabi
    
    circlearrayi = xlocs**2 + ylocs**2 <= (newgabi.shape[0]/2+1)**2
    newgabi = circle_crop(circlearrayi, newgabi, normalize=normalize)
    
    
    circlearrayr = xlocs**2 + ylocs**2 <= (newgabr.shape[0]/2)**2
    newgabr = circle_crop(circlearrayr, newgabr, normalize=normalize)
    return newgabr, newgabi


def circle_crop(circlearray, gaborarray, normalize=True):
    """
    Circular crop function to be used in crop_gabor_filter for better readability. Filter is normalized to have
    a min value of 0 and max value of 1
    
    Parameters
    ----------
    circlearray: 2-D array
        The circular mask array.
    gaborarray: 2-D array
        The imaginary or the real part of the Gabor array.
    normalize: boolean
        If True, cropped Gabor is normalized by its max value.
    Returns
    --------
    croppedarray: 2-D array
        The circularly cropped Gabor array.
    """
    croppedarray = gaborarray * circlearray
    if normalize == True:
        croppedarray /= np.max(croppedarray)
    return croppedarray


def filters_activity(img, flt):
    """
    Extract the filter activity from the filter bank of equal size for a given image.
    
    Parameters
    ----------
    img: 2-D array
        The stimulus image.
    fltarray: 2-D
        The filter kernel array.
    
    Returns
    -------
    fltact: 2-D array
        The activity of the filter summed over all pixels of each image patch. The array contains each patch activity
        starting from top left then iterating over each row horizontally. 
    xext: 2-D array
        The start and stop index of the image patches along x direction.
    yext: 2-D array
        The start and stop indices of the image patches along y direction.
    """
    fltsize = flt.shape[0] #since the filters are quadratic arrays, getting the shape in one dimension is
                                   #sufficient to get the size 
    imgxdiv = img.shape[1] // fltsize #the number of patches along x direction
    imgydiv = img.shape[0] // fltsize #the number of patches along y direction
    fltact = np.zeros(imgxdiv*imgydiv) #shape number of patches
    xext = np.zeros([imgxdiv*imgydiv,2]) #start and stop of x indexes for each image patch
    yext = np.zeros([imgxdiv*imgydiv,2]) #start and stop of y indexes for each image patch

    for i in range(imgxdiv):
        for j in range(imgydiv):
            patch = img[fltsize*j:fltsize*(j+1), fltsize*i:fltsize*(i+1)]
            yext[imgydiv*i+j, :] = [fltsize*j, fltsize*(j+1)]
            xext[imgydiv*i+j, :] = [fltsize*i, fltsize*(i+1)]
            fltact[imgydiv*i+j] = np.sum(patch*flt) #the filter activity summed over each pixel.
            #print(imgydiv*i+j, i ,j) #this to debug, imgydiv*... ensures the flattening of the index
    return fltact, xext.astype(int), yext.astype(int)       


def population_activity(gfsf, gfph, gfsz, gfilters, img):
    """
    Get the population activity of all filters considered.
    
    Parameters
    ----------
    gfsf: 1-D array
        The spatial frequencies in degrees used for filters.
    gfph: 1-D array
        The phases in degrees used for filters.
    gfsz: 1-D array
        The sizes in pixels used for filters
    gfilter: Nested array
        The array containing all the filter kernels. First 2 dimensions are size and spatial frequency, each element
        is then n*n*m array where n is the length of the kernel side (e.g. 20 pixels, can also be thought as the 
        radius of the kernel) and m is the number of phases considered.
    img: 2-D array
        The image array
        
    Returns
    -------
    gacts: 3-D array
        The array containing the filter activity for each patch of the image. 1st dimension is for spatial frequency,
        2nd for phase and 3rd for size. Each element contains the filter activity for each of the image patch outlined
        by the indices given in pext
    pext: 2-D array
        The array containing the indices of the image patches for each size. First dimension is filter size, 2nd 
        dimension is for x and y coordinates (e.g. pext[0,0] returns the x indices of the patches for the first size
        i.e. gfsz[0] and pest[0,1] the y indices respectively)
    """
    gacts = np.zeros([len(gfsf), len(gfph), len(gfsz)], dtype=object) #Gabor kernel activities
    pext = np.zeros([len(gfsz), 2], dtype=object) #Patch extends for each Gabor filter

    for i, fszs in enumerate(gfilters): #iterate over size
        for j, fsfs in enumerate(fszs): #iterate over spatial frequency
            for k in range(fsfs.shape[2]): #iterate over phase
                fltact, xext, yext = filters_activity(img, fsfs[:,:,k])        
                gacts[j,k,i] = fltact

        #since for the same sized filters the patches also have the same size, the patch x and y indices only have to be 
        #given for different sizes of filters (hence shape sz*2 where 2d dimension is for x and y index arrays 
        #respectively)
        pext[i,0] = xext #the x indices of the patches
        pext[i,1] = yext #the y indices of the patches
    return gacts, pext


def circle_stimulus(rdot, rsl, dxcent=None, dycent=None):
    """
    Create an visual field array where a circle stimulus with radius rdot is drawn somewhere randomly.
    
    Parameters
    -----------
    rdot: float
        The radius of the circle in pixels
    rsl: float
        The image resolution (pixel per degrees)
    dxcent: float
        The stimulus center x coordinate in degrees
    dycent: float
        The stimulus center y coordinate in degrees
    Returns
    --------
    img: 2-D array
        The image containing the circle.
    circcent: 1-D array
        The x and y locations of the circle center in degrees
    """
    #TODO: ENSURE THE NEGATIVE ELEVATIONS AND SIMILAR ARE HANDLED CORRECTLY. FOR NOW ONLY AZIMUTH CORRECTIONS 
    #ARE IMPLEMENTED. SEE ALSO FLORIAN_MODEL FOR THE IMPLEMENTATION
    img = np.zeros([180*rsl,360*rsl])
    xvals, yvals = np.meshgrid(np.linspace(-rdot, rdot, 2*rdot), np.linspace(-rdot, rdot, 2*rdot))
    dot = xvals**2 + yvals**2 <= rdot**2

    if dxcent == None or dycent == None:
        #choose random location center for the dot in the visual image if no center is specified.
        dxcent = np.random.randint(rdot, (360-rdot))
        dycent = np.random.randint(rdot, (180-rdot))
    
    #WRAP THE STIMULUS AROUND AZIMUTH WHEN NEEDED!
    if np.round(dxcent*rsl-rdot) < 0:
        dxcent = dxcent + img.shape[1]/rsl
    
    if np.round(dycent*rsl-rdot) < 0:
        dycent = dxcent + img.shape[1]
    
    if (dxcent*rsl-rdot) > (dxcent*rsl+rdot)%img.shape[1]:
        xext = img.shape[1]
        img[np.round(dycent*rsl-rdot).astype(int):np.round(dycent*rsl+rdot).astype(int), \
            np.round(dxcent*rsl-rdot).astype(int):] = dot[:, :xext-np.round((dxcent*rsl-rdot)).astype(int)]
        img[np.round(dycent*rsl-rdot).astype(int):np.round(dycent*rsl+rdot).astype(int), \
            :np.round((dxcent*rsl+rdot)%img.shape[1]).astype(int)] = \
                                                    dot[:, xext-np.round((dxcent*rsl-rdot)).astype(int):]
        
    else:
        img[np.round(dycent*rsl-rdot).astype(int):np.round(dycent*rsl+rdot).astype(int), \
            np.round(dxcent*rsl-rdot).astype(int):np.round(dxcent*rsl+rdot).astype(int)] = dot
    return img, np.array([dxcent, dycent])


def florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img, params=None):
    """
    The function to get the random shuffled parameter combinations from a given parameter pool
    
    Parameters
    ----------
    nfilters: integer
        Number of filters used in the model
    rsl: float
        The resolution of the stimulus image
    gfsf: 1-D array
        The array containing the possible filter spatial frequency values
    gfsz: 1-D array
        The array containing the possible filter size values
    gfph: 1-D array
        The array containing the possible filter phas values
    jsigma: float
        The standard deviation of the filter center location randomizer
    img: float
        The stimulus image
    params: 2-D array or None
        If None, this variable is not considered. If this array is not None, then the parameter triplet combinations
        are not generated, since this variable contains all model parameters for each Gabor unit.
    
    Returns
    -------
    parameters: 2-D array
        The array containing the parameter values (2nd dimension in the order of spatial frequency, size and phase)
        for each filter
    fltcenters: 2-D array
        The array containing the x and y coordinates of the Gabor filter centers.
    """
    if params == None:
        parametertriplets = [(i,j,k) for i in gfsf for j in gfsz for k in gfph] #order is sf sz ph
        randomidxs = np.random.randint(0, len(parametertriplets), nfilters)
        parameters = [parametertriplets[randomidxs[i]] for i in range(len(randomidxs))]
        parameters = np.array(parameters)
    else:
        parameters = params
    #tile the image then jitter the locations
    #take the center of each patch (x y pixel values)
    xcenters = np.linspace(0, 360, 2*np.sqrt(nfilters/2).astype(int), endpoint=False) +\
                                                                                  90/np.sqrt(nfilters/2).astype(int)
    ycenters = np.linspace(0, 180, np.sqrt(nfilters/2).astype(int), endpoint=False) +\
                                                                                  90/np.sqrt(nfilters/2).astype(int)
    xcenters *= rsl #convert to pixels
    ycenters *= rsl
    
    (xloc, yloc) = np.meshgrid(xcenters, ycenters)
    jitterxy = np.random.normal(0, jsigma, [2, *xloc.shape]) * rsl #jitter in pixels
    xloc += jitterxy[0]
    yloc += jitterxy[1]
    xloc = np.round(xloc).astype(int)
    yloc = np.round(yloc).astype(int)
    xloc = xloc % img.shape[1] #for center values bigger than 360° in azimuth, wrap the index around and get the
                               #correxponding smaller angle value location (i.e. 363° is the same as 3°)
    #for elevation set the center values outside of the image borders to the image border
    yloc[yloc<0] = 0
    yloc[yloc>img.shape[0]-1] = img.shape[0]-1                            
    xloc = xloc.reshape(len(parameters)) #make it a list so circular modulo operator works.
    yloc = yloc.reshape(len(parameters))
    fltcenters = np.array([xloc, yloc]).T
    return parameters, fltcenters


def florian_model_filter_population(nfilters, img, rsl, sigma, parameters, xloc, yloc):
    """
    Generate a population of filters with the given settings
    
    Parameters
    ----------
    nfilters: integer
        Number of filters used in the population
    img: 2-D array
        The stimulus image
    rsl: float
        The resolution of the stimulus image
    sigma: float
        The standard deviation used for the Gabor filters
    parameters: 2-D array
        The array containing the parameters. 2nd dimension is for spatial frequency, size and phase values in the 
        respective order
    xloc: 1-D array
        The array containing the x coordinates of the filter centers
    yloc: 1-D array
        The array containing the y coordinates of the filter centers
    
    Returns
    --------
    filtersarray: 3-D array
        The array containing the filter receptive fields. First dimension is for filter number, second for elevation
        and last for azimuth
    filtersimg: 2-D array
        The array containing all the filters at once within the visual field. The Gabor filters show overlap.    
    """
    filtersarray = np.zeros([nfilters, *img.shape]) #each filter has a specific position. (xloc yloc determines the 
                                                    #center). This array is hence nfilters*img.shape
    filtersimg = np.zeros(img.shape)
    for idx, params in enumerate(parameters):
        _, flt = crop_gabor_filter(rsl, sigma, *params)
        fltext = np.floor(flt.shape[1] / 2) #radius of the filter
        ypixs = img.shape[0] #pixel numbers along elevation
        xpixs = img.shape[1] #pixel numbers along azimuth
        
        #ensure the start y value for the filter in the image is bigger or equal to zero
        ystartcorr = np.int(np.abs(np.min([0, yloc[idx]-fltext]))) #correction factor, zero if yloc[idx]-fltext)>=0 
        #ensure the stop y value for the filter in the image is bigger than 180*rsl
        ystopcorr = ypixs - np.max([ypixs, yloc[idx]+fltext]) #correction factor, zero if yloc[idx]+fltext)<180    
        
        #x and y start and stop indices
        ystart = np.int(yloc[idx]-fltext+ystartcorr)
        ystop = np.int(yloc[idx]+fltext+ystopcorr)
        xstart = np.int((xloc[idx]-fltext)%xpixs)
        xstop = np.int((xloc[idx]+fltext)%xpixs)
        
        if xstop > xstart:
            filtersarray[idx, ystart:ystop, xstart:xstop] = flt[ystartcorr:ystop-ystart+ystartcorr, :]
            filtersimg[ystart:ystop, xstart:xstop][flt[ystartcorr:ystop-ystart+ystartcorr, :]!=0] = \
                             flt[ystartcorr:ystop-ystart+ystartcorr, :][flt[ystartcorr:ystop-ystart+ystartcorr, :]!=0]
        else: #if the filter is wrapping around in azimuth, we need to index first until the end of the image, then 
              #index the remainder to the beginning.
            #index until which the first part of the filter is to be taken. (part until the end of the image).
            fltidx = np.int(xpixs-xstart) 
            #index until the end of the azimuth
            filtersarray[idx, ystart:ystop, xstart:] = flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx]
            filtersimg[ystart:ystop, xstart:][flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx]!=0] = \
                  flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx][flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx]!=0]
            #index from zero until the remainder
            filtersarray[idx, ystart:ystop, :xstop] = flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:]
            filtersimg[ystart:ystop, :xstop][flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:]!=0] = \
                  flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:][flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:]!=0]
                  
    return filtersarray, filtersimg
    

def florian_model_population_activity(filtersarray, fltcenters, img):
    """
    Read out the population activity by using Florian model (random RF location, get filter activity and estimate
    stimulus location)
    
    Parameters
    ----------
    filtersarray: 3-D array
        The array containing the Gabor filters, First dimension is for different filters, last 2 dimensions are
        azimuth and elevation
    fltcenters: 2-D array
        The array containing the filter center locations. First dimension is for filters, second for x and y values
        of the center
    img: 2-D array
        The image array
        
    Returns
    -------
    popact: 1-D array
        The filter population activity
    stimcenters: 1-D array
        The estimated center of stimulus from population activity (x and y locations). 
        !Note that the stimulus centers are not in degrees but in pixel values. Furthermore the x pixel values go from 0 
        to 360*rsl from left to right, and y pixel values from 0 to 180*rsl from top to bottom.
    """
    popact = np.zeros(filtersarray.shape[0])
    for idx, filt in enumerate(filtersarray):
        filtact = np.abs(np.sum(filt*img))
        popact[idx] = filtact
    stimcenter = popact/np.sum(popact) @ fltcenters
    return popact, stimcenter


def florian_model_species_edition(img, rsl, parameters, xloc, yloc, imgrec=False):
    """
    Filter population and population activity functions have to be fused together for species simulations, since you
    have to go through each filter one by one to get the respective activity and get the stimulus position estimate.
    Or you can find a hardware with 2 TB RAM and use the upper 2 functions!
    
    Parameters
    ----------
    img: 2-D array
        The stimulus image
    rsl: integer
        Resolution in pixel per degrees
    parameters: 2-D array
        The parameters of the model in the order sf, sz, ph
    xloc: 1-D array
        The filter centers (x coordinate)
    yloc: 1-D array
        The filter centers (y coordinate)
    imgrec: boolean
        If true, return the image reconstruction by multiplying each filter with its absolute population activity
        
    Returns
    -------
    filtersimg: 2-D array
        The image array containing all the filters at the same time.
    popact: 1-D array
        The activity of each filter caused by the stimulus
    """
    if imgrec == True:
        recimg = np.zeros(img.shape)
    filtersimg = np.zeros(img.shape)
    popact = np.zeros(len(parameters))
    for idx, params in enumerate(parameters):
        #print(idx+1)
        #generate the filter
        filterarray = np.zeros(img.shape) #Preallocate the array for ONE filter.
        flt = species_model_gabor(rsl, *params) #the gabor func. is very slow, so wrote my own gabor function
        fltext = flt.shape[1] / 2 #radius of the filter
        ypixs = img.shape[0] #pixel numbers along elevation
        xpixs = img.shape[1] #pixel numbers along azimuth
        #ensure the start y value for the filter in the image is bigger or equal to zero
        ystartcorr = np.int(np.abs(np.min([0, yloc[idx]-fltext]))) #correction factor, zero if yloc[idx]-fltext)>=0 
        #ensure the stop y value for the filter in the image is bigger than 180*rsl
        ystopcorr = ypixs - np.max([ypixs, yloc[idx]+fltext]) #correction factor, zero if yloc[idx]+fltext)<180    
        
        #x and y start and stop indices
        ystart = np.int(yloc[idx]-fltext+ystartcorr)
        ystop = np.int(yloc[idx]+fltext+ystopcorr)
        xstart = np.int((xloc[idx]-fltext)%xpixs)
        xstop = np.int((xloc[idx]+fltext)%xpixs)
        
        if xstop > xstart:
            filterarray[ystart:ystop, xstart:xstop] = flt[ystartcorr:ystop-ystart+ystartcorr, :]
            filtersimg[ystart:ystop, xstart:xstop][flt[ystartcorr:ystop-ystart+ystartcorr, :]!=0] = \
                             flt[ystartcorr:ystop-ystart+ystartcorr, :][flt[ystartcorr:ystop-ystart+ystartcorr, :]!=0]
        else: #if the filter is wrapping around in azimuth, we need to index first until the end of the image, then 
              #index the remainder to the beginning.
            #index until which the first part of the filter is to be taken. (part until the end of the image).
            fltidx = np.int(xpixs-xstart) 
            #index until the end of the azimuth
            filterarray[ystart:ystop, xstart:] = flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx]
            filtersimg[ystart:ystop, xstart:][flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx]!=0] = \
                  flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx][flt[ystartcorr:ystop-ystart+ystartcorr, :fltidx]!=0]
            #index from zero until the remainder
            filterarray[ystart:ystop, :xstop] = flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:]
            filtersimg[ystart:ystop, :xstop][flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:]!=0] = \
                  flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:][flt[ystartcorr:ystop-ystart+ystartcorr, fltidx:]!=0]
        
        #get the activity for each filter per loop
        if xstop < xstart:
            filtact = np.sum(filterarray[ystart:ystop, xstart:]*img[ystart:ystop, xstart:]) + \
                             np.sum(filterarray[ystart:ystop, :xstop]*img[ystart:ystop, :xstop])
        else:
            filtact = np.sum(filterarray[ystart:ystop, xstart:xstop]*img[ystart:ystop, xstart:xstop])
        popact[idx] = np.abs(filtact)
        
        if imgrec == True and filtact != 0:
            recimg[filterarray!=0] += filterarray[filterarray!=0] * filtact
        else:   
            pass
        
    if imgrec == True:
        recimg /= len(parameters)
        recimg[recimg < np.max(recimg)-np.std(recimg)] = 0
        return filtersimg, popact, recimg  
    else:
        return filtersimg, popact
        
        
def species_model_gabor(rsl, sf, sz, ph, normalize=False):
    """
    Since for higher resolutions (e.g. 20 pixels per degree) the Gabor function is very computationaly heavy, I 
    implemented my simpler version of the Gabor where there is no orientation parameter but other parameters of 
    interest are implemented.
    
    Parameters
    ----------
    rsl: integer
        Image resolution in pixels per degrees
    sf: float
        Spatial frequency of the Gabor filter in cyc/deg
    sz: float
        Size of the filter (diameter)
    ph: float
        Phase of the Gabor filter
    normalize: boolean
        If True, cropped Gabor is normalized by its max value.

    Returns
    -------
    filterimg: 2-D array
        The Gabor filter 
    """ 
    #Generate first the rectangular Gabor
    degrees = np.linspace(0,360, 360*rsl)
    gaborsin = np.sin(2*np.pi * sf * degrees + ph)
    filterimg = np.zeros([(np.round(sz*rsl)).astype(int), (np.round(sz*rsl)).astype(int)])
    filterimg += gaborsin[:filterimg.shape[0]]
    #Crop circularly
    xlocs, ylocs = np.meshgrid(np.linspace(-filterimg.shape[0]/2, filterimg.shape[0]/2, filterimg.shape[0]),
                           np.linspace(-filterimg.shape[0]/2, filterimg.shape[0]/2, filterimg.shape[0]))
    circlearray = xlocs**2 + ylocs**2 <= (filterimg.shape[0]/2+1)**2
    filterimg = circle_crop(circlearray, filterimg, normalize=normalize)
    filterimg[filterimg<-1] = -1
    filterimg[filterimg>1] = 1
    return filterimg


def plot_outliers_to_violinplot(data, pcutoff, ax, positions=None):
    """
    Plot the outlier datapoints to the violinplot as small black dots
    
    Parameters
    ----------
    data: 2-D array
        The data from which the violinplot is generated
    pcutoff: 1-D array
        The percentile cutoffs for a data to be considered as an outlier, between 0 and 100
    ax: axis object
        The axis to which the outliers are to be plotted.
    positions: 1-D array
        The tick positions to where the outliers should be plotted. If None, all outlier sets are plotted with 1
        tick apart from each other
    Returns
    --------
    """
    for didx, dat in enumerate(data):
        percentiles = np.percentile(np.array(dat)[~np.isnan(dat)], pcutoff)
        outliers = np.array(dat)[(dat<percentiles[0]) | (dat>percentiles[1])]
        if positions is None:
            ax.plot(np.repeat(didx+1, len(outliers)), outliers, 'k.', markersize=1.5)
        else:
            ax.plot(np.repeat(positions[didx], len(outliers)), outliers, 'k.', markersize=1.5) 
    return


def violin_plot_xticklabels_add_nsimul(xtcks, data):
    """
    Format the x tick labels so that under the real label the corresponding number of simulations are shown.
    
    Parameters
    ----------
    xtcks: 1-D array
        The initial tick labels.
    data: 1-D or 2-D array
        The dataset from which the violinplots are generated.
    
    Returns
    -------
    xtcks: 1-D array
        The new x ticks array formatted to show the number of simulations in a new line.
    """
    for idx, label in enumerate(xtcks[1:]):
        xtcks[idx+1] = str(label) + ' \n (%i)'%(len(data[idx]))
    return xtcks
    

def florian_model_population_activity_img_reconstruction(filtersarray, fltcenters, img):
    """
    Read out the population activity by using Florian model (random RF location, get filter activity to reconstruct
    the image)
    
    Parameters
    ----------
    filtersarray: 3-D array
        The array containing the Gabor filters, First dimension is for different filters, last 2 dimensions are
        azimuth and elevation
    fltcenters: 2-D array
        The array containing the filter center locations. First dimension is for filters, second for x and y values
        of the center
    img: 2-D array
        The image array
        
    Returns
    -------
    popact: 1-D array
        The filter population activity
    stimcenters: 1-D array
        The estimated center of stimulus from population activity (x and y locations)
    """
    popact = np.zeros(filtersarray.shape[0])
    for idx, filt in enumerate(filtersarray):
        filtact = np.sum(filt*img)
        popact[idx] = filtact
    stimcenter = popact/np.sum(popact) @ fltcenters
    return popact, stimcenter


class generate_species_parameters:
    """
    Generate model parameters for different species
    
    Species so far:
    ---------------
    Macaque
    Zebrafish
    """
    def __init__(self, nfilters):
        self.nfilters = nfilters

        
        
    def macaque(self):
        """
        SF:
        Chen et al 2018 fig. 7 45 0.5 cyc/deg, 30 1.11, 10 2.22, 20 4.44 2 11.11
        Thus app. 107 neurons, making the percentages as follows:
        0.42056075, 0.28037383, 0.09345794, 0.18691589, 0.01869159 
        number matches that of the reported in the paper (42%).   
        """
        macsfs = np.array([0.5, 1.1, 2.2, 4.4, 10])
        macsfprobs = np.array([45, 30, 10, 20, 2]) / np.sum(np.array([45, 30, 10, 20, 2]))
        macsfparams = []
        for idx, sf in enumerate(macsfs):
            #sample for each probability interval from uniform distribution the appropriate number of units.
            if idx == len(macsfs)-1:    
                sfpar = np.random.uniform(sf-2, sf, np.ceil(self.nfilters*macsfprobs[idx]).astype(int))
            else:
                sfpar = np.random.uniform(sf, macsfs[idx+1], np.ceil(self.nfilters*macsfprobs[idx]).astype(int))
            macsfparams.append(list(sfpar))
        macsfparams = np.array([a for b in macsfparams for a in b])
        #take out random values to match nfilters
        macsfremoveidx = np.random.randint(0, len(macsfparams)-1, len(macsfparams)-self.nfilters)
        macsfparams = np.delete(macsfparams, macsfremoveidx)
        #shuffle the parameters.
        random.shuffle(macsfparams)
        
        """
        RF size:
        Chen et al. 2018 : Diameter between 0.618-19.642
        Cyander and Berman 1971: Diameter 0.75-70
        Schiller and Stryker: Diameter 0.125-more than 10
        Use Chen et al. 2018 for consistency.
        Radius between 10-0.3° (diameter extent 19.024)
        mean diameter 10 degrees median diameter 6 degrees
        !Probably distribution not Gaussian, since mean and median show huge deviations
        """
        macszparams = np.random.uniform(0.5, 20, self.nfilters) #Shuffle not necessary, smaller RFs here underrepresented.
        
        #Finally the phase: simply sample nfilter times from the gfph
        macphparams = np.random.uniform(0, 360, self.nfilters)
        macparameters = [[macsfparams[i], macszparams[i], macphparams[i]] for i in range(self.nfilters)]
        return macparameters
        """
        #try the model - works but very slow ;( AFTER LAST OPTIMIZATION ITS LIKE LIGHTNING MCQUEEN!
        rsl = 24
        img, circcent = circle_stimulus(rdot, rsl, *(180, 90))    
        _, fltcenters = florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, params=macparameters)
        filtersimg, popact = florian_model_species_edition(img, rsl, macparameters, fltcenters[:,0], fltcenters[:,1])
        
        stimcenter = popact/np.sum(popact) @ fltcenters
        stimcenter /= rsl
        """

       
    def zebrafish(self, sfdist='logunif'):
        """
        SF:
        Only info I found was upper resolution limit is 1° so
        max SF is 1 cyc/° (Sajovic & Levinthal 1982) so I guess sample uniformly between 0-1
        !UPDATE: As of 21.02.2021 the spatial frequencies are log normal distributed, so any simulations after this
        date differ in zebrafish DF distribution.
        """
        if sfdist == 'logunif':
            zfsfparams = reciprocal(0.000001,1).rvs(self.nfilters)
        elif sfdist == 'unif':
            zfsfparams = np.random.uniform(0,1, self.nfilters)    
        #since 0 does not work in log, a workaround is to choose a positive very tiny number  
        #(i.e. infinitesimally small).
                                                          
        """
        RF size:
        Wang et al. 2020: Biggest RF 168x80, 86% 30x13-90x39, remaining 14% bigger than that 
        Preuss et al. 2014: 2-8 diameter 40%, 16-64 diameter 43°
        Sajovic and Levinthal 1982 73% 34x24.9, 18% 38.6x28.2, 8% 35.6x25.4
        Use Wang et al. 2020 since that data is the most informative. Calculate the areas and estimate the diameter assuming
        circular RF shape (obviously wrong but an acceptable simplification.)
        """
        #zfszs = np.sqrt(np.array([30*13,90*39,168*80])/np.pi)*2 #use to check diameter calculation.
        zfszparams = []
        zfszparams.append(np.random.uniform(22.2837, 66.8511, np.round(self.nfilters*0.86).astype(int)))
        zfszparams.append(np.random.uniform(66.8511, 130.8141, np.round(self.nfilters*0.14).astype(int)))
        zfszparams = [a for b in zfszparams for a in b]
        zfszremoveidx = np.random.randint(0, len(zfszparams)-1, len(zfszparams)-self.nfilters)
        zfszparams = np.delete(zfszparams, zfszremoveidx)
        random.shuffle(zfszparams)
        
        #Finally the phase: simply sample nfilter times from the gfph
        zfphparams = np.random.uniform(0, 360, self.nfilters)
        zfparameters = [[zfsfparams[i], zfszparams[i], zfphparams[i]] for i in range(self.nfilters)]
        return zfparameters
        """
        #try the model - slower than macaque since now the RFs are way bigger and thus it takes longer. I am not sure how
        #to optimize even further. OR REDUCE THE rsl DUMMY!
        rsl = 5
        img, circcent = circle_stimulus(rdot, rsl, *(180, 90))    
        _, fltcenters = florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, params=zfparameters)
        filtersimg, popact = florian_model_species_edition(img, rsl, zfparameters, fltcenters[:,0], fltcenters[:,1])
        
        stimcenter = popact/np.sum(popact) @ fltcenters
        stimcenter /= rsl
        """

def detect_saccades_v2(rawdata, smoothsigma=15, velthres=1000, accthres=100000, savgollength=51, savgolorder=4, rf=1000, a=0.05, b=0.05, velperc=90):
    """
    Improved saccade onset and offset detection algorithm adapted from Nyström & Holmqvist 2010
    
    Parameters
    ----------
    rawdata: 1-D array
        The raw saccade trace data
    rf: float, optional
        Temporal sampling frequency in Hz
    smoothsigma: float, optional
        The standard deviation of the Gaussian smoothing kernel
    savgollength: float, optional
        The length of the Savitzky-Golay filter in ms
    savgolorder: float, optional
        The polynomial order of the Savitzky-Golay fit. See also scipy.signal.savgol_filter documentation
    velthres: float, optional
        The velocity threshold for the filtered data in °/s. Any velocity bigger than this value is physiologically unrealistic
        and is therefore discarded.
    accthres: float, optional
        The acceleration threshold for the filtered data in °/s^2. Any acceleration bigger than this value is physiologically 
        unrealistic and is therefore discarded.
    a: float, optional
        Weight factor of the global noise for saccade offset detection
    b: float, optional
        Weight factor of the local noise for saccade offset detection
    saconidx: integer, optional
        In one instance, saccade onset cannot be reliably detected by the algorithm, since the onset is at 0. Thus, this
        optional variable is used to specify the saccade onset index manually when needed.
    velperc: float, optional
        Percentile for the saccade velocity threshold. Can be between 0 and 100.
    Returns
    -------
    saconidx: integer
        The index of the saccade onset
    sacoffidx: integer
        The index of the saccade offset
    """
    radius = np.int(4 * smoothsigma**2 + 0.5)

    x = np.arange(-radius, radius+1)
    gausskern = np.exp(-0.5 / smoothsigma**2 * x ** 2)
    gausskern[x<-10] = 0 #causal kernel, so for x<0 all values are zero
    gausskern /= np.sum(gausskern)
    smoothdata = correlate1d(rawdata, gausskern[::-1])
        
    velocitydata = np.abs(savgol_filter(smoothdata, savgollength, savgolorder, deriv=1) * rf) #filtered LE velocity in °/s
    #if the last velocity point in the data is extremely high, this likely indicates a recording artifact, so velocity is set to 0
    if velocitydata[-1] > 20: 
        velocitydata[-50:] = 0
    #discard velocity values bigger than threshold
    velocitydata[velocitydata>velthres] = None
    accdata = np.abs(savgol_filter(smoothdata, savgollength, savgolorder, deriv=2) * rf**2) #filtered LE acceleration in °/s^2
    #discard data points exceeding acceleration threshold.
    velocitydata[accdata>accthres] = None
    
    #saccade velocity threshold estimation: iterative and data-driven approach
    sacvelthres = np.percentile(velocitydata[~np.isnan(velocitydata)],velperc) #set the initial threshold to 99th percentile to be on the safe side
    #iteration
    thres = False
    iternum = 0
    stdval = 6
    while thres == False:
        iternum += 1
        underthres = velocitydata[velocitydata<sacvelthres]
        previoussacvelthres = sacvelthres
        sacvelthres = np.mean(underthres[~np.isnan(underthres)]) + stdval*np.std(underthres[~np.isnan(underthres)])
        if sacvelthres >= np.max(velocitydata):
            stdval -= 0.5
        elif np.abs(previoussacvelthres-sacvelthres) < 1:
            #print(iternum)
            thres = True
    
    #saccade onset detection: saccade onset velocity threshold is defined as mean+3*std for eye traces lower than peak threshold
    uthresidx = np.where(velocitydata>=sacvelthres)[0][0] #index of first peak exceeding velocity threshold
    if uthresidx == 0:
        uthresidxs = np.where(velocitydata>=sacvelthres)[0] #indices of the datapoints over threshold
        print('first index skipped')
        #first datapoints likely to have some fluctuations, thus take the index which shows a difference bigger than 1 
        #compared to previous index.
        uthresidx = uthresidxs[np.where(np.diff(uthresidxs)>1)[0][0]+1] 
    underthres = velocitydata[0 : uthresidx]
    onstd = 3
    saconthres = np.mean(underthres) + onstd*np.std(underthres)
    saconidx = np.where((underthres[1:] < saconthres) & (np.diff(underthres)>=0))[0][-1] 
    
    
    #Saccade offset detection: choose the leftmost velocity peak and search forward from there to find wished saccade.
    offpeakidx = np.where(velocitydata>=sacvelthres)[0][-1]
    underthres = velocitydata[offpeakidx:] #saccade velocity trace from last velocity peak on
    
    if saconidx < 40:
        presaccade = velocitydata[:saconidx] #velocity curve before saccade onset 
    else:
        presaccade = velocitydata[saconidx-40:saconidx] #velocity curve before saccade onset
    
    noisefacs = False
    while noisefacs == False:
        noisefac = np.mean(presaccade) + 3*np.std(presaccade) #adaptive noise factor
        sacoffthres = a*saconthres + b*noisefac
        
        while sacoffthres < np.min(underthres):
            a += 0.005
            b += 0.005
            sacoffthres = a*saconthres + b*noisefac
        sacoffidx = np.where((underthres[1:] < sacoffthres) & (np.diff(underthres)<=0))[0][0] + offpeakidx
        
        glitw = 400 #glissade time window
        if sacoffidx + glitw > len(velocitydata):
            glitw = len(velocitydata) - glitw #if time window bigger than total eye trace length, it is set to the end point of the trace
        else:
            pass
        
        if True in (velocitydata[sacoffidx:sacoffidx+glitw] >= sacoffthres):
           print("glissade detected")
           #Find the glissade offset
           glipeakidx = np.where(velocitydata[sacoffidx:sacoffidx+glitw]>=sacoffthres)[0][-1] + sacoffidx
           gliunderthres = velocitydata[glipeakidx:]
           if len(np.where((gliunderthres[1:] < sacoffthres) & (np.diff(gliunderthres)<=0))[0]) > 0:
               noisefacs = True
            
           else:
               a += 0.005
               b += 0.005
               continue
               
           if len(gliunderthres) == 1:
               sacoffidx = glipeakidx
           else:
               sacoffidx = np.where((gliunderthres[1:] < sacoffthres) & (np.diff(gliunderthres)<=0))[0][0] + glipeakidx    
        else:
            noisefacs = True
            
    return saconidx, sacoffidx, smoothdata


#Fit function from Dai et al. 2016
def saccade_fit_func(t, c, nu, tau, t_0, s_0):
    """
    Parametric fit function for saccade trace from Dai et al. 2016
    
    Parameters
    ----------
    t: 1-D array
        The time array in ms
    c: float
        Model parameter
    nu: float
        Model parameter
    tau: float
        Model parameter
    t_0: float
        Model parameter
    s_0: float
        Model parameter
        
    Returns
    -------
    fitfunc: 1-D array
        The fitted function to the saccade trace    
    """
    comp1 = c*saccade_fit_func_f(nu*(t-t_0)/c)
    comp2 = -c*saccade_fit_func_f(nu*((t-t_0)-tau)/c)
    fitfunc = comp1 + comp2 + s_0    
    #if return_comps == True:
    #    return fitfunc, comp1, comp2
    #else:
        #return fitfunc
    return fitfunc


def saccade_fit_func_f(t):
    """
    Part of the fit function to be used in the saccade fitting.
    
    Parameters
    ----------
    t: 1-D array
        The time array
    
    Returns
    -------
    f: 1-D array
        Ramp function used for saccade fitting
    """
    f = np.zeros(len(t))
    f[t<=0] = 0.25*np.e**(2*t[t<=0])
    f[t>=0] = t[t>=0] + 0.25*np.e**(-2*t[t>=0])
    return f

def linear_fit(x, a, b):
    """
    Function used for linear fit.
    
    Parameters
    ----------
    x: 1-D array
        The data used for the linear fit
    a: float
        Slope of the fit curve
    b: float
        Offset of the fit curve
    """
    return a*x+b


class cylindrical_random_pix_stimulus:
    """
    Drifting random pixel stimulus used in Giulia Soto's experiments. There are 2 functions. First one (__init) 
    generates the image array, the second one returns the current stimulus for the given time. This approach is 
    chosen for minimizing the RAM usage.
    """
    
    def __init__(self, rsl, shiftmag, tdur, fps=200, arenaaz=168, maxelev=40):
        """
        Generate the stimulus image array

        Parameters
        ----------
        arenaaz: float
            Azimuth of the arena in degrees. Note that this value spans in both positive and negative.
        maxelev: float
            Maximum elevation of the cylindrical arena in degrees. Note that this value spans 
            in both positive and negative.
        rsl: float
            Stimulus resolution in pixel per degrees.
        shiftmag: float
            Total amount of stimulus shift in degrees.
        tdur: float
            Duration of the stimulus shift in ms
        fps: int
            Number of frames per second
            
        Returns
        -------
        None.

        """
        #define initial parameters
        self.arenaaz = arenaaz
        self.maxelev = maxelev
        self.rsl = rsl
        self.shiftmag = shiftmag 
        self.tdur = tdur
        self.fps = fps
        
        #calculations about the cylinder parameters
        self.maxz = self.maxelev #maximum z value (cylinder height). Note that z extends between +- this value.
                                 #This value is set to the maximum elevation for implementation ease.
        self.radius = self.maxz / np.tan(np.deg2rad(self.maxelev))
        
        #frame and stimulus timing calculations
        self.frametot = np.int(fps*tdur/1000) #total number of frames (excluding start position)
        self.shiftperfr = np.round(self.shiftmag/(tdur/1000)/fps*self.rsl) #shift size in pixels per frame
        
        #generate the stimulus array:
        self.zrest = 5 #additional extension to z, so that smallest stimulus unit issue is less pronounced
        #make the stimulus coarser: in Giulia's matlab code, smallest unit in stimulus is 5x5 pixels, and resolution is 1.5 pix/deg. 
        #Thus, one side of the stimulus is 3.333 deg. 
        stimunit = np.round(rsl*5/1.5).astype(int) #length of the stimulus unit in pixels (i.e. length of one pixel side)
        #each stimulus unit should be stimunit*stimunit in size (cropping in visual field limits is allowed)
        #total stimulus array : azimuth is extended +-20°, so that shifts in both to the left and to the right can be chosen. 
        #rsl*elevrest pixels in elevation are additional
        #Generate coarse grating
        coarsegrating = np.random.rand(np.ceil((self.maxz*2+self.zrest)*rsl/stimunit).astype(int), \
                                       np.ceil(np.int(arenaaz+2*self.shiftmag)*rsl/stimunit).astype(int)) < 0.5 
        coarseimg = Image.fromarray(coarsegrating)
        #resize the coarse grating to fit to the whole visual field
        coarseimg = coarseimg.resize(np.flip(np.array(coarsegrating.shape)*stimunit), resample=Image.NEAREST)                               
        #final stimulus array
        self.finalstim = np.array(coarseimg)

    
    def move_stimulus(self, frameidx, shiftdir):
        """
        Get the current stimulus frame for the given frame index

        Parameters
        ----------
        frameidx: int
            Index of the current frame. This value is between 0 and self.frametot
        shiftdir: str
            Direction of shift, can be 'right' or 'left'.

        Returns
        -------
        finalarr : 2-D array
            Stimulus in current frame represented in whole visual field.

        """
        #ensure correct direction is specified.
        dircorrect = False
        while dircorrect == False:
            if shiftdir == 'right':
                dirfac = -1 #direction factor ensuring correct slice is taken for the directional shift
                dircorrect = True    
            elif shiftdir == 'left':
                dirfac = 1
                dircorrect = True
            else:
                shiftdir = input('Wrong shift direction given. Please write "right" or "left" \n')
        
        #start and stop indices for slicing the current frame from the big finalstim array
        startidx, stopidx = (np.array([self.shiftmag,self.arenaaz+self.shiftmag])*self.rsl \
                             + dirfac * frameidx*self.shiftperfr).astype(int)
        
        #preallocate current frame
        currentframe = np.zeros([2*self.maxz*self.rsl,self.arenaaz*2*self.rsl])
        #determine current frame (first half)
        currentframe[:,:self.arenaaz*self.rsl] = \
            self.finalstim[self.zrest*self.rsl:(self.zrest+2*self.maxz)*self.rsl, startidx:stopidx]
        #copy the stimulus to the other cylindrical half
        currentframe[:,self.arenaaz*self.rsl:] = currentframe[:,:self.arenaaz*self.rsl]
        #extend the stimulus array to whole visual field.
        finalarr = np.zeros(np.array([180,360])*self.rsl)
        elidxs = self.rsl*np.array([np.int(90-self.maxelev), np.int(90+self.maxelev)]) #elevation indices for the stimulus
        azidxs = self.rsl*np.array([np.int(180-self.arenaaz), np.int(180+self.arenaaz)]) #azimuth indices for the stimulus
        finalarr[elidxs[0]:elidxs[1], azidxs[0]:azidxs[1]] = currentframe

        return finalarr
        

class coordinate_transformations():
    """
    Class for functions to convert geographical coordinates to cartesian and vice versa
    """
    
    def car2geo(x,y,z):
        """
        Convert cartesian coordinate system to geographical.
                
        Parameters
        ----------
        x : float/ 1-D array
            x value in cartesian coordinates.
        y : float/ 1-D array
            y in cartesian coordinates.
        z : float/ 1-D array
            z in cartesian coordinates.

        Returns
        -------
        r: float/ 1-D array
            radius (distance from origin)
        azimuth: float/ 1-D array
            azimuth in deg (horizontal angle)
        elevation: float/ 1-D array
            elevation in deg (vertical angle)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        elevation = np.arcsin(z / r)
        azimuth = np.arctan2(y, x)
        
        return r, np.rad2deg(azimuth), np.rad2deg(elevation)
    

    def geo2car(r,azimuth,elevation):
        """
        Convert geographical coordinate system to cartesian.

        Parameters
        -------
        r: float/ 1-D array
            radius (distance from origin)
        azimuth: float/ 1-D array
            azimuth in deg (horizontal angle)
        elevation: float/ 1-D array
            elevation in deg (vertical angle)
                
        Returns
        ----------
        x : float/ 1-D array
            x value in cartesian coordinates.
        y : float/ 1-D array
            y in cartesian coordinates.
        z : float/ 1-D array
            z in cartesian coordinates.    
        """
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        
        return x, y, z    
    
  
def crop_rf_geographic(azidx, elevidx, radius, rsl, radiusfac=1):
    """
    Generate the receptive field mask for the given radius in geographical coordinates

    Parameters
    ----------
    azidx: int
        Index of the azimuth angle for given rsl
    elevidx: int
        Index of the elevation angle for given rsl
    radius: float
        Radius of the receptive field along elevation in degrees.
    rsl: float
        Image resolution in pixel per degrees.
    radiusfac: float, optional
        Factor determining the ratio between horizontal and vertical radius. The default is 1 leading to circle.
        For now elliptic shape is not implemented.

    Returns
    -------
    circarr: 2-D array
        The receptive field mask array in geographical coordinates.
    ccent: 1-D array
        Receptive field center in geographic coordinates (azimuth, elevation) in degrees
    ccentxyz: 1-D array
        Receptive field center in cartesian coordinates
    """
    #generate the grids for geographical and cartesian coordinates
    gaz, gel = np.meshgrid(np.linspace(-180,180,np.int(360*rsl)), np.linspace(90,-90,np.int(180*rsl))) 
    r = np.ones(gaz.shape) #radius (sphere) constant 1
    cx, cy, cz = coordinate_transformations.geo2car(r, gaz, gel)#cartesian coordinates

    #circle center in geographical coordinates
    ccent = np.array([gaz[elevidx,azidx], gel[elevidx,azidx]])
    
    #calculate the circle radius in cartesian coordinates
    addidx = np.int(radius*rsl) #index to be added to elevation for defining the circle radius
    if elevidx + addidx < gaz.shape[0]:
        cedge = np.array([gaz[elevidx,azidx], gel[elevidx+addidx,azidx]]) #circle edge offset in elevation
    else:
        cedge = np.array([gaz[elevidx,azidx], gel[elevidx-addidx,azidx]]) #circle edge offset in elevation
    
    #RF center in cartesian
    ccentx = cx[(gaz == ccent[0]) & (gel == ccent[1])]
    ccenty = cy[(gaz == ccent[0]) & (gel == ccent[1])]
    ccentz = cz[(gaz == ccent[0]) & (gel == ccent[1])]
    #edge in cartesian
    cedgex = cx[(gaz == cedge[0]) & (gel == cedge[1])]
    cedgey = cy[(gaz == cedge[0]) & (gel == cedge[1])]
    cedgez = cz[(gaz == cedge[0]) & (gel == cedge[1])]

    #circle radius in cartesian
    crc = np.sqrt((ccentx-cedgex)**2 + (ccenty-cedgey)**2 + (ccentz-cedgez)**2) #along y direction
    #crcx = radiusfac*crc
    
    #circle crop mask (ELLIPSE TO BE IMPLEMENTED LATER...)
    circarr = ((cx-ccentx))**2 + ((cy-ccenty))**2 + ((cz-ccentz))**2 <= crc**2
    return circarr, ccent, np.squeeze(np.array([ccentx, ccenty, ccentz]))


def generate_RF_geographic(sf, sz, ph, azidx, elevidx, rsl, radiusfac=1):
    """
    Generate the receptive field in geographic coordinates

    Parameters
    ----------
    sf: float
        RF spatial frequency in cyc/deg.
    sz: float
        RF radius in deg.
    ph: float
        Gabor phase offset in deg.
    azidx: int
        RF center azimuth index.
    elevidx: int
        RF center elevation index.
    rsl: float
        Image resolution in pix/deg.
    radiusfac: float, optional
        Ratio factor between horizontal and vertical radii. The default is 1. Cylindrical shape not yet implemented

    Returns
    -------
    filterimg: 2-D array
        Gabor RF array in geographical coordinates.
    rfcentcar: 1-D array
        The center of the receptive field in cartesian coordinates, order is x,y,z.
    """
    #Generate RF mask
    maskparams = [azidx, elevidx, sz, rsl, radiusfac]
    rfarr, rfcent, rfcentcar = crop_rf_geographic(*maskparams)
    
    #Generate first the rectangular Gabor in geographical coordinates
    degrees = np.linspace(0,360, 360*rsl)
    gaborsin = np.sin(2*np.pi * sf * degrees - rfcent[0] + ph)
    filterimg = np.tile(gaborsin,[180*rsl,1])
    

    filterimg[rfarr==False] = 0
    return filterimg, rfcentcar
