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

#Script for functions used in the saccadic suppression project

#Plot style: General figure parameters:
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 15,
           'ytick.labelsize' : 15,
           'legend.fontsize' : 15}
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
    N: integer:
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
    sacthres =  0.4+ 10*np.std(saccade[:15])
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