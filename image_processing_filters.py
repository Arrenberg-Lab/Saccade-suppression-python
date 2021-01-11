# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:21:37 2020

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
from imageio import imread
from skimage.color import rgb2gray


#https://www.allaboutcircuits.com/technical-articles/two-dimensional-convolution-in-image-processing/
#a nice website for intuitive understanding of 2-D image convolution.

arr = imread(r'D:\ALPEREN\junk\IMG_9380.jpg') # 800x634x3 array
grarr = rgb2gray(arr)
example = False #if true, examples of gabor and gauss filters are shown

if example == True:
    #try a gabor kernel
    gkernparams = {'gwl': 10,
                   'theta': np.deg2rad(45),
                   'sigma_x' : 10,
                   'sigma_y' : 10,
                   'offset' : np.deg2rad(0),
                   'n_stds': 3}
    kerngab, gaborr, gabori = hlp.gabor_image_filter(grarr, **gkernparams)
    
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Gabor filtering of an image', size=25)
    axs[0].imshow(np.real(kerngab), cmap='gray')
    axs[1].imshow(gaborr, cmap='gray')
    axs[2].imshow(grarr, cmap='gray')
    axs[0].set_title('Gabor kernel')
    axs[1].set_title('Filtered image')
    axs[2].set_title('Raw image')
    
    gkernparams['theta'] = np.deg2rad(0)
    kerngab, gaborr, gabori = hlp.gabor_image_filter(grarr, **gkernparams)
    
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Gabor filtering of an image', size=25)
    axs[0].imshow(np.real(kerngab), cmap='gray')
    axs[1].imshow(gaborr, cmap='gray')
    axs[2].imshow(grarr, cmap='gray')
    axs[0].set_title('Gabor kernel')
    axs[1].set_title('Filtered image')
    axs[2].set_title('Raw image')
    
    #now try Gaussian
    sigma = 5
    gaussimg = hlp.gaussian_image_filter(grarr, sigma)
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Gaussian filtering of an image', size=25)
    axs[0].imshow(gaussimg, cmap='gray')
    axs[1].imshow(grarr, cmap='gray')
    axs[0].set_title('Filtered image')
    axs[1].set_title('Raw image')
    
else:
    #Gabor kernel
    gkernparams = {'gwl': 20,
                       'theta': np.deg2rad(45),
                       'sigma_x' : 10,
                       'sigma_y' : 10,
                       'offset' : np.deg2rad(0),
                       'n_stds': 3}
    kerngab = hlp.gabor_image_filter(None, **gkernparams, returnimg=False)
    
    #choose an image patch the size of Gabor filter
    
    #gabor filter x and y half size
    halfx = kerngab.shape[1]/2
    halfy = kerngab.shape[0]/2
    
    #minimum and maximum x and y pixel locations to choose as the center of the image patch.
    minmaxx = [np.ceil(halfx).astype(int), np.floor((grarr.shape[1]-halfx)).astype(int)]
    minmaxy = [np.ceil(halfy).astype(int), np.floor((grarr.shape[0]-halfy)).astype(int)] 
    randx = np.random.randint(*minmaxx)
    randy = np.random.randint(*minmaxy)
    imgpatch = grarr[np.floor(randy-halfy).astype(int) : np.floor(randy+halfy).astype(int), 
                     np.floor(randx-halfx).astype(int) : np.floor(randx+halfx).astype(int)]
    imgpatch /= np.max(imgpatch)
    gabresp = kerngab.real * imgpatch
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
    axs[0].imshow(kerngab.real)
    axs[1].imshow(gabresp)
    axs[2].imshow(imgpatch)

    #try the LIF
    #example 1: no input at all
    #stim = np.zeros(100000)
    #stimes = hlp.leaky_integrate_and_fire(stim,input_scaling=500, v_offset=0.2) #few spikes due to random noise
    
    ntimes = 100000
    t_delta = 5e-5
    #example 2: static image
    stim = np.ones(ntimes) * np.sum(gabresp)
    sptimes = hlp.leaky_integrate_and_fire(stim, input_scaling=500, v_offset=0.2)
    
    #example 3: best stimulus (the gabor filter itself as the grating)
    stimbest = np.ones(ntimes) * np.sum(kerngab.real * kerngab.real / np.max(kerngab.real))
    sptimesbest = hlp.leaky_integrate_and_fire(stimbest, input_scaling=500, v_offset=0.2)
    
    #example 4: baseline firing rate, where image patch is all white
    baseimg = np.ones(kerngab.shape)
    basegabresp = kerngab.real * baseimg
    stimbase = np.ones(ntimes) * np.sum(basegabresp)
    sptimesbase = hlp.leaky_integrate_and_fire(stimbase, input_scaling=500, v_offset=0.2)
    
    #check the firing rates (1/ISI), for now average. 
    meanfrbest = len(sptimesbest) / (ntimes*t_delta)
    meanfrimg = len(sptimes) / (ntimes*t_delta)
    meanfrbase = len(sptimesbase) / (ntimes*t_delta)
    #nice, that best firing rate is way above than image firing rate and baseline firing rate as expected.
    #the baseline firing rate is above the image patch firing rate, meaning that image does not contain the feature
    #filter is responsive to.
    
    #the LIF neuron response to the moving image
    tmove = np.arange(0,1000)
    x = randx
    y = randy
    angle = np.random.uniform(0,2*np.pi)
    speed = 10 #pixels per second
    imggabors = np.zeros(tmove.shape[0])
    for i in tmove:
        xnew, ynew = hlp.img_move(x,y,angle,speed/1000)
        imgpatch1 = grarr[np.floor(ynew-halfy).astype(int) : np.floor(ynew+halfy).astype(int), 
                          np.floor(xnew-halfx).astype(int) : np.floor(xnew+halfx).astype(int)]
        imgpatch1 /= np.max(imgpatch)
        gresp = np.sum(kerngab.real * imgpatch1)
        imggabors[i] = gresp
        x = xnew
        y = ynew
        
    #TODO: use a random image instead, value mean at 0.5        
'''
#There is a problem with the image color values, I am downloading a different file now.   
#data handling in the image dataset:
import os
root = r'E:\Ibrahim_images\cps20100428.ppm'
imglist = os.listdir(root)
asd = imread(r'E:\Ibrahim_images\cps20100428.ppm\\'+imglist[10], format='.ppm')
'''