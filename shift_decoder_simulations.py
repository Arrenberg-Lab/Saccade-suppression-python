# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:07:35 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
import pandas as pd
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data\simulations'

#Run the model many times for each image shift values, generate a violin plot for the decoder error distribution
#initial image values
rsl = 1 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
rdot = 10*rsl #the radius of the dot in pixels
jsigma = 4 #the standard deviation of the jitter noise in degrees
centoffs = np.round(np.logspace(0, np.log10(100), 5))
centoffs[0] = 0

#filter parameters
gfsz = np.round(np.logspace(np.log10(20), np.log10(60), 3) * rsl) #3 sizes sampled logarithmically between 10-100°
gfsz = np.ceil(gfsz / 2)*2 #ensure the filter sizes are even so defining the radius in pixels is not an issue.
gfsf = [0.1, 0.15, 0.2, 0.25] #the filter spatial frequencies in degrees
gfph = np.arange(0,360, 45) #the filter phases in degrees
sigma = 100*rsl #the standard deviation of the initial gabor filter
nfilters = 200 #number of filters
nsimul = 1000 #number of model simulations

decodingerrors = np.zeros([nsimul, len(centoffs)]) #the decoding error for each image shift and each simulation

#tgenerate the stimulus images.
imgs = np.zeros([len(centoffs), *(180, 360)*rsl]) #preallocated array for each stimulus image
ccents = np.zeros([len(centoffs), 2]) #preallocated array for the circle centers, first dimension is image idx second 
                                      #dimension is x and y coordinates.
scents = np.zeros([nsimul, len(centoffs), 2]) #preallocated array for the decoded centers, first dimension is
                                              #simulation idx, secon dimension image idx and last dimension is x and 
                                              #y coordinates.                                      

for idx, off in enumerate(centoffs):
    img, circcent = hlp.circle_stimulus(rdot, rsl, *(50, 50)+off)    
    imgs[idx,:,:] = img
    ccents[idx,:] = circcent

for n in range(nsimul): 
    print('Current simulation number: %i' %(n+1))
    #generate random gabor population in each simulation
    parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img)
    filtersarray, _ = hlp.florian_model_filter_population(nfilters, img, rsl, sigma, parameters, fltcenters[:,0],
                                                                                                      fltcenters[:,1])

    for idx, img in enumerate(imgs):
        #get the population decoding readout for each image
        _, stimcent = hlp.florian_model_population_activity(filtersarray, fltcenters, img)
        scents[n, idx,:] = stimcent

ddict = {'XY shift': np.tile(centoffs, nsimul),
         'Stimulus center X': np.tile(ccents[:,0], nsimul),
         'Stimulus center Y': np.tile(ccents[:,1], nsimul), 
         'Decoded center X': scents[:,:,0].flatten(), 
         'Decoded center Y': scents[:,:,1].flatten()}
simuldf = pd.DataFrame.from_dict(ddict)
filename = r'\nsimul=%s_rsl=%s_rdot=%s_jsigma=%s_centoffs=%s_gfsz=%s_gfsf=%s_gfph=%s_nfilters=%s' \
           %(nsimul, rsl, rdot, jsigma, centoffs, gfsz, gfsf, gfph, nfilters)
#maybe find a better way for saving the simulation parameters, like in a separate txt file or something.
simuldf.to_csv(savepath+filename)

#run the simulations for different neuron numbers, probably first make the above code to a function for readability.
#!!! np.sqrt(nfilters/2) has to be a rational number so that the visual scene tiling works properly.
nfilters = np.round(np.logspace(np.log10(np.sqrt(20/2)), np.log10(np.sqrt(1000/2)), 5))
nfilters = (2*nfilters**2).astype(int)
centoffs = np.round(np.logspace(0, np.log10(50), 3))
centoffs[0] = 0

#tgenerate the stimulus images.
imgs = np.zeros([len(centoffs), *(180, 360)*rsl]) #preallocated array for each stimulus image
ccents = np.zeros([len(centoffs), 2]) #preallocated array for the circle centers, first dimension is image idx second 
                                      #dimension is x and y coordinates.
scents = np.zeros([nsimul, len(nfilters), len(centoffs), 2]) #preallocated array for the decoded centers, first 
                                                             #dimension is simulation idx, second dimension the
                                                             # number of filters in the population, third dimension
                                                             # image idx  and last dimension is x and y coordinates.                                      

for idx, off in enumerate(centoffs):
    img, circcent = hlp.circle_stimulus(rdot, rsl, *(50, 50)+off)    
    imgs[idx,:,:] = img
    ccents[idx,:] = circcent

for n in range(nsimul): 
    print('Current simulation number: %i' %(n+1))
    for nidx, nfilter in enumerate(nfilters):
        #generate random gabor population in each simulation
        parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilter, rsl, gfsf, gfsz, gfph, jsigma, img)
        filtersarray, _ = hlp.florian_model_filter_population(nfilter, img, rsl, sigma, parameters, fltcenters[:,0]
                                                                                                    ,fltcenters[:,1])
        print('nfilter is %d' %(nfilter))
        
        for idx, img in enumerate(imgs):
            #get the population decoding readout for each image
            _, stimcent = hlp.florian_model_population_activity(filtersarray, fltcenters, img)
            scents[n, nidx, idx,:] = stimcent


ddict = {'XY shift': np.tile(centoffs, nsimul*len(nfilters)),
         'Stimulus center X': np.tile(ccents[:,0], nsimul*len(nfilters)),
         'Stimulus center Y': np.tile(ccents[:,1], nsimul*len(nfilters)), 
         'Decoded center X': scents[:,:,:,0].flatten(), 
         'Decoded center Y': scents[:,:,:,1].flatten(),
         'Number of filters': np.tile(np.repeat(nfilters,len(centoffs)), nsimul)}
simuldf = pd.DataFrame.from_dict(ddict)
filename = r'\nsimul=%s_rsl=%s_rdot=%s_jsigma=%s_centoffs=%s_gfsz=%s_gfsf=%s_gfph=%s_nfilters=%s' \
           %(nsimul, rsl, rdot, jsigma, centoffs, gfsz, gfsf, gfph, nfilters)
simuldf.to_csv(savepath+filename)

#Run the last simulation where image center also varies between +-avg(filterdistance)/2, this corresponds to randomly
#shifting the receptive fields in the direction opposite to the image shift. Do this for one image in the center 
#(i.e. image center is at 180,90 azimuth&elevation) and shift the image center randomly at each iteration.
ccents = np.zeros([nsimul, len(nfilters), 2]) #preallocated array for the circle centers, first dimension is
                                              #simulation idx, second dimension the number of filters in the
                                              #population and last dimension is x and y coordinates.

scents = np.zeros([nsimul, len(nfilters), 2]) #preallocated array for the decoded centers, first dimension is
                                              #simulation idx, second dimension the number of filters in the
                                              #population and last dimension is x and y coordinates.
for n in range(nsimul): 
    print('Current simulation number: %i' %(n+1))
    for nidx, nfilter in enumerate(nfilters):
        #generate random gabor population in each simulation
        parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilter, rsl, gfsf, gfsz, gfph, jsigma, img)
        filtersarray, _ = hlp.florian_model_filter_population(nfilter, img, rsl, sigma, parameters, fltcenters[:,0]
                                                                                                    ,fltcenters[:,1])
        print('nfilter is %d' %(nfilter))
        #generate the image, center offset sampled randomly from discrete uniform distribution for each iteration.
        fltdist = 90*rsl /np.sqrt(nfilter/2) #in pixels
        print('fltdist is %.2f' %(fltdist))
        off = np.random.randint(np.round(-fltdist/2), np.round(fltdist/2), 2) #1st value y jitter 2nd x of the img
        img, circcent = hlp.circle_stimulus(rdot, rsl, *(180, 90)+off)
        _, stimcent = hlp.florian_model_population_activity(filtersarray, fltcenters, img)
        scents[n, nidx, :] = stimcent
        ccents[n, nidx, :] = circcent

ddict = {'Stimulus center X': np.squeeze(ccents[:,:,0].flatten()),
         'Stimulus center Y': np.squeeze(ccents[:,:,1].flatten()), 
         'Decoded center X': np.squeeze(scents[:,:,0].flatten()), 
         'Decoded center Y': np.squeeze(scents[:,:,1].flatten()),
         'Number of filters': np.squeeze(np.tile(nfilters, nsimul))}
simuldf = pd.DataFrame.from_dict(ddict)
filename = r'\nsimul=%s_rsl=%s_rdot=%s_jsigma=%s_gfsz=%s_gfsf=%s_gfph=%s_nfilters=%s_nimg=1' \
           %(nsimul, rsl, rdot, jsigma, gfsz, gfsf, gfph, nfilters)
simuldf.to_csv(savepath+filename)