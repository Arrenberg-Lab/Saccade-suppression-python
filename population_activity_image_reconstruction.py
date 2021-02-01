# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:34:39 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp

#Use the population activity model of FLorian's, weight each of the filter with their respective activity in order to
#reconstruct the image.
#initial image values
rsl = 1 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
rdot = 5*rsl #the radius of the dot in pixels
jsigma = 4 #the standard deviation of the jitter noise in degrees

#filter parameters
gfsz = np.round(np.logspace(np.log10(20), np.log10(60), 3) * rsl) #3 sizes sampled logarithmically between 10-100°
gfsz = np.ceil(gfsz / 2)*2 #ensure the filter sizes are even so defining the radius in pixels is not an issue.
gfsf = [0.1, 0.15, 0.2, 0.25] #the filter spatial frequencies in degrees
gfph = np.arange(0,360, 45) #the filter phases in degrees
sigma = 100*rsl #the standard deviation of the initial gabor filter
nfilters = 2048 #number of filters


#the image with the dot.
img, circcent = hlp.circle_stimulus(rdot, rsl, *(100, 100)) #works like magic :)
parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img)
filtersarray, filtersimg = hlp.florian_model_filter_population(nfilters, img, rsl, sigma, parameters, fltcenters[:,0],
                                                                                                      fltcenters[:,1])
#population activity - for now single image
popact, stimcent = hlp.florian_model_population_activity_img_reconstruction(filtersarray, fltcenters, img)

#weigh each filter with the respective population activity
weightedfilters = np.zeros(filtersarray.shape)
for idx, flt in enumerate(filtersarray):
    fltcontribution = flt/parameters[idx,1] * popact[idx] / np.sum(popact)
    weightedfilters[idx,:,:] = fltcontribution

reconstructedimg = np.sum(weightedfilters,0)

#the image with the dot.

img2, circcent2 = hlp.circle_stimulus(rdot, rsl, *np.array([100, 100])+60) #works like magic :)
parameters2, fltcenters2 = hlp.florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img2)
filtersarray2, __ = hlp.florian_model_filter_population(nfilters, img2, rsl, sigma, parameters, fltcenters2[:,0],
                                                                                                      fltcenters2[:,1])
#population activity - for now single image
popact2, stimcent2 = hlp.florian_model_population_activity_img_reconstruction(filtersarray2, fltcenters2, img2)

#weigh each filter with the respective population activity
weightedfilters2 = np.zeros(filtersarray2.shape)
for idx, flt in enumerate(filtersarray2):
    fltcontribution = flt/parameters2[idx,1] * popact2[idx] / np.sum(popact2)
    weightedfilters2[idx,:,:] = fltcontribution

reconstructedimg2 = np.sum(weightedfilters2,0)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].imshow(img)
axs[0,1].imshow(reconstructedimg)
axs[1,0].imshow(img2)
axs[1,1].imshow(reconstructedimg2)
axs[0,0].set_title('Real image')
axs[0,1].set_title('Reconstructed image')
axs[0,0].invert_yaxis()
axs[0,0].set_ylabel('Elevation [°]')
axs[1,0].set_xlabel('Azimuth [°]')