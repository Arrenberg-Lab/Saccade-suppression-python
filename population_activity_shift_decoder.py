# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:15:31 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp

#Florian's model of image shift decoding.
#initial image values
rsl = 1 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
rdot = 10*rsl #the radius of the dot in pixels
jsigma = 4 #the standard deviation of the jitter noise in degrees
#the image with the dot.
img, circcent = hlp.circle_stimulus(rdot, rsl, *(100, 100)) #works like magic :)

#filter parameters
gfsz = np.round(np.logspace(np.log10(20), np.log10(60), 3) * rsl) #3 sizes sampled logarithmically between 10-100°
gfsz = np.ceil(gfsz / 2)*2 #ensure the filter sizes are even so defining the radius in pixels is not an issue.
gfsf = [0.1, 0.15, 0.2, 0.25] #the filter spatial frequencies in degrees
gfph = np.arange(0,360, 45) #the filter phases in degrees
sigma = 100*rsl #the standard deviation of the initial gabor filter
nfilters = 200 #number of filters

parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img)
filtersarray, filtersimg = hlp.florian_model_filter_population(nfilters, img, rsl, sigma, parameters, fltcenters[:,0],
                                                                                                      fltcenters[:,1])

fig, ax = plt.subplots(1,1)
ax.imshow(filtersimg, vmin=-1, vmax=1)    
ax.set_ylabel('Elevation [$^\circ$]')
ax.set_xlabel('Azimuth [$^\circ$]')
ax.invert_yaxis()
ax.set_title('Receptive fields of the filter population')

#population activity
centoffs = np.array([0,1,5,10,20])
circcents = np.zeros([len(centoffs),2]) #real center
popacts = np.zeros([len(centoffs), len(parameters)])
stimcents = np.zeros(circcents.shape) #decoded center
imgs = np.zeros([len(centoffs),*(180,360)*rsl])
for idx, centoff in enumerate(centoffs):
    img, circcent = hlp.circle_stimulus(rdot, rsl, *(100, 100)+centoff) #works like magic :)
    circcents[idx, :] = circcent
    popact, stimcent = hlp.florian_model_population_activity(filtersarray, fltcenters, img)
    popacts[idx,:] = popact
    stimcents[idx,:] = stimcent
    imgs[idx,:,:] = img

fig, axs = plt.subplots(3,2)
for idx, ax in enumerate(np.squeeze(axs.reshape(1,6))):
    if idx == 5:
        break
    ax.imshow(imgs[idx,:,:])
    ax.scatter(*circcents[idx], label='Real', s=2)
    ax.scatter(*stimcents[idx], label='Decoded', s=2)
    ax.set_title('Shift %.2f pixels' %(np.sqrt(2)*centoffs[idx]))
    ax.invert_yaxis()
axs[1,1].legend(loc='best')
stimcircdiff = np.sqrt(np.diff(circcents[:,0])**2 + np.diff(circcents[:,1])**2)
deccircdiff = np.sqrt(np.diff(stimcents[:,0])**2 + np.diff(stimcents[:,1])**2)
axs[2,1].plot(stimcircdiff,deccircdiff, 'r.')  
axs[2,1].plot(stimcircdiff,stimcircdiff, 'k-')  
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,0].set_xticks([])
axs[2,1].set_xlabel('Real shift')
axs[2,1].set_ylabel('Decoded shift')
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.98, top=0.94, wspace=0.04, hspace=0.26)
axs[1,0].set_ylabel('Elevation [$^\circ$]')
axs[2,0].set_xlabel('Azimuth [$^\circ$]')
