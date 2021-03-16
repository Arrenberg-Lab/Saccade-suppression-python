# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:15:31 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from scipy.stats import ttest_1samp

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
ntiling = 10 #number of patches of the visual space along elevation.
nfilters = 2*ntiling**2 #number of filters

parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img)
filtersarray, filtersimg = hlp.florian_model_filter_population(nfilters, img, rsl, sigma, parameters, fltcenters[:,0],
                                                                                                      fltcenters[:,1])

fig, ax = plt.subplots(1,1)
ax.imshow(filtersimg, vmin=-1, vmax=1, extent=(-180,180,-90,90))    
ax.set_ylabel('Elevation [$^\circ$]')
ax.set_xlabel('Azimuth [$^\circ$]')
ax.invert_yaxis()
ax.set_yticks(np.linspace(-90, 90, 5))
ax.set_xticks(np.linspace(-180, 180, 9))
ax.set_title('Receptive fields of the filter population')

#population activity
centoffs = np.array([0,1,5,15,30,50,20])
circcents = np.zeros([len(centoffs),2]) #real center
popacts = np.zeros([len(centoffs), len(parameters)])
stimcents = np.zeros(circcents.shape) #decoded center
imgs = np.zeros([len(centoffs),*(180,360)*rsl])
fltdist = 90 / np.sqrt(nfilters/2) #in degrees
for idx, centoff in enumerate(centoffs):
    #shuffle stimulus position in accordance with the average distance between Gabor units.
    stimjit = np.random.randint(np.round(-fltdist/2), np.round(fltdist/2), 2)
    img, circcent = hlp.circle_stimulus(rdot, rsl, *(180, 90)+centoff+stimjit) #works like magic :)
    circcents[idx, :] = circcent
    popact, stimcent = hlp.florian_model_population_activity(filtersarray, fltcenters, img)
    popacts[idx,:] = popact
    stimcents[idx,:] = stimcent
    imgs[idx,:,:] = img

circcents[:,0] -= 180
circcents[:,1] = 90 - circcents[:,1]
stimcents[:,0] -= 180
stimcents[:,1] = 90 - stimcents[:,1]

fig, axs = plt.subplots(3,3)
gs = axs[1,2].get_gridspec() #big subplot for the scatterplot
for ax in np.squeeze(axs[1:, :-1].reshape(1,4)):
    ax.remove()
axbig = fig.add_subplot(gs[1:, :-1])
axs = np.squeeze(axs.reshape(1,9))

iidx = 0
for idx in [0,1,2,5,8]:
    if iidx == 1:
        iidx += 1
    axs[idx].imshow(imgs[iidx,:,:], extent=(-180,180,-90,90))
    axs[idx].scatter(*circcents[iidx], label='Real', s=2)
    axs[idx].scatter(*stimcents[iidx], label='Decoded', s=10)
    axs[idx].set_title('Shift %.2f°' %(np.sqrt(2)*centoffs[iidx]/rsl))
    axs[idx].set_yticks(np.linspace(-90, 90, 5))
    axs[idx].set_xticks(np.linspace(-180, 180, 5))
    iidx += 1 #show the bigger shifts.

for ax in axs[[1,2]]:
    ax.set_yticklabels('')
for ax in axs[[2,5]]:
    ax.set_xticklabels('')    

axs[idx].legend(loc='best')
axs[0].set_ylabel('Elevation [$^\circ$]')
axs[-1].set_xlabel('Azimuth [$^\circ$]')
plt.subplots_adjust(hspace=0.533)

nsimul = 50
stimcircdiff = np.sqrt(np.diff(circcents[:,0])**2 + np.diff(circcents[:,1])**2)
deccircdiff = np.sqrt(np.diff(stimcents[:,0])**2 + np.diff(stimcents[:,1])**2)
decerr = np.zeros([len(stimcircdiff), nsimul])
#axbig.plot(stimcircdiff,deccircdiff, 'r.')  

maxdeccircdiff = np.max(deccircdiff)
mindeccircdiff = np.min(deccircdiff)
for n in range(nsimul):
    print(n)
    stimcents = np.zeros(circcents.shape) #decoded center
    parameters, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, gfsf, gfsz, gfph, jsigma, img)
    filtersarray, filtersimg = hlp.florian_model_filter_population(nfilters, img, rsl, sigma, parameters, 
                                                                   fltcenters[:,0], fltcenters[:,1])
    
    for idx, centoff in enumerate(centoffs):
        stimjit = np.random.randint(np.round(-fltdist/2), np.round(fltdist/2), 2)
        img, circcent = hlp.circle_stimulus(rdot, rsl, *(180, 90)+centoff+stimjit) #works like magic :)
        circcents[idx, :] = [180 , 90] + centoff
        popact, stimcent = hlp.florian_model_population_activity(filtersarray, fltcenters, img)
        popacts[idx,:] = popact
        stimcents[idx,:] = stimcent
        imgs[idx,:,:] = img
    stimcircdiff = np.abs(np.diff(centoffs)) * np.sqrt(2)
    deccircdiff = np.sqrt(np.diff(stimcents[:,0])**2 + np.diff(stimcents[:,1])**2)
    decerr[:, n] = deccircdiff - stimcircdiff
    if maxdeccircdiff < np.max(deccircdiff):
        maxdeccircdiff = np.max(deccircdiff)
    if mindeccircdiff > np.min(deccircdiff):
        mindeccircdiff = np.min(deccircdiff)
    axbig.plot(stimcircdiff,deccircdiff, 'r.')  
#axbig.plot([mindeccircdiff, maxdeccircdiff+5],[mindeccircdiff, maxdeccircdiff+5], 'k-')  
axbig.set_xlabel('Real shift [$^\circ$]')
axbig.set_ylabel('Decoded shift [$^\circ$]')
axbig.set_yticks(np.arange(0, np.ceil(maxdeccircdiff),10))
axbig.set_xticks(np.arange(0, np.ceil(maxdeccircdiff),10))
axbig.set_ylim(0, maxdeccircdiff+2)
axbig.set_xlim(0, maxdeccircdiff+2)
axbig.set_aspect('equal')
axbig.plot(axbig.get_xlim(), axbig.get_ylim(), 'k-')
plt.subplots_adjust(left=0.129, bottom=0.09, right=0.9, top=0.933, wspace=0.0, hspace=0.464)

decerr = decerr.reshape(np.product(decerr.shape))
fig, ax = plt.subplots(1,1)
ax.hist(decerr, density=True, bins=20)
ax.set_xlabel('Decoding error [°]')
ax.set_ylabel('Density')
ax.set_title('Pooled decoding error distribution')
t, p = ttest_1samp(decerr, 0, axis=0)
lineyvals = ax.get_ylim()
ax.plot([np.mean(decerr),np.mean(decerr)], lineyvals, 'r-', label='Mean')
ax.plot([0,0], lineyvals, 'k-')
