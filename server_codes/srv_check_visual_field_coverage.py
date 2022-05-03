# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 14:06:29 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
#from pathlib import Path
import sys
#sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import tarfile
import os
import matplotlib.pyplot as plt

#check the RF coverage in the visual field. 
#General parameters -> some of those will be sys argv arguments
seed = 450
animals = ['macaque', 'zebrafish'] 
nfilterss = (2*np.floor(np.logspace(np.log10(10),np.log10(72),5))**2).astype(int) #50 #sys argv -> Double the amount of RF for macaque
nfilterss = np.concatenate([nfilterss, (2*np.floor(np.logspace(np.log10(100),np.log10(200),3))**2).astype(int)])
rsl = 20 #pixel per degrees
jsigma = 4

#Plot style: General figure parameters:
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 20,
           'ytick.labelsize' : 20,
           'legend.fontsize' : 20,
           'figure.titlesize' : 25,
           'image.cmap' : 'gray'}
plt.style.use(figdict)


np.random.seed(seed)

#Simplify even further -> 1D coverage map along azimuth, 

fig, axs = plt.subplots(2,2, sharex='col') 

for a, animal in enumerate(animals):
    print('Animal : %s \n'%animal)
    
    if animal == 'zebrafish':
        az = 180
    else:
        az = 90
        
    azimuth = np.linspace(-az,az, 2*az*rsl+1) #azimuth array    
    
    for j, nfilters in enumerate(nfilterss):
        coverage = np.zeros(azimuth.shape) #coverage array
        print('nfilters : %s \n'%nfilters)
        #Prepare species parameters
        if animal == 'zebrafish':
            m = False #macaque is false
            speciesparams = hlp.generate_species_parameters(nfilters)
            params = speciesparams.zebrafish_updated()

        elif animal == 'macaque': #needs update, macaque part not yet done, e.g. visual field has to be limited to +-90° azimuth
            m = True #macaque is true now    
            nfilters = int(np.ceil(np.sqrt(nfilters))**2)    
            speciesparams = hlp.generate_species_parameters(nfilters)
            params = speciesparams.macaque_updated()

        #shuffle rf locations       
        __, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, params=params, 
                                                                  macaque=m)
        
        #get only azimuth
        fltcenters = fltcenters[:, 0]
        #phase shifted second set of filters with all the same parameters except with a phase shift as much as rf radius
        fltcenters2 = fltcenters.copy()
        params = np.squeeze(params)

        for idx, param in enumerate(params):
            rds = param[1]*rsl/2 #RF radius in pixels
            fltcenters2[idx] += rds
    
        if animal == 'zebrafish':
            #loop around the index (azimuth) if it exceeds current max index
            fltcenters2[:][fltcenters2[:]>=360*rsl] -= 360*rsl
        else:
            #For macaque, if value exceeds +90 azimuth, set the second filter center location to +90 azimuth.
            fltcenters2[:][fltcenters2[:]>=180*rsl] = 180*rsl - 1 #-1 since 
        
        #convert back to degrees
        fltcenters = fltcenters/rsl - az
        fltcenters2 = fltcenters2/rsl - az
        
        mns = fltcenters - params[:,1] / 2
        mxs = fltcenters + params[:,1] / 2
                
        mns2 = fltcenters2 - params[:,1] / 2
        mxs2 = fltcenters2 + params[:,1] / 2
                

        for i in range(len(mns)):
            if mns[i] < -180:
                #in this case the RF is split in edges of the visual field
                mn = mns[i]+360 #this is the left most part of the RF
                coverage[azimuth>=mn] += 1
                coverage[azimuth<=mxs[i]] += 1 #cover also the right most part of the RF
            elif mxs[i] > 180:
                #in this case the RF is split in edges of the visual field
                mx = mxs[i] - 360
                coverage[azimuth<=mx] += 1
                coverage[azimuth>=mns[i]] += 1 #cover also the right most part of the RF
            else:
                coverage[(azimuth>=mns[i]) & (azimuth<=mxs[i])] += 1            
            
            if mns2[i] < -180:
                #in this case the RF is split in edges of the visual field
                mn = mns2[i]+360 #this is the left most part of the RF
                coverage[azimuth>=mn] += 1
                coverage[azimuth<=mxs2[i]] += 1 #cover also the right most part of the RF
            elif mxs2[i] > 180:
                #in this case the RF is split in edges of the visual field
                mx = mxs2[i] - 360
                coverage[azimuth<=mx] += 1
                coverage[azimuth>=mns2[i]] += 1 #cover also the right most part of the RF
            else:
                coverage[(azimuth>=mns2[i]) & (azimuth<=mxs2[i])] += 1            

        axs[0, a].plot(azimuth,coverage, '.-', label=nfilters)
        axs[1, a].plot(azimuth,coverage>0, '.', label=nfilters)
    axs[0, a].legend(loc='best', bbox_to_anchor=(1,1))
    
axs[0,0].set_ylabel('Number of RFs')
axs[0,0].set_title('Macaque')
axs[0,1].set_title('Zebrafish')
fig.suptitle('Visual field coverage maps')
axs[1,0].set_ylabel('RF coverage')
axs[1,0].set_xlabel('Azimuth [°]', x=1.25)

for ax in axs[1,:]:
    ax.set_yticks([0,1])
    ax.set_yticklabels(['No', 'Yes'])

plt.get_current_fig_manager().window.showMaximized()
plt.subplots_adjust(top=0.897, bottom=0.09, left=0.058, right=0.887, hspace=0.097, wspace=0.495)
