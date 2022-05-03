# -*- coding: utf-8 -*-
"""
Created on Thu May 20 12:13:29 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

#Plot before and after results for coordinate transformations.

#General parameters
nfilters = 50
rsl = 5
jsigma = 4

speciesparams = hlp.generate_species_parameters(nfilters)
zfparams = speciesparams.zebrafish()
img = np.zeros(np.array([180,360])*rsl)
#shuffle rf locations
__, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, params=zfparams)

zfparams = np.squeeze(zfparams)

#preallocate arrays for the model
rfsimg = np.zeros(img.shape) #all receptive fields in geographical coordinates

#calculate the rf activities for the given stimulus.    
for i in range(len(zfparams)):    
    rfarr, rfcentcar = hlp.generate_RF_geographic(zfparams[i,0], zfparams[i,1]/2, zfparams[i,2], *fltcenters[i], rsl)
    rfsimg[rfarr!=0] = rfarr[rfarr!=0]
oldimg, __ = hlp.florian_model_species_edition(img, rsl, zfparams, fltcenters[:,0], fltcenters[:,1], imgrec=False) 
fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
axs[0].imshow(oldimg, origin='lower', extent=[-180,180,-90,90], vmin=-1, vmax=1)
axs[1].imshow(rfsimg, origin='lower', extent=[-180,180,-90,90], vmin=-1, vmax=1)
axs[0].set_ylabel('Elevation [°]')
axs[0].set_xlabel('Azimuth [°]')
axs[0].set_title('Old')
axs[1].set_title('New')
fig.suptitle('Example RF set (zebrafish)')

rfarr, __ = hlp.generate_RF_geographic(zfparams[10,0], zfparams[10,1]/2, zfparams[10,2], *fltcenters[10], rsl)
plt.figure()
plt.imshow(rfarr, origin='lower', extent=[-180,180,-90,90], vmin=-1, vmax=1)
plt.ylabel('Elevation [°]')
plt.xlabel('Azimuth [°]')
plt.title('Example RF')