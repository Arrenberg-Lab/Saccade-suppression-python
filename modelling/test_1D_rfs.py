# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 19:36:49 2022

@author: Ibrahim Alperen Tunc
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs_ as hlp
import tarfile
import os
from IPython import embed
import matplotlib.pyplot as plt


#Test the 1-D RFs

seed = 666
nfilters = 100 #int(sys.argv[3]) #50 #sys argv -> Double the amount of RF for macaque
rsl = 10 #int(sys.argv[4])

np.random.seed(seed)

#Prepare species parameters - first zebrafish
speciesparams = hlp.generate_species_parameters(nfilters)
params = speciesparams.zebrafish_updated()
azilims = [0, 360] #azimuth limits

azvals = (np.random.uniform(*azilims, nfilters) * rsl).astype(int) #The pixel location of azimuth values

rfs = np.zeros([nfilters, 360*rsl])
covarr = np.zeros(360*rsl)
fig, ax = plt.subplots(2, sharex=True)  
for i, par in enumerate(params):
    azarr, rfarr = hlp.gaussian_rf_geog_1D(par[0], par[1], azvals[i], rsl)
    rfs[i, :] = rfarr
    covarr[rfarr!=0] += 1
    ax[0].plot(azarr-180, rfarr)

ax[1].plot(azarr-180, covarr)

ax[1].set_xlabel('Azimuth [°]')
ax[0].set_ylabel('Unit activity [a.u.]')
ax[1].set_ylabel('RF coverage')
fig.suptitle('1D receptive field and coverage map for zebrafish (n=%i)'%nfilters)

#Now macaque
nfilters = 5000 #int(sys.argv[3]) #50 #sys argv -> Double the amount of RF for macaque
speciesparams = hlp.generate_species_parameters(nfilters)
params = speciesparams.macaque_updated()
azvals = (np.random.uniform(*azilims, nfilters) * rsl).astype(int) #The pixel location of azimuth values

rfs = np.zeros([nfilters, 360*rsl])
covarr = np.zeros(360*rsl)
fig, ax = plt.subplots(2, sharex=True)  
for i, par in enumerate(params):
    azarr, rfarr = hlp.gaussian_rf_geog_1D(par[0], par[1], azvals[i], rsl)
    rfs[i, :] = rfarr
    covarr[rfarr!=0] += 1
    ax[0].plot(azarr-180, rfarr)

ax[1].plot(azarr-180, covarr)

ax[1].set_xlabel('Azimuth [°]')
ax[0].set_ylabel('Unit activity [a.u.]')
ax[1].set_ylabel('RF coverage')
fig.suptitle('1D receptive field and coverage map for macaque (n=%i)'%nfilters)
