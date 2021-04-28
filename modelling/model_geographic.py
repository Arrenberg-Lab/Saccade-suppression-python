# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:59:25 2021

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

#Generate the model in geographical coordinates
rsl = 5
jsigma = 4
n = 23
nfilters = 2*n**2
nsimul = 10
tdur = 50
shiftmag = 150

#use zf parameters
speciesparams = hlp.generate_species_parameters(nfilters)
zfparams = speciesparams.zebrafish()
img = np.zeros(np.array([180,360])*rsl)
#shuffle rf locations
__, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, params=zfparams)

zfparams = np.squeeze(zfparams)

#stimulus
stim = hlp.cylindrical_random_pix_stimulus(rsl, shiftmag, tdur)

#preallocate arrays for the model
rfsimg = np.zeros(img.shape) #all receptive fields in geographical coordinates
rfcents = np.zeros([nfilters,3]) #rf centers in cartesian coordinates, 2nd dimension in the order x,y,z
rfacts = np.zeros([nfilters,stim.frametot]) #rf activities, nfilters x total number of frames


#calculate the rf activities for the given stimulus.    
for i in range(len(zfparams)):    
    rfarr, rfcentcar = hlp.generate_RF_geographic(zfparams[i,0], zfparams[i,1]/2, zfparams[i,2], *fltcenters[i], rsl, radiusfac=1)
    rfsimg[rfarr!=0] = rfarr[rfarr!=0]
    rfcents[i,:] = rfcentcar #use the receptive field centers in cartesian coordinates for future decoding
    for j in range(stim.frametot):
        stimulus = stim.move_stimulus(j, 'right') #shift in positive
        rfact = np.sum(rfarr[rfarr!=0]*stimulus[rfarr!=0])
        rfacts[i,j] = np.abs(rfact)
    print(i)

#decode stimulus center using cartesian coordinates
stimcent = (rfacts/np.sum(rfacts, axis=0)).T @ rfcents
#convert the decoded stim center to geographical coordinates
__, stimaz, stimel = hlp.coordinate_transformations.car2geo(stimcent[:,0], stimcent[:,1], stimcent[:,2])
#Problem is with this decoding the radius becomes way smaller than 1, if we are only interested in azimuth and 
#elevation all is good. Else we have some issues at hand
shiftbef = np.array(hlp.coordinate_transformations.geo2car(1,0,0))
shiftaf = np.array(hlp.coordinate_transformations.geo2car(1,stim.shiftperfr/rsl,0))
realshift = np.sum((shiftaf-shiftbef)**2)

decshift = np.sum(np.diff(stimcent, axis = 0)**2, axis=1)
plt.plot(np.tile(realshift,len(decshift)), decshift, 'k.')
shiftdecerr = decshift-realshift