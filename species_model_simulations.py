# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:19:23 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
import pandas as pd
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data\simulations'

#Species specific model simulations

#For now do only macaque and zebrafish, keep reading literature to get more info.

nfiltersarray = [128, 1058, 10082]
nsimul = 100

gfph = np.arange(0,360, 45) #the filter phases in degrees
#general parameters for image and model                 
rsl = 24 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
rdot = 5*rsl #the radius of the dot in pixels
jsigma = 4 #the standard deviation of the jitter noise in degrees
centoffs = np.round(np.logspace(0, np.log10(50), 3))
centoffs[0] = 0

#ZF AND MACAQUE STUFF ARE MOVED TO DISTINCT FUNCTIONS

#Everything looks good, you can start with simulations
#Preallocate arrays, first dim zf or mac, 2nd for nfilters, 3rd for centoffs, 4th for nsimul and last for xy
realcenters = np.zeros([2,len(nfiltersarray),len(centoffs),nsimul,2])
decodedcenters = np.zeros([2,len(nfiltersarray),len(centoffs),nsimul,2])

for nfidx, nfilters in enumerate(nfiltersarray):
    speciesparams = hlp.generate_species_parameters(nfilters)
    for nidx, nn in enumerate(range(nsimul)):
        print('Current simulation number: %i, nfilters=%i' %(nn+1, nfilters))
        #Automatically generate the parameters random for each species     
        zfparams = speciesparams.zebrafish()
        macparams = speciesparams.macaque()
        for nc, centoff in enumerate(centoffs):
            #shuffle stimulus position in accordance with the average distance between Gabor units.
            fltdist = 90 / np.sqrt(nfilters/2) #in degrees
            centoff += np.random.randint(np.round(-fltdist/2), np.round(fltdist/2), 2)
            #1st value x jitter 2nd y of the img
            
            #1st for zebrafish
            print('zebrafish, centoff (x,y)=%.2f,%.2f'%(centoff[0],centoff[1]))
            rsl = 5
            img, circcent = hlp.circle_stimulus(rdot, rsl, *(180, 90)+centoff)    
            _, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, \
                                                             params=zfparams)
            _, popact = hlp.florian_model_species_edition(img, rsl, zfparams, fltcenters[:,0], fltcenters[:,1])
        
            stimcenterzf = popact/np.sum(popact) @ fltcenters
            stimcenterzf /= rsl
            #add the values to arrays
            realcenters[0,nfidx,nc,nidx,:] = circcent
            decodedcenters[0,nfidx,nc,nidx,:] = stimcenterzf 
                       
            #now for macaque
            print('macaque, centoff (x,y)=%.2f,%.2f'%(centoff[0],centoff[1]))
            rsl = 24
            img, circcent = hlp.circle_stimulus(rdot, rsl, *(180, 90)+centoff)    
            _, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, \
                                                                 params=macparams)
            _, popact = hlp.florian_model_species_edition(img, rsl, macparams, fltcenters[:,0], fltcenters[:,1])
        
            stimcentermac = popact/np.sum(popact) @ fltcenters
            stimcentermac /= rsl
            #add the values to arrays
            realcenters[1,nfidx,nc,nidx,:] = circcent
            decodedcenters[1,nfidx,nc,nidx,:] = stimcentermac            
