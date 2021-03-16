# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 01:22:41 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import pandas as pd
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data\simulations'

#Species specific model simulations, this time stimulus shift (only in the horizontal for now) is completely randomized.

#CODE ALL READY, RUN AFTER CURRENT SIMULATION IS DONE
#NOTE: You DUMBO forgot to update rdot while changing resolution! DUDE rdot IS IN PIXELS ARE YOU DUMBDUMB? xD
#NOTE 2: AS OF THIS SIMULATION, the zebrafish SF distribution is log uniform (i.e. reciprocal) distribution.
#NOTE 3: THERE IS (STILL) AN ERROR IN THE DECODING ALGORITHM, YOU NEED TO CONVERT THE GEOGRAPHICAL SPACE TO THE
#CARTESIAN COORDINATES BEFORE READ-OUT; SO LEAVE OUT THE SIMULATIONS FOR NOW, THIS IS THE FIRST THING TO IMPROVE
#AFTER LAB REPORT!

#For now do only macaque and zebrafish, keep reading literature to get more info.
nfiltersarray = [128, 1058, 10082]
nsimul = 50 #reduced number due to time constraints.
nshifts = 50 #number of shifts to be considered, thus 51 stimuli including the central one.
gfph = np.arange(0,360, 45) #the filter phases in degrees
#general parameters for image and model                 
rsl = 24 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
jsigma = 4 #the standard deviation of the jitter noise in degrees
stimrang = 5 #the radius of the stimulus in degrees

#Preallocate arrays, first dim zf or mac, 2nd for nfilters, 3rd for centoffs, 4th for nsimul and last for xy
realcenters = np.zeros([2,len(nfiltersarray),nshifts+1,nsimul,2])
decodedcenters = np.zeros([2,len(nfiltersarray),nshifts+1,nsimul,2])
realcentersbeforejitter = np.zeros([2,len(nfiltersarray),nshifts+1,nsimul,2]) #the stimulus center before jitter
                                                                              #in degrees

for nfidx, nfilters in enumerate(nfiltersarray):
    speciesparams = hlp.generate_species_parameters(nfilters)
    for nidx, nn in enumerate(range(nsimul)):
        print('Current simulation number: %i, nfilters=%i' %(nn+1, nfilters))
        #Automatically generate the parameters random for each species     
        zfparams = speciesparams.zebrafish()
        macparams = speciesparams.macaque()
        centoffs = np.random.uniform(-180+5, 180-5, 51)
        centoffs[0] = 0
        realcentersbeforejitter[:, nfidx, :, nidx, 0] = centoffs
        realcentersbeforejitter[:, nfidx, :, nidx, 1] = 90
        for nc, centoff in enumerate(centoffs):
            #shuffle stimulus position in accordance with the average distance between Gabor units.
            fltdist = 90 / np.sqrt(nfilters/2) #in degrees
            sjitter = np.random.uniform(np.round(-fltdist/2), np.round(fltdist/2), 2) #stimulus jitter in x&y
            #1st value x jitter 2nd y of the img
            
            #1st for zebrafish
            rsl = 4
            rdot = stimrang*rsl
            centoffzf = np.round(centoff*rsl)/rsl
            sjitterzf = np.round(sjitter*rsl)/rsl
            print('zebrafish, centoff (x,y)=%.2f,%.2f'%(centoffzf+sjitterzf[0],sjitterzf[1]))
            img, circcent = hlp.circle_stimulus(rdot, rsl, 180+centoffzf+sjitter[0], 90+sjitter[1])    
            _, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, \
                                                             params=zfparams)
            _, popact = hlp.florian_model_species_edition(img, rsl, zfparams, fltcenters[:,0], fltcenters[:,1])

            stimcenterzf = popact/np.sum(popact) @ fltcenters
            stimcenterzf /= rsl
            #add the values to arrays
            realcenters[0,nfidx,nc,nidx,:] = circcent
            decodedcenters[0,nfidx,nc,nidx,:] = stimcenterzf 
                       
            #now for macaque
            rsl = 24
            rdot = stimrang*rsl
            centoffmac = np.round(centoff*rsl)/rsl
            sjittermac = np.round(sjitter*rsl)/rsl
            print('macaque, centoff (x,y)=%.2f,%.2f'%(centoffmac+sjitter[0],sjitter[1]))
            img, circcent = hlp.circle_stimulus(rdot, rsl, 180+centoffmac+sjitter[0], 90+sjitter[1])    
            _, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, \
                                                                 params=macparams)
            _, popact = hlp.florian_model_species_edition(img, rsl, macparams, fltcenters[:,0], fltcenters[:,1])
        
            stimcentermac = popact/np.sum(popact) @ fltcenters
            stimcentermac /= rsl
            #add the values to arrays
            realcenters[1,nfidx,nc,nidx,:] = circcent
            decodedcenters[1,nfidx,nc,nidx,:] = stimcentermac            

#Save the results
ddict = {'Species': np.repeat(['Zebrafish', 'Macaque'],np.product(realcenters.shape)/4),
         'Number of filters': np.tile(np.repeat(nfiltersarray, len(centoffs)*nsimul),2), 
         'Real center X' : realcenters[:,:,:,:,0].flatten(),
         'Real center Y' : realcenters[:,:,:,:,1].flatten(),
         'Real center X before jitter' : realcentersbeforejitter[:,:,:,:,0].flatten(),
         'Real center Y before jitter' : realcentersbeforejitter[:,:,:,:,1].flatten(),
         'Decoded center X' : decodedcenters[:,:,:,:,0].flatten(),
         'Decoded center Y' : decodedcenters[:,:,:,:,1].flatten(),
         'Simulation number' : np.tile(np.tile(np.arange(1,nsimul+1),len(centoffs)*len(nfiltersarray)),2)
        }

simuldf = pd.DataFrame.from_dict(ddict)
filename = r'\zf_mac_simul_nsimul=%s_rsl_mac=%i_rsl_zf=5_rdot=%i_jsigma=%i_nfilters=%s_nshifts=%s' \
           %(nsimul, rsl, rdot/rsl, jsigma, nfiltersarray, nshifts)
simuldf.to_csv(savepath+filename)

#NOTE THAT REAL CENTER BEFORE JITTER X ENTRY IS RELATIVE TO VISUAL FIELD CENTERS WHILE THE OTHER STIMULUS POSITION
#ENTRIES ARE 0-360 FOR X AND 0-180 FOR Y