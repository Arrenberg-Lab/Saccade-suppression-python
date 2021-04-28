# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:19:23 2021

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

#Species specific model simulations

#For now do only macaque and zebrafish, keep reading literature to get more info.
#Update: also negative shifts, and calculate shifts between the stimulus in center.
nfiltersarray = [128, 1058, 10082]
nsimul = 200

gfph = np.arange(0,360, 45) #the filter phases in degrees
#general parameters for image and model                 
rsl = 24 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
jsigma = 4 #the standard deviation of the jitter noise in degrees
centoffp = np.round(np.logspace(0, np.log10(60), 4))
centoffp[0] = 0
centoffs = np.zeros(len(centoffp)*2-1) #Note that for now you change both x and y the same, in future you can change x and 
                                       #y independently
centoffs[:len(centoffp)-1] = -np.flip(centoffp[1:])
centoffs[len(centoffp)-1:] = centoffp


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
            rsl = 5
            rdot = 5*rsl
            print('zebrafish, centoff (x,y)=%.2f,%.2f, rsl=%i, rdot=%i'%(centoff[0],centoff[1], rsl, rdot))
            img, circcent = hlp.circle_stimulus(rdot, rsl, *np.array((180, 90))+centoff)    
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
            rdot = 5*rsl
            print('macaque, centoff (x,y)=%.2f,%.2f, rsl=%i, rdot=%i'%(centoff[0],centoff[1], rsl, rdot))
            img, circcent = hlp.circle_stimulus(rdot, rsl, *(180, 90)+centoff)    
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
         'Decoded center X' : decodedcenters[:,:,:,:,0].flatten(),
         'Decoded center Y' : decodedcenters[:,:,:,:,1].flatten(),
         'Simulation number' : np.tile(np.tile(np.arange(1,nsimul+1),len(centoffs)*len(nfiltersarray)),2)
        }

simuldf = pd.DataFrame.from_dict(ddict)
filename = r'\zf_mac_simul_nsimul=%s_rsl_mac=%i_rsl_zf=5_rdot=%i_jsigma=%i_nfilters=%s_centoffs=%s' \
           %(nsimul, rsl, rdot/rsl, jsigma, nfiltersarray, centoffs)
simuldf.to_csv(savepath+filename)


#Plot one exemplary parameter histogram for zf and mac each
sfzf = []
szzf = []
for params in zfparams:
    sfzf.append(params[0])
    szzf.append(params[1])

sfmac = []
szmac = []
for params in macparams:
    sfmac.append(params[0])
    szmac.append(params[1])

fig, axs = plt.subplots(2,2)
axs[0,0].hist(sfzf, bins=100, density=True)
axs[0,1].hist(szzf, bins=100, density=True)
axs[1,0].hist(sfmac, bins=100, density=True)
axs[1,1].hist(szmac, bins=100, density=True)
fig.suptitle('Parameter histograms')
axs[0,0].set_title('Spatial frequency')
axs[0,1].set_title('Size')
axs[0,0].set_ylabel('Density')
axs[1,0].set_xlabel('Spatial frequency [cyc/°]')
axs[1,1].set_xlabel('RF diameter [°]')
fig.text(0.5, 0.87, 'Zebrafish', size=25, horizontalalignment='center')
fig.text(0.5, 0.45, 'Macaque', size=25, horizontalalignment='center')
