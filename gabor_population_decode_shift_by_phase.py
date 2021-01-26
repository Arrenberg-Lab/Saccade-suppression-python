# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:34:39 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
from zf_helper_funcs import rt

#Try the phase shift decoder with the location decoder model
rsl = 1 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
sigma = 100*rsl #the standard deviation of the initial gabor filter
rdot = 17*rsl #the radius of the dot in pixels
jsigma = 4 #the standard deviation of the jitter noise in degrees
centoffs = np.round(np.logspace(0, np.log10(100), 5))
centoffs[0] = 0

imgs = np.zeros([len(centoffs), *(180, 360)*rsl]) #preallocated array for each stimulus image
ccents = np.zeros([len(centoffs), 2]) #preallocated array for the circle centers, first dimension is image idx second 
                                      #dimension is x and y coordinates.
for idx, off in enumerate(centoffs):
    img, circcent = hlp.circle_stimulus(rdot, rsl, *(50, 50)+off)    
    imgs[idx,:,:] = img
    ccents[idx,:] = circcent

#Generate Gabor filters of different spatial frequency and size
gfsz = [20, 30, 60] #the filter sizes in degrees
gfsf = [0.05, 0.1, 0.15, 0.2] #the filter spatial frequencies in degrees
gfph = np.arange(0,360, 45) #the filter phases in degrees
phidx = 0 #the phase index to plot the filters

gfilters = [[] for _ in range(len(gfsz))] 

for i, sz in enumerate(gfsz):
    for j, sf in enumerate(gfsf):
        fltarray = np.zeros([sz,sz, len(gfph)])
        
        for k, ph in enumerate(gfph):
            _, flt = hlp.crop_gabor_filter(rsl, sigma, sf, sz, ph)
            print('The filter with the size %d, phase %d and spatial frequency %.2f is generated' %(sz, ph, sf))
            fltarray[:,:,k] = flt #the last dimension denotes filters of different phases
            
        gfilters[i].append(fltarray)
       
gfilters = np.array(gfilters, dtype=object)

for img in imgs:
    
    gacts, pext = hlp.population_activity(gfsf, gfph, gfsz, gfilters, img)
    
    fltspersizeidx = [(j,k) for j in range(gacts.shape[0]) for k in range(gacts.shape[1])]
    
    #rearrange the filter activity array to be sublists separated by size, each sublist has the all filter activity per
    #each image patch
    fltactsperpatch = [[] for _ in range(gacts.shape[2])] #each filter activity sorted by size (in the sublists) and by
    #the patch location (each sublist has the shape patch number * spatial frequency * phase)
    
    #iterate separately for each filter size
    for sidx in range(gacts.shape[2]):
        fltspersize = gacts[:,:,sidx] #the array containing all filter activities of same size for all image patches.
        patchsize = gfsz[sidx] #the size of each patch for the given filter set
        fltactspersize = np.zeros([np.product(np.array(img.shape)//patchsize), *fltspersize.shape]) 
        #the array containing the filter activites for all patches. Difference is each element corresponds to a patch
        for fltidx in fltspersizeidx:
            for pidx, patch in enumerate(fltspersize[fltidx]):
                #print(pidx, fltidx, patch)
                fltactspersize[tuple([pidx, *fltidx])] = patch #pidx, fltidx[0], fltidx[1]
        #print(fltactspersize)
        fltactsperpatch[sidx] = fltactspersize
    
    #1) Location: the easiest case: find for which patches at least 1 of the filters are showing activity, use this also
    #to sort out what filters show activity. You can do this for different size filters, the outcome should match anyways.
    #With the implementation below the size estimation is also solved.
    activepatches = []
    fltpatchactivities = [] #activity of the considered filters in the active patches, chosen by size.
    for szidx, patchszacts in enumerate(fltactsperpatch):
        sizepatches = []
        sizepatchactivity = []
        for pidx, patch in enumerate(patchszacts):
            if np.any(patch != 0) == True: #if this statement true at least 1 filter in the patch of the given size shows
                                          #some activity (reduction or increase from baseline) so note the location
                (xloc, yloc) = (pext[szidx,0][pidx], pext[szidx,1][pidx])
                sizepatches.append([xloc, yloc])
                sizepatchactivity.append(patch)
        fltpatchactivities.append(np.array(sizepatchactivity))
        activepatches.append(np.array(sizepatches))
    
    fltpatchactivities = np.array(fltpatchactivities)
    activepatches = np.array(activepatches)
    #Activepatches: first dimension size, second dimension number of patches evoking activity, third dimension if x or y, 
    #last dimension start and stop points along x or y.
    
    #In order to determine the location of the patches, start by the biggest filter set, create a logical array where 
    #within the whole visual field each patch with an activity has 1 or otherwise 0.
    #At the same time in this iteration, you can also determine the spatial frequency and phase
    loclogarr = np.ones(img.shape) #location logical array within the entire visual field
    sfphmaps = np.zeros([len(activepatches), *img.shape]) #the map for decoded spatial frequency and phase for the 
                                                              #given gabor filter set sorted by size in descending order.
    actmaps = np.zeros([len(activepatches), *img.shape]) #the activity map for decoded spatial frequency and phase 
                                                             #for the given gabor filter set sorted by size in descending
                                                             #order.
    sfphszarr = [] #the spatial frequency, phase and size outcomes sublist for all filters. First sublist is size sorted,
                   #within each sublist the nested sublists are for active patches containing the spatial frequency, phase
                   #, size, maximum activity and patch location parameters n the given order. Size sorting in descending 
                   #order
    for sidx, sizepatches in enumerate(np.flip(activepatches, 0)): #iterate over the filter sets sorted by size in descending
                                                                #order.    
        loclogszarr = np.zeros(img.shape) #generate the whole visual field logical array for each filter sorted 
                                            #by size    
        sfphmap = np.zeros(img.shape) #the map for each filter set sorted by size for the spatial frequency and phase
        actmap = np.zeros(img.shape) #the map for each filter set sorted by size for the maximum activity
        sfphszarrps = [] #the spatial frequency, phase and size outcomes sublist for each filter set (size sorted)                                                     
        for pidx, activepatch in enumerate(sizepatches): #iterate over each active patch within the same size filter set.
            popact = np.flip(fltpatchactivities)[sidx][pidx] #the population activity for the given size and given patch.
            maxactidx = np.squeeze(np.where(popact == np.max(popact))) #this returns the indices of the unit showing max
                                                                       #activity to be used for spatial frequency decoding 
            sf = gfsf[maxactidx[0]] #the decoded spatial frequency for the given patch with winner takes all
            ph = gfph[maxactidx[1]] #the decoded phase for the given patch winner takes all        
            sz = np.flip(gfsz)[sidx]
            _ , decodedgabor = hlp.crop_gabor_filter(rsl, sigma, sf, sz, ph, circlecrop=False) 
            loclogszarr[activepatch[1,0]:activepatch[1,1], activepatch[0,0]:activepatch[0,1]] = 1
            sfphmap[activepatch[1,0]:activepatch[1,1], activepatch[0,0]:activepatch[0,1]] = decodedgabor
            actmap[activepatch[1,0]:activepatch[1,1], activepatch[0,0]:activepatch[0,1]] = np.max(popact) / \
                                                                                            np.product(decodedgabor.shape)
            sfphszarrps.append(np.array([sf, ph, sz, np.max(popact), activepatch]))
        loclogarr *= loclogszarr
        sfphmaps[sidx, : ,:] = sfphmap
        actmaps[sidx, : ,:] = actmap
        sfphszarr.append(np.array(sfphszarrps))
    
    sfphszarr = np.array(sfphszarr)
    
    #x and pair indices off all pixels from top left to bottom right going from left to right then from up to down.
    xyidx = [(k,j) for j in range(img.shape[1]) for k in range(img.shape[0])] 
    
    decodedimg = np.zeros(img.shape)
    totalactmap = np.zeros(img.shape)
    for idx, actmap in enumerate(actmaps):
        for lidx in xyidx:
            if totalactmap[lidx] < actmap[lidx]:
                totalactmap[lidx] = actmap[lidx]
                decodedimg[lidx] = sfphmaps[idx][lidx]
    decodedimg *= loclogarr
    
    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(loclogarr)
    axs[0].set_title('Real image')
    axs[1].set_title('Population read-out')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Elevation [$^\circ$]')
    fig.text(0.475, 0.02, 'Azimuth [$^\\circ$]', size=20)
    fig.suptitle('Location decoding', size=30)
    
    fig, axs = plt.subplots(1,5, sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[0].set_title('Real image')
    axs[4].imshow(decodedimg)
    axs[4].set_title('Model readout')
    for idx, imgmap in enumerate(sfphmaps):
        axs[idx+1].imshow(imgmap)
        axs[idx+1].set_title('size=%d'%(np.flip(gfsz)[idx]))
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Elevation [$^\circ$]')
    axs[2].set_xlabel('Azimuth [$^\circ$]')
    fig.text(0.45,0.85,'Filter maps',size=30)
    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break
