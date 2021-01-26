# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:33:16 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
from zf_helper_funcs import rt

#Gabor population to decode the spatial frequency, location

#create image grating
#initial values
rsl = 1 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
imgsf = 0.2 #spatial frequency of the image in degrees

img = hlp.generate_grating(rsl, imgsf)

plt.imshow(img, vmin=0, vmax=1)
plt.colorbar()

"""
Generate exemplary gratings
fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
sfs = [0.1, 0.2]
for idx, sf in enumerate(sfs):   
    img, gr = hlp.generate_grating(rsl, sf)
    axs[idx].imshow(img)
    axs[idx].set_title(r'f = %.1f [$1/^\circ$]' %(sf))
    axs[idx].invert_yaxis()
    axs[idx].set_xlim(0,360)   
    axs[idx].set_ylim(0,180)
    axs[idx].set_xticks(np.linspace(0,360,9))   
    axs[idx].set_yticks(np.linspace(0,180,5))

axs[0].set_ylabel('Elevation [$^\circ$]')
axs[1].set_xlabel('Azimuth [$^\circ$]')
plt.subplots_adjust(left=0.07, bottom=0.11, right=0.98, top=0.88, wspace=0.07, hspace=0.2)
"""

"""
Regarding Gabor pixel size: Dumb thing is the filter extent is also dependent on the angle theta. Therefore, aim is  
to generate an initial Gabor with very big extent (huge sigma for both x and y) and then crop the appropriate size 
this huge initial Gabor out of
"""
sigma = 100*rsl #the standard deviation of the initial gabor filter

#Generate Gabor filters of different spatial frequency and size

gfsz = [20, 30, 60] #the filter sizes in degrees
gfsf = [0.05, 0.1, 0.15, 0.2] #the filter spatial frequencies in degrees
gfph = np.arange(0,360, 45) #the filter phases in degrees
phidx = 0 #the phase index to plot the filters

gfilters = [[] for _ in range(len(gfsz))] 
"""
Empty list to dump each Gabor filter one by one. Each empty sublist is for the filter size. This approach is chosen 
for future ease in filter sorting. Using array not possible as different filter sizes also mean different array sizes
"""

"""
#Quick gfilters example:
gfilters[0] #gives all filters of the size gfsz[0]
gfilters[0,1] #gives all filters of the size gfsz[0] and spatial frequency gfsf[1]
gfilters[0,1].shape #the shape is sz*sz*len(gfph), last dimension is for filter phase parameter.
"""

#empty figure for plotting the resulting filters for demonstration.
fig, axs = plt.subplots(len(gfsz), len(gfsf), sharex='row', sharey='row') 
for i, sz in enumerate(gfsz):
    for j, sf in enumerate(gfsf):
        fltarray = np.zeros([sz,sz, len(gfph)])
        
        for k, ph in enumerate(gfph):
            _, flt = hlp.crop_gabor_filter(rsl, sigma, sf, sz, ph)
            print('The filter with the size %d, phase %d and spatial frequency %.2f is generated' %(sz, ph, sf))
            fltarray[:,:,k] = flt #the last dimension denotes filters of different phases
            
        gfilters[i].append(fltarray)
        #show the 0 phase filter (or the first value in phase array)
        axs[i,j].imshow(fltarray[:,:,phidx], vmin=-1, vmax=1)
        axs[i,j].set_xticks(rsl*np.arange(0, sz+1, 10))
        axs[i,j].set_yticks(rsl*np.arange(0, sz+1, 10))
        axs[i,j].invert_yaxis()

gfilters = np.array(gfilters, dtype=object)
        
axs[1,0].set_ylabel('Elevation [$^\circ$]')
fig.text(0.475, 0.01, 'Azimuth [$^\\circ$]', size=20)
fig.suptitle('Filters used for decoding, phase = %d$^\\circ$' %(gfph[phidx]))
plt.subplots_adjust(left=0.06, bottom=0.09, right=1, top=0.93, wspace=0, hspace=0.2)

#example plot to see the phase
fig, axs = plt.subplots(2,4)
axs = axs.reshape([1,8])
for idx, ax in enumerate(np.squeeze(axs)):
    ax.imshow(gfilters[0,0][:,:,idx])
    ax.set_xticks([])
    ax.set_yticks([])

#see if the code works properly
gacts, pext = hlp.population_activity(gfsf, gfph, gfsz, gfilters, img)
    
#Read-outs
#First create a 20x20 grating within black visual field for simplicity
patch = img[0:20, 0:20]
eximg = np.zeros(img.shape)
eximg[0:20, 0:20] = patch
eximg[30:50, 30:50] = patch
eximg[55:65, 40:50] = patch[0:10, 0:10]
eximg[100:110, 42:52] = patch[0:10, 0:10]



gacts, pext = hlp.population_activity(gfsf, gfph, gfsz, gfilters, eximg)

#first write a loop to access each element one by one
#generate the triplet combinations of all gact size sublists indices from 0 to dimension shape with maximum element 
#number
"""
Ultra noob way of index generation, but I do not know how to iterate over gacts.shape[1] while keeping i,j unique
"""
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
loclogarr = np.ones(eximg.shape) #location logical array within the entire visual field
sfphmaps = np.zeros([len(activepatches), *eximg.shape]) #the map for decoded spatial frequency and phase for the 
                                                          #given gabor filter set sorted by size in descending order.
actmaps = np.zeros([len(activepatches), *eximg.shape]) #the activity map for decoded spatial frequency and phase 
                                                         #for the given gabor filter set sorted by size in descending
                                                         #order.
sfphszarr = [] #the spatial frequency, phase and size outcomes sublist for all filters. First sublist is size sorted,
               #within each sublist the nested sublists are for active patches containing the spatial frequency, phase
               #, size, maximum activity and patch location parameters n the given order. Size sorting in descending 
               #order
for sidx, sizepatches in enumerate(np.flip(activepatches,0)): #iterate over the filter sets sorted by size in descending
                                                            #order.    
    loclogszarr = np.zeros(eximg.shape) #generate the whole visual field logical array for each filter sorted 
                                        #by size    
    sfphmap = np.zeros(eximg.shape) #the map for each filter set sorted by size for the spatial frequency and phase
    actmap = np.zeros(eximg.shape) #the map for each filter set sorted by size for the maximum activity
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
xyidx = [(k,j) for j in range(eximg.shape[1]) for k in range(eximg.shape[0])] 

decodedimg = np.zeros(eximg.shape)
totalactmap = np.zeros(eximg.shape)
for idx, actmap in enumerate(actmaps):
    for lidx in xyidx:
        if totalactmap[lidx] < actmap[lidx]:
            totalactmap[lidx] = actmap[lidx]
            decodedimg[lidx] = sfphmaps[idx][lidx]
decodedimg *= loclogarr

fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
axs[0].imshow(eximg)
axs[1].imshow(loclogarr)
axs[0].set_title('Real image')
axs[1].set_title('Population read-out')
axs[0].invert_yaxis()
axs[0].set_ylabel('Elevation [$^\circ$]')
fig.text(0.475, 0.02, 'Azimuth [$^\\circ$]', size=20)
fig.suptitle('Location decoding', size=30)

fig, axs = plt.subplots(1,5, sharex=True, sharey=True)
axs[0].imshow(eximg)
axs[0].set_title('Real image')
axs[4].imshow(decodedimg)
axs[4].set_title('Model readout')
for idx, imgmap in enumerate(sfphmaps):
    axs[idx+1].imshow(imgmap[0:120, 0:60])
    axs[idx+1].set_title('size=%d'%(np.flip(gfsz)[idx]))
axs[0].invert_yaxis()
axs[0].set_ylabel('Elevation [$^\circ$]')
axs[2].set_xlabel('Azimuth [$^\circ$]')
fig.text(0.45,0.85,'Filter maps',size=30)



