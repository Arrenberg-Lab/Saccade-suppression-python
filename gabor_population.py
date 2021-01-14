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
imgsf = 0.1 #spatial frequency of the image in degrees

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
Regarding Gabor pixel size: Dumb thing is the filter extent is also dependent on the angle theta. Therefore, aim is to 
generate an initial Gabor with very big extent (huge sigma for both x and y) and then crop the appropriate size out of
this huge initial Gabor
"""
sigma = 100*rsl #the standard deviation of the initial gabor filter

#Generate Gabor filters of different spatial frequency and size

gfsz = [20, 30, 60] #the filter sizes in degrees
gfsf = [0.05, 0.1, 0.15, 0.2] #the filter spatial frequencies in degrees
gfph = np.arange(0,360, 45) #the filter phases in degrees
pidx = 0 #the phase index to plot the filters

gfilters = [[] for _ in range(len(gfsz))] #empty list to dump each Gabor filter one by one. Each empty sublist is for 
                                          #the filter size. This approach is chosen for future ease in filter sorting.
                                          #array not possible as different filter sizes also mean different array sizes
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
        axs[i,j].imshow(fltarray[:,:,pidx], vmin=-1, vmax=1) #show the 0 phase filter (or the first value in phase array)
        axs[i,j].set_xticks(rsl*np.arange(0, sz+1, 10))
        axs[i,j].set_yticks(rsl*np.arange(0, sz+1, 10))
        axs[i,j].invert_yaxis()

gfilters = np.array(gfilters, dtype=object)
        
axs[1,0].set_ylabel('Elevation [$^\circ$]')
fig.text(0.475, 0.01, 'Azimuth [$^\\circ$]', size=20)
fig.suptitle('Filters used for decoding, phase = %d$^\\circ$' %(gfph[pidx]))
plt.subplots_adjust(left=0.06, bottom=0.09, right=1, top=0.93, wspace=0, hspace=0.2)

#example plot to see the phase
fig, axs = plt.subplots(2,4)
axs = axs.reshape([1,8])
for idx, ax in enumerate(np.squeeze(axs)):
    ax.imshow(gfilters[0,0][:,:,idx])

#see if the code works properly
gacts, pext = hlp.population_activity(gfsf, gfph, gfsz, gfilters, img)
    
#Read-outs
#First choose a 60x60 image for simplicity
eximg = img[0:60, 0:60]
#1) The spatial frequency: use the max operation within the filters
gacts, pext = hlp.population_activity(gfsf, gfph, gfsz, gfilters, eximg)
plt.plot(gacts[1,:,2])
