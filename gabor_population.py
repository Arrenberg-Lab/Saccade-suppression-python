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
Regarding Gabor pixel size: Dumb thing is the filter extent is also dependent on the angle theta. Therefore, aim is to 
generate an initial Gabor with very big extent (huge sigma for both x and y) and then crop the appropriate size out of
this huge initial Gabor
"""
sigma = 100*rsl #the standard deviation of the initial gabor filter

#Generate Gabor filters of different spatial frequency and size

gfsz = [20, 30, 60] #the filter sizes in degrees
gfsf = [0.05, 0.1, 0.15, 0.2] #the filter spatial frequencies in degrees

gfilters = [[] for _ in range(len(gfsz))] #empty list to dump each Gabor filter one by one. Each empty sublist is for 
                                          #the filter size. This approach is chosen for future ease in filter sorting.

#empty figure for plotting the resulting filters for demonstration.
fig, axs = plt.subplots(len(gfsz), len(gfsf), sharex='row', sharey='row') 
for i, sz in enumerate(gfsz):
    for j, sf in enumerate(gfsf):
        _, flt = hlp.crop_gabor_filter(rsl, sigma, sf, sz)
        gfilters[i].append(flt)
        print('The filter with the size %d and spatial frequency %.2f is generated' %(sz, sf))
        axs[i,j].imshow(flt, vmin=-1, vmax=1)        
        axs[i,j].set_xticks(rsl*np.arange(0, sz+1, 10))
        axs[i,j].set_yticks(rsl*np.arange(0, sz+1, 10))
        axs[i,j].invert_yaxis()
        
axs[1,0].set_ylabel('Elevation [$^\circ$]')
fig.text(0.475, 0.01, 'Azimuth [$^\\circ$]', size=20)
fig.suptitle('Filters used for decoding')
plt.subplots_adjust(left=0.06, bottom=0.09, right=1, top=0.93, wspace=0, hspace=0.2)

#Write the code for tiling the image to appropriate sizes of the given filter sets and get the activity as the 
#correlation of the image patch with the filter.
#Question: Shall i normalize the filter maximum activity to 1?
gfilters = np.array(gfilters, dtype=object)
gacts = [] #ordered according to size

for idx, fsar in enumerate(gfilters):
    fltsacts, xext, yext = hlp.filters_activity(img, fsar)        
    #fltsacts[fltsacts<0] = 0 #as negative activity makes no sense, set them to 0
    gacts.append(fltsacts)
