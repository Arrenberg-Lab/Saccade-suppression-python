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
imgsf = 0.05 #spatial frequency of the image in degrees

img, gr = hlp.generate_grating(rsl, imgsf)

plt.imshow(img, vmin=0, vmax=1)
plt.colorbar()

#improve on gabor: generate gabor with correct weavelength etc so that you can express it in degrees and/or pixels.

"""
Regarding Gabor pixel size: Dumb thing is the filter extent is also dependent on the angle theta. Therefore, aim is to 
generate an initial Gabor with very big extent (huge sigma for both x and y) and then crop the appropriate size out of
this huge initial Gabor
"""
sigma = 100*rsl #the standard deviation of the initial gabor filter

#Generate Gabor filters of different spatial frequency and size

gfsz = [20, 30, 60] #the filter sizes in degrees
gfsf = [0.1, 0.12, 0.15, 0.18] #the filter spatial frequencies in degrees

gfilters = [[] for _ in range(len(gfsz))] #empty list to dump each Gabor filter one by one. Each empty sublist is for 
                                          #the filter size. This approach is chosen for future ease in filter sorting.

#empty figure for plotting the resulting filters for demonstration.
fig, axs = plt.subplots(len(gfsz), len(gfsf), sharex='row', sharey='row') 
for i, sz in enumerate(gfsz):
    for j, sf in enumerate(gfsf):
        _, flt = hlp.crop_gabor_filter(rsl, sigma, sf, sz)
        gfilters[i].append(flt)
        print('The filter with the size %d and spatial frequency %.2f is generated' %(sz, sf))
        axs[i,j].imshow(flt, vmin=0, vmax=1)        
        axs[i,j].set_xticks(np.arange(0, sz+1, 10))
        axs[i,j].set_yticks(np.arange(0, sz+1, 10))
        axs[i,j].invert_yaxis()
        
axs[1,0].set_ylabel('Elevation [$^\circ$]')
fig.text(0.475, 0.01, 'Azimuth [$^\\circ$]', size=20)
fig.suptitle('Filters used for decoding')
plt.subplots_adjust(left=0.06, bottom=0.09, right=1, top=0.93, wspace=0, hspace=0.2)

#Write the code for tiling the image to appropriate sizes of the given filter sets and get the activity as the 
#correlation of the image patch with the filter.
#Question: Shall i normalize the filter maximum activity to 1?
gfilters = np.array(gfilters, dtype=object)

fltsacts, xext, yext = hlp.filters_activity(img, gfilters[0])        
