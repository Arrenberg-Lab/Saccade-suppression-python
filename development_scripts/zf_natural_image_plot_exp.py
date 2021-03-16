# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 01:55:03 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from imageio import imread
from skimage.color import rgb2gray
from skimage.transform import resize

#Plot an example natural image.
loadpath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\underwater_img_database\raw-890'
imgs = os.listdir(loadpath)
arr = imread(loadpath+r'\\'+imgs[790]) # 800x634x3 array
grarr = rgb2gray(arr)
#upsample the image to match the visual field size of the model.
rsl = 5
newimg = resize(grarr, np.array([180,360])*rsl)

#shift the image horizontally:
shiftang = 90 #the horizontal shift angle in degrees
shiftedimg = np.zeros(newimg.shape)
shiftedimg[:,shiftang*rsl:] = newimg[:, :-shiftang*rsl]
shiftedimg[:,:shiftang*rsl] = newimg[:, -shiftang*rsl:]

#do the stimulus position decoding
nfilters = 1058
jsigma = 4 #the standard deviation of the jitter noise in degrees
zfparams = hlp.generate_species_parameters(nfilters).zebrafish()
__, zffltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, newimg, \
                                                             params=zfparams)
fltimg, popact = hlp.florian_model_species_edition(newimg, rsl, zfparams, zffltcenters[:,0], zffltcenters[:,1])

stimcenterzf = popact/np.sum(popact) @ zffltcenters
stimcenterzf /= rsl
stimcenterzf[0] -= 180
stimcenterzf[1] = -stimcenterzf[1] + 90

_, popact2 = hlp.florian_model_species_edition(shiftedimg, rsl, zfparams, zffltcenters[:,0], zffltcenters[:,1])

stimcenterzf2 = popact2/np.sum(popact2) @ zffltcenters
stimcenterzf2 /= rsl
stimcenterzf2[0] -= 180
stimcenterzf2[1] = -stimcenterzf2[1] + 90

#plotting
fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
axs[0].imshow(newimg, extent=(-180,180,-90,90))
axs[0].set_yticks(np.linspace(-90, 90, 5))
axs[0].set_xticks(np.linspace(-180, 180, 5))
axs[0].set_title('Example image')
axs[0].set_ylabel('Elevation [°]')
axs[0].set_xlabel('Azimuth [°]')
axs[1].imshow(shiftedimg, extent=(-180,180,-90,90))
axs[1].set_title('Example image (shifted)')
axs[0].plot(*stimcenterzf, 'r.')
axs[1].plot(*stimcenterzf2, 'r.')

fig, ax = plt.subplots()
#plot image
ax.imshow(newimg, extent=(-180,180,-90,90))
ax.set_title('Example image')
ax.set_ylabel('Elevation [°]')
ax.set_xlabel('Azimuth [°]')
ax.set_yticks(np.linspace(-90, 90, 5))
ax.set_xticks(np.linspace(-180, 180, 5))

#plot image patches
elevations = np.linspace(-90, 90, 5) 
azimuths = np.linspace(-180, 180, 10)
for x in azimuths:
    ax.plot([x,x], [-90,90], 'b-', linewidth=2)
for y in elevations:
    ax.plot([-180,180], [y,y], 'b-', linewidth=2)
#plot example decoding within each patch (totally random for now)
#DO LATER NOW CHEAT THE SYSTEM COZ ITS ALMOST 4 AM ;'(


    
