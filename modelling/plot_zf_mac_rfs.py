# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:32:00 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


#Plot example model RFs from zebrafish and macaque
nfilters = 10082
rslzf = 5 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
rslmac = 24 #resolution as pixel per degrees, use to transform the degrees to pixels as this is the no of pixels 
        #corresponding to 1 degree.
rdot = 5 #the radius of the dot in degrees

jsigma = 4 #the standard deviation of the jitter noise in degrees

speciesparams = hlp.generate_species_parameters(nfilters)
zfparamslog = speciesparams.zebrafish() #sf distribution log uniform
zfparamsunif = speciesparams.zebrafish('unif')

macparams = speciesparams.macaque()
imgzf, circcentzf = hlp.circle_stimulus(rdot*rslzf, rslzf, *(180, 90))    
__, zfcenterslog = hlp.florian_model_shuffle_parameters(nfilters, rslzf, None, None, None, jsigma, imgzf, \
                                                             params=zfparamslog)
zfimglog, __ = hlp.florian_model_species_edition(imgzf, rslzf, zfparamslog, zfcenterslog[:,0], zfcenterslog[:,1])

__, zfcentersunif = hlp.florian_model_shuffle_parameters(nfilters, rslzf, None, None, None, jsigma, imgzf, \
                                                             params=zfparamsunif)
zfimgunif, __ = hlp.florian_model_species_edition(imgzf, rslzf, zfparamsunif, zfcentersunif[:,0], zfcentersunif[:,1])


imgmac, circcentmac = hlp.circle_stimulus(rdot*rslmac, rslmac, *(180, 90))
__, maccenters = hlp.florian_model_shuffle_parameters(nfilters, rslmac, None, None, None, jsigma, imgmac, \
                                                             params=macparams)
macimg, __ = hlp.florian_model_species_edition(imgmac, rslmac, macparams, maccenters[:,0], maccenters[:,1])

fig, axs = plt.subplots(2, 3, sharex='row', sharey='row')
plt.subplots_adjust(left=0.064, bottom=0, right=0.983, top=0.95, wspace=0.129, hspace=0)
#first plot the whole visual field
axs[0,0].imshow(zfimglog, extent=(-180,180,-90,90))
axs[0,1].imshow(zfimgunif, extent=(-180,180,-90,90))
axs[0,2].imshow(macimg, extent=(-180,180,-90,90))
axs[0,0].set_yticks(np.linspace(-90, 90, 5))
axs[0,0].set_xticks(np.linspace(-180, 180, 5))
fig.suptitle('Example RF distribution for different species')
axs[0,0].set_title('Zebrafish (reciprocal)')
axs[0,1].set_title('Zebrafish (uniform)')
axs[0,2].set_title('Macaque')
axs[0,0].set_ylabel('Elevation [°]')
axs[1,0].set_xlabel('Azimuth [°]')
#choose a small patch from central, plot a thick square to the plot area
dgex = 10 #the extent in degrees to plot the small patch
for ax in axs[0,:]:
    ax.plot([-2*dgex,2*dgex],[-dgex,-dgex], 'r-', linewidth=5)
    ax.plot([-2*dgex,2*dgex],[dgex,dgex], 'r-', linewidth=5)
    ax.plot([-2*dgex,-2*dgex],[-dgex,dgex], 'r-', linewidth=5)
    ax.plot([2*dgex,2*dgex],[-dgex,dgex], 'r-', linewidth=5)

#plot the small patch

mididxs = np.array(zfimglog.shape)/2 #the middle y and x indices (the pixel at 0° azimuth and elevation)    
zfsmallpatchlog = zfimglog[(mididxs[0]-dgex*rslzf).astype(int):(mididxs[0]+dgex*rslzf).astype(int), \
                     (mididxs[1]-2*dgex*rslzf).astype(int):(mididxs[1]+2*dgex*rslzf).astype(int)]
zfsmallpatchunif = zfimgunif[(mididxs[0]-dgex*rslzf).astype(int):(mididxs[0]+dgex*rslzf).astype(int), \
                     (mididxs[1]-2*dgex*rslzf).astype(int):(mididxs[1]+2*dgex*rslzf).astype(int)]

mididxs = np.array(macimg.shape)/2 #the middle y and x indices (the pixel at 0° azimuth and elevation)
macsmallpatch = macimg[(mididxs[0]-dgex*rslmac).astype(int):(mididxs[0]+dgex*rslmac).astype(int), \
                     (mididxs[1]-2*dgex*rslmac).astype(int):(mididxs[1]+2*dgex*rslmac).astype(int)]
axs[1,0].imshow(zfsmallpatchlog, extent=(-2*dgex,2*dgex,-dgex,dgex))
axs[1,1].imshow(zfsmallpatchunif, extent=(-2*dgex,2*dgex,-dgex,dgex))
axs[1,2].imshow(macsmallpatch, extent=(-2*dgex,2*dgex,-dgex,dgex))
axs[1,0].set_yticks(np.linspace(-dgex,dgex,5))
axs[1,0].set_xticks(np.linspace(-2*dgex,2*dgex,5))


#Example plot of how the stimulus shift looks like.
centoffp = np.round(np.logspace(0, np.log10(60), 4))
centoffp[0] = 0
centoffs = np.zeros(len(centoffp)*2-1) #Note that for now you change both x and y the same, in future you can change x and 
                                       #y independently
centoffs[:len(centoffp)-1] = -np.flip(centoffp[1:])
centoffs[len(centoffp)-1:] = centoffp

fig, ax = plt.subplots(1,1)
img, __ = hlp.circle_stimulus(rdot, rslzf, *(180, 90))   
for idx, centoff in enumerate(centoffs):
    if centoff == 0:
        continue
    img2, __ =  hlp.circle_stimulus(rdot, rslzf, *(180, 90)+centoff)
    img[img2>0] = 1
    ax.plot(centoff,-centoff, 'k.', markersize=5)
    if np.abs(centoff) == 60:
        ax.arrow(*np.array([0, 0, centoff, -centoff]), color='red', width=0.5, length_includes_head=True)
ax.imshow(img, extent=(-180,180,-90,90))
ax.plot(0,0, 'b.', markersize=5)
ax.set_yticks(np.linspace(-90, 90, 5))
ax.set_xticks(np.linspace(-180, 180, 5))
ax.set_title('Stimulus shifts used in the simulation')
ax.set_ylabel('Elevation [°]')
ax.set_xlabel('Azimuth [°]')

#plot parameter histograms
sfzfl = [] #spatial frequency zebrafish log uniform
sfzfu = [] #spatial frequency zebrafish uniform
szzf = [] #RF size zebrafish
for i in range(len(zfparamslog)):
    sfzfl.append(zfparamslog[i][0])
    sfzfu.append(zfparamsunif[i][0])
    szzf.append(zfparamslog[i][1])

sfmac = []
szmac = []
for params in macparams:
    sfmac.append(params[0])
    szmac.append(params[1])

fig, axs = plt.subplots(2,2)
plt.subplots_adjust(top=0.843, hspace=0.317)
sps1, sps2, sps3, sps4 = GridSpec(2,2)
axs[0,0].axes.xaxis.set_visible(False)
axs[0,0].axes.yaxis.set_visible(False)
brokaxlowlim = np.max(np.histogram(sfzfu,bins=100, density=True)[0])
brokaxuplim = np.max(np.histogram(sfzfl,bins=100, density=True)[0])

bax = brokenaxes(ylims=((0, brokaxlowlim), (brokaxuplim-0.5,brokaxuplim+0.5)), subplot_spec=sps1, hspace=.05)
bax.hist(sfzfu, bins=100, density=True, color='b')
bax.hist(sfzfl, bins=100, density=True, color='r', alpha=0.7)

axs[0,1].hist(szzf, bins=100, density=True)
axs[1,0].hist(sfmac, bins=100, density=True)
axs[1,1].hist(szmac, bins=100, density=True)
fig.suptitle('Parameter histograms')
axs[0,0].set_title('Spatial frequency')
axs[0,1].set_title('Size')
bax.set_ylabel('Density')
axs[1,0].set_xlabel('Spatial frequency [cyc/°]')
axs[1,1].set_xlabel('RF diameter [°]')
fig.text(0.5, 0.87, 'Zebrafish', size=25, horizontalalignment='center')
fig.text(0.5, 0.45, 'Macaque', size=25, horizontalalignment='center')
l1 = mpatches.Patch(color='r', label='reciprocal')
l2 = mpatches.Patch(color='b', label='uniform')
plt.legend(handles=[l1,l2])
