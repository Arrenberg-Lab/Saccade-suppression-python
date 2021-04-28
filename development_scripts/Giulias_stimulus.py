# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:41:00 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from PIL import Image

#Python implementation of Giulia's stimulus:
arenaaz = 168 #azimuth extent of half arena (168). Two half arenas are combined horizontally, so this is half of total azimuth.
arenaelev = 40 #elevation extent of total arena (+-40)
rsl = 1 #visual field resolution in pix/deg
fps = 200 #number of frames per second
shiftmag = 20 #magnitude of shift (horizontal) in degrees
tdur = 70 #duration of shift in ms
frametot = np.int(fps*tdur/1000) #total number of frames (excluding start position)
elevrest = 5 #additional extension to elevation, so that smallest stimulus unit issue is less pronounced

#make the stimulus coarser: in Giulia's matlab code, smallest unit in stimulus is 5x5 pixels, and resolution is 1.5 pix/deg. 
#Thus, one side of the stimulus is 3.333 deg. 
stimunit = np.round(rsl*5/1.5).astype(int) #length of the stimulus unit in pixels (i.e. length of one pixel side)
#each stimulus unit should be stimunit*stimunit in size (cropping in visual field limits is allowed)
#total stimulus array : azimuth is extended +-20°, so that shifts in both to the left and to the right can be chosen. 
#rsl*elevrest pixels in elevation are additional
#Generate coarse grating
coarsegrating = np.random.rand(np.ceil((arenaelev*2+elevrest)*rsl/stimunit).astype(int), \
                               np.ceil(np.int(arenaaz+2*shiftmag)*rsl/stimunit).astype(int)) < 0.5 
coarseimg = Image.fromarray(coarsegrating)
#resize the coarse grating to fit to the whole visual field
coarseimg = coarseimg.resize(np.flip(np.array(coarsegrating.shape)*stimunit), resample=Image.NEAREST)                               
#final stimulus array
finalstim = np.array(coarseimg)

#implement the shift for each frame
shiftperfr = np.round(shiftmag/(tdur/1000)/fps*rsl) #shift size in pixels per frame
frames = np.zeros([2*arenaelev*rsl,arenaaz*2*rsl,frametot+1])
frames[:,:arenaaz*rsl,0] = finalstim[elevrest*rsl:(elevrest+2*arenaelev)*rsl, shiftmag*rsl:(arenaaz+shiftmag)*rsl]
frames[:,arenaaz*rsl:,0] = frames[:,:arenaaz*rsl,0] 

#preallocate shift frames
shiftright = frames.copy()
shiftleft = frames.copy()

for i in range(frametot):
    xintleft = (np.array([shiftmag,arenaaz+shiftmag])*rsl + (i+1)*shiftperfr).astype(int) #shifted slice chosen from finalstim
    xintright = (np.array([shiftmag,arenaaz+shiftmag])*rsl - (i+1)*shiftperfr).astype(int)
    #print(xintright,xintleft)
    
    shiftright[:,:arenaaz*rsl,i+1] = finalstim[elevrest*rsl:(elevrest+2*arenaelev)*rsl, xintright[0]:xintright[1]]
    shiftleft[:,:arenaaz*rsl,i+1] = finalstim[elevrest*rsl:(elevrest+2*arenaelev)*rsl, xintleft[0]:xintleft[1]]
    
    shiftright[:,arenaaz*rsl:,i+1] = shiftright[:,:arenaaz*rsl,i+1]
    shiftleft[:,arenaaz*rsl:,i+1] = shiftleft[:,:arenaaz*rsl,i+1]


fig, ax = plt.subplots()
plt.get_current_fig_manager().window.showMaximized()
ax.imshow(shiftleft[:,:,0], extent=[-168,168,-40,40])
plt.pause(0.5)
for i in range(frametot):
    ax.imshow(shiftleft[:,:,i+1], extent=[-168,168,-40,40])
    plt.pause(0.5)
                
        
#Stimulus -> omit the black stripes in between, as they are for luminance modulation. ASK FLORIAN THO!