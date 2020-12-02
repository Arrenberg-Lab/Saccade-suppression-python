# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:30:24 2020

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp


#import saccade data 
"""
-TODOs: 
    -implement saving procedure for the saccade data to prevent long time waiting. Do this the latest, when you can reliably extract the saccades
    +set a saccade magnitude threshold, and discard all eye movements not being unilateral. Then you need to do further stuff for error estimation
    DONE
    -
"""
root = r'E:\HSC_saccades'
angthres = 5
flthres = 0.3
"""
Data from 'Easter Jr, S. S., & Nicola, G. N. (1997). The development of eye movements in the zebrafish (Danio rerio). 
Developmental Psychobiology: The Journal of the International Society for Developmental Psychobiology, 31(4), 267-276.'
suggests a minimal saccade to be 5 degrees for a stimulus turning 2.4 degrees/s with 22.5 or 45 degrees wide.
"""

saccadedata, datlens, nmnposidx, saccadedataout, saccadedatanoise = hlp.extract_saccade_data(root, angthres, flthres)


#Plot each saccade
rt = 1000 #sampling rate
t = np.arange(0, saccadedata.shape[0]) / rt


#plot remaining saccades
for trial in range(saccadedata.shape[2]):
    print(trial)
    fig, axs = plt.subplots(1,2)
    figdata = saccadedata[:,:,trial]
    for idx, ax in enumerate(axs):
        ax.plot(t, figdata[:, idx], 'k')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break

t = np.arange(0, saccadedataout.shape[0]) / rt

#plot discarded saccades
for trial in range(saccadedataout.shape[2]):
    print(trial)
    fig, axs = plt.subplots(1,2)
    figdata = saccadedataout[:,:,trial]
    for idx, ax in enumerate(axs):
        ax.plot(t, figdata[:, idx], 'k')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break        

t = np.arange(0, saccadedatanoise.shape[0]) / rt

#plot noisy saccades
for trial in range(saccadedatanoise.shape[2]):
    print(trial)
    fig, axs = plt.subplots(1,2)
    figdata = saccadedatanoise[:,:,trial]
    for idx, ax in enumerate(axs):
        ax.plot(t, figdata[:, idx], 'k')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break     
        
"""
*02.12.2020: +Filtered data: idx 0 (can discard if we are conservative with data),
             32 (can discard if we are conservative with data), 33 (can discard if we are conservative with data), 
             41 (can discard if we are conservative with data), 50 (weird), 53 (weird),  91 (weird), 92 (can discard if we are conservative).
             *Thus, 3 out of 107 is very weird, 5 are questionable
             +Noise: idx 3 (can add if we are liberal with data), 5, 17 (due to onset/offset)
             *So 3 out of 26 with 2 being miss and 1 is questionable
             +Removed data: idx 3 (can add if we are liberal with data), 4 (can add if we are liberal with data), 47, 98 (can add if we are liberal),
             *1 missed and 3 questionable out of 124          
"""
