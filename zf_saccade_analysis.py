# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:33:21 2020

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
import sys
sys.path.insert(0, r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\OnlineSaccadeDetection-master\python')
import matplotlib as mpl

#import saccade data 
root = r'E:\HSC_saccades'
angthres = 5
flthres = 0.3
plot = False

saccadedata, datlens, nmnposidx, saccadedataout, saccadedatanoise = hlp.extract_saccade_data(root, angthres, flthres)
saccadedata = np.delete(saccadedata,[0, 32, 33, 41, 50, 53, 91, 92], axis=2)

sacdat = saccadedata.reshape((saccadedata.shape[0],saccadedata.shape[2]*2)) #eye info is discarded
#separate saccades in their direction (positive/negative)
separray = np.ones(sacdat.shape[1]) #this has 1 for positive direction and -1 for negative.
saconsets = np.empty(sacdat.shape[1]) #preallocate the saccade onset index array.
sacoffsets = np.empty(sacdat.shape[1]) #preallocate the saccade offset index array.
saccades = [] #preallocate the saccade array (with onset value starting.)
avgonset = np.empty(sacdat.shape[1]) #the average eye position before saccade onset
avgoffset = np.empty(sacdat.shape[1]) #the average eye position after saccade offset
normfacs = np.empty(sacdat.shape[1]) #normalization factors

for idx in range(sacdat.shape[1]):
    print(idx)
    saccade = sacdat[:,idx]
    saccade = saccade[~np.isnan(saccade)]
    saccade = hlp.running_median(saccade, 5)
    separray[idx] *= np.sign(np.mean(saccade[-20:-15]) - np.mean(saccade[10:15]))
    saconset, sacoffset = hlp.detect_saccade(saccade, separray[idx])
    saconsets[idx] = saconset
    sacoffsets[idx] = sacoffset
    t = np.arange(0, saccade.shape[0])/rt
    saccades.append(saccade[saconset:sacoffset])
    avgonset[idx] = np.mean(saccade[saconset-10:saconset]) #the average eye position 10 ms before saccade
    avgoffset[idx] = np.mean(saccade[sacoffset:sacoffset+10]) #the average eye position 10 ms before saccade
    normfacs[idx] = np.abs(saccade[sacoffset] - saccade[saconset])
    
    if plot == True:
        plt.figure()
        plt.plot(t, saccade)
        plt.plot(t[saconset], saccade[saconset], '.', markersize=10)
        plt.plot(t[sacoffset], saccade[sacoffset], '.', markersize=10)
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
"""
#To add legend to an example saccade trace, run this code snippet
import matplotlib.patches as mpatches
plt.title('Example saccade')
plt.xlabel('Time [s]')
plt.ylabel('Eye position [$^\circ$]')
labels = ['saccade trace', 'saccade onset', 'saccade offset']
patches = []
for idx in range(len(labels)):
    patch = mpatches.Patch(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][idx], label=labels[idx])
    patches.append(patch)
plt.legend(handles=patches)
"""
#normalize saccades to have 1 amplitude and 0 at onset    
sacarray = np.array(saccades) / normfacs
saclens = np.empty(len(sacarray))
for idx in range(len(sacarray)):
    sacarray[idx] -= sacarray[idx][0] 
    saclens[idx] = len(sacarray[idx])

#sort saccades in their direction
possaccades = sacarray[separray>0]
negsaccades = sacarray[separray<0]

#check the saccades to decide what to discard
check = False
if check == True:
    for idx, saccade in enumerate(possaccades):
        print(idx)
        plt.figure()
        plt.plot(saccade)
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
    
    for idx, saccade in enumerate(negsaccades):
        print(idx)
        plt.figure()
        plt.plot(saccade)
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
        
"""
Saccades to be discarded (idx):
    -Pos: 22, 82
    -Neg: 16, 71
"""

possaccades = np.delete(possaccades, [22, 82])
negsaccades = np.delete(negsaccades, [16, 71])

#overshoots
overshootsp = np.empty(len(possaccades)) #in the positive
overshootsn = np.empty(len(negsaccades)) #in the negative

#plotting  
fig, ax = plt.subplots(1,1)
fig.suptitle('Saccade Amplitude Distribution', size=20)
ax.hist(normfacs, bins=30)
ax.set_xlabel('Saccade amplitude $[^\circ]$')
ax.set_ylabel('# of occurence')

fig, axs = plt.subplots(1,2,sharex=True)
fig.suptitle('All detected saccades from Giulia data, n=%i' %(len(possaccades) + len(negsaccades)), size=20)
axs[0].set_ylabel('Saccade amplitude [norm.]')
axs[0].set_xlabel('Time [ms]')
axs[0].set_title('Positive saccades')
axs[1].set_title('Negative saccades')

for idx, sac in enumerate(possaccades):
    #estimate the degree of overshoot in positive
    overshootsp[idx] = np.max(sac)-1
    axs[0].plot(sac)

for idx, sac in enumerate(negsaccades):
    #estimate the degree of overshoot in negative
    #print(np.min(sac)+1)
    overshootsn[idx] = np.min(sac)+1
    axs[1].plot(sac)

fig, axs = plt.subplots(1,2)
fig.suptitle('Overshoot distribution', size=30)
axs[0].hist(overshootsp*100, bins=30)
axs[1].hist(overshootsn*100, bins=30)
axs[0].set_title('Overshoot (positive)')
axs[0].set_ylabel('# of occurence')
axs[0].set_xlabel('Overshoot %')
axs[1].set_title('Overshoot (negative)')

"""
#Log y scale
axs[0].set_ylabel('# of occurence (log)')
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].set_yticks(np.linspace(2,14,7))
axs[0].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axs[1].set_yticks(np.linspace(2,20,10))
axs[1].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
"""