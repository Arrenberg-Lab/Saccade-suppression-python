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

#find average saccade for positive and negative saccades
anglocsps = [ [] for _ in range(np.max(saclens).astype(int)) ] #nested list containing the values over all trials for 
                                                               #a given time bin
anglocsns = [ [] for _ in range(np.max(saclens).astype(int)) ] #same as above for negative saccades


for saccade in possaccades:
    for idx, angle in enumerate(saccade):
        anglocsps[idx].append(angle)

for saccade in negsaccades:
    for idx, angle in enumerate(saccade):
        anglocsns[idx].append(angle)
        
anglocsps = np.array(anglocsps)
anglocsns = np.array(anglocsns)

meantimebinsp = np.empty(anglocsps.shape[0]) #the array containing average eye poisition per time bin for positive saccades
meantimebinsp[:] = None
vartimebinsp = np.empty(anglocsps.shape[0]) #the array containing eye poisition variance per time bin for positive saccades
vartimebinsp[:] = None
setimebinsp = np.empty(anglocsps.shape[0]) #the array containing eye poisition SE per time bin for positive saccades
setimebinsp[:] = None

meantimebinsn = np.empty(anglocsns.shape[0]) #same as above for negative saccades
meantimebinsn[:] = None
vartimebinsn = np.empty(anglocsps.shape[0])
vartimebinsn[:] = None
setimebinsn = np.empty(anglocsps.shape[0]) 
setimebinsn[:] = None

for idx, timebin in enumerate(anglocsps):
    timebin = np.array(timebin)[~np.isnan(timebin)]
    if len(timebin) == 0: #skip time bins which have no data point in the given trial
        continue
    meantimebinsp[idx] = np.mean(timebin)
    vartimebinsp[idx] = np.var(timebin) #variance
    setimebinsp[idx] = np.std(timebin) / np.sqrt(len(timebin)) #standard error of the mean
    
for idx, timebin in enumerate(anglocsns):
    timebin = np.array(timebin)[~np.isnan(timebin)]
    if len(timebin) == 0: #skip time bins which have no data point in the given trial
        continue
    meantimebinsn[idx] = np.mean(timebin)
    vartimebinsn[idx] = np.var(timebin) #variance
    setimebinsn[idx] = np.std(timebin) / np.sqrt(len(timebin))

#plotting
fig, axs = plt.subplots(1,2,sharex=True)
fig.suptitle('Average saccades', size=30)

#plot the average
axs[0].plot(meantimebinsp, 'r-', label='Average')
axs[1].plot(meantimebinsn, 'b-', label='Average')

#shade the standard error
t = np.arange(0, len(meantimebinsp))
axs[0].fill_between(t, meantimebinsp+2*setimebinsp, meantimebinsp-2*setimebinsp, facecolor='r', alpha=0.5, 
                   label='Average$\pm 2\cdot SE$')
axs[1].fill_between(t, meantimebinsn+2*setimebinsn, meantimebinsn-2*setimebinsn, facecolor='b', alpha=0.5,
                   label='Average$\pm 2\cdot SE$')

axs[0].set_ylabel('Eye position (norm)')
axs[0].set_xlabel('Time [ms]')
axs[0].set_title('Positive saccade (n=%i)' %(len(possaccades)))
axs[1].set_title('Negative saccade (n=%i)' %(len(negsaccades)))
axs[0].legend()
axs[1].legend()

#do the same plot but with all traces in grey
fig, axs = plt.subplots(1,2,sharex=True)
fig.suptitle('Average saccades', size=30)

#Plot the traces in gray with less linewidth
lab = None
#positive
for idx, sac in enumerate(possaccades):
    if idx == len(possaccades)-1:
        lab='Single trial'
    axs[0].plot(sac, 'gray', linewidth=1, label=lab)

lab = None
#negative
for idx, sac in enumerate(negsaccades):
    if idx == len(negsaccades)-1:
        lab='Single trial'
    axs[1].plot(sac, 'gray', linewidth=1, label=lab)

#plot the average
axs[0].plot(meantimebinsp, 'r-', label='Average', linewidth=2)
axs[1].plot(meantimebinsn, 'b-', label='Average', linewidth=2)
axs[0].set_ylabel('Eye position (norm)')
axs[0].set_xlabel('Time [ms]')
axs[0].set_title('Positive saccades (n=%i)' %(len(possaccades)))
axs[1].set_title('Negative saccades (n=%i)' %(len(negsaccades)))
axs[0].legend()
axs[1].legend()

#calculate root mean square error for each timebin -> this is the standard deviation by definition
#see https://en.wikipedia.org/wiki/Root-mean-square_deviation

#calculate root mean square error for each trial (positive/negative)
RMSp = np.empty(possaccades.shape[0]) #the array of average RMS per trial (positive saccades)
RMSp[:] = None
RMSn = np.empty(negsaccades.shape[0]) #the array of average RMS per trial (negative saccades)
RMSn[:] = None

for idx, sac in enumerate(possaccades):
    n = sac.shape[0]
    meansac = meantimebinsp[:n]
    RMS = np.sqrt( np.sum((meansac-sac)**2) / n)
    RMSp[idx] = RMS

for idx, sac in enumerate(negsaccades):
    n = sac.shape[0]
    meansac = meantimebinsn[:n]
    RMS = np.sqrt( np.sum((meansac-sac)**2) / n)
    RMSn[idx] = RMS    

#plot the standard deviation for each time bin and RMS per each trial
fig, axs = plt.subplots(1,2)
fig.suptitle('Saccadic noise', size=30)
axs[0].plot(np.sqrt(vartimebinsp), 'r.', label='Positive', markersize=3)
axs[0].plot(np.sqrt(vartimebinsn), 'b.', label='Negative', markersize=3)
axs[1].plot(RMSp, 'r.', label='Positive', markersize=3)
axs[1].plot(RMSn, 'b.', label='Negative', markersize=3)

axs[0].set_title('Per time bin')
axs[0].set_ylabel('Standard deviation $\sigma$')
axs[0].set_xlabel('Time [ms]')
axs[1].set_title('Per trial')
axs[1].set_ylabel('RMSE')
axs[1].set_xlabel('Trial ID')
axs[0].legend()
axs[1].legend()

#plot saccade velocity to compare with saccade std per timebin
fig, ax = plt.subplots(1,1)
ax.set_title('Velocity of the average saccade')
ax.plot(np.diff(meantimebinsp)*rt, 'r.', label='Positive')
ax.plot(np.diff(meantimebinsn)*rt, 'b.', label='Negative')
ax.set_ylabel(r'Angular velocity [$\frac{\circ}{s}]$')
ax.set_xlabel('Time [ms]')
ax.legend()