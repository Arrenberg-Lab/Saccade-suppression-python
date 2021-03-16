# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:33:21 2020

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
import matplotlib as mpl
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter
from scipy.stats import mannwhitneyu

#import saccade data 
root = r'\\172.25.250.112\arrenberg_data\shared\Ibrahim\HSC_saccades'
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
sactracebef = []#saccade trace before saccade onset
sactraceaf = []#saccade trace after saccade offset 

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
    sactracebef.append(saccade[:saconset] - np.mean(saccade[:saconset])) #saccade trace before saccade onset - mean
    sactraceaf.append(saccade[sacoffset:] - np.mean(saccade[sacoffset:])) #saccade trace before saccade onset - mean
    avgonset[idx] = np.mean(saccade[:saconset]) #the average eye position 10 ms before saccade
    avgoffset[idx] = np.mean(saccade[sacoffset:]) #the average eye position 10 ms after saccade
    normfacs[idx] = np.abs(saccade[sacoffset] - saccade[saconset])
    
    if plot == True:
        plt.figure()
        plt.plot(t, saccade, 'k-')
        plt.plot(t[saconset], saccade[saconset], 'g.', markersize=10)
        plt.plot(t[sacoffset], saccade[sacoffset], 'm.', markersize=10)
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
colors = ['k', 'g', 'm']
labels = ['saccade trace', 'saccade onset', 'saccade offset']
patches = []
for idx in range(len(labels)):
    patch = mpatches.Patch(color=colors[idx], label=labels[idx])
    patches.append(patch)
plt.legend(handles=patches)
"""
#normalize saccades to have 1 amplitude and 0 at onset    
sacarray = np.array(saccades)
sacarraynorm = np.array(saccades) / normfacs
saclensnorm = np.empty(len(sacarraynorm))
for idx in range(len(sacarraynorm)):
    sacarraynorm[idx] -= sacarraynorm[idx][0] 
    saclensnorm[idx] = len(sacarraynorm[idx])

#sort saccades in their direction
possaccadesnorm = sacarraynorm[separray>0]
negsaccadesnorm = sacarraynorm[separray<0]
possaccades = sacarray[separray>0]
negsaccades = sacarray[separray<0]

#check the saccades to decide what to discard
check = False
if check == True:
    for idx, saccade in enumerate(possaccadesnorm):
        print(idx)
        plt.figure()
        plt.plot(saccade)
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
    
    for idx, saccade in enumerate(negsaccadesnorm):
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

possaccadesnorm = np.delete(possaccadesnorm, [22, 82])
negsaccadesnorm = np.delete(negsaccadesnorm, [16, 71])

#overshoots
overshootsp = np.empty(len(possaccadesnorm)) #in the positive
overshootsn = np.empty(len(negsaccadesnorm)) #in the negative

#plotting  
fig, ax = plt.subplots(1,1)
fig.suptitle('Saccade Amplitude Distribution', size=20)
ax.hist(normfacs, bins=30, color='k')
ax.set_xlabel('Saccade amplitude $[^\circ]$')
ax.set_ylabel('# of occurence')

fig, axs = plt.subplots(1,2,sharex=True)
fig.suptitle('All detected saccades from Giulia data, n=%i' %(len(possaccadesnorm) + len(negsaccadesnorm)), size=20)
axs[0].set_ylabel('Saccade amplitude [norm.]')
axs[0].set_xlabel('Time [ms]')
axs[0].set_title('Positive saccades')
axs[1].set_title('Negative saccades')

for idx, sac in enumerate(possaccadesnorm):
    #estimate the degree of overshoot in positive
    overshootsp[idx] = np.max(sac)-1
    axs[0].plot(sac)

for idx, sac in enumerate(negsaccadesnorm):
    #estimate the degree of overshoot in negative
    #print(np.min(sac)+1)
    overshootsn[idx] = np.min(sac)+1
    axs[1].plot(sac)

fig, ax = plt.subplots(1,1)
fig.suptitle('Overshoot distribution', size=30)
binwidth = 1.7
ax.hist(overshootsp*100, bins=np.arange(min(overshootsp*100), max(overshootsp*100) + binwidth, binwidth), \
        color='r', alpha=0.5, label='positive')
ax.hist(-overshootsn*100, bins=np.arange(min(-overshootsn*100), max(-overshootsn*100) + binwidth, binwidth), \
        color='b', alpha=0.5, label='negative')
ax.set_ylabel('# of occurence')
ax.set_xlabel('Overshoot %')
ax.legend()
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
anglocsps = [ [] for _ in range(np.max(saclensnorm).astype(int)) ] #nested list containing the values over all trials for 
                                                               #a given time bin
anglocsns = [ [] for _ in range(np.max(saclensnorm).astype(int)) ] #same as above for negative saccades


for saccade in possaccadesnorm:
    for idx, angle in enumerate(saccade):
        anglocsps[idx].append(angle)

for saccade in negsaccadesnorm:
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
axs[0].set_title('Positive saccade (n=%i)' %(len(possaccadesnorm)))
axs[1].set_title('Negative saccade (n=%i)' %(len(negsaccadesnorm)))
axs[0].legend()
axs[1].legend()

#do the same plot but with all traces in grey
fig, axs = plt.subplots(1,2,sharex=True)
fig.suptitle('Average saccades', size=30)

#Plot the traces in gray with less linewidth
lab = None
#positive
for idx, sac in enumerate(possaccadesnorm):
    if idx == len(possaccadesnorm)-1:
        lab='Single trial'
    axs[0].plot(sac, 'gray', linewidth=1, label=lab)

lab = None
#negative
for idx, sac in enumerate(negsaccadesnorm):
    if idx == len(negsaccadesnorm)-1:
        lab='Single trial'
    axs[1].plot(sac, 'gray', linewidth=1, label=lab)

#plot the average
axs[0].plot(meantimebinsp, 'r-', label='Average', linewidth=2)
axs[1].plot(meantimebinsn, 'b-', label='Average', linewidth=2)
axs[0].fill_between(t, meantimebinsp+2*setimebinsp, meantimebinsp-2*setimebinsp, facecolor='r', alpha=0.5, 
                   label='Average$\pm 2\cdot SE$', zorder=100)
axs[1].fill_between(t, meantimebinsn+2*setimebinsn, meantimebinsn-2*setimebinsn, facecolor='b', alpha=0.5,
                   label='Average$\pm 2\cdot SE$', zorder=100)
axs[0].set_ylabel('Eye position (norm)')
axs[0].set_xlabel('Time [ms]')
axs[0].set_title('Positive saccades (n=%i)' %(len(possaccadesnorm)))
axs[1].set_title('Negative saccades (n=%i)' %(len(negsaccadesnorm)))
axs[0].legend()
axs[1].legend()

#calculate root mean square error for each timebin -> this is the standard deviation by definition
#see https://en.wikipedia.org/wiki/Root-mean-square_deviation

#calculate root mean square error for each trial (positive/negative)
RMSp = np.empty(possaccadesnorm.shape[0]) #the array of average RMS per trial (positive saccades)
RMSp[:] = None
RMSn = np.empty(negsaccadesnorm.shape[0]) #the array of average RMS per trial (negative saccades)
RMSn[:] = None

for idx, sac in enumerate(possaccadesnorm):
    n = sac.shape[0]
    meansac = meantimebinsp[:n]
    RMS = np.sqrt( np.sum((meansac-sac)**2) / n)
    RMSp[idx] = RMS

for idx, sac in enumerate(negsaccadesnorm):
    n = sac.shape[0]
    meansac = meantimebinsn[:n]
    RMS = np.sqrt( np.sum((meansac-sac)**2) / n)
    RMSn[idx] = RMS    

#plot the standard deviation for each time bin and RMS per each trial
fig, axs = plt.subplots(1,2)
fig.suptitle('Saccadic noise', size=30)
axs[0].plot(np.sqrt(vartimebinsp), 'r.', label='Positive', markersize=3)
axs[0].plot(np.sqrt(vartimebinsn), 'b.', label='Negative', markersize=3)
binwidth = 0.006
axs[1].hist(RMSp, color='r', label='Positive',  bins=np.arange(min(RMSn), max(RMSp) + binwidth, binwidth), alpha=0.5, density=True)
axs[1].hist(RMSn, color='b', label='Negative', bins=np.arange(min(RMSn), max(RMSn) + binwidth, binwidth), alpha=0.5, density=True)

axs[0].set_title('Per time bin')
axs[0].set_ylabel('Standard deviation')
axs[0].set_xlabel('Time [ms]')
axs[1].set_title('Per trial')
axs[1].set_ylabel('Frequency density')
axs[1].set_xlabel('Standard deviation')
axs[0].legend()
axs[1].legend()

#plot saccade velocity to compare with saccade std per timebin
fig, ax = plt.subplots(1,1)
ax.set_title('Velocity of the average saccade')
ax.plot(np.diff(meantimebinsp)*rt, 'r.', label='Positive')
ax.plot(np.diff(meantimebinsn)*rt, 'b.', label='Negative')
ax.set_ylabel(r'Angular velocity $[\frac{norm}{s}]$')
ax.set_xlabel('Time [ms]')
ax.legend()

#Motor noise I: xi, the noise without saccades
tracesbefsac = [a for b in sactracebef for a in b]
tracesafsac = [a for b in sactraceaf for a in b]
alltraceswithoutsac = [c for b in [sactracebef,sactraceaf] for a in b for c in a]
xi = np.std(alltraceswithoutsac)
#Motor noise II: epsilon, the noise arising during saccades (on top of xi) which scales with the corollary discharge
#corollary discharge u is the eye position derivative in time
#use the average saccade as template
#in the saccades array, the onset is the first data point, and offset is the last data point.

usp = [] #template corollary discharge for positive
for idx, sac in enumerate(possaccades):
    template = meantimebinsp[:len(sac)]*(sac[-1]-sac[0])+sac[0]
    """
    #Check if template matches the saccades
    plt.figure()
    plt.plot(template)
    plt.plot(sac)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    #Template looks good.
    """
    #change the way of analysis: first calculate u for each template, then sort s_epsilon for the given u values 
    #(100 different bins). Finally, average s_epsilon for each u and then average over all epsilons related to u
    uvals = np.diff(template) #u of template so that xi is removed (since average saccade).
    usp.append(np.abs(uvals))

usp = [a for b in usp for a in b]
usp = np.unique(usp)    

#same analysis for negative saccades:
usn = [] #template corollary discharge for positive
for idx, sac in enumerate(negsaccades):
    template = meantimebinsp[:len(sac)]*(sac[-1]-sac[0])+sac[0]
    """
    #Check if template matches the saccades
    plt.figure()
    plt.plot(template)
    plt.plot(sac)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    #Template looks good.
    """
    #change the way of analysis: first calculate u for each template, then sort s_epsilon for the given u values 
    #(100 different bins). Finally, average s_epsilon for each u and then average over all epsilons related to u
    uvals = np.diff(template) #u of template so that xi is removed (since average saccade).
    usn.append(np.abs(uvals))

usn = [a for b in usn for a in b]
usn = np.unique(usn)    

#pool usp and usn together to generate shared bins.
uspooled = [b for a in [usp,usn] for b in a]
uspooled = np.unique(uspooled)

#bins for positive ang negative saccades:
_, ubinsp = np.histogram(uspooled, bins=100)
_, ubinsn = np.histogram(uspooled, bins=100)
epsbinnedp = [[] for a in range(len(ubinsp)-1)]
epsbinnedn = [[] for a in range(len(ubinsn)-1)]

for idx, sac in enumerate(possaccades):
    template = meantimebinsp[:len(sac)]*(sac[-1]-sac[0])+sac[0]
    uvals = np.diff(template) #u of template so that xi is removed (since average saccade).
    uvals = np.abs(uvals)

    mnoise = (template-sac)**2 #total motor noise
    
    eps = np.sqrt(mnoise[1:]-np.var(alltraceswithoutsac))/uvals #s_eps
    uvals = uvals[(~np.isnan(eps)) & (~np.isinf(eps))] 
    eps = eps[(~np.isnan(eps)) & (~np.isinf(eps))]
    
    for i, u in enumerate(uvals): #separate each eps to the respective bin of u
        epsbin = np.where(ubinsp<u)[0][-1]
        epsbinnedp[epsbin].append(eps[i])
                                                  
for idx, sac in enumerate(negsaccades):
    template = meantimebinsp[:len(sac)]*(sac[-1]-sac[0])+sac[0]
    uvals = np.diff(template) #u of template so that xi is removed (since average saccade).
    uvals = np.abs(uvals)

    mnoise = (template-sac)**2 #total motor noise
    
    eps = np.sqrt(mnoise[1:]-np.var(alltraceswithoutsac))/uvals
    uvals = uvals[(~np.isnan(eps)) & (~np.isinf(eps))] 
    eps = eps[(~np.isnan(eps)) & (~np.isinf(eps))]
    
    for i, u in enumerate(uvals): #separate each eps to the respective bin of u
        epsbin = np.where(ubinsn<u)[0][-1]
        epsbinnedn[epsbin].append(eps[i])

#check if eps distribution differs 
epspooledp = [a for b in epsbinnedp for a in b]
epspooledn = [a for b in epsbinnedn for a in b]
epspooledp = np.array(epspooledp)
epspooledn = np.array(epspooledn)
_, peps = mannwhitneyu(epspooledp, epspooledn) #0.274 so not significant, you can pool

#pool epsbinnedp and n according to u:
epsbinnedpooled = [[epsbinnedp[a] + epsbinnedn[a]] for a in range(len(epsbinnedp))]
ubinspooled = ubinsp
epsmeanperu = [np.mean(a) for a in epsbinnedpooled]
ubinspooled = ubinspooled[1:][~np.isnan(epsmeanperu)]
epsmeanperu = np.array(epsmeanperu[1:])[~np.isnan(epsmeanperu[1:])]


#plot eps histogram for different u values
fig, ax = plt.subplots(1,1)
ax.hist(epsmeanperu, bins=20, color='k', density=True)
ax.set_ylabel('Frequency density')
ax.set_xlabel('Average $s_{\epsilon^\prime_i}$ per $\overline{|u(t)|}$ bin')
ax.set_title('Average $s_{\epsilon^\prime_i}$ distribution for all $\overline{|u(t)|}$ bins')

epsmediantot = np.median(epsmeanperu) #5.1

"""
#Check saccade traces before and after saccade
plt.figure()
for i in sactracebef:
    plt.plot(i)
for i in sactraceaf:
    plt.plot(i)    
"""
"""
#check velocity-noise relationship
#positive saccades
dsmtrp = sm.add_constant(np.sqrt(vartimebinsp[1:]))
dsmtrp
regp = sm.OLS(np.diff(meantimebinsp)*rt, dsmtrp)
pmod = regp.fit()
p_valuep = pmod.summary2().tables[1]['P>|t|'][1] #in some sessions this somehow returns nan!
t_valuep = pmod.summary2().tables[1]['t'][1]

#negative saccades
dsmtrn = sm.add_constant(np.sqrt(vartimebinsn[~np.isnan(vartimebinsn)][1:]))
regn = sm.OLS(np.diff(meantimebinsn[~np.isnan(meantimebinsn)])*rt, dsmtrn)
nmod = regn.fit()
p_valuen = nmod.summary2().tables[1]['P>|t|'][1]
t_valuen = nmod.summary2().tables[1]['t'][1]

fig, axs = plt.subplots(1,2, sharex=True)
fig.suptitle('Noise-velocity relationship', size=30)
axs[0].plot(np.sqrt(vartimebinsp[1:]), np.diff(meantimebinsp)*rt, 'r.', label='Data')
axs[0].plot(np.sqrt(vartimebinsp[1:]), pmod.fittedvalues, 'k-', label='Linear fit')
axs[1].plot(np.sqrt(vartimebinsn[~np.isnan(vartimebinsn)][1:]), np.diff(meantimebinsn[~np.isnan(meantimebinsn)])*rt, 
           'b.', label='Data')
axs[1].plot(np.sqrt(vartimebinsn[~np.isnan(vartimebinsn)][1:]), nmod.fittedvalues, 'k-', label='Linear fit')

axs[0].set_ylabel(r'Eye velocity $[\frac{norm}{s}]$')
axs[0].set_xlabel('Noise $\sigma$')
axs[0].set_title('Positive average saccade, t=%.2f' %(t_valuep))
axs[1].set_title('Negative average saccade, t=%.2f' %(t_valuen))

for ax in axs:
    ax.legend()
"""