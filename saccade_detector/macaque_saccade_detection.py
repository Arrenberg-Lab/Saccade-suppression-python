# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:29:37 2021

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
from scipy.stats import kruskal, mannwhitneyu
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import scipy.io as sio
from scipy.ndimage.filters import correlate1d
from scipy.signal import savgol_filter

#A good idea -> comment all the imports for what you are doing. Not necessary in the imminent future!

#Macaque saccade data
filename = r'../data/Hafed_dataset/example_saccades.mat'
dataset = sio.loadmat(filename)
globals().update(dataset)
mytime = np.squeeze(mytime) #this is from updated globals, dont mind the error

checksaccades = True

#convert saccades
dl = hlp.macaque_saccade_convert_eye_position(down_left_horizontal, down_left_vertical)
dr = hlp.macaque_saccade_convert_eye_position(down_right_horizontal, down_right_vertical)
ul = hlp.macaque_saccade_convert_eye_position(up_left_horizontal, up_left_vertical)
ur = hlp.macaque_saccade_convert_eye_position(up_right_horizontal, up_right_vertical)

if checksaccades == True:
    #check dl and ur -> standard notation is top and bottom, matter of taste.
    for i in range(dr.shape[0]):
        fig, axs = plt.subplots(1,2)
        axs[0].set_title('Down left')
        axs[1].set_title('Up right')
        axs[0].plot(dl[i,:], 'k-', label='converted')
        axs[0].plot(down_left_horizontal[i,:], 'r-', label='horizontal')
        axs[0].plot(down_left_vertical[i,:], 'b-', label='vertical')
        axs[1].plot(ur[i,:], 'k-', label='converted')
        axs[1].plot(up_right_horizontal[i,:], 'r-', label='horizontal')
        axs[1].plot(up_right_vertical[i,:], 'b-', label='vertical')
        axs[0].set_xlabel('Time [ms]')
        axs[0].set_ylabel('Eye position [°]')
        axs[1].legend()
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        plt.tight_layout()
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
        a = input('c to continue, q to quit \n')
        if a == 'c':
            kill = False
        elif a == 'q':
            kill = True
        if kill == True:
            break
    
    #check dr and ul
    for i in range(dr.shape[0]):
        fig, axs = plt.subplots(1,2)
        axs[0].set_title('Down right')
        axs[1].set_title('Up left')
        axs[0].plot(dr[i,:], 'k-', label='converted')
        axs[0].plot(down_right_horizontal[i,:], 'r-', label='horizontal')
        axs[0].plot(down_right_vertical[i,:], 'b-', label='vertical')
        axs[1].plot(ul[i,:], 'k-', label='converted')
        axs[1].plot(up_left_horizontal[i,:], 'r-', label='horizontal')
        axs[1].plot(up_left_vertical[i,:], 'b-', label='vertical')
        axs[0].set_xlabel('Time [ms]')
        axs[0].set_ylabel('Eye position [°]')
        axs[1].legend()
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        plt.tight_layout()        
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
        a = input('c to continue, q to quit \n')
        if a == 'c':
            kill = False
        elif a == 'q':
            kill = True
        if kill == True:
            break
    #saccades seem to be good. Now do the detection.

macdet = {'savgollength' : 51,
          'savgolorder' : 3,
          'a' : 0.8,
          'b' : 0.5,
          'onstd' : 10,
          'macaque' : True}

onsetdl = np.zeros(dl.shape[0]).astype(int) #saccade onset towards down left
offsetdl = np.zeros(dl.shape[0]).astype(int) #saccade offset towards down left

onsetdr = np.zeros(dr.shape[0]).astype(int) #saccade onset down right
offsetdr = np.zeros(dr.shape[0]).astype(int) #saccade offset down right

onsetul = np.zeros(ul.shape[0]).astype(int) #saccade onset up left
offsetul = np.zeros(ul.shape[0]).astype(int) #saccade offset up left

onsetur = np.zeros(ur.shape[0]).astype(int) #saccade onset up right
offsetur = np.zeros(ur.shape[0]).astype(int) #saccade offset up right

#! CHECK THE SACCADE DETECTION PARAMETERS, ADJUST THEM ACCORDINGLY TO GET A BETTER OUTCOME FOR MACAQUE.

for i in range(0,dl.shape[0]):
    print(i)
    saconidx, sacoffidx, smoothtr = hlp.detect_saccades_v2(dl[i,:], **macdet)
    onsetdl[i] = np.int(saconidx) #onset idx
    offsetdl[i] = np.int(sacoffidx) #offset idx
    #Check onset/offset detection for both eyes
    """
    fig, axs = plt.subplots(1,1)
    axs.plot(mytime, dl[i,:], 'k-', label='raw')
    axs.plot(mytime, smoothtr, 'b-', label='smooth')
    axs.plot(mytime[sacoffidx], dl[i,sacoffidx], 'r|', label='on/offset', markersize=30, mew=3)
    axs.plot(mytime[saconidx], dl[i,saconidx], 'r|', markersize=30, mew=3)
    axs.legend(loc='best')
    axs.set_ylabel('Eye position [°]')
    axs.set_xlabel('Time [ms]')
    plt.get_current_fig_manager().window.showMaximized()
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    a = input('c to continue, q to quit \n')
    if a == 'c':
        kill = False
    elif a == 'q':
        kill = True
    if kill == True:
        break
    """
#problematic traces -> 375!, 405, 434!, 496!, 528!, 549!, 607!

for i in range(0,dr.shape[0]):
    print(i)
    saconidx, sacoffidx, smoothtr = hlp.detect_saccades_v2(dr[i,:], **macdet)
    onsetdr[i] = np.int(saconidx) #onset idx
    offsetdr[i] = np.int(sacoffidx) #offset idx
    """
    fig, axs = plt.subplots(1,1)
    axs.plot(mytime, dr[i,:], 'k-', label='raw')
    axs.plot(mytime, smoothtr, 'b-', label='smooth')
    axs.plot(mytime[sacoffidx], dr[i,sacoffidx], 'r|', label='on/offset', markersize=30, mew=3)
    axs.plot(mytime[saconidx], dr[i,saconidx], 'r|', markersize=30, mew=3)
    axs.legend(loc='best')
    axs.set_ylabel('Eye position [°]')
    axs.set_xlabel('Time [ms]')
    plt.get_current_fig_manager().window.showMaximized()
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    a = input('c to continue, q to quit \n')
    if a == 'c':
        kill = False
    elif a == 'q':
        kill = True
    if kill == True:
        break
    #problematic traces -> 29, 45, 
    """
    
for i in range(0,ul.shape[0]):
    print(i)
    saconidx, sacoffidx, smoothtr = hlp.detect_saccades_v2(ul[i,:], **macdet)
    onsetul[i] = np.int(saconidx) #onset idx
    offsetul[i] = np.int(sacoffidx) #offset idx
    """
    fig, axs = plt.subplots(1,1)
    axs.plot(mytime, ul[i,:], 'k-', label='raw')
    axs.plot(mytime, smoothtr, 'b-', label='smooth')
    axs.plot(mytime[sacoffidx], ul[i,sacoffidx], 'r|', label='on/offset', markersize=30, mew=3)
    axs.plot(mytime[saconidx], ul[i,saconidx], 'r|', markersize=30, mew=3)
    axs.legend(loc='best')
    axs.set_ylabel('Eye position [°]')
    axs.set_xlabel('Time [ms]')
    plt.get_current_fig_manager().window.showMaximized()
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    a = input('c to continue, q to quit \n')
    if a == 'c':
        kill = False
    elif a == 'q':
        kill = True
    if kill == True:
        break
    #problematic traces -> !409,  
    """
 
for i in range(0,ur.shape[0]):
    print(i)
    saconidx, sacoffidx, smoothtr = hlp.detect_saccades_v2(ur[i,:], **macdet)
    onsetur[i] = np.int(saconidx) #onset idx
    offsetur[i] = np.int(sacoffidx) #offset idx
    
    """
    fig, axs = plt.subplots(1,1)
    axs.plot(mytime, ur[i,:], 'k-', label='raw')
    axs.plot(mytime[sacoffidx], ur[i,sacoffidx], 'r|', label='on/offset', markersize=30, mew=3)
    axs.plot(mytime[saconidx], ur[i,saconidx], 'r|', markersize=30, mew=3)
    axs.legend(loc='best')
    axs.set_ylabel('Eye position [°]')
    axs.set_xlabel('Time [ms]')
    plt.get_current_fig_manager().window.showMaximized()
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    a = input('c to continue, q to quit \n')
    if a == 'c':
        kill = False
    elif a == 'q':
        kill = True
    if kill == True:
        break
    #problematic traces -> !75, 105, 107, 168, 
"""    

#check overshoot distribution for different directions 
overshootsdl = np.zeros(dl.shape[0])
overshootsdr = np.zeros(dr.shape[0])
overshootsul = np.zeros(ul.shape[0])
overshootsur = np.zeros(ur.shape[0])


for idx in range(dl.shape[0]):
    saccade = dl[idx, onsetdl[idx] : offsetdl[idx]]    
    saccade = (saccade - saccade[0])/(saccade[-1] - saccade[0])
    overshootsdl[idx] = np.max(saccade)-1
    
    
for idx in range(dr.shape[0]):
    saccade = dr[idx, onsetdr[idx] : offsetdr[idx]]    
    saccade = (saccade - saccade[0])/(saccade[-1] - saccade[0])
    overshootsdr[idx] = np.max(saccade)-1
    
for idx in range(ul.shape[0]):
    saccade = ul[idx, onsetul[idx] : offsetul[idx]]    
    saccade = (saccade - saccade[0])/(saccade[-1] - saccade[0])
    overshootsul[idx] = np.max(saccade)-1
    
    
for idx in range(ur.shape[0]):
    saccade = ur[idx, onsetur[idx] : offsetur[idx]]    
    saccade = (saccade - saccade[0])/(saccade[-1] - saccade[0])
    overshootsur[idx] = np.max(saccade)-1

#check if distributions differ:
stats = kruskal(overshootsdl, overshootsdr, overshootsul, overshootsur) 
#Difference highly significant so I will separate them

fig, axs = plt.subplots(2,2)
fig.suptitle('Overshoot distributions for the saccades in different directions')
axs = axs.reshape(4)
dists = [overshootsul, overshootsur, overshootsdl, overshootsdr]
dnames = ['Upper left', 'Upper right', 'Lower left', 'Lower right']
for idx, ax in enumerate(axs):
    ax.hist(dists[idx], bins=np.floor(len(dists[idx])/10).astype(int))
    ax.set_title(dnames[idx])
    
diststog = np.zeros(np.sum([len(a) for a in dists])) #pooled version of all the overshoot distributions.
lenn = 0
for dist in dists:
    print(lenn,lenn+len(dist))
    diststog[lenn:lenn+len(dist)] = dist
    lenn += len(dist)

fig, ax = plt.subplots(1,1)
ax.hist(diststog, bins=np.floor(len(diststog)/10).astype(int))
ax.set_title('Pooled overshoot distribution')

#Motor noise 1: xi -> fit a linear curve to nonsaccadic traces, find the deviation from the fit and calculate s_xi
#down left
xidl, prenumdl, postnumdl, sxidl = hlp.macaque_xi_estimation(dl, onsetdl, offsetdl)
#down right
xidr, prenumdr, postnumdr, sxidr = hlp.macaque_xi_estimation(dr, onsetdr, offsetdr)
#up left
xiul, prenumul, postnumul, sxiul = hlp.macaque_xi_estimation(ul, onsetul, offsetul)
#down right
xiur, prenumur, postnumur, sxiur = hlp.macaque_xi_estimation(ur, onsetur, offsetur)

#pool all xi together
xipooled = np.array([a for b in [xidl,xidr, xiul, xiur] for a in b])
xipooled = xipooled[~np.isnan(xipooled)]
#take 99th percentile -> remove first 0.5 (negative outliers) and last 0.5 (positive outliers)
xipperc = xipooled[(xipooled<np.percentile(xipooled, 99.5)) & (xipooled>np.percentile(xipooled, 0.5))] 
sxipooled = hlp.calculate_s_xi(xipooled)

#Motor noise 2: epsilon -> trace during saccade
ucut = 0.01
s_epsdl, udl, s_epsdlall, tndl, nbdl = hlp.macaque_eps_estimation(dl, xipperc, onsetdl, offsetdl, ucutoff= ucut, RMSfac=0.5, 
                                                 onsrem = 5, offsrem=5, fplot=False, nplot=False)

s_epsdr, udr, s_epsdrall, tndr, nbdr = hlp.macaque_eps_estimation(dr, xipperc, onsetdr, offsetdr, ucutoff= ucut, RMSfac=0.5, 
                                                 onsrem = 5, offsrem=5, fplot=False, nplot=False)

s_epsul, uul, s_epsulall, tnul, nbul = hlp.macaque_eps_estimation(ul, xipperc, onsetul, offsetul, ucutoff= ucut, RMSfac=0.5, 
                                                 onsrem = 5, offsrem=5, fplot=False, nplot=False)

s_epsur, uur, s_epsurall, tnur, nbur = hlp.macaque_eps_estimation(ur, xipperc, onsetur, offsetur, ucutoff= ucut, RMSfac=0.5, 
                                                 onsrem = 5, offsrem=5, fplot=False, nplot=False)
s_epsavg = np.sum(np.array([s_epsdl,s_epsdr, s_epsul, s_epsur]) * 
                  np.array([dl.shape[0], dr.shape[0], ul.shape[0], ur.shape[0]])) / \
                  np.sum(np.array([dl.shape[0], dr.shape[0], ul.shape[0], ur.shape[0]])) #0.0961

#descriptive figures
totnoise = np.concatenate([tndl, tndr, tnul, tnur]) 
fig, axs = plt.subplots(1, 3)
axs[0].hist(totnoise, color='k')
axs[0].set_xlabel(r'$y_{t} - y_{f}$')
axs[0].set_ylabel(r'# of occurences')

totnb = np.concatenate([nbdl, nbdr, nbul, nbur])
axs[1].hist(totnb, color='k', label='$s_m$')
axs[1].plot([sxipooled[-1]]*2, axs[1].get_ylim(), 'r-', label=r'$s_{\xi}$')
axs[1].set_xlabel(r'$s_m$')
axs[1].set_ylabel(r'# of occurences')
axs[1].legend()


for i, (u,n) in enumerate(zip([udl, udr, uul, uur], [nbdl, nbdr, nbul, nbur])):
    if i == 0:
        axs[2].plot(u, n, 'k-', label='$s_m$')
    else:
        axs[2].plot(u, n, 'k-')
axs[2].plot(axs[2].get_xlim(), [sxipooled[-1]]*2, 'r-', label=r'$s_{\xi}$')
axs[2].set_xlabel(r'$u$')
axs[2].set_ylabel(r'Standard deviation')
axs[2].legend()

#do histogram for resting state
fig, ax = plt.subplots()
ax.hist(xipooled, color='k', density=True, bins=500)
ax.set_xlabel(r'$y_{t} - y_{f}$')
ax.set_ylabel(r'Density')
ax.set_title('Additive noise distribution')

