# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:39:08 2021

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
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


#Refine the saccade detection algorithm:
#Separation of saccades as temporal-nasal and nasal-temporal
#Smooth the curves before onset detection
#Extend offset detection (see slack)
#Try the algorithm in Nyström & Holmqvist 2010

#import saccade data 
root = r'\\172.25.250.112\arrenberg_data\shared\Ibrahim\HSC_saccades'
angthres = 5
flthres = 0.3
plot = False
saconsmoothsigma = 15 #gaussian smoothing filter standard deviation for saccade onset detection
rf = 1000 #sampling rate in Hz.
saccadedata, datlens, nmnposidx, saccadedataout, saccadedatanoise = hlp.extract_saccade_data(root, angthres, flthres)
#velocity and acceleration thresholds (from Nyström and Holmqvist), values exceeding these thresholds are set to 0 so they do 
#not interfere with saccade detection (assuming this happens at the saccade edges etc). !Check this procedure in case of errors.
velthres = 1000 
accthres = 100000
savgollength = 51 #The vindow length of savgol filter, must be odd number. NOTE IN PAPER LENGTH IS 2 x min saccade duration
savgolorder = 4 #order of the polynomial fit used to derive the savgol filter. NOTE IN PAPER ORDER IS 2

tracesnt = [] #eye traces nasal to temporal
tracestn = [] #eye traces temporal to nasal
tracesntsmooth = [] #eye traces nasal to temporal
tracestnsmooth = [] #eye traces temporal to nasal

onsetnt = [] #saccade onset nasal to temporal
offsetnt = [] #saccade offset nasal to temporal
onsettn = [] #saccade onset temporal to nasal
offsettn = [] #saccade offset temporal to nasal

for i in range(saccadedata.shape[2]):
    print(i)
    leraw = saccadedata[:,0,i][~np.isnan(saccadedata[:,0,i])]
    reraw = saccadedata[:,1,i][~np.isnan(saccadedata[:,1,i])]
    
    if i == 7: #for this value a narrower kernel works better.    
        saconidxle, sacoffidxle, lesmooth = hlp.detect_saccades_v2(leraw, smoothsigma=10)
        saconidxre, sacoffidxre, resmooth = hlp.detect_saccades_v2(reraw, smoothsigma=10)
    
    else: 
        saconidxle, sacoffidxle, lesmooth = hlp.detect_saccades_v2(leraw)            
        saconidxre, sacoffidxre, resmooth = hlp.detect_saccades_v2(reraw)
        
    if plot == True:
        #Check onset/offset detection for both eyes
        #offset mostly chosen as the last point, maybe you should increase a and b values for both eyes.
        fig, axs = plt.subplots(1,2)
        axs[0].plot(leraw, 'k-', label='raw')
        axs[0].plot(lesmooth, 'b-', label='smooth')
        axs[0].plot(sacoffidxle, leraw[sacoffidxle], 'r|', label='on/offset', markersize=30, mew=3)
        axs[0].plot(saconidxle, leraw[saconidxle], 'r|', markersize=30, mew=3)
        axs[0].legend(loc='best')
        axs[1].plot(reraw, 'k-', label='eye trace')
        axs[1].plot(resmooth, 'b-', label='smooth trace')
        axs[1].plot(sacoffidxre, reraw[sacoffidxre], 'r|', label='saccade onset', markersize=30, mew=3)
        axs[1].plot(saconidxre, reraw[saconidxre], 'r|', label='saccade onset', markersize=30, mew=3)
        axs[0].set_title('Left eye')
        axs[1].set_title('Right eye')
        axs[0].set_ylabel('Eye position [°]')
        axs[0].set_xlabel('Time [ms]')
        plt.get_current_fig_manager().window.showMaximized()
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break

    #separate eye traces into nasal-->temporal and temporal-->nasal
    #for left eye, nasal-->temporal is positive saccades, and temporal-->nasal negative. This is opposite for right eye.
    if leraw[-1]-leraw[0] < 0:
        tracestn.append(leraw)
        tracestnsmooth.append(lesmooth)

        onsettn.append(saconidxle)
        offsettn.append(sacoffidxle)

    else:
        tracesnt.append(leraw)
        tracesntsmooth.append(lesmooth)

        onsetnt.append(saconidxle)
        offsetnt.append(sacoffidxle)
    
    if reraw[-1]-reraw[0] > 0:
        tracestn.append(reraw)
        tracestnsmooth.append(resmooth)

        onsettn.append(saconidxre)
        offsettn.append(sacoffidxre)
    
    else:
        tracesnt.append(reraw)
        tracesntsmooth.append(resmooth)

        onsetnt.append(saconidxre)
        offsetnt.append(sacoffidxre)
    #codes used for debugging/development
    """
    #Check velocity curves for left eye
    fig, axs = plt.subplots(1,2, sharex=True)
    #axs[0].plot(leraw)
    axs[0].plot(lesmooth)
    axs[1].plot(velocityle, 'k-', label='eye trace')
    axs[1].plot([0,len(leraw)], [sacvelthresle, sacvelthresle], 'r-', label='eye trace')
    axs[1].plot([0,len(leraw)], np.array([sacoffthresle, sacoffthresle]), 'r-', label='eye trace')

    axs[1].plot(saconidxle, velocityle[saconidxle], 'r.', label='saccade onset')
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    
    """
    #Check offset detection for right eye
    fig, ax = plt.subplots(1,1)
    ax.plot(resmooth, 'k-', label='eye trace')
    ax.plot(sacoffidxre, resmooth[sacoffidxre], 'r.', label='saccade onset')
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    
    """
    #Ari's suggestion: threshold the saccade velocity, and eye position change in the last few frames should be smaller than a
    #threshold
    #find noise in 40 ms presaccade, if trace before saccade is < 40ms, then take all trace from beginning until saccade onset
    nonsaccade = velocityre[-40:] #velocity curve before saccade onset 
    
    #saccade offset threshold as 3 standard deviations of the presaccade trace
    underthres = velocityre[saconidxre:][velocityre[saconidxre:]<saconthresre]
    sacoffthres = np.std(nonsaccade)
    sacoffidxre = np.where(underthres < sacoffthres)[0][0] + saconidxre
    """    
    
    """
    #Check velocity curves for right eye
    fig, axs = plt.subplots(1,2, sharex=True)
    #axs[0].plot(reraw)
    axs[0].plot(resmooth)
    axs[1].plot(velocityre, 'k-', label='eye trace')
    axs[1].plot([0,len(reraw)], [sacvelthresre, sacvelthresre], 'r-', label='eye trace')
    axs[1].plot([0,len(reraw)], np.array([sacoffthresre, sacoffthresre]), 'r-', label='eye trace')

    axs[1].plot(saconidxre, velocityre[saconidxre], 'r.', label='saccade onset')
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    
    """
    #Check how well onset detection works for both eyes:
    #Works well for most cases.
    fig, axs = plt.subplots(1,2)
    axs[0].plot(leraw, 'k-', label='eye trace')
    axs[0].plot(saconidxle, leraw[saconidxle], 'r.', label='saccade onset')
    axs[1].plot(reraw, 'k-', label='eye trace')
    axs[1].plot(saconidxre, reraw[saconidxre], 'r.', label='saccade onset')    
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    
    """    
    #Check how well onset detection works for left eye:
    #All but first LE traces look good!
    fig, ax = plt.subplots(1,1)
    ax.plot(leraw, 'k-', label='eye trace')
    ax.plot(saconidxle, leraw[saconidxle], 'r.', label='saccade onset')    
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    
    """
    #try an iterative approach with velocity threshold
    velocityle = np.abs(np.diff(lesmooth))
    sacvelthresle = np.percentile(velocityle, 99)
    
    lethres = False
    while lethres == False:
        underthres = velocityle[velocityle<sacvelthresle]
        previoussacvelthresle = sacvelthresle
        sacvelthresle = np.mean(underthres) + 6*np.std(underthres)
        if np.abs(previoussacvelthresle-sacvelthresle) < 1:
            lethres = True
    
    underthres = velocityle[velocityle<sacvelthresle]
    saconthresle = np.mean(underthres) + 2*np.std(underthres)
    onpeakidxle = np.where(velocityle>sacvelthresle)[0][0] #index of first peak exceeding velocity threshold
    if onpeakidxle == 0: #choose the next point if first peak already happens at the first data point.
        saconidxle = 0
    else: 
        #flip the velocity profile from start of recording until onpeakle
        saconsetvelle = np.flip(velocityle[:onpeakidxle])
        #index of first datapoint in saconsetvelle exceeding saconthresle is the saccade onset index
        saconidxle = len(saconsetvelle) - np.where(saconsetvelle>saconthresle)[0][0]
    
    #Check saccades and fits
    fig, axs = plt.subplots(1,2, sharex=True)
    axs[0].plot(leraw, 'k-') #LE
    axs[1].plot(reraw, 'k-') #RE
    axs[0].plot(lesmooth, 'r-') #LE
    axs[1].plot(resmooth, 'r-') #RE
    axs[0].plot(saconidxle, lesmooth[saconidxle], 'b.') #LE
    
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break    
    """
    """
    #Recycle code if Nystör and Holmqvist is to be implemented. First try and I failed!
    #Possible reasons: longer window length required, noisy eye drifts in some instances etc.
    #Nyström & Holmqvist smoothens velocity and acceleration with a Savitzky-Golay filter:
    velocityle = np.abs(savgol_filter(leraw, savgollength, savgolorder, deriv=1) * rf) #filtered LE velocity in °/s
    velocityle[velocityle>velthres] = 0
    accle = np.abs(savgol_filter(leraw, savgollength, savgolorder, deriv=2) * rf**2) #filtered LE acceleration in °/s^2
    accle[accle>accthres] = 0
    velocityre = np.abs(savgol_filter(reraw, savgollength, savgolorder, deriv=1) * rf) #filtered RE velocity in °/s
    velocityre[velocityre>velthres] = 0
    accre = np.abs(savgol_filter(reraw, savgollength, savgolorder, deriv=2) * rf**2) #filtered RE acceleration in °/s^2
    accre[accre>accthres] = 0
    #saccade velocity threshold estimation: iterative and data-driven approach
    sacvelthresle = np.percentile(velocityle, 99) #set the initial threshold to 90th percentile to be on the safe side
    #iteration
    lethres = False
    while lethres == False:
        underthres = velocityle[velocityle<sacvelthresle]
        previoussacvelthresle = sacvelthresle
        sacvelthresle = np.mean(underthres) + 6*np.std(underthres)
        if np.abs(previoussacvelthresle-sacvelthresle) < 1:
            lethres = True
    
    #saccade onset detection: saccade onset velocity threshold is defined as mean+3*std for eye traces lower than peak threshold
    underthres = velocityle[velocityle<sacvelthresle]
    saconthresle = np.mean(underthres) + 3*np.std(underthres)
    #find the first peak exceeding saccade velocity threshold
    onpeakidxle = np.where(velocityle>sacvelthresle)[0][0] #index of first peak exceeding velocity threshold
    if onpeakidxle == 0: #choose the next point if first peak already happens at the first data point.
        onpeakidxle = np.where(velocityle>sacvelthresle)[0][1] #index of first peak exceeding velocity threshold
    #flip the velocity profile from start of recording until onpeakle
    saconsetvelle = np.flip(velocityle[:onpeakidxle])
    #index of first datapoint in saconsetvelle exceeding saconthresle is the saccade onset index
    saconidxle = len(saconsetvelle) - np.where(saconsetvelle>saconthresle)[0][0]
    
    #Check how well onset detection works for left eye:
    fig, ax = plt.subplots(1,1)
    ax.plot(leraw, 'k-', label='eye trace')
    ax.plot(saconidxle, leraw[saconidxle], 'r.', label='saccade onset')    
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    """
    #Check smoothed and raw velocity/acceleration
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Left eye')
    axs[0].plot(np.abs(np.diff(leraw))*rt, 'k-')
    axs[0].plot(velocityle, 'r-')
    axs[1].plot(np.abs(np.diff(np.diff(leraw)))*rt**2, 'k-', label='raw')
    axs[1].plot(accle, 'r-', label='filtered')
    axs[1].legend()
    axs[0].set_title('Velocity')
    axs[1].set_title('Acceleration')  
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Right eye')
    axs[0].plot(np.abs(np.diff(reraw))*rt, 'k-')
    axs[0].plot(velocityre, 'r-')
    axs[1].plot(np.abs(np.diff(np.diff(reraw)))*rt**2, 'k-', label='raw')
    axs[1].plot(accre, 'r-', label='filtered')
    axs[1].legend()
    axs[0].set_title('Velocity')
    axs[1].set_title('Acceleration')
    plt.get_current_fig_manager().window.state('zoomed')
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break    
    """

"""
#check the traces and discard if necessary
for i, trace in enumerate(tracesnt):
    fig, ax = plt.subplots(1,1)
    ax.plot(trace, 'k-')
    ax.plot(onsetnt[i], trace[onsetnt[i]], 'r.')
    ax.plot(offsetnt[i], trace[offsetnt[i]], 'r.')    
    plt.get_current_fig_manager().window.state('zoomed')
    print(i)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    #27, 50, 90, 91 -> check these values again.
for i in [27, 50, 90, 91]:
    trace = tracesnt[i]
    fig, ax = plt.subplots(1,1)
    ax.plot(trace, 'k-')
    ax.plot(onsetnt[i], trace[onsetnt[i]], 'r.')
    ax.plot(offsetnt[i], trace[offsetnt[i]], 'r.')    
    plt.get_current_fig_manager().window.state('zoomed')
    print(i)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    #remove 50,91. 27 and 90 onset is detected suboptimally because of the presaccadic noise level, not much to do against it.
"""

"""
for i, trace in enumerate(tracestn):
    fig, ax = plt.subplots(1,1)
    ax.plot(trace, 'k-')
    ax.plot(onsettn[i], trace[onsettn[i]], 'r.')
    ax.plot(offsettn[i], trace[offsettn[i]], 'r.')    
    plt.get_current_fig_manager().window.state('zoomed')
    print(i)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    #50, 53, 91 -> check these values again.
for i in [50,53,91]:
    trace = tracestn[i]
    fig, ax = plt.subplots(1,1)
    ax.plot(trace, 'k-')
    ax.plot(onsettn[i], trace[onsettn[i]], 'r.')
    ax.plot(offsettn[i], trace[offsettn[i]], 'r.')    
    plt.get_current_fig_manager().window.state('zoomed')
    print(i)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    #remove 50,53,91
"""

tracesnt = [j for i, j in enumerate(tracesnt) if i not in [50,91]]
tracesntsmooth = [j for i, j in enumerate(tracesntsmooth) if i not in [50,91]]

onsetnt = [j for i, j in enumerate(onsetnt) if i not in [50,91]]
offsetnt = [j for i, j in enumerate(offsetnt) if i not in [50,91]]

tracestn = [j for i, j in enumerate(tracestn) if i not in [50,53,91]]
tracestnsmooth = [j for i, j in enumerate(tracestnsmooth) if i not in [50,53,91]]

onsettn = [j for i, j in enumerate(onsettn) if i not in [50,53,91]]
offsettn = [j for i, j in enumerate(offsettn) if i not in [50,53,91]]

#check overshoot distribution for different nasal-temporal / temporal-nasal 
overshootstn = np.zeros(len(tracestn))
overshootsnt = np.zeros(len(tracesnt))

for idx, trace in enumerate(tracesnt):
    saccade = trace[onsetnt[idx] : offsetnt[idx]]    
    saccade = (saccade - trace[onsetnt[idx]])/(trace[offsetnt[idx]] - trace[onsetnt[idx]])
    overshootsnt[idx] = np.max(saccade)-1

for idx, trace in enumerate(tracestn):
    saccade = trace[onsettn[idx] : offsettn[idx]]    
    saccade = (saccade - trace[onsettn[idx]])/(trace[offsettn[idx]] - trace[onsettn[idx]])
    overshootstn[idx] = np.max(saccade)-1

#check if distributions differ:
_, overp = mannwhitneyu(overshootstn, overshootsnt) 
#the distributions differ significantly, so xi should be calculated separately

#for the noise estimate, you need to keep in mind we are dealing with discrete time steps, making the process
#a random walk. Therefore you need to scale the noise according to the time step: 7
#https://www.softcover.io/read/bf34ea25/math_for_finance/random_walks

dt = 1# for now set to 1 0.001 #sampling frequency is 1000 Hz (Giulia data) therefore dt is 1 ms = 0.001 s

#Motor noise 1: xi -> fit a linear curve to nonsaccadic traces, find the deviation from the fit and calculate s_xi
#nasal->temporal
xint = [] #xi nasal to temporal
prenumnt = 0 #number of presaccadic traces used in the analysis
postnumnt = 0 #number of postsaccadic traces used in the analysis
for idx, trace in enumerate(tracesnt):
    presaccade = trace[:onsetnt[idx]]
    postsaccade = trace[offsetnt[idx]:]

    if len(presaccade) < 20: #if less than 20 ms available for presaccade, discard the presaccadic curve from analysis
        None        
    else:
        prenumnt += 1
        presaccade = trace[5:onsetnt[idx]-5]#discard 5 ms from beginning and end of presaccadic fixation
        xpre = np.arange(0,len(presaccade))
        preparams, _ = curve_fit(hlp.linear_fit, xpre, presaccade)
        prefit = hlp.linear_fit(xpre, *preparams)
        xint.append((presaccade-prefit)/np.sqrt(dt)) #AS OF 28.06 RMS is left out, noise is simply template minus raw data
                                                     #and is scaled by the time step of the sampling (i.e. sampling freq).
                                                     #See also Florian's slack post on motor noise estimation from 24.06.
                                                     
    if len(postsaccade) < 20: #same story
        None
    else:
        postnumnt += 1
        postsaccade = trace[offsetnt[idx]+5:-5] #discard again
        xpost = np.arange(0,len(postsaccade))
        postparams, _ = curve_fit(hlp.linear_fit, xpost, postsaccade)
        postfit = hlp.linear_fit(xpost, *postparams)
        xint.append((postsaccade-postfit)/np.sqrt(dt))
    
xint = np.array([a for b in xint for a in b])
#find outliers
xintiqr = np.percentile(xint,75)-np.percentile(xint,25)
xintout = xint[(xint<np.median(xint)-1.5*xintiqr) | (xint>np.median(xint)+1.5*xintiqr)]
xintfiltered = xint[(xint>np.median(xint)-1.5*xintiqr) & (xint<np.median(xint)+1.5*xintiqr)]
s_xintraw = np.std(xint) #2.348° now smaller (after removing 5 ms from beginning and right before saccade onset)
#outliers as 1.5 IQR
s_xintfiltered = np.std(xintfiltered) #0.0075 this the same as before
#outliers as above 99th percentile (probably this is the safest way to go)
s_xintpercentiled = np.std(xint[xint<np.percentile(xint,99)]) #0.044

#temporal->nasal
xitn = [] #xi temporal to nasal
prenumtn = 0
postnumtn = 0
for idx, trace in enumerate(tracestn):
    presaccade = trace[:onsettn[idx]]
    postsaccade = trace[offsettn[idx]:]
    
    if len(presaccade) < 20: #if less than 20 ms available for presaccade, discard the presaccadic curve from analysis
        None        
    else:
        prenumtn += 1
        presaccade = trace[5:onsettn[idx]-5]#discard 5 ms from beginning and end of presaccadic fixation
        xpre = np.arange(0,len(presaccade))
        preparams, _ = curve_fit(hlp.linear_fit, xpre, presaccade)
        prefit = hlp.linear_fit(xpre, *preparams)
        xitn.append((prefit-presaccade)/np.sqrt(dt))
    if len(postsaccade) < 20: #same story
        None
    else:
        postnumtn += 1
        postsaccade = trace[offsettn[idx]+5:-5] #discard again
        xpost = np.arange(0,len(postsaccade))
        postparams, _ = curve_fit(hlp.linear_fit, xpost, postsaccade)
        postfit = hlp.linear_fit(xpost, *postparams)
        xitn.append((postfit-postsaccade)/np.sqrt(dt))

xitn = np.array([a for b in xitn for a in b])
xitn = xitn[~np.isnan(xitn)]
#find outliers
xitniqr = np.percentile(xitn,75)-np.percentile(xitn,25)
xitnout = xitn[(xitn<np.median(xitn)-1.5*xitniqr) | (xitn>np.median(xitn)+1.5*xitniqr)]
xitnfiltered = xitn[(xitn>np.median(xitn)-1.5*xitniqr) & (xitn<np.median(xitn)+1.5*xitniqr)]
s_xitnraw = np.std(xitn) #0.686° smaller after removing data points from beginning and right before saccade
#outliers as 1.5 IQR
s_xitnfiltered = np.std(xitnfiltered) #0.0101 slightly bigger (0.001)
#outliers as above 99th percentile (probably this is the safest way to go)
s_xitnpercentiled = np.std(xitn[xitn<np.percentile(xitn,99)]) #0.0386 slightly smaller (0.0001)

#see how s_xi is when all traces are pooled
xipooled = np.array([a for b in [xitn,xint] for a in b])
#find outliers
xipoolediqr = np.percentile(xipooled,75)-np.percentile(xipooled,25)
xipooledout = xipooled[(xipooled<np.median(xipooled)-1.5*xipoolediqr) | (xipooled>np.median(xipooled)+1.5*xipoolediqr)]
xipooledfiltered = xipooled[(xipooled>np.median(xipooled)-1.5*xipoolediqr) & (xipooled<np.median(xipooled)+1.5*xipoolediqr)]
s_xipooledraw = np.std(xipooled) #0.238° as of 4.3.2022
#outliers as 1.5 IQR
s_xipooledfiltered = np.std(xipooledfiltered) #0.0615
#outliers as above 99th percentile (probably this is the safest way to go)
s_xipooledpercentiled = np.std(xipooled[(xipooled<np.percentile(xipooled,99.5)) & 
                                        (xipooled>np.percentile(xipooled,0.5))]) #0.1435° as of 4.3.2022
#since values (especially outlier filtered) are similar, it is plausible to use pooled data.

#Motor noise 2: epsilon -> trace during saccade
#u is temporal derivative of the saccadic trace template, epsilon is calculated as s_eps = sqrt(var_tot-var_xi)/u.
#Template from Dai et al. 2016
#choose nice saccades -> ones fitting the template well, use RMS to quantify this.
#AS OF 23.06 the additive noise is updated to not use the squared error from the linear fits but the square root of it.
#This lead to a better estimate without removing outliers (i.e. difference of standard deviation with and without outliers
#is much much smaller now (0.2 to 0.09 as opposed to 1.7 to 0.04 from previous squared error case)).
#Therefore now I am using the standard deviation estimate with outliers.
RMSfac = 0.5 #rms factor for exclusion criterion

#temporal->nasal
ntn = [] #temporal-nasal deviation of eye traces from saccade fit
utn = []
RMSstn = np.zeros(len(tracestn))
fittedsacs = []
onsrem = 10 #number of datapoints to remove from saccade start
offsrem = 150 #number of datapoints to remove from saccade end
plot = False
for idx, trace in enumerate(tracestn):
    saccade = trace[onsettn[idx]+onsrem:offsettn[idx]-offsrem]
    t = np.arange(0, len(saccade))
    fitparams, _ = curve_fit(hlp.saccade_fit_func, t, saccade, method='lm', maxfev=100000)
    fittedsac = hlp.saccade_fit_func(t, *fitparams)
    #NOTE THAT 2-3 CASES FIT A STUPID FLAT LINE! Discard them since u will be zero
    totnoise = (saccade-fittedsac)/np.sqrt(dt)
    utemp = np.abs(np.diff(fittedsac)) #template u
    #I am not very sure if totnoise**2 is appropriate here.
    #eps = np.sqrt(totnoise**2-s_xipooledpercentiled**2)[1:] / utemp #PROBABLY BUG!!!
    #!!! TODO: MISTAKE
    #FIRST CAlCULATE ALL U VALUES, ALONG WITH ALL TOTNOISE (i.e. deviation between fit and trace).
    #YOU BIN U TOGETHER WITH TOTNOISE (i.e. WITHOUT CALCULATING EPSILON RIGHT OFF THE BAT).
    #FOR EACH U BIN YOU HAVE A TOTNOISE BIN; EACH TOTNOISE BIN HAS A DISTRIBUTION -> u-dependent total noise distributions
    #THEN YOU CAN USE THE EQUATION FOR DETERMINING ALPHA*EPSILON. IN CASES WHERE XI IS SMALLER THAN S_M
    ntn.append(totnoise)
    utn.append(utemp)
    RMSstn[idx] = np.sqrt(np.mean(totnoise**2))
    fittedsacs.append(fittedsac)
    print(idx)
    if plot == True:
        fig, axs = plt.subplots(1,3)
        axs[0].plot(saccade, 'k-', label='trace')
        axs[0].plot(fittedsac, 'r-', label='fit')
        axs[0].legend()
        axs[0].set_title('Saccade')
        axs[1].plot(totnoise, 'k.', label=r'$y_{t} - y_{f}$')
        axs[1].plot([0, len(totnoise)], [s_xipooledraw,s_xipooledraw], 'r-', label=r'$s_\xi$')
        axs[1].legend()
        axs[1].set_title('Noises')
        axs[2].plot(utemp, 'k.')
        axs[2].set_title(r'$u$')
        plt.get_current_fig_manager().window.showMaximized()
        plt.tight_layout()
        plt.pause(0.1)

        a = True
        while a == True:
            plt.tight_layout()
            plt.waitforbuttonpress()
            plt.close('all')
            inp = input('c for continue, s for stop \n')
            if inp == 'c':
                break
            elif inp == 's':
                sys.exit()
            else:
                inp = input('Wrong button, c for continue, s for stop \n')
        
#visual inspection to "nice" saccades
plot = False
nicesacidxstn = np.where(RMSstn<=np.mean(RMSstn)-RMSfac*np.std(RMSstn))[0]
if plot == True:
    for idx in nicesacidxstn:
        print(idx, nicesacidxstn, RMSstn[idx])
        fig, ax = plt.subplots(1,1)
        ax.plot(tracestn[idx][onsettn[idx]+onsrem:offsettn[idx]-offsrem], label='raw trace')
        ax.plot(fittedsacs[idx], label='template')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Eye position [°]')
        plt.legend()
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        a = True
        while a == True:
            inp = input('c for continue, s for stop \n')
            if inp == 'c':
                plt.close()
                break
            elif inp == 's':
                sys.exit()
            else:
                inp = input('Wrong button, c for continue, s for stop \n')
            
ntn = np.array(ntn)
ntn = np.array([a for b in ntn[nicesacidxstn] for a in b])
utn = np.array(utn)
utn = np.array([a for b in utn[nicesacidxstn] for a in b])
(utn, ntn) = zip( *sorted( zip(utn, ntn) ) ) #sort u and noise arrays in ascending u order
utn = np.array(utn)
ntn = np.array(ntn)

#bin u values and epsilon values -> 03.05.2022 : Bin it per percentile, that you take first 100 smallest u values 
#and bin is the average value of these 100 values. -> This is almost like logarithmic.
#Alternative idea is logarithmic binning of u
nperbin = 100 #number of datapoints per bin

ubinstn = []
binnedutn = [] #u bins
binnedntn = [] #total noise bins

for k in range(np.ceil(len(utn)/nperbin).astype(int)):
    print(k*nperbin, k*nperbin+nperbin)
    if k*nperbin+nperbin < len(utn):
        us = utn[k*nperbin : k*nperbin+nperbin]
        ns = ntn[k*nperbin : k*nperbin+nperbin]
    else:
        us = utn[k*nperbin:]
        ns = ntn[k*nperbin:]
    ubinstn.append(np.mean(us))
    binnedutn.append(us)
    binnedntn.append(ns)
    
ubinstn = np.array(ubinstn)  

#find average s_eps per bin and then average over all
ucutoff = 0.01 #u cutoff value since smaller than this causes huge epsilon inflation
usidx = np.where(ubinstn>ucutoff)[0][0] #the first u bin index value bigger than cutoff

stdnperbintn = np.array([np.std(a) for a in binnedntn]) #standard deviation of total noise
s_epstns = 1/np.abs(ubinstn[usidx:])*np.sqrt(stdnperbintn[usidx:]**2 - s_xipooledpercentiled**2)
s_epstn = np.mean(s_epstns[~np.isnan(s_epstns)]) #1.824 as of 4.3.2022, 7.818 (!) after binning u according to percentiles.

#descriptive figures 
fig, axs = plt.subplots(1, 3)
axs[0].hist(ntn, color='k', density=True)
axs[0].set_xlabel(r'$y_{t} - y_{f}$')
axs[0].set_ylabel(r'Density')


axs[1].hist(stdnperbintn, color='k', label='$s_m$', density=True)
axs[1].plot([s_xipooledpercentiled]*2, axs[1].get_ylim(), 'r-', label=r'$s_{\xi}$')
axs[1].set_xlabel(r'$s_m$')
axs[1].set_ylabel(r'Density')
axs[1].legend()

axs[2].plot(ubinstn[usidx:], stdnperbintn[usidx:], 'k.-', label='$s_m$')
axs[2].plot(axs[2].get_xlim(), [s_xipooledpercentiled]*2, 'r-', label=r'$s_{\xi}$')
axs[2].set_xlabel(r'$u$')
axs[2].set_ylabel(r'Standard deviation')
axs[2].legend()

#!!!do an example saccade plot withh fits and stuff
explot = False
if explot == True:
    #Plot style: General figure parameters:
    figdict = {'axes.titlesize' : 30,
               'axes.labelsize' : 25,
               'xtick.labelsize' : 25,
               'ytick.labelsize' : 25,
               'legend.fontsize' : 25,
               'figure.titlesize' : 25,
               'image.cmap' : 'gray'}
    plt.style.use(figdict)
    idx = 8
    saccadedat = tracesnt[8] #tracesnt[2]
    """
    for itr, tr in enumerate(tracesnt):
        if tr.shape != saccadedat.shape:
            continue
        elif False not in (tr == saccadedat):
            print(itr)
    """
    plt.figure()
    plt.plot(saccadedat)
    t = np.arange(saccadedat.shape[0])
    fig, ax = plt.subplots()
    onset = onsetnt[2]
    offset = offsetnt[2]
    
    #plot the eye trace
    ax.plot(offset, saccadedat[offset], 'k|', label='saccade onset', markersize=30, mew=3)
    ax.plot(onset, saccadedat[onset], 'k|', label='saccade onset', markersize=30, mew=3)
    ax.plot(t[:onset], saccadedat[:onset], 'gray')
    ax.plot(t[offset:], saccadedat[offset:], 'gray')
    ax.plot(t[onset:offset], saccadedat[onset:offset], 'k')
    
    #nonsaccadic fits - pre
    presaccade = saccadedat[5:onset-5] #discard 5 ms from beginning and end of presaccadic fixation
    xpre = np.arange(5,onset-5)
    preparams, _ = curve_fit(hlp.linear_fit, xpre, presaccade)
    prefit = hlp.linear_fit(xpre, *preparams)
    #nonsaccadic fits - post    
    postsaccade = saccadedat[offset+5:-5] #discard again
    xpost = np.arange(offset+5,len(saccadedat)-5)
    postparams, _ = curve_fit(hlp.linear_fit, xpost, postsaccade)
    postfit = hlp.linear_fit(xpost, *postparams)
    #plots
    ax.plot(xpost, postfit, 'r-')
    ax.plot(xpre, prefit, 'r-')
    
    #saccadic fit
    saccade = saccadedat[onset+onsrem:offset-offsrem]
    t = np.arange(0, len(saccade))
    fitparams, _ = curve_fit(hlp.saccade_fit_func, t, saccade, method='lm', maxfev=100000)
    fittedsac = hlp.saccade_fit_func(t, *fitparams)
    #plot
    t = np.arange(onset+onsrem, offset-offsrem)
    ax.plot(t, fittedsac, 'b-')
    
    #figure adjustments
    ax.set_ylabel('Eye position [°]')
    ax.set_xlabel('t [ms]')
    ax.set_title('Example zebrafish eye trace')    
#nasal->temporal
nnt = []
unt = []
RMSsnt = np.zeros(len(tracesnt))
fittedsacs = []
offsrem = 100
plot = False
for idx, trace in enumerate(tracesnt):
    saccade = trace[onsetnt[idx]+onsrem:offsetnt[idx]-offsrem] #!offsrem shorter here
    t = np.arange(0, len(saccade))
    fitparams, _ = curve_fit(hlp.saccade_fit_func, t, saccade, method='lm', maxfev=100000)
    fittedsac = hlp.saccade_fit_func(t, *fitparams)
    #NOTE THAT 2-3 CASES FIT A STUPID FLAT LINE! Discard them since u will be zero
    totnoise = (saccade-fittedsac)/np.sqrt(dt)
    utemp = np.abs(np.diff(fittedsac)) #template u
    #eps = np.sqrt(totnoise**2-s_xipooledpercentiled**2)[1:] / utemp
    #epsnt.append(eps)
    nnt.append(totnoise)
    unt.append(utemp)
    RMSsnt[idx] = np.sqrt(np.mean(totnoise))
    fittedsacs.append(fittedsac)
    print(idx)
    if plot == True:
        fig, axs = plt.subplots(1,3)
        axs[0].plot(saccade, 'k-', label='trace')
        axs[0].plot(fittedsac, 'r-', label='fit')
        axs[0].legend()
        axs[0].set_title('Saccade')
        axs[1].plot(totnoise, 'k.', label=r'$y_{t} - y{f}$')
        axs[1].plot([0, len(totnoise)], [s_xipooledraw,s_xipooledraw], 'r-', label=r'$s_\xi$')
        axs[1].legend()
        axs[1].set_title('Noises')
        axs[2].plot(utemp, 'k.')
        axs[2].set_title(r'$u$')
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        a = True
        while a == True:
            inp = input('c for continue, s for stop \n')
            if inp == 'c':
                plt.close()
                break
            elif inp == 's':
                sys.exit()
            else:
                inp = input('Wrong button, c for continue, s for stop \n')
    
    #25(!) saccades are fitted with a flat line -> search for better parametric saccade fit functions or use the fit function
    #on smoothed saccade

#visual inspection to "nice" saccades
plot = False
rmsidxnt = ~np.isnan(RMSsnt)
nicesacidxsnt = np.where(RMSsnt<=np.mean(RMSsnt[rmsidxnt])-RMSfac*np.std(RMSsnt[rmsidxnt]))
if plot == True:
    for idx in nicesacidxsnt:
        print(idx, nicesacidxsnt, RMSsnt[idx])
        fig, ax = plt.subplots(1,1)
        ax.plot(tracesnt[idx][onsetnt[idx]+onsrem:offsetnt[idx]-offsrem], label='raw trace')
        ax.plot(fittedsacs[idx], label='template')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Eye position [°]')
        plt.legend()
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        a = True
        while a == True:
            inp = input('c for continue, s for stop \n')
            if inp == 'c':
                plt.close()
                break
            elif inp == 's':
                sys.exit()
            else:
                inp = input('Wrong button, c for continue, s for stop \n')
    

nnt = np.array(nnt)
nnt = np.array([a for b in nnt[nicesacidxsnt] for a in b])
unt = np.array(unt)
unt = np.array([a for b in unt[nicesacidxsnt] for a in b])
(unt, nnt) = zip( *sorted( zip(unt, nnt) ) ) #sort u and noise arrays in ascending u order
unt = np.array(unt)
nnt = np.array(nnt)

#bin u values and epsilon values, again do similar percentile approach -> first 100 smallest values first bin etc
nperbin = 100 #number of datapoints per bin

ubinsnt = []
binnedunt = [] #u bins
binnednnt = [] #total noise bins

for k in range(np.ceil(len(unt)/nperbin).astype(int)):
    print(k*nperbin, k*nperbin+nperbin)
    if k*nperbin+nperbin < len(unt):
        us = unt[k*nperbin : k*nperbin+nperbin]
        ns = nnt[k*nperbin : k*nperbin+nperbin]
    else:
        us = unt[k*nperbin:]
        ns = nnt[k*nperbin:]
    ubinsnt.append(np.mean(us))
    binnedunt.append(us)
    binnednnt.append(ns)
    
ubinsnt = np.array(ubinsnt)  

#find average s_eps per bin and then average over all
ucutoff = 0.01 #u cutoff value since smaller than this causes huge epsilon inflation
usidx = np.where(ubinsnt>ucutoff)[0][0] #the first u bin index value bigger than cutoff

stdnperbinnt = np.array([np.std(a) for a in binnednnt]) #standard deviation of total noise
s_epsnts = 1/np.abs(ubinsnt[usidx:])*np.sqrt(stdnperbinnt[usidx:]**2 - s_xipooledpercentiled**2)
s_epsnt = np.mean(s_epsnts[~np.isnan(s_epsnts)]) #1.824 as of 4.3.2022, 10.208(!) as of 03.05 after percentile binning

#descriptive figures 
fig, axs = plt.subplots(1, 3)
axs[0].hist(nnt, color='k', density=True)
axs[0].set_xlabel(r'$y_{t} - y_{f}$')
axs[0].set_ylabel(r'Density')

axs[1].hist(stdnperbinnt, color='k', label='$s_m$', density=True)
axs[1].plot([s_xipooledpercentiled]*2, axs[1].get_ylim(), 'r-', label=r'$s_{\xi}$')
axs[1].set_xlabel(r'$s_m$')
axs[1].set_ylabel(r'Density')
axs[1].legend()

axs[2].plot(ubinsnt[usidx:], stdnperbinnt[usidx:], 'k.-', label='$s_m$')
axs[2].plot(axs[2].get_xlim(), [s_xipooledpercentiled]*2, 'r-', label=r'$s_{\xi}$')
axs[2].set_xlabel(r'$u$')
axs[2].set_ylabel(r'Standard deviation')
axs[2].legend()

#do also for nonsaccadic error
fig, ax = plt.subplots()
ax.hist(xipooled[(xipooled<np.percentile(xipooled,99.5)) & (xipooled>np.percentile(xipooled,0.5))], 
        color='k', density=True)
ax.set_xlabel(r'$y_{t} - y_{f}$')
ax.set_ylabel(r'Density')


#LEAVE THIS FOR NOW! Removing data points improved, you need to come up with a better interpolation.
#sometimes the eye tracker generates stupid artefacts, create an interpolator which detects these outliers and removes
#them
extrc = np.copy(tracestn[3])
t = np.arange(0,len(extrc))
trcvel = np.diff(extrc)*rf #velocity curve in °/s
#take 100 ms snippets:
stdvelg = np.std(trcvel) # standard deviation of all traces
sudfluc = np.abs(trcvel)>3.5*stdvelg #looks like it works well for this current example
sudflucidxs = np.where(sudfluc==1)[0]
#remove data points between each adjacent datapoints where sudfluc==1
for idx in sudflucidxs:
    if idx == sudflucidxs[0]:
        continue
    pos = extrc[idx]
    rmv = True
    rmvidx = 0
    while rmv == True:
        if np.abs(np.diff([extrc[idx+rmvidx],pos])) < 0.5:
            extrc[idx] = np.nan
            rmvidx += 1
        else:
            rmv = False

tvals = np.where(~np.isnan(extrc[sudflucidxs[0]:]))[0] + sudflucidxs[0]
trcvals = extrc[sudflucidxs[0]:][~np.isnan(extrc[sudflucidxs[0]:])]
intp = interp1d(tvals, trcvals)
tintp = np.where(np.isnan(extrc))[0]
intptrc = intp(tintp) 
extrc[tintp] = intptrc  

#check position and velocity
fig, axs = plt.subplots(1,2, sharex=True)
axs[0].plot(t, extrc)
#axs[0].plot(t, tracestn[3])
axs[1].plot(t[1:], trcvel)
axs[0].plot(t[1:][sudfluc>0], extrc[1:][sudfluc>0], 'r.')

"""
for idx, trace in enumerate(tracestn):
    print(idx)
    plt.plot(trace)
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.1)
    a = True
    while a == True:
        inp = input('c for continue, s for stop \n')
        if inp == 'c':
            plt.close()
            break
        elif inp == 's':
            sys.exit()
        else:
            inp = input('Wrong button, c for continue, s for stop \n')
"""