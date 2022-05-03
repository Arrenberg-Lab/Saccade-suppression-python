# -*- coding: utf-8 -*-
"""
Created on Wed May 19 01:40:48 2021

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
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit

#initial Reichardt like detector implementation
#General parameters
nfilters = 8
rsl = 5
jsigma = 4
nsimul = 10
shiftmag = 340
startloc = -170
tdur = shiftmag/200*rsl*1000/ 10
shiftdir = 'right'
speciesparams = hlp.generate_species_parameters(nfilters)
#zfparams = speciesparams.zebrafish()
zfparams = speciesparams.zebrafish_updated()
img = np.zeros(np.array([180,360])*rsl)
#shuffle rf locations
__, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, params=zfparams)
#phase shifted second set of filters with all the same parameters except with a phase shift as much as rf radius
fltcenters2 = fltcenters.copy()
for idx, params in enumerate(zfparams):
    rds = params[1]*rsl/2 #RF radius in pixels
    fltcenters2[idx,0] += rds
#loop around the index (azimuth) if it exceeds current max index
fltcenters2[:,0][fltcenters2[:,0]>=360*rsl] -= 360*rsl

zfparams = np.squeeze(zfparams)

#stimulus
stim = hlp.cylindrical_random_pix_stimulus(rsl, shiftmag, tdur, maxelev=70)
stim.test_stimulus(5, -170)

#preallocate arrays for the model
rfsimg = np.zeros(img.shape) #all receptive fields in geographical coordinates
rfcents = np.zeros([nfilters,3]) #rf centers in cartesian coordinates, 2nd dimension in the order x,y,z
rfacts = np.zeros([nfilters,stim.frametot+1]) #rf activities, nfilters x total number of frames
rfacts2 = np.zeros([nfilters,stim.frametot+1]) #rf activities, nfilters x total number of frames
rfactsnormed = rfacts.copy() #activities normed by flt size.
rfactsnormed2 = rfacts.copy() #activities normed by flt size.

#calculate the rf activities for the given stimulus.    
checkrfs = False
checkframe = False
for i in range(len(zfparams)):    
    rfarr, rfcentcar = hlp.gaussian_rf_geog(zfparams[i,0], zfparams[i,1]/2, *fltcenters[i], rsl)
    rfarr2, rfcentcar2 = hlp.gaussian_rf_geog(zfparams[i,0], zfparams[i,1]/2, *fltcenters2[i], rsl)
    if checkrfs == True:
        print('wl=%.5f (sf=%.5f), sz=%.3f'%(1/zfparams[i,0], zfparams[i,0], zfparams[i,1]))
        fig, ax = plt.subplots(1,1)
        pic = rfarr
        pic[rfarr2!=0] = rfarr2[rfarr2!=0]
        pos = ax.imshow(pic, cmap='jet', origin='lower', extent=[-180,180,-90,90])
        fig.colorbar(pos)
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
            checkrfs = False
            pass
    rfsimg[rfarr!=0] = rfarr[rfarr!=0]
    rfsimg[rfarr2!=0] = rfarr2[rfarr2!=0]
    rfcents[i,:] = rfcentcar #use the receptive field centers in cartesian coordinates for future decoding
    for j in range(stim.frametot+1):
        stimulus, __ , __ = stim.move_stimulus(j, shiftdir) #shift in positive
        rfact = np.sum(rfarr[rfarr!=0]*stimulus[rfarr!=0])
        rfact2 = np.sum(rfarr2[rfarr2!=0]*stimulus[rfarr2!=0])
        rfacts[i,j] = rfact
        rfacts2[i,j] = rfact2
        if checkframe == True:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(stimulus, origin='lower', extent=[-180,180,-90,90])
            axs[1].imshow(rfarr, origin='lower', extent=[-180,180,-90,90])
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
                checkframe = False
                pass
        #print(rfact)
        rfactsnormed[i,j] = rfact / (np.pi*(zfparams[i,1]/2)**2) #normed to unit area
        rfactsnormed2[i,j] = rfact / (np.pi*(zfparams[i,1]/2)**2) #normed to unit area

    print(i)

#Look at the horizontal shift estimate -> azimuth difference between units 
#YOU CANNOT DO LIKE THIS SINCE THIS APPROACH REFUTES THE FACT THAT -180 AZIMUTH IS CLOSER TO 179 THAN TO -50 AZIMUTH!
#This is a problem for long shifts and when the whole field stimulus is repeating itself, either of which is not even the case
#for our purposes. Keep it simple and use azimuth angle difference like a normal human being.


#! FOUND AN ERROR: since in geographic coordinates, the distance between 2 units is assumed the shortest one. Yet, stimulus can
#also move from the opposite (longer) direction, causing an almost 0 estimate of the speed tuning (since very short distance is
#assumed but maximum correlation happens with a very big shift)

#

#DO THE THING BETWEEN PHASE SHIFTED RF PAIRS
rfcentaz = fltcenters[:,0]/rsl - 180
rfcentaz2 = fltcenters2[:,0]/rsl - 180
motionsig = np.zeros([nfilters, stim.frametot+1]) #pairwise unit correlation array, first dim unit pair
                                                               #second dimension frame difference
xdif = rfcentaz2 - rfcentaz #azimuth angle difference between unit centers (2nd unit - 1st unit)
xdif[xdif<0] +=360
sptuning = np.zeros(motionsig.shape) #speed tuning for the given array. This is azimuth angle difference / shift
pshs = []
nshs = []
for tdx in range(stim.frametot+1):       
    print(tdx) #in frames
    ticker = 0 #dummy variable to follow the pair array indices
    
    if tdx == 0:
        pshift = np.sum(rfacts2*rfacts)
        nshift = np.sum(rfacts2*rfacts)
        sptuning[ticker, tdx] = 0

    else:
        #second cell activity is shifted ahead in time: max correlation if second cell shows similar excitement
        #later. First frame of first cell correlated with a later frame of second cell.
        pshift = np.sum(rfacts[:,:-tdx]*rfacts2[:,tdx:])
        #first cell activity shifted ahead in time 
        nshift = np.sum(rfacts[:,tdx:]*rfacts2[:,:-tdx])
        sptuning[:, tdx] = xdif / (tdx/stim.fps) #convert the unit of speed tuning to °/s
            
        mosig = pshift - nshift
        pshs.append(pshift)
        nshs.append(nshift)
        """
        if mosig < 0:
            mosig = np.abs(mosig)
            sptuning[ticker,tdx] = -sptuning[ticker,tdx]
        """
        motionsig[ticker, tdx] = mosig
        #print(ticker)
        ticker += 1
        
sptun = sptuning.flatten()
msig = motionsig.flatten()

#bin speed tuning to sum up motion signal
sptunbins = np.linspace(np.min(sptun),np.max(sptun), 1000)
binnedspt = [[] for a in range(len(sptunbins))]
binnedmsig = [[] for a in range(len(sptunbins))]

#sort sptun and msig into bins
for idx, ms in enumerate(msig):
    curbin = np.where(sptunbins<=sptun[idx])[0][-1]
    binnedmsig[curbin].append(ms)
    binnedspt[curbin].append(sptun[idx])

netmsig = [np.sum(a) for a in binnedmsig]
netmsig = np.array(netmsig)
startidx = np.where(netmsig>0)[0][0]
mspt = sptunbins[np.where(netmsig==np.max(netmsig))[0][0]] #speed tuning of maximum motion signal (°/sec)
popt,pcov = curve_fit(hlp.gauss,sptunbins[startidx:],netmsig[startidx:])
gfit = hlp.gauss(sptunbins[startidx:], *popt)

expstim, __, __ = stim.move_stimulus(stim.frametot, shiftdir)
f1, __, __ = stim.move_stimulus(0, shiftdir)
expstim[f1!=0] = f1[f1!=0]

fig, axs = plt.subplots(2,1, sharex=True, sharey=True, constrained_layout=True)
axs[0].imshow(rfsimg, cmap='jet', origin='lower', extent=[-180,180,-90,90])
axs[1].imshow(expstim, origin='lower', extent=[-180,180,-90,90])
axs[0].set_ylabel('Elevation', x=0, y=-0.15)
axs[1].set_xlabel('Azimuth')
if stim.test == True:
    axs[1].arrow((stim.spbar)/rsl-180,0,shiftmag,0, width=5, length_includes_head=True, color='red')

"""
#Check the stimulus shift rounding error correction with complex stimulus
expstim, __, __ = stim.move_stimulus(stim.frametot, shiftdir)
f1, __, __ = stim.move_stimulus(0, shiftdir)
f1[500:600, 1000:1100] = 2
expstim[500:600, np.int(1000+shiftmag*rsl):np.int(1100+shiftmag*rsl)] = 2

fig, axs = plt.subplots(1,2)
axs[0].imshow(expstim, origin='lower', extent=[-180,180,-90,90])
axs[1].imshow(f1, origin='lower', extent=[-180,180,-90,90])
"""
cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots()
for i in range(rfacts.shape[0]):
    ax.plot(rfacts[i,:], '.-', color=cmap(i))
    ax.plot(rfacts2[i,:], '--', color=cmap(i))
ax.set_ylabel('Model unit activity [a.u.]')
ax.set_xlabel('Frame number')
ax.set_title('Example receptive field set activity')


fig, ax = plt.subplots(1,2)
titles = ['Spatial frequency [cyc/°]', 'RF diameter [°]']
for idx in range(zfparams.shape[1]-1):
    ax[idx].hist(zfparams[:,idx], bins=np.int(nfilters/2), color='k')
    ax[idx].set_title(titles[idx])
fig.suptitle('Zebrafish parameter distributions used in the simulation')


"""
#Try the multineuron approach (Warland et al. 1997)
flen = 3 #length of the filter is 3 frames
M = stim.frametot+1-flen
Rrowsize = flen * nfilters
R = np.ones((M, Rrowsize+1))
#fill in the R matrix
for idx in range(M):
    acts = rfactsnormed[:,idx:flen+idx].flatten()
    R[idx, 1:] = acts
"""

frametot = stim.frametot #total number of frames
shperfr = shiftmag/frametot #the amount of shift per frame in degrees
fps = stim.fps

#start with considering the decoded motion error in all shifts
rsh = [] #real shift
dsh = [] #decoded shift
ss = [] #start and stop positions (degrees) for the considered shift, maybe for future analysis to see if stimulus position
        #shows a bias in motion estimate (highly likely for single RF set, highly unlikely for multiple sets).


for start in range(0, 1): #reduce the sampling and take every 5th frame as start for motion decoding -> even better take one 
                          #starting position  
    startang = startloc + (start*shperfr) #starting position of the bar (azimuth angles)
    for end in range(frametot, start, -1):
        endang = startloc + (end*shperfr) #starting position of the bar (azimuth angles)

        print('start %i end %i \n'%(start,end))
        nf = end-start #number of frames used to find out shift magnitude and duration
        sm = shperfr * nf #the magnitude of the current stimulus shift
        td = nf / fps #duration of the current signal
        
        motionsig = np.zeros([nfilters, nf+1]) #pairwise unit correlation array, first dim unit pair
                                               #second dimension frame difference
        sptuning = np.zeros(motionsig.shape) #speed tuning for the given array. This is azimuth angle difference / shift

        #motion decoding for each shift
        for i in range(nfilters):
        
            xdif = rfcentaz2[i] - rfcentaz[i] #azimuth angle difference between unit centers (2nd unit - 1st unit)
            if xdif < 0:
                xdif = 360 + xdif
            #print(xdif)
            
            pshs = []
            nshs = []
            for tdx in range(nf+1):       
                #print(tdx) #in frames
                if tdx == 0:
                    pshift = np.sum(rfacts2[i]*rfacts[i])
                    nshift = np.sum(rfacts2[i]*rfacts[i])
                    sptuning[0, tdx] = 0
            
                else:
                    #second cell activity is shifted ahead in time: max correlation if second cell shows similar excitement
                    #later. First frame of first cell correlated with a later frame of second cell.
                    pshift = np.sum(rfacts[i, :-tdx]*rfacts2[i, tdx:])
                    #first cell activity shifted ahead in time 
                    nshift = np.sum(rfacts[i, tdx:]*rfacts2[i, :-tdx])
                    sptuning[i, tdx] = xdif / (tdx/fps) #convert the unit of speed tuning to °/s
                        
                mosig = pshift - nshift
                pshs.append(pshift)
                nshs.append(nshift)
                motionsig[i, tdx] = mosig
                
        sptun = sptuning.flatten()
        msig = motionsig.flatten()
        
        #bin speed tuning to sum up motion signal
        sptunbins = np.linspace(np.min(sptun),np.max(sptun), 1000)
        binnedspt = [[] for a in range(len(sptunbins))]
        binnedmsig = [[] for a in range(len(sptunbins))]
        
        #sort sptun and msig into bins
        for idx, ms in enumerate(msig):
            curbin = np.where(sptunbins<=sptun[idx])[0][-1]
            binnedmsig[curbin].append(ms)
            binnedspt[curbin].append(sptun[idx])
        
        netmsig = [np.sum(a) for a in binnedmsig]
        netmsig = np.array(netmsig)
        try:
            startidx = np.where(netmsig>0)[0][0] #if all motion signal is zero, the decoded motion etc also zero.
            mspt = sptunbins[np.where(netmsig==np.max(netmsig))[0][0]] #speed tuning of maximum motion signal (°/sec)
            dmot = (mspt*td) #decoded motion signal
            #decerr = np.abs(shiftmag-dmot) #decoding error - you can calculate this in the very end. 
            #popt,pcov = curve_fit(hlp.gauss,sptunbins[startidx:][netmsig[startidx:]!=0],netmsig[startidx:][netmsig[startidx:]!=0])
            #gfit = hlp.gauss(sptunbins[startidx:], *popt)
        except:
            dmot = 0
        #fill in the values for real and decoded motion signals    
        dsh.append(dmot)            
        rsh.append(sm)
        ss.append((startang, endang))

        if end == int(frametot/4) or end == int(frametot/2):
            fig, ax = plt.subplots()
            ax.plot(sptunbins[startidx:], netmsig[startidx:])
            #ax.plot(sptunbins[startidx:],gfit)
            ax.plot([mspt,mspt],[0,np.max(netmsig)], 'r-')
            ax.set_ylabel('Net motion signal [a.u]')
            ax.set_xlabel(r'Speed tuning [$\frac{°}{s}$]')
            ax.set_title('Stimulus shift of %.1f degrees over %.2f seconds' %(rsh[-1], td))
            ax.text(x=0.65, y=0.8, s='Decoded motion : %.2f°' %(mspt*td), transform=ax.transAxes, fontsize=25)
    
dsh = np.array(dsh)
rsh = np.array(rsh)
derr = dsh-rsh

fig, axs = plt.subplots(2,sharex=True)
axs[0].plot(rsh,dsh, 'k.')
axs[0].plot([0,340], [0,340], 'r-')
axs[0].set_xlim(-1,350)
axs[1].plot(rsh,derr, 'k.')
axs[1].plot([0,340],[0,0], 'k-')
axs[1].set_xlabel('Real shift [°]')
axs[0].set_ylabel('Decoded shift shift [°]')
axs[1].set_ylabel('Decoding error [°]')

