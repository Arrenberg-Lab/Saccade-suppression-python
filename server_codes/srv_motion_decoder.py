# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 08:07:01 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
#from pathlib import Path
import sys
#sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
import os 
import tarfile
from io import BytesIO

#Decode the stimulus motion for the given settings (server)

#initial Reichardt like detector implementation
#General parameters
#From generate RFs
seed = int(sys.argv[1]) #666
animal = sys.argv[2] #'zebrafish' #sys.argv[3] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = int(sys.argv[3]) #50 #sys.argv[4] #50 #sys argv
rsl = int(sys.argv[4]) #5 #sys.argv[5]
jsigma = int(sys.argv[5]) #4

#From simulation
shiftmag = float(sys.argv[6]) #340 #sys.argv[2] #340
startloc = int(sys.argv[7]) #-170
bwidth = int(sys.argv[8]) #5

if animal == 'macaque':
    #see simulation function for description
    shiftmag -= 180
    startloc += 90
    nfilters = int(np.ceil(np.sqrt(nfilters))**2)
    
fps = 200
test = True #if False, random dot stimulus is used, else a bar sweeps through. False is not yet implemented. 
#ensure shift is one pixel per frame -> frame per frame shift is determined by rsl
tdur = shiftmag/200*rsl*1000/ 10 #200 is fps (default value), 1000 is to convert the duration to ms. This gives a duration
                                #which allows 1 pixel shift per frame for the given settings. Divide by 10 for time reasons

frametot = int(tdur/1000*fps) #total number of frames
shperfr = shiftmag/frametot #the amount of shift per frame in degrees

if test == True:
    stm = 'test'
else:
    stm = 'randomdot'

checkmotsig = False #if True, motion signals are plotted

#load the tar file
sdir = r'simulation/stimuli/%s/tdur_%05i_shiftmag_%3i_rsl_%02i_startloc_%i_barwidth_%i'%(stm, tdur, shiftmag, 
                                                                                         rsl, startloc, bwidth)

#save directory for the given RF set
fname = sdir + '/activities/%s/seed_%i_nfilters_%06i_jsigma_%i'%(animal, seed, nfilters, jsigma) 

tf = tarfile.open(fname+'.tar')

names = tf.getnames()

"""
#this is the main idea in considering all possible shifts in the simulation, go frame by frame, consider from biggest
#shift to the smallest shift all possible combinations of activity arrays, along which you also calculate the maximum of
#the net motion signal, decoded shift magnitde, shift error etc.
for i in range(frametot):
    for j in range(frametot,i,-1):
        print(i,j)
"""
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
        
        motionsig = np.zeros([len(names), nf+1]) #pairwise unit correlation array, first dim unit pair
                                               #second dimension frame difference
        sptuning = np.zeros(motionsig.shape) #speed tuning for the given array. This is azimuth angle difference / shift

        #motion decoding for each shift
        for i,n in enumerate(names):
            #load the activity file
            array_file = BytesIO()
            array_file.write(tf.extractfile(n).read())
            array_file.seek(0)
            rfdat = np.load(array_file, allow_pickle=True)
            rfacts=rfdat['rfacts'][start:end]
            rfacts2=rfdat['rfacts2'][start:end] 
            
            rfcentaz = rfdat['rfcent'][0]/rsl - 180
            rfcentaz2 = rfdat['rfcent2'][0]/rsl - 180
            if rfcentaz < 0:
                rfcentaz += 360
            if rfcentaz2 < 0:
                rfcentaz2 += 360    
        
            xdif = rfcentaz2 - rfcentaz #azimuth angle difference between unit centers (2nd unit - 1st unit)
            if xdif < 0:
                xdif = 360 + xdif
            #print(xdif)
            
            pshs = []
            nshs = []
            for tdx in range(nf+1):       
                #print(tdx) #in frames
                if tdx == 0:
                    pshift = np.sum(rfacts2*rfacts)
                    nshift = np.sum(rfacts2*rfacts)
                    sptuning[0, tdx] = 0
            
                else:
                    #second cell activity is shifted ahead in time: max correlation if second cell shows similar excitement
                    #later. First frame of first cell correlated with a later frame of second cell.
                    pshift = np.sum(rfacts[:-tdx]*rfacts2[tdx:])
                    #first cell activity shifted ahead in time 
                    nshift = np.sum(rfacts[tdx:]*rfacts2[:-tdx])
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
        
        if checkmotsig == True:
            #check the motion signal
            fig, ax = plt.subplots()
            ax.plot(sptunbins[startidx:][netmsig[startidx:]!=0], netmsig[startidx:][netmsig[startidx:]!=0], 'k.-')
            #ax.plot(sptunbins[startidx:],gfit)
            ax.plot([mspt,mspt],[0,np.max(netmsig)], 'r-')
            ax.set_ylabel('Net motion signal [a.u]')
            ax.set_xlabel(r'Speed tuning [$\frac{°}{s}$]')
            ax.set_title('Stimulus shift of %.1f degrees over %3.f ms' %(sm, td*1000))
            ax.text(x=0.65, y=0.8, s='Decoded motion : %.2f°' %(dmot), transform=ax.transAxes, fontsize=25)
            plt.get_current_fig_manager().window.showMaximized()
            while True:
                if plt.waitforbuttonpress():
                    plt.close('all')
                    break

dsh = np.array(dsh)
rsh = np.array(rsh)
derr = dsh-rsh

#saving part
svdir = r'simulation/analysis/%s/%s_stim_tdur_%05i_shiftmag_%3i_rsl_%02i_startloc_%i_barwidth_%i'%(animal, stm, tdur, shiftmag, 
                                                                                         rsl, startloc, bwidth) #save to an analysis file
#save the arrays
try:
    os.makedirs(svdir)
except:
    pass
np.savez(svdir+ '/seed_%i_nfilters_%06i_jsigma_%i.npz'%(seed, nfilters, jsigma), 
         decsh=dsh, realsh=rsh, decerr=derr,stimpos=ss)


"""
#Plot style: General figure parameters:
figdict = {'axes.titlesize' : 30,
           'axes.labelsize' : 25,
           'xtick.labelsize' : 25,
           'ytick.labelsize' : 25,
           'legend.fontsize' : 25,
           'figure.titlesize' : 30,
           'image.cmap' : 'gray'}
plt.style.use(figdict)

#check how model performs for all stimulus shifts
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(rsh, np.array(dsh), 'r.')
axs[0].plot(axs[0].get_xlim(), axs[0].get_xlim(), 'k-')

axs[1].plot(rsh, derr, 'k.')

axs[0].set_xlim([0,np.max(rsh)+np.max(rsh)/50])
axs[0].set_ylim([0,np.max(dsh)+np.max(dsh)/50])
axs[0].set_ylabel('Decoded shift [°]')
axs[1].set_ylabel('Decoding error [°]')
axs[1].set_xlabel('Real shift [°]')
fig.suptitle('Example simulation for %s with %i RFs'%(animal, nfilters))
"""

"""
#Former plot to check if the motion decoder worked well for the maximum amount of shift         
fig, ax = plt.subplots()
ax.plot(sptunbins[startidx:][netmsig[startidx:]!=0], netmsig[startidx:][netmsig[startidx:]!=0], 'k.-')
#ax.plot(sptunbins[startidx:],gfit)
ax.plot([mspt,mspt],[0,np.max(netmsig)], 'r-')
ax.set_ylabel('Net motion signal [a.u]')
ax.set_xlabel(r'Speed tuning [$\frac{°}{s}$]')
ax.set_title('Stimulus shift of %.1f degrees over %1.f ms' %(shiftmag, tdur))
ax.text(x=0.65, y=0.8, s='Decoded motion : %.2f°' %(mspt*tdur/1000), transform=ax.transAxes, fontsize=25)
"""