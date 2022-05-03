# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:04:40 2022

@author: Ibrahim Alperen Tunc
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs_ as hlp
import tarfile
import os
from IPython import embed
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Test 1-D stimulation and model decoding

#1st test 1-D stimulus: Test stimulus works as intended.
animal = 'zebrafish' #sys.argv[2] #'zebrafish' #sys.argv[3] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = 200 #int(sys.argv[3]) #sys.argv[4] #50 #sys argv
rsl = 200 #int(sys.argv[4]) #sys.argv[5]
#Required here
shiftmag = 340 #float(sys.argv[6]) #340 #sys.argv[2] #340
startloc = -170 #float(sys.argv[7]) #-170 #start location of the test stimulus in degrees
bwidth = 5 #int(sys.argv[8]) #bar width of the test stimulus in degrees
seed = 666
np.random.seed(seed)


#Fixed parameters
test = True #if False, random dot stimulus is used, else a bar sweeps through. False is not yet implemented. 
#ensure shift is one pixel per frame -> frame per frame shift is determined by rsl
pxsh = 100
fps = 200

tdur = shiftmag*rsl / (pxsh * fps) #stimulus duration in seconds.

frametot = int(tdur*fps)
stop = False
check = False
niter = 10
dvel = [] #decoded velocities

for n in range(niter):
    print(n)
    #Prepare the RFs -> first with phase shifted pairs
    speciesparams = hlp.generate_species_parameters(nfilters)
    params = speciesparams.zebrafish_updated()
    #DO the same randomization as you updated in 2D -> sample all Gaussian xyz, then get azimuth elevation and just use azimuth 
    xyz = np.random.multivariate_normal([0]*3, np.identity(3), nfilters).T
    __, azvals, __ = hlp.coordinate_transformations.car2geo(xyz[0], xyz[1], xyz[2])
    #convert azimuth values to index
    azidx = ((180+azvals) * rsl).astype(int)
    rfs = np.zeros([nfilters, 360*rsl])
    rfs2 = np.zeros(rfs.shape) #phase shifted RFs
    covarr = np.zeros(360*rsl)
    fig, ax = plt.subplots(3)  
    for i, par in enumerate(params):
        azarr, rfarr = hlp.gaussian_rf_geog_1D(par[0], par[1], azidx[i], rsl)
        __, rfarr2 = hlp.gaussian_rf_geog_1D(par[0], par[1], azidx[i]+int(par[1]/2*rsl), rsl)
        rfs[i, :] = rfarr
        rfs2[i, :] = rfarr2
        covarr[rfarr!=0] += 1
        covarr[rfarr2!=0] += 1
        ax[0].plot(azarr-180, rfarr, 'k-')
        ax[0].plot(azarr-180, rfarr2, 'r-')
    
    ax[1].plot(azarr-180, covarr)
    
    ax[1].set_xlabel('Azimuth [°]')
    ax[0].set_ylabel('Unit activity [a.u.]')
    ax[1].set_ylabel('RF coverage')
    fig.suptitle('1D receptive field and coverage map for zebrafish (n=%i)'%nfilters)
    
    rfacts = np.zeros([nfilters, frametot+1])
    rfacts2 = np.zeros([nfilters, frametot+1])
    
    #simulation with RFs (phase shifted pairs)
    for i in range(frametot+1):
        if stop == True:
            break
        a, s = hlp.test_bar_stimulus_1D(rsl, shiftmag, tdur, bwidth, startloc, fps, i)
        rfacts[:,i] = np.sum(s*rfs, axis=1)
        rfacts2[:,i] = np.sum(s*rfs2, axis=1)
        if check == True:
            fig, ax = plt.subplots()
            ax.plot(a,s, 'k-')
            while True:
                if plt.waitforbuttonpress():
                    plt.close('all')
                    break
            inp = input('c to continue, s to stop \n')
            if inp == 's':
                stop = True
    
    
    #coincidence detector
    start = 0
    end = frametot
    shperfr = shiftmag/frametot #shift magnitude in a given frame
    print('start %i end %i \n'%(start,end))
    nf = end-start #number of frames used to find out shift magnitude and duration
    sm = shperfr * nf #the magnitude of the current stimulus shift
    td = nf / fps #duration of the current signal
    shps = shiftmag/tdur 
         
    #start with considering the decoded motion error in all shifts
    rsh = [] #real shift
    dsh = [] #decoded shift
    ss = [] #start and stop positions (degrees) for the considered shift, maybe for future analysis to see if stimulus position
            #shows a bias in motion estimate (highly likely for single RF set, highly unlikely for multiple sets).
        
    startang = startloc + (start*shperfr) #starting position of the bar (azimuth angles)
    endang = startloc + (end*shperfr) #starting position of the bar (azimuth angles)
            
    rfcentaz = azvals
    rfcentaz2 = azvals + np.array(params)[:,1]/2
    
    #loop around azimuth values outside the bounds
    for x in (rfcentaz,rfcentaz2):
        x[x<-180] += 360
        x[x>180] -= 360
    
    rfpars = np.zeros([*rfacts.shape,2])
    rfpars[:,:,0] = rfacts
    rfpars[:,:,1] = rfacts2
    del(rfacts,rfacts2)
    
    xdifs = rfcentaz2 - rfcentaz
    
    swidxs = np.where(xdifs<0)[0]
    for i in swidxs:
        rfpars[i,:,:] = np.flip(rfpars[i,:,:], axis=1)
    
    xdifs[xdifs<0] += 360            
                

    msig = np.zeros([nf, nfilters])        
    for tdx in range(nf):       
        #print(tdx) #in frames
        if tdx == 0:
            pshift = np.sum(rfpars[:,:,1]*rfpars[:,:,0], axis=1)
            nshift = np.sum(rfpars[:,:,0]*rfpars[:,:,1], axis=1)
            mosig = np.zeros(nfilters)
            
        else:
            #second cell activity is shifted ahead in time: max correlation if second cell shows similar excitement
            #later. First frame of first cell correlated with a later frame of second cell.
            pshift = np.sum(rfpars[:,:-tdx, 0]*rfpars[:, tdx:,1], axis=1)
            #first cell activity shifted ahead in time 
            nshift = np.sum(rfpars[:, tdx:, 0]*rfpars[:, :-tdx, 1], axis=1)
            mosig = pshift - nshift
        msig[tdx, :] = mosig
    
    
    
    maxfs = np.argmax(msig, axis=0) #frames maximizing the motion signal for each cell pair       
    sptuns = xdifs / (maxfs/fps)
    
    sptuns[sptuns==np.inf] = 0 #replace divide by zero cases with no motion -> model sees nothing

    #find the population readout -> maximum count of the speed tuning distribution
    rng = int(np.max(sptuns-shps)) - int(np.min(sptuns-shps))
    counts, bins = np.histogram(sptuns-shps, 
                    bins=np.linspace(int(np.min(sptuns-shps)), int(np.max(sptuns-shps)), int(rng*np.ceil(10000/rng))+1))
    intrp = interp1d(bins[1:], counts, kind=3)#quadratic interpolation
    bur = 10
    if bur > np.max(bins[1:]):
        bur = np.floor(np.max(bins[1:]))
    bns = np.linspace(np.ceil(np.min(bins[1:])), bur, 100001)
    intcnts = intrp(bns)
    dcsp = bns[np.argmax(intcnts)] #decoded speed estimate error
    #fill in the values for real and decoded motion signals    
    dsh.append(dcsp*td)            
    rsh.append(sm)
    ss.append((startang, endang))
    
         
    dsh = np.array(dsh)
    rsh = np.array(rsh)
    derr = dsh-rsh
    
    

    ax[2].plot(bns, intcnts, 'k.-')
    #ax.plot(sptunbins[startidx:],gfit)
    ymax = ax[2].get_ylim()[1]
    ax[2].plot([dcsp,dcsp],[0,ymax], 'r-')
    ax[2].plot([0,0],[0,ymax], 'b-')
    ax[2].set_ylabel('Net motion signal [a.u]')
    ax[2].set_xlabel(r'Speed tuning [$\frac{°}{s}$]')
    ax[2].text(x=0.65, y=0.8, s='Velocity error : %.2f°/s' %dcsp, transform=ax[2].transAxes, fontsize=25)
    fig.tight_layout()
    plt.pause(0.05)

    dvel.append(dcsp)
    
fig, ax = plt.subplots()
ax.hist(dvel)
ax.set_xlabel('Velocity error [°/s]')
ax.set_title('Velocity error distribution for example simulations')

