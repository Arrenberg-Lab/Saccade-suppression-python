# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:29:55 2022

@author: Ibrahim Alperen Tunc
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs_ as hlp
import tarfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
from IPython import embed
from scipy.interpolate import interp1d

#General parameters -> first argument in sys argv is rsl, second is shift magnitude
#From generate_RFs
seed = 666 #int(sys.argv[1]) #sys.argv[1]
animal = 'zebrafish' #sys.argv[2] #'zebrafish' #sys.argv[3] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = 500 #int(sys.argv[3]) #sys.argv[4] #50 #sys argv
rsl = 5 #int(sys.argv[4]) #sys.argv[5]
#Required here
shiftmag = 340 #float(sys.argv[6]) #340 #sys.argv[2] #340
startloc = -170 #float(sys.argv[7]) #-170 #start location of the test stimulus in degrees
bwidth = 5 #int(sys.argv[8]) #bar width of the test stimulus in degrees

#Fixed parameters
test = True #if False, random dot stimulus is used, else a bar sweeps through. False is not yet implemented. 
#ensure shift is one pixel per frame -> frame per frame shift is determined by rsl
pxsh = 2
fps = 200

tdur = shiftmag/fps*rsl*1000 / pxsh #200 is fps (default value), 1000 is to convert the duration to ms. This gives a duration
                                    #which allows 1 pixel shift per frame for the given settings.
                                    #For time considerations, divide by 10 (i.e. 10 pixel shift per frame = 2 degrees)
shiftdir = 'right'


if test == True:
    stm = 'test'
else:
    stm == 'randomdot'

#save directory for all files
sdir = r'simulation/stimuli/%s/tdur_%05i_shiftmag_%3i_rsl_%02i_startloc_%i_barwidth_%i'%(stm, tdur, shiftmag, 
                                                                                         rsl, startloc, bwidth)
"""
#save directory for the given RF set
fname = sdir + '/activities/%s/seed_%i_nfilters_%06i'%(animal, seed, nfilters) 

try: 
    os.makedirs(fname)
except:
    pass
"""
#stimulus
stim = hlp.cylindrical_random_pix_stimulus(rsl, shiftmag, tdur, maxelev=70)
stim.test_stimulus(bwidth, startloc)

#save dir for activity arrays:
fname = r'simulation/testacts'

try:
    os.makedirs(fname)
except:
    pass

#load the RFs tar file
fn = r'simulation/RFs/zebrafish/seed_666_nfilters_000500_rsl_05'
rfns = os.listdir(fn)

#run simulations for all units
for i, n in enumerate(rfns):    
    print('Current RF number=%i, %.2f%% complete'%(i, i/(len(rfns))*100))

    #preallocate the activity array for the given RF and given stimulus frame
    rfacts = np.zeros(stim.frametot+1)
    rfacts2 = np.zeros(stim.frametot+1)
    rfactsnormed = np.zeros(stim.frametot+1)
    rfactsnormed2 = np.zeros(stim.frametot+1)
    
    #load the RF parameters
    rfdat = np.load(os.path.join(fn, n))
    rfarr = rfdat['rfarr']
    rfarr2 = rfdat['rfarr2']
    rfdiam = rfdat['rfpars'][1]
    rfcent = rfdat['rfcent'] 
    rfcent2 = rfdat['rfcent2'] 
    for j in range(stim.frametot+1):    
        stimulus, __ , __= stim.move_stimulus(j, shiftdir) #shift in positive
       
        #Get the RF activity
        if animal == 'macaque':
            stimulus = stimulus[:,90*rsl:(360-90)*rsl] #very dirty trick: clip stimulus to macaque visual field
        rfacts[j] = np.sum(rfarr[rfarr!=0]*stimulus[rfarr!=0])
        rfacts2[j] = np.sum(rfarr2[rfarr2!=0]*stimulus[rfarr2!=0])
        rfactsnormed[j] = rfacts[j] / (np.pi*(rfdiam/2)**2) #normed to unit area
        rfactsnormed2[j] = rfacts2[j] / (np.pi*(rfdiam/2)**2) #normed to unit area
    
    #save the arrays
    np.savez(fname+'/'+n[-10:-4]+'_pixshiftperfr_%i'%(pxsh) + '.npz', rfacts=rfacts, rfacts2=rfacts2, 
                                                            rfactsnormed=rfactsnormed, rfactsnormed2=rfactsnormed2,
                                                            rfcent=rfcent, rfcent2=rfcent2)


#from here on do 5 different simulations
niter = 10 #number of trials
nunits = 100 #number of units for each trial

checkmotsig = False
stop = False
nopair = False
npairs = 500 #number of pairs only used when nopair is true


names = os.listdir(fname)
names = [n for n in names if '.npz' in n]
dvel = [] #decoded velocities
for n in range(niter):
    print(n)
    fig, axs = plt.subplots(2)
    #choose 100 units
    allidxs = np.arange(0, len(names))
    unitidxs = np.random.choice(allidxs, nunits, replace=False)
    
    #!!!decoding -> no pair case is still to be developed
    frametot = stim.frametot #int(tdur/1000*fps) #total number of frames
    shperfr = stim.shiftperfr #the amount of shift per frame in pixels
    shps = shiftmag / (tdur/1000) #shift per second i.e. shift speed -> tdur was in ms, so converted to s
    
    start = 0
    end = frametot
    
    print('start %i end %i \n'%(start,end))
    nf = end-start #number of frames used to find out shift magnitude and duration
    sm = shperfr * nf #the magnitude of the current stimulus shift
    td = nf / fps #duration of the current signal
        
    
    if nopair == False:
        pass

    else:
        pairs = []
        for i in range(npairs):
            pairidxs = np.random.choice(unitidxs, 2, replace=False)
            rfc1 = np.load(os.path.join(fname, names[pairidxs[0]]))['rfcent'][0]/rsl - 180 #azimuth center of 1st receptive field
            rfc2 = np.load(os.path.join(fname, names[pairidxs[1]]))['rfcent'][0]/rsl - 180 #azimuth center of 2nd receptive field
            if rfc1 > rfc2:
                pairs.append(np.flip(pairidxs))
            else:
                pairs.append(pairidxs)
    
    #check the coverage map
    covmap = np.zeros(np.array([180,360])*rsl) #RF coverage map
    if nopair == False:
        for i, udx in enumerate(unitidxs):
            print(i)
            rfdat = np.load(os.path.join(fn, rfns[udx]))
            rfarr = rfdat['rfarr']
            rfarr2 = rfdat['rfarr2']
            covmap[rfarr!=0] += 1
            covmap[rfarr2!=0] += 1
        covmap /= 2*nunits
    
    else:
        pairs = np.array(pairs)
        usedunits = np.unique(pairs)
        for i, udx in enumerate(usedunits):
            print(i)
            rfdat = np.load(os.path.join(fn, rfns[udx]))
            rfarr = rfdat['rfarr']
            covmap[rfarr!=0] += 1
        covmap /= nunits
    
    axs[0].imshow(covmap, cmap='jet', origin='lower', extent=[-180,180,-90,90])

    if test == True:
        stm = 'test'
    else:
        stm = 'randomdot'
    

    #start with considering the decoded motion error in all shifts
    sptun = np.zeros(len(unitidxs)) #speed tuning of each pair
    maxfs = []
    startang = startloc + (start*shperfr) #starting position of the bar (azimuth angles)
    endang = startloc + (end*shperfr) 
    
    if nopair == False:
        for i, udx in enumerate(unitidxs):
            #load the activity file
            rfdat1 = np.load(os.path.join(fname, names[udx]))
            rfacts=rfdat1['rfacts'][start:end]
            rfacts2=rfdat1['rfacts2'][start:end] 
        
            rfcentaz = rfdat1['rfcent'][0]/rsl - 180
            rfcentaz2 = rfdat1['rfcent2'][0]/rsl - 180
                        
            
            
            xdif = rfcentaz2 - rfcentaz #azimuth angle difference between unit centers (2nd unit - 1st unit)
            if xdif < 0:
                xdif = 360 + xdif
            #print(xdif, rfcentaz2, rfcentaz)
            
            msig = [] #motion signal
            for tdx in range(nf):       
                
                if tdx == 0:
                    pshift = np.sum(rfacts2*rfacts)
                    nshift = np.sum(rfacts2*rfacts)
                    mosig = 0
                
                else:
                    #second cell activity is shifted ahead in time: max correlation if second cell shows similar excitement
                    #later. First frame of first cell correlated with a later frame of second cell.
                    pshift = np.sum(rfacts[:-tdx]*rfacts2[tdx:])
                    #first cell activity shifted ahead in time 
                    nshift = np.sum(rfacts[tdx:]*rfacts2[:-tdx])
                    mosig = pshift - nshift
                
                msig.append(mosig)
                
            #find the frame at motion signal maximum for the given phase shifted pair.
            maxf = np.argmax(msig) #frame shift maximizing motion signal
            maxfs.append(maxf)
            if maxf == 0:
                sptun[i] = 0 #if no motion signal is detected, model assumes no motion happened
            else:    
                sptun[i] = xdif / (maxf/fps)
        
    else:
        #nopair case 
        for i, (n1, n2) in enumerate(pairs):
            #load the activity file
            rfdat1 = np.load(os.path.join(fname, names[n1]))
            rfdat2 = np.load(os.path.join(fname, names[n2]))

            rfacts=rfdat1['rfacts'][start:end]
            rfacts2=rfdat2['rfacts'][start:end] 
        
            rfcentaz = rfdat1['rfcent'][0]/rsl - 180
            rfcentaz2 = rfdat2['rfcent'][0]/rsl - 180
                        
                            
            xdif = rfcentaz2 - rfcentaz #azimuth angle difference between unit centers (2nd unit - 1st unit)
            print(rfcentaz2, rfcentaz, xdif)
            if xdif < 0:
                xdif = 360 + xdif
            #print(xdif)
            
            pshs = []
            nshs = []
            
            for tdx in range(nf):       
                #print(tdx) #in frames
                if tdx == 0:
                    pshift = np.sum(rfacts2*rfacts)
                    nshift = np.sum(rfacts2*rfacts)
                    mosig = 0
                    
                else:
                    #second cell activity is shifted ahead in time: max correlation if second cell shows similar excitement
                    #later. First frame of first cell correlated with a later frame of second cell.
                    pshift = np.sum(rfacts[:-tdx]*rfacts2[tdx:])
                    #first cell activity shifted ahead in time 
                    nshift = np.sum(rfacts[tdx:]*rfacts2[:-tdx])    
                    mosig = pshift - nshift
    
    #find the population readout -> maximum count of the speed tuning distribution
    rng = int(np.max(sptun-shps)) - int(np.min(sptun-shps))
    counts, bins = np.histogram(sptun-shps, 
                    bins=np.linspace(int(np.min(sptun-shps)), int(np.max(sptun-shps)), int(rng*np.ceil(10000/rng))+1))
    intrp = interp1d(bins[1:], counts, kind=3)#quadratic interpolation
    bns = np.linspace(-10,10, 100001)
    intcnts = intrp(bns)
    dcsp = bns[np.argmax(intcnts)] #decoded speed estimate error
    dsh = (dcsp+shps) * tdur/1000 #decoded shift for the given trial
    rsh = shiftmag
    dvel.append(dcsp)
    
    """
    #check how model performs for all stimulus shifts
    fig2, axss = plt.subplots(2, sharex=True)
    axss[0].plot(rsh, np.array(dsh), 'r.')
    axss[0].plot(axs[0].get_xlim(), axs[0].get_xlim(), 'k-')
    
    axss[1].plot(rsh, derr, 'k.')
    
    axss[0].set_xlim([0,np.max(rsh)+np.max(rsh)/50])
    axss[0].set_ylim([0,np.max(dsh)+np.max(dsh)/50])
    axss[0].set_ylabel('Decoded shift [°]')
    axss[1].set_ylabel('Decoding error [°]')
    axss[1].set_xlabel('Real shift [°]')
    fig2.suptitle('Example simulation for %s with %i RFs'%(animal, nfilters))
    """

    #Former plot to check if the motion decoder worked well for the maximum amount of shift         
    axs[1].plot(bns, intcnts, 'k.-')
    #ax.plot(sptunbins[startidx:],gfit)
    ymax = axs[1].get_ylim()[1]
    axs[1].plot([dcsp,dcsp],[0,ymax], 'r-')
    axs[1].plot([0,0],[0,ymax], 'b-')
    axs[1].set_ylabel('Net motion signal [a.u]')
    axs[1].set_xlabel(r'Speed tuning error [$\frac{°}{s}$]')
    axs[1].text(x=0.65, y=0.8, s='Velocity error : %.2f°/s' %dcsp, transform=axs[1].transAxes, fontsize=25)


fig, ax = plt.subplots()
ax.hist(np.array(dvel), bins=10)
if nopair == False:
    ax.set_title('Velocity estimation error for 10 trials, shift per frame = %.2f degrees'%(pxsh/rsl))
else:
    ax.set_title('Velocity estimation error for 10 trials, shift per frame = %.2f degrees (random pairs)'%(pxsh/rsl))
ax.set_xlabel('Velocity error [°/s]')

"""
#No pair approach will be implemented soon...
#try the 1000 pairs approach
idxs = np.arange(len(names))
pairs = []
for i in range(1000):
    pairs.append(np.random.choice(names, 2, replace=False))

else:
    motionsig = np.zeros([len(pairs), nf]) #pairwise unit correlation array, first dim unit pair
                                           #second dimension frame difference
    sptuning = np.zeros(motionsig.shape) #speed tuning for the given array. This is azimuth angle difference / shift
    for i, (n1, n2) in enumerate(pairs):
        None
        #load the activity file
        rfdat1 = np.load(os.path.join(fname, n1))
        rfdat2 = np.load(os.path.join(fname, n2))

        rfacts=rfdat1['rfacts'][start:end]
        rfacts2=rfdat2['rfacts'][start:end] 
        
        rfcentaz = rfdat1['rfcent'][0]/rsl - 180
        rfcentaz2 = rfdat2['rfcent'][0]/rsl - 180
        
        if rfcentaz < 0:
            rfcentaz += 360
        if rfcentaz2 < 0:
            rfcentaz2 += 360
        
        #if second RF has smaller azimuth, convert the motion signal sign
        if rfcentaz > rfcentaz2:
            msfac = -1
        else:
            msfac = 1        
"""
