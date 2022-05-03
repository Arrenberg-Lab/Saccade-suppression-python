# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:39:28 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import pandas as pd
import matplotlib.ticker as ticker
loadpath = r'..\data\simulations'

#Species simulation violinplots (macaque and zebrafish)

#percentile to plot datapoints as outliers
pcutoff = [1, 99] #%1 percent and 99% as cutoff
#load the datafiles
df = r'\zf_mac_simul_nsimul=100_rsl_mac=24_rsl_zf=5_rdot=5_jsigma=4_nfilters=[128, 1058, 10082]_centoffs=[ 0.  7. 50.]'
simul = pd.read_csv(loadpath+df)
#convert the azimuth and elevation so that center is 0 and elevation decreases from top to bottom
simul['Real center X'] -= 180
simul['Decoded center X'] -= 180
simul['Real center Y'] = -simul['Real center Y']+90
simul['Decoded center Y'] = -simul['Decoded center Y']+90

zebrafish = simul[simul['Species']=='Zebrafish']
macaque = simul[simul['Species']=='Macaque']

#As always first the position decoding accuracy for different stimulus positions:
#position decoding error for zebrafish
zflocdecerr = np.array(np.sqrt((zebrafish['Real center X'] - zebrafish['Decoded center X'])**2 + \
                               (zebrafish['Real center Y'] - zebrafish['Decoded center Y'])**2))
maclocdecerr = np.array(np.sqrt((macaque['Real center X'] - macaque['Decoded center X'])**2 + \
                                (macaque['Real center Y'] - macaque['Decoded center Y'])**2))

#separate by filter numbers: first 300 100 filters next 300 1000 filters and last 300 10k filters.
#separate by stimulus position: first 100 middle of the screen (180-90) next 7° shifted (avg) next 50° shifted (avg)
#within each set differing in filter number.
nstimuli = 3 #number of stimuli used for simulation
nnfilters = 3 #number of different filter numbers used for simulation
nsimul = 100 #number of simulations for each
zfsortedlocdecerr = np.zeros([nnfilters,nstimuli,nsimul]) #shape nfilters x nstimuli x nsimul
macsortedlocdecerr = np.zeros([nnfilters,nstimuli,nsimul])
for nidx in range(nnfilters):
    sizeseparatedzf = zflocdecerr[nidx * nsimul*nstimuli : nidx * nsimul*nstimuli + nsimul*nstimuli]
    sizeseparatedmac = maclocdecerr[nidx * nsimul*nstimuli : nidx * nsimul*nstimuli + nsimul*nstimuli]    
    for cidx in range(nstimuli):
        zfsortedlocdecerr[nidx, cidx, :] = sizeseparatedzf[cidx*nsimul : cidx*nsimul + nsimul]
        macsortedlocdecerr[nidx, cidx, :] = sizeseparatedmac[cidx*nsimul : cidx*nsimul + nsimul]

#plotting        
fig, axs = plt.subplots(1, nnfilters, sharey=True)
for idx in range(nnfilters):
    zfdat = [b[~np.isnan(b)] for b in zfsortedlocdecerr[idx, :, :]]
    vp1 = axs[idx].violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
                       quantiles=[[0.01,0.99] for a in range(len(zfdat))])
    hlp.plot_outliers_to_violinplot(zfdat, pcutoff, axs[idx], positions=np.arange(len(zfdat))*2)   
    
    
    macdat = [b[~np.isnan(b)] for b in macsortedlocdecerr[idx, :, :]]
    vp2 = axs[idx].violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
                       quantiles=[[0.01,0.99] for a in range(len(macdat))])
    hlp.plot_outliers_to_violinplot(macdat, pcutoff, axs[idx], positions=np.arange(len(macdat))*2+1)   
    labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
    labeldat = [b for a in labeldat for b in a]
    axs[idx].set_xticks(np.arange(0,6,2)+0.5)
    axs[idx].set_xticklabels([0, 7, 50])
    for i in range(len(zfdat)):
        axs[idx].text(0+i*2,-10, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
        axs[idx].text(1+i*2,-10, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')
    axs[idx].set_title('nfilters = %i'%(np.unique(zebrafish['Number of filters'])[idx]))
fig.suptitle('Stimulus position decoding error [$\circ$]')
axs[0].set_ylabel('Decoding error [$\circ$]')
axs[1].set_xlabel('X-Y shift relative to visual field center [$\circ$]', labelpad=20)
axs[2].legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.057, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)

#shift decoding error
#you need to compare the simulations with the same ID (i.e. same simulation number and same number of filters)
#sort the real and decoded xy positions first by filter number, then by simulation ID
"""
#You might need these values for debugging purposes (see also gigantic nested for loops monstrosity)
realxyzf = np.zeros([nnfilters,nsimul,nstimuli,2]) #shape nfilters x nsimul x nstimuli x xy 
decxyzf = np.zeros([nnfilters,nsimul,nstimuli,2])
realxymac = np.zeros([nnfilters,nsimul,nstimuli,2])
decxymac = np.zeros([nnfilters,nsimul,nstimuli,2])
decxymac = np.zeros([nnfilters,nsimul,nstimuli,2])
"""
shifterrzf = np.zeros([nnfilters, 3, nsimul]) #number of different filters tested x number of shifts x nsimul
shifterrmac = np.zeros([nnfilters, 3, nsimul])
decshiftszf = np.zeros([nnfilters, 3, nsimul]) #just necessary for the square plots way below, so only doing for zf.

realshifts = np.round(np.array([7,43,50])*np.sqrt(2),2)

for nidx, nf in enumerate(np.unique(zebrafish['Number of filters'])):
    
    subdfzf = zebrafish[zebrafish["Number of filters"]==nf]
    subdfmac = macaque[macaque["Number of filters"]==nf]
    
    for sidx, sID in enumerate(np.unique(zebrafish['Simulation number'])):
        realxzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Real center X'])
        realyzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Real center Y'])
        decxzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Decoded center X'])
        decyzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Decoded center Y'])
        
        realxmac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Real center X'])
        realymac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Real center Y'])
        decxmac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Decoded center X'])
        decymac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Decoded center Y'])
        
        """
        #You might need this for debug purposes
        realxyzf[nidx, sidx, :, 0] = realxzf
        realxyzf[nidx, sidx, :, 1] = realyzf
        decxyzf[nidx, sidx, :, 0] = decxzf
        decxyzf[nidx, sidx, :, 1] = decyzf
        
        
        realxymac[nidx, sidx, :, 0] = np.array(realxmac)
        realxymac[nidx, sidx, :, 1] = np.array(realymac)
        decxymac[nidx, sidx, :, 0] = np.array(decxmac)
        decxymac[nidx, sidx, :, 1] = np.array(decymac)
        """
        
        for idx in range(len(realxzf)):
            r = len(realxzf) #to use modulo in a shorter way. Modulo is used to wrap around the array in the last
                             #index to get the shift between the last and first stimulus.

            decshiftzf = np.sqrt((decxzf[idx] - decxzf[(idx+1)%r])**2 + (decyzf[idx] - decyzf[(idx+1)%r])**2)
            if np.isnan(decshiftzf) == True:
                decshiftzf = 0
            decerrzf = decshiftzf - realshifts[idx]
            shifterrzf[nidx, idx, sidx] = decerrzf
            decshiftszf[nidx, idx, sidx] = decshiftzf
            
            decshiftmac = np.sqrt((decxmac[idx] - decxmac[(idx+1)%r])**2 + (decymac[idx] - decymac[(idx+1)%r])**2)
            if np.isnan(decshiftmac) == True:
                #print(realshifts)
                decshiftmac = 0
            decerrmac = decshiftmac - realshifts[idx]
            shifterrmac[nidx, idx, sidx] = decerrmac
            #print(realshiftzf, realshiftmac) #comparable to the shifts calculated by using values 0,7,50
            
#plotting
fig, axs = plt.subplots(1, nnfilters, sharey=True)
for idx in range(nnfilters):
    zfdat = [b for b in shifterrzf[idx, :, :]]
    vp1 = axs[idx].violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
                       quantiles=[[0.01,0.99] for a in range(len(zfdat))])
    hlp.plot_outliers_to_violinplot(zfdat, pcutoff, axs[idx], positions=np.arange(len(zfdat))*2)   
    
    
    macdat = [b for b in shifterrmac[idx, :, :]]
    vp2 = axs[idx].violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
                       quantiles=[[0.01,0.99] for a in range(len(macdat))])
    hlp.plot_outliers_to_violinplot(macdat, pcutoff, axs[idx], positions=np.arange(len(macdat))*2+1)   
    labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
    labeldat = [b for a in labeldat for b in a]
    axs[idx].set_xticks(np.arange(0,6,2)+0.5)
    axs[idx].set_xticklabels(realshifts)
    """
    for i in range(len(zfdat)):
        axs[idx].text(0+i*2,-70, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
        axs[idx].text(1+i*2,-70, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')
    """
    axs[idx].set_title('nfilters = %i'%(np.unique(zebrafish['Number of filters'])[idx]))
fig.suptitle('Stimulus shift decoding error [$\circ$]')
axs[0].set_ylabel('Decoding error [$\circ$]')
axs[1].set_xlabel('Stimulus shift magnitude [$\circ$]')#, labelpad=20)
axs[2].legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.067, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)

#Square plots:
fig, axs = plt.subplots(1,2, sharex =True, sharey=True)
for idx in range(decshiftszf.shape[1]):
    errdata1 = decshiftszf[1,idx,:]
    errdata2 = decshiftszf[2,idx,:]
    axs[0].plot(np.repeat(realshifts[idx], len(errdata1)), errdata1, 'r.')
    axs[1].plot(np.repeat(realshifts[idx], len(errdata2)), errdata2, 'r.')
axs[0].set_xlim(axs[0].get_ylim())
axs[1].set_xlim(axs[0].get_ylim())
axs[1].set_ylim(axs[0].get_ylim())
axs[0].plot(axs[0].get_xlim(), axs[0].get_ylim(), 'k-')
axs[1].plot(axs[0].get_xlim(), axs[0].get_ylim(), 'k-')
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
axs[0].set_title('nfilters=%i'%(np.unique(zebrafish['Number of filters'])[1]))
axs[1].set_title('nfilters=%i'%(np.unique(zebrafish['Number of filters'])[2]))
axs[0].set_ylabel('Decoded shift [°]')
axs[0].set_xlabel('Real shift [°]')

#calculate error factors for macaque and zf:
abserrmac = np.abs(shifterrmac)
abserrzf = np.abs(shifterrzf)
avgerrmacpernfilters = np.mean(abserrmac[:,2,:],axis=1)
avgerrzfpernfilters = np.mean(abserrzf[:,2,:],axis=1)

#second simulation with 200 iterations and more shifts (positive and negative directions.)
df2 = r'\zf_mac_simul_nsimul=200_rsl_mac=24_rsl_zf=5_rdot=5_jsigma=4_nfilters=[128, 1058, 10082]_' + \
        'centoffs=[-60. -15.  -4.   0.   4.  15.  60.]'
simul2 = pd.read_csv(loadpath+df2)
#convert the azimuth and elevation so that center is 0 and elevation decreases from top to bottom
simul2['Real center X'] -= 180
simul2['Decoded center X'] -= 180
simul2['Real center Y'] = -simul2['Real center Y']+90
simul2['Decoded center Y'] = -simul2['Decoded center Y']+90

xylocs = [-60, -15, -4, 0, 4, 15, 60] #x,y angular positions relative to visual field center

zebrafish2 = simul2[simul2['Species']=='Zebrafish']
macaque2 = simul2[simul2['Species']=='Macaque']
#As always first the position decoding accuracy for different stimulus positions:
#position decoding error for zebrafish
zflocdecerr2 = np.array(np.sqrt((zebrafish2['Real center X'] - zebrafish2['Decoded center X'])**2 + \
                               (zebrafish2['Real center Y'] - zebrafish2['Decoded center Y'])**2))
maclocdecerr2 = np.array(np.sqrt((macaque2['Real center X'] - macaque2['Decoded center X'])**2 + \
                                (macaque2['Real center Y'] - macaque2['Decoded center Y'])**2))
nstimuli = 7 #number of stimuli used for simulation
nnfilters = 3 #number of different filter numbers used for simulation
nsimul = 200 #number of simulations for each
zfsortedlocdecerr2 = np.zeros([nnfilters,nstimuli,nsimul]) #shape nfilters x nstimuli x nsimul
macsortedlocdecerr2 = np.zeros([nnfilters,nstimuli,nsimul])
for nidx in range(nnfilters):
    sizeseparatedzf2 = zflocdecerr2[nidx * nsimul*nstimuli : nidx * nsimul*nstimuli + nsimul*nstimuli]
    sizeseparatedmac2 = maclocdecerr2[nidx * nsimul*nstimuli : nidx * nsimul*nstimuli + nsimul*nstimuli]    
    for cidx in range(nstimuli):
        zfsortedlocdecerr2[nidx, cidx, :] = sizeseparatedzf2[cidx*nsimul : cidx*nsimul + nsimul]
        macsortedlocdecerr2[nidx, cidx, :] = sizeseparatedmac2[cidx*nsimul : cidx*nsimul + nsimul]

#plotting        
fig, axs = plt.subplots(1, nnfilters, sharey=True)
for idx in range(nnfilters):
    zfdat = [b[~np.isnan(b)] for b in zfsortedlocdecerr2[idx, :, :]]
    vp1 = axs[idx].violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
                       quantiles=[[0.01,0.99] for a in range(len(zfdat))])
    hlp.plot_outliers_to_violinplot(zfdat, pcutoff, axs[idx], positions=np.arange(len(zfdat))*2)   
    
    
    macdat = [b[~np.isnan(b)] for b in macsortedlocdecerr2[idx, :, :]]
    print([len(b) for b in macdat])
    vp2 = axs[idx].violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
                       quantiles=[[0.01,0.99] for a in range(len(macdat))])
    hlp.plot_outliers_to_violinplot(macdat, pcutoff, axs[idx], positions=np.arange(len(macdat))*2+1)   
    labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
    labeldat = [b for a in labeldat for b in a]
    axs[idx].set_xticks(np.arange(0,14,2)+0.5)
    axs[idx].set_xticklabels(xylocs)
    #for i in range(len(zfdat)):
    #    axs[idx].text(0+i*2,-10, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
    #    axs[idx].text(1+i*2,-10, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')
    axs[idx].set_title('nfilters = %i'%(np.unique(zebrafish['Number of filters'])[idx]))
fig.suptitle('Stimulus position decoding error [$\circ$]')
axs[0].set_ylabel('Decoding error [$\circ$]')
axs[1].set_xlabel('Stimulus position relative to visual field center (X, -Y) [$\circ$]')
axs[2].legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.057, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)


#shift decoding error
#you need to compare the simulations with the same ID (i.e. same simulation number and same number of filters)
#sort the real and decoded xy positions first by filter number, then by simulation ID"""
shifterrzf2 = np.zeros([nnfilters, np.int((len(xylocs)-1)), nsimul]) #number of different filters tested x number of shifts x nsimul
shifterrmac2 = np.zeros([nnfilters, np.int((len(xylocs)-1)), nsimul])
realshifts = np.round(np.sqrt(2*np.array(xylocs)**2),2)
realshifts = np.delete(realshifts, np.where(realshifts==0)[0])

for nidx, nf in enumerate(np.unique(zebrafish2['Number of filters'])):
    
    subdfzf = zebrafish2[zebrafish2["Number of filters"]==nf]
    subdfmac = macaque2[macaque2["Number of filters"]==nf]
    
    for sidx, sID in enumerate(np.unique(zebrafish2['Simulation number'])):
        realxzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Real center X'])
        realyzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Real center Y'])
        decxzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Decoded center X'])
        decyzf = np.array(subdfzf[subdfzf['Simulation number']==sID]['Decoded center Y'])
        
        realxmac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Real center X'])
        realymac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Real center Y'])
        decxmac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Decoded center X'])
        decymac = np.array(subdfmac[subdfmac['Simulation number']==sID]['Decoded center Y'])
        
        """
        #You might need this for debug purposes
        realxyzf[nidx, sidx, :, 0] = realxzf
        realxyzf[nidx, sidx, :, 1] = realyzf
        decxyzf[nidx, sidx, :, 0] = decxzf
        decxyzf[nidx, sidx, :, 1] = decyzf
        
        
        realxymac[nidx, sidx, :, 0] = np.array(realxmac)
        realxymac[nidx, sidx, :, 1] = np.array(realymac)
        decxymac[nidx, sidx, :, 0] = np.array(decxmac)
        decxymac[nidx, sidx, :, 1] = np.array(decymac)
        """
        
        visfieldcntidx = np.ceil(len(realxzf)/2-1).astype(int) #index of the stimulus at the visual field center
        for idx in range(len(realxzf)-1):
            if idx >= visfieldcntidx:
                idx += 1 #skip the iteration if the stimulus position index is the one in the visual field center.
                rsidx = idx-1
            else:
                rsidx = idx #real shift index, this is used since stimulus position array 
            #print(idx, rsidx)
            decshiftzf = np.sqrt((decxzf[idx] - decxzf[visfieldcntidx])**2 + \
                                 (decyzf[idx] - decyzf[visfieldcntidx])**2)
            
            if np.isnan(decshiftzf) == True:
                decshiftzf = 0
            decerrzf = decshiftzf - realshifts[rsidx]
            shifterrzf2[nidx, rsidx, sidx] = decerrzf
            
            decshiftmac = np.sqrt((decxmac[idx] - decxmac[visfieldcntidx])**2 + \
                                  (decymac[idx] - decymac[visfieldcntidx])**2)
            
            if np.isnan(decshiftmac) == True:
                #print(realshifts)
                decshiftmac = 0
            decerrmac = decshiftmac - realshifts[rsidx]
            shifterrmac2[nidx, rsidx, sidx] = decerrmac
            #print(realshiftzf, realshiftmac) #comparable to the shifts calculated by using values 0,7,50
 
#plotting
#1) Same shift magnitudes of different directions are shifted.
fig, axs = plt.subplots(1, nnfilters, sharey=True)
for idx in range(nnfilters):

    zfdat = [b for b in shifterrzf2[idx, :, :]]
    #merge the shifts of same magnitude but different direction together
    zfdat = [list(zfdat[i])+list(zfdat[len(zfdat)-1-i]) for i in range(np.int(len(zfdat)/2))]
    zfdat = list(np.flip(zfdat, axis=0)) #flip the list so shifts go from smallest to biggest
    
    vp1 = axs[idx].violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
                       quantiles=[[0.01,0.99] for a in range(len(zfdat))])
    hlp.plot_outliers_to_violinplot(zfdat, pcutoff, axs[idx], positions=np.arange(len(zfdat))*2)   
    
    
    macdat = [b for b in shifterrmac2[idx, :, :]]
    #merge the shifts of same magnitude but different direction together
    macdat = [list(macdat[i])+list(macdat[len(macdat)-1-i]) for i in range(np.int(len(macdat)/2))]
    macdat = list(np.flip(macdat, axis=0)) #flip the list so shifts go from smallest to biggest
    
    vp2 = axs[idx].violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
                       quantiles=[[0.01,0.99] for a in range(len(macdat))])
    hlp.plot_outliers_to_violinplot(macdat, pcutoff, axs[idx], positions=np.arange(len(macdat))*2+1)   
    labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
    labeldat = [b for a in labeldat for b in a]
    axs[idx].set_xticks(np.arange(0,6,2)+0.5)
    axs[idx].set_xticklabels(np.flip(realshifts[:np.int(len(realshifts)/2)]))
    axs[idx].plot(axs[idx].get_xlim(), [0,0], 'k-', linewidth=0.8)

    """
    for i in range(len(zfdat)):
        axs[idx].text(0+i*2,-70, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
        axs[idx].text(1+i*2,-70, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')
    """
    axs[idx].set_title('nfilters = %i'%(np.unique(zebrafish['Number of filters'])[idx]))
fig.suptitle('Stimulus shift decoding error [$\circ$]')
axs[0].set_ylabel('Decoding error [$\circ$]')
axs[1].set_xlabel('Stimulus shift magnitude [$\circ$]')#, labelpad=20)
axs[2].legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.067, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)

#2) Plot each direction separately.
#plotting
fig, axs = plt.subplots(1, nnfilters, sharey=True)
for idx in range(nnfilters):
    zfdat = [b for b in shifterrzf2[idx, :, :]]
    vp1 = axs[idx].violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
                       quantiles=[[0.01,0.99] for a in range(len(zfdat))])
    hlp.plot_outliers_to_violinplot(zfdat, pcutoff, axs[idx], positions=np.arange(len(zfdat))*2)   
    
    
    macdat = [b for b in shifterrmac2[idx, :, :]]
    vp2 = axs[idx].violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
                       quantiles=[[0.01,0.99] for a in range(len(macdat))])
    hlp.plot_outliers_to_violinplot(macdat, pcutoff, axs[idx], positions=np.arange(len(macdat))*2+1)   
    labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
    labeldat = [b for a in labeldat for b in a]
    axs[idx].set_xticks(np.arange(0,12,2)+0.5)
    xtcklabels = realshifts.copy()
    xtcklabels[:np.int(len(xtcklabels)/2)] = -xtcklabels[:np.int(len(xtcklabels)/2)]
    axs[idx].set_xticklabels(xtcklabels)
    """
    for i in range(len(zfdat)):
        axs[idx].text(0+i*2,-70, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
        axs[idx].text(1+i*2,-70, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')
    """
    axs[idx].set_title('nfilters = %i'%(np.unique(zebrafish['Number of filters'])[idx]))
fig.suptitle('Stimulus shift decoding error [$\circ$]')
axs[0].set_ylabel('Decoding error [$\circ$]')
axs[1].set_xlabel('Stimulus shift magnitude [$\circ$]')#, labelpad=20)
axs[2].legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.067, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)
