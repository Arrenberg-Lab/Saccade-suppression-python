# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:39:28 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
import pandas as pd
import matplotlib.ticker as ticker
loadpath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data\simulations'

#Species simulation violinplots (macaque and zebrafish)

#percentile to plot datapoints as outliers
pcutoff = [1, 99] #%1 percent and 99% as cutoff
#load the datafiles
df = r'\zf_mac_simul_nsimul=100_rsl_mac=24_rsl_zf=5_rdot=5_jsigma=4_nfilters=[128, 1058, 10082]_centoffs=[ 0.  7. 50.]'
simul = pd.read_csv(loadpath+df)
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
shifterrzf = np.zeros([nnfilters, 3, nsimul])
shifterrmac = np.zeros([nnfilters, 3, nsimul])

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
            realshiftzf = np.sqrt((realxzf[idx] - realxzf[(idx+1)%r])**2 + (realyzf[idx] - realyzf[(idx+1)%r])**2)
            decshiftzf = np.sqrt((decxzf[idx] - decxzf[(idx+1)%r])**2 + (decyzf[idx] - decyzf[(idx+1)%r])**2)
            decerrzf = decshiftzf - realshiftzf
            shifterrzf[nidx, idx, sidx] = decerrzf
            
            realshiftmac = np.sqrt((realxmac[idx]-realxmac[(idx+1)%r])**2 + (realymac[idx] - realymac[(idx+1)%r])**2)
            decshiftmac = np.sqrt((decxmac[idx] - decxmac[(idx+1)%r])**2 + (decymac[idx] - decymac[(idx+1)%r])**2)
            decerrmac = decshiftmac - realshiftmac
            shifterrmac[nidx, idx, sidx] = decerrmac
            #print(realshiftzf, realshiftmac) #comparable to the shifts calculated by using values 0,7,50

realshifts = np.round(np.array([7,43,50])*np.sqrt(2),2)
            
#plotting
fig, axs = plt.subplots(1, nnfilters, sharey=True)
for idx in range(nnfilters):
    zfdat = [b[~np.isnan(b)] for b in shifterrzf[idx, :, :]]
    vp1 = axs[idx].violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
                       quantiles=[[0.01,0.99] for a in range(len(zfdat))])
    hlp.plot_outliers_to_violinplot(zfdat, pcutoff, axs[idx], positions=np.arange(len(zfdat))*2)   
    
    
    macdat = [b[~np.isnan(b)] for b in shifterrmac[idx, :, :]]
    vp2 = axs[idx].violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
                       quantiles=[[0.01,0.99] for a in range(len(macdat))])
    hlp.plot_outliers_to_violinplot(macdat, pcutoff, axs[idx], positions=np.arange(len(macdat))*2+1)   
    labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
    labeldat = [b for a in labeldat for b in a]
    axs[idx].set_xticks(np.arange(0,6,2)+0.5)
    axs[idx].set_xticklabels(realshifts)
    for i in range(len(zfdat)):
        axs[idx].text(0+i*2,-70, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
        axs[idx].text(1+i*2,-70, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')
    axs[idx].set_title('nfilters = %i'%(np.unique(zebrafish['Number of filters'])[idx]))
fig.suptitle('Stimulus shift decoding error [$\circ$]')
axs[0].set_ylabel('Decoding error [$\circ$]')
axs[1].set_xlabel('Stimulus shift magnitude [$\circ$]', labelpad=20)
axs[2].legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.067, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)

