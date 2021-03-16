# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:26:13 2021

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
loadpath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data\simulations'
from scipy.stats import ttest_1samp


#THERE WAS AN UNEXPECTED BUG IN THE DECODING ALGORITHM. FOR NOW ONLY POOL THE STIMULI SHIFTED SMALLER THAN 90°
#IN AZIMUTH 

#Species simulation plots for randomized horizontal stimulus shifts.
#percentile to plot datapoints as outliers
pcutoff = [1, 99] #%1 percent and 99% as cutoff
#load the datafiles
df = r'\zf_mac_simul_nsimul=50_rsl_mac=24_rsl_zf=5_rdot=5_jsigma=4_nfilters=[128, 1058, 10082]_nshifts=50'
simul = pd.read_csv(loadpath+df)

#transform the stimulus positions to azimuth and elevations respectively.
simul['Real center X'] -=180
simul['Real center Y'] = 90 - simul['Real center Y']
simul['Real center Y before jitter'] = 90 - simul['Real center Y before jitter']
simul['Decoded center X'] -=180
simul['Decoded center Y'] = 90 - simul['Decoded center Y']

zebrafish = simul[simul['Species']=='Zebrafish']
macaque = simul[simul['Species']=='Macaque']
zebrafish = zebrafish[np.abs(zebrafish['Real center X before jitter']) < 90]
macaque = macaque[np.abs(macaque['Real center X before jitter']) < 90]
zfdfidxs = np.array(zebrafish.index)
macdfidxs = np.array(macaque.index)

#1) Stimulus position decoding error
#azimuth difference between real and decoded
zfxdif = np.abs(zebrafish['Real center X before jitter'] - zebrafish['Decoded center X']) 
zfxdif[zfxdif>180] = 360 - zfxdif[zfxdif>180] #the max azimuth distance between the stimulus positions can be 180°,
                                              #as azimuth wraps around the whole visual field. Thus, if  
                                              #abs(real-decoded)>360, real distance is 360-abs(real-decoded)
                                                                                          
zfydif = zebrafish['Real center Y before jitter'] - zebrafish['Decoded center Y'] #elevation difference between real
                                                                                  #and decoded                                                                                  
zflocdecerr = np.array(np.sqrt(zfxdif**2 +  zfydif**2))

macxdif = np.abs(macaque['Real center X before jitter'] - macaque['Decoded center X']) 
macxdif[macxdif>180] = 360 - macxdif[macxdif>180] #the max azimuth distance between the stimulus positions can be 180°,
                                                  #as azimuth wraps around the whole visual field. Thus, if  
                                                  #abs(real-decoded)>360, real distance is 360-abs(real-decoded)
                                                                                          
macydif = macaque['Real center Y before jitter'] - macaque['Decoded center Y'] #elevation difference between real
                                                                               #and decoded                                                                                  
maclocdecerr = np.array(np.sqrt(macxdif**2 + macydif**2))

#separate by filter numbers  
nnfilters = 3 #number of different filter numbers used for simulation
nsimul = 50
nfilters = list(np.unique(zebrafish['Number of filters']))
#as you take some values from the data frames, you cannot use the n dimensional array approach, use nested list.
zfsortedlocdecerr = [[] for i in range(nnfilters)]
macsortedlocdecerr = [[] for i in range(nnfilters)]

for idx in range(len(zebrafish)):
    nfilter = zebrafish['Number of filters'][zfdfidxs[idx]]
    nidx = nfilters.index(nfilter)
    zfsortedlocdecerr[nidx].append(zflocdecerr[idx])
    
    nfilter = macaque['Number of filters'][macdfidxs[idx]]
    nidx = nfilters.index(nfilter)
    macsortedlocdecerr[nidx].append(maclocdecerr[idx])
    
#plotting        
fig, ax = plt.subplots(1,1, sharey=True)
zfdat = [np.array(b)[~np.isnan(b)] for b in zfsortedlocdecerr]
vp1 = ax.violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
         quantiles=[[0.01,0.99] for a in range(len(zfdat))])
hlp.plot_outliers_to_violinplot(zfdat, pcutoff, ax, positions=np.arange(len(zfdat))*2)   

    
macdat = [np.array(b)[~np.isnan(b)] for b in macsortedlocdecerr]
vp2 = ax.violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
         quantiles=[[0.01,0.99] for a in range(len(macdat))])
hlp.plot_outliers_to_violinplot(macdat, pcutoff, ax, positions=np.arange(len(macdat))*2+1)   
labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
labeldat = [b for a in labeldat for b in a]
ax.set_xticks(np.arange(0,6,2)+0.5)
ax.set_xticklabels(np.unique(zebrafish['Number of filters']))
for i in range(len(zfdat)):
    ax.text(0+i*2,-2, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
    ax.text(1+i*2,-2, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')

fig.suptitle('Stimulus position decoding error (shift<90°)')
ax.set_ylabel('Decoding error [$\circ$]')
ax.set_xlabel('Number of units', labelpad=10)
ax.legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'])
plt.subplots_adjust(left=0.057, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)

#2) Stimulus shift decoding error
#This is easy now since you only have horizontal shifts. So real shift is given already. For decoded shift do the 
#same calculations as before.
shifterrszf = [[] for i in range(nnfilters)] #number of different filters tested x number of shifts x nsimul
shifterrsmac = [[] for i in range(nnfilters)]
decshiftszf = [[] for i in range(nnfilters)] #necessary for the square plots.
decshiftsmac = [[] for i in range(nnfilters)] #necessary for the square plots.
realshiftszf = [[] for i in range(nnfilters)] #necessary for the square plots.
realshiftsmac = [[] for i in range(nnfilters)] #necessary for the square plots.


#Take the before stimuli for each simulation idx.
decxbeforezf = [[] for i in range(nnfilters)]
decybeforezf = [[] for i in range(nnfilters)]
decxbeforemac = [[] for i in range(nnfilters)]
decybeforemac = [[] for i in range(nnfilters)]

for nidx, nfilter in enumerate(nfilters):
    zfdf = zebrafish[zebrafish['Number of filters']==nfilter]
    decxbeforezf[nidx].append(np.array(zfdf['Decoded center X'][:50]))
    decybeforezf[nidx].append(np.array(zfdf['Decoded center Y'][:50]))
    
    macdf = macaque[macaque['Number of filters']==nfilter]
    decxbeforemac[nidx].append(np.array(macdf['Decoded center X'][:50]))
    decybeforemac[nidx].append(np.array(macdf['Decoded center Y'][:50]))
    
#compare each shift to the before stimulus based on the simulation index
for i in range(len(zebrafish)):
    zfdf = zebrafish.iloc[i] #works
    simulID = zfdf['Simulation number']
    simulID -= 1
    decxafterzf = zfdf['Decoded center X']
    decyafterzf = zfdf['Decoded center Y']
    nfilter = zfdf['Number of filters']
    decshiftzf = np.sqrt((decxafterzf - decxbeforezf[nfilters.index(nfilter)][0][simulID])**2 + \
                         (decyafterzf - decybeforezf[nfilters.index(nfilter)][0][simulID])**2)

    realshiftzf = np.abs(zfdf['Real center X before jitter'])
    if realshiftzf == 0:
        continue
    elif np.isnan(decshiftzf):
        decshiftzf = 0 #no shift decoded if null result.
    decshiftszf[nfilters.index(nfilter)].append(decshiftzf)
    shifterrzf = decshiftzf - realshiftzf
    
    shifterrszf[nfilters.index(nfilter)].append(shifterrzf)
    realshiftszf[nfilters.index(nfilter)].append(realshiftzf)
    
    macdf = macaque.iloc[i] #works
    simulID = macdf['Simulation number']
    simulID -= 1
    decxaftermac = macdf['Decoded center X']
    decyaftermac = macdf['Decoded center Y']
    nfilter = macdf['Number of filters']
    decshiftmac = np.sqrt((decxaftermac - decxbeforemac[nfilters.index(nfilter)][0][simulID])**2 + \
                         (decyaftermac - decybeforemac[nfilters.index(nfilter)][0][simulID])**2)
    
    realshiftmac = np.abs(macdf['Real center X before jitter'])
    if realshiftmac == 0:
        continue
    elif np.isnan(decshiftmac):
        decshiftmac = 0 #no shift decoded if null result.
    decshiftsmac[nfilters.index(nfilter)].append(decshiftmac)
    shifterrmac = decshiftmac - realshiftmac

    shifterrsmac[nfilters.index(nfilter)].append(shifterrmac)
    realshiftsmac[nfilters.index(nfilter)].append(realshiftmac)

   
#Violinplots
fig, ax = plt.subplots(1, 1, sharey=True)
zfdat = [np.array(b) for b in shifterrszf]
vp1 = ax.violinplot(zfdat, showmedians=True, showextrema=False, positions=np.arange(len(zfdat))*2,\
         quantiles=[[0.01,0.99] for a in range(len(zfdat))])
hlp.plot_outliers_to_violinplot(zfdat, pcutoff, ax, positions=np.arange(len(zfdat))*2)   

    
macdat = [np.array(b) for b in shifterrsmac]
vp2 = ax.violinplot(macdat, showmedians=True, showextrema=False, positions=np.arange(len(macdat))*2+1, \
         quantiles=[[0.01,0.99] for a in range(len(macdat))])
hlp.plot_outliers_to_violinplot(macdat, pcutoff, ax, positions=np.arange(len(macdat))*2+1)   
labeldat = [[zfdat[i], macdat[i]] for i in range(len(zfdat))]
labeldat = [b for a in labeldat for b in a]
ax.set_xticks(np.arange(0,6,2)+0.5)
ax.set_xticklabels(np.unique(zebrafish['Number of filters']))

for i in range(len(zfdat)):
    ax.text(0+i*2,59, '(%i)'%(len(zfdat[i])), size=15, horizontalalignment='center')
    ax.text(1+i*2,59, '(%i)'%(len(macdat[i])), size=15, horizontalalignment='center')

fig.suptitle('Stimulus shift decoding error (shift<90°)')
ax.set_ylabel('Decoding error [$\circ$]')
ax.set_xlabel('Number of units')
ax.legend([vp1['bodies'][0],vp2['bodies'][0]], ['Zebrafish', 'Macaque'], loc='lower right')
plt.subplots_adjust(left=0.067, bottom=0.13, right=0.99, top=0.88, wspace=0.06, hspace=0.2)


#Square plots
fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
axs = axs.reshape(6)
for idx, ax in enumerate(axs[:3]):
    ax.plot(realshiftszf[idx], decshiftszf[idx], '.')
    ax.set_xlim([-5,185])
    ax.set_ylim([-5,185])
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k-')
    ax.set_title('nfilters=%i'%(nfilters[idx]))
    #plt.gca().set_aspect('equal')   

for idx, ax in enumerate(axs[3:]):
    ax.plot([]) #trick the color cycle to the second one
    ax.plot(realshiftsmac[idx], decshiftsmac[idx], '.')
    ax.set_xlim([-5,185])
    ax.set_ylim([-5,185])
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k-')
    #plt.gca().set_aspect('equal')  
for ax in axs:
    ax.set_aspect('equal')   
plt.suptitle('Comparison of real and decoded shifts (shift<90°)')
axs[4].set_xlabel('Real shift [°]')
axs[0].set_ylabel('Decoded shift [°]')
axs[0].set_xticks(np.linspace(0,180,5))
axs[0].set_yticks(np.linspace(0,180,5))

tzf, pzf = ttest_1samp(shifterrszf[-1], 0, axis=0)
tmac, pmac = ttest_1samp(shifterrsmac[-1], 0, axis=0)
