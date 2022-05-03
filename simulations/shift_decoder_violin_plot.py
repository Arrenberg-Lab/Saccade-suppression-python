# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:55:35 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import pandas as pd
loadpath = r'../data/simulations'

#percentile to plot datapoints as outliers
pcutoff = [1, 99] #%1 percent and 99% as cutoff
#load the datafiles
df1 = r'\nsimul=1000_rsl=1_rdot=10_jsigma=4_centoffs=[  0.   3.  10.  32. 100.]_gfsz=[20. 36. 60.]_' + \
            r'gfsf=[0.1, 0.15, 0.2, 0.25]_gfph=[  0  45  90 135 180 225 270 315]_nfilters=200'
simul1 = pd.read_csv(loadpath+df1)

#generate a location decoding error violin plot (distance between decoded and real stimulus centers)
locerrs = np.sqrt((simul1['Stimulus center X']-simul1['Decoded center X'])**2 + \
                  (simul1['Stimulus center Y']-simul1['Decoded center Y'])**2)

idxs = [] #the indexes for each different X-Y locations
for center in np.unique(simul1['Stimulus center X']):
    idxs.append(np.squeeze(np.where(simul1['Stimulus center X']==center)))
idxs = np.array(idxs)

#sort locerrs by idxs
sortedlocerrs = []
for i, idxarr in enumerate(idxs):
    sortedlocerrs.append(np.array(locerrs[idxarr]))    
sortedlocerrs.append(np.array([y for x in sortedlocerrs for y in x])) #to plot all together

fig, ax = plt.subplots(1,1)
ax.violinplot(sortedlocerrs, showmedians=True, showextrema=False, \
                                                  quantiles=[[0.01,0.99] for a in range(len(sortedlocerrs))])
hlp.plot_outliers_to_violinplot(sortedlocerrs, pcutoff, ax)   

ax.set_xticklabels([0, *np.unique(simul1['Stimulus center X']), 'Pooled'])
ax.set_xlabel('Stimulus center (XY) [°]')
ax.set_ylabel('Decoding error [°]')
ax.set_title('Absolute location decoding error (distance)')

#gerenate a shift decoding error violin plot (mismatch between the decoded and real shifts for different shift sizes)
shifterror = []
shifterrornorm = []
realshifts = np.zeros(len(np.unique(simul1['Stimulus center X']))-1)
for i in range(len(idxs)-1):
    #calculate real shifts
    xdiffreal = np.array(simul1['Stimulus center X'][idxs[i+1]])-np.array(simul1['Stimulus center X'][idxs[i]])
    ydiffreal = np.array(simul1['Stimulus center Y'][idxs[i+1]])-np.array(simul1['Stimulus center Y'][idxs[i]])
    realshift = np.sqrt(xdiffreal**2 + ydiffreal**2)
    
    
    #calculate decoded shifts
    xdiffdec = np.array(simul1['Decoded center X'][idxs[i+1]])-np.array(simul1['Decoded center X'][idxs[i]])
    ydiffdec = np.array(simul1['Decoded center Y'][idxs[i+1]])-np.array(simul1['Decoded center Y'][idxs[i]])
    decodedshift = np.sqrt(xdiffdec**2 + ydiffdec**2)
    
    shifterror.append(decodedshift - realshift)
    shifterrornorm.append((decodedshift - realshift) / np.unique(realshift))
    realshifts[i] = np.round(np.unique(realshift),2)
shifterror.append(np.array([y for x in shifterror for y in x])) #to plot all together
shifterrornorm.append(np.array([y for x in shifterrornorm for y in x])) #to plot all together

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].violinplot(shifterror, showmedians=True, showextrema=False, \
                                                  quantiles=[[0.01,0.99] for a in range(len(shifterror))])
hlp.plot_outliers_to_violinplot(shifterror, pcutoff, ax[0])   

ax[0].set_xticklabels([0, *realshifts, 'Pooled'])
ax[0].set_xlabel('Real shift magnitude [°]')
ax[0].set_ylabel('Decoding error [°]')
ax[0].set_title('Absolute shift decoding error')

ax[1].violinplot(shifterrornorm, showmedians=True, showextrema=False, \
                                                  quantiles=[[0.01,0.99] for a in range(len(shifterrornorm))])
hlp.plot_outliers_to_violinplot(shifterrornorm, pcutoff, ax[1])   
ax[1].set_xticklabels([0, *realshifts, 'Pooled'])
ax[1].set_xlabel('Real shift magnitude [°]')
ax[1].set_ylabel('Decoding error')
ax[1].set_title('Relative shift decoding error')

#now the same as a function of neuron number:
df2 = r'\nsimul=1000_rsl=1_rdot=10_jsigma=4_centoffs=[ 0.  7. 50.]_gfsz=[20. 36. 60.]_gfsf=[0.1, 0.15, 0.2, 0.25]_' + \
            r'gfph=[  0  45  90 135 180 225 270 315]_nfilters=[ 18  50 128 392 968]'
simul2 = pd.read_csv(loadpath+df2)
nsimul = np.int(len(simul2) / len(np.unique(simul2['Stimulus center X'])) / \
                                                                        len(np.unique(simul2['Number of filters'])))
#get the indices sorted by the stimulus location
idxsloc = [] #the indexes for each different X-Y locations
for center in np.unique(simul2['Stimulus center X']):
    idxsloc.append(np.squeeze(np.where(simul2['Stimulus center X']==center)))
idxsloc = np.array(idxsloc)

#get the indices sorted by the filter number
idxsnf = [] #the indexes for each filter number
for nf in np.unique(simul2['Number of filters']):
    idxsnf.append(np.squeeze(np.where(simul2['Number of filters']==nf)))
idxsnf = np.array(idxsnf)

locerrs = np.sqrt((simul2['Stimulus center X']-simul2['Decoded center X'])**2 + \
                  (simul2['Stimulus center Y']-simul2['Decoded center Y'])**2)

#sort locerrs by filter numbers
#first generate a violin plot as a function of neuron numbers, without considering the shift size.
sortedlocerrsnf = []
for i, idxnf in enumerate(idxsnf):
    locerrnf = np.array(locerrs[idxnf])
    sortedlocerrsnf.append(locerrnf[~np.isnan(locerrnf)])
sortedlocerrsnf.append(np.array([y for x in sortedlocerrsnf for y in x])) #to plot all together

fig, ax = plt.subplots(1,1)
ax.violinplot(sortedlocerrsnf, showmedians=True, showextrema=False, \
                                                  quantiles=[[0.01,0.99] for a in range(len(sortedlocerrsnf))])
hlp.plot_outliers_to_violinplot(sortedlocerrsnf, pcutoff, ax)   

xtcklabels = [0, *np.unique(simul2['Number of filters']), 'Pooled']
newlabels = hlp.violin_plot_xticklabels_add_nsimul(xtcklabels, sortedlocerrsnf)
ax.set_xticklabels(newlabels)
ax.set_xlabel('Number of filters')
ax.set_ylabel('Decoding error [°]')
ax.set_title('Absolute location decoding error (distance)')

#now the subplots considering each size
fig, axs = plt.subplots(1,3, sharey=True)
fig.suptitle('Absolute location decoding error (distance)')
for idxl, loc in enumerate(np.unique(simul2['Stimulus center X'])):
    loclist = []
    for idxn, nf in enumerate(np.unique(simul2['Number of filters'])):
        locerr = locerrs[(simul2['Number of filters']==nf) & (simul2['Stimulus center X']==loc)]
        locerr = locerr[~np.isnan(locerr)]
        loclist.append(np.array(locerr))
    loclist.append(np.array([y for x in loclist for y in x])) #to plot all together
    axs[idxl].violinplot(loclist, showmedians=True, showextrema=False, \
                                                  quantiles=[[0.01,0.99] for a in range(len(loclist))])
    hlp.plot_outliers_to_violinplot(loclist, pcutoff, axs[idxl])   

    axs[idxl].set_title('XY location=%d°'%(loc))
    axs[idxl].set_xticks(np.arange(0,len(np.unique(simul2['Number of filters']))+2))
    xtcklabels = [0, *np.unique(simul2['Number of filters']), 'P']
    newlabels = hlp.violin_plot_xticklabels_add_nsimul(xtcklabels, loclist)
    print(newlabels)
    axs[idxl].set_xticklabels(newlabels)
axs[1].set_xlabel('Number of filters')
axs[0].set_ylabel('Decoding error [°]')
plt.subplots_adjust(left=0.05, bottom=0.11, right=0.995, top=0.88, wspace=0.057, hspace=0.2)

#generate the shift decoding error violin plot 
shifterror = [] #each sublist for number of filters, each of which for different shift sizes.
shifterrornorm = [] #same as above but shift magnitude normalized
locs = np.unique(simul2['Stimulus center X'])
for idxn, nf in enumerate(np.unique(simul2['Number of filters'])):
    bysize = simul2[simul2['Number of filters'] == nf]
    realsbz = []
    decsbz = []
    shifterrsbz = []
    shifterrsbznorm = []
    for idxloc in range(len(locs[:-1])):
        #calculate real shifts
        xdiffreal = np.array(bysize[bysize['Stimulus center X'] == locs[idxloc+1]]['Stimulus center X'])-\
                    np.array(bysize[bysize['Stimulus center X'] == locs[idxloc]]['Stimulus center X'])
        ydiffreal = np.array(bysize[bysize['Stimulus center Y'] == locs[idxloc+1]]['Stimulus center Y'])-\
                    np.array(bysize[bysize['Stimulus center Y'] == locs[idxloc]]['Stimulus center Y'])
        realshift = np.sqrt(xdiffreal**2 + ydiffreal**2)
        realsbz.append(realshift)
        
        #calculate decoded shifts
        xdiffdec = np.array(bysize[bysize['Stimulus center X'] == locs[idxloc+1]]['Decoded center X'])-\
                   np.array(bysize[bysize['Stimulus center X'] == locs[idxloc]]['Decoded center X'])
        ydiffdec = np.array(bysize[bysize['Stimulus center Y'] == locs[idxloc+1]]['Decoded center Y'])-\
                   np.array(bysize[bysize['Stimulus center Y'] == locs[idxloc]]['Decoded center Y'])
        decodedshift = np.sqrt(xdiffdec**2 + ydiffdec**2)
        decsbz.append(decodedshift)
        #for now take out the nan values
        shifterrsbz.append((decodedshift - realshift)[~np.isnan(decodedshift - realshift)])
        shifterrsbznorm.append((decodedshift - realshift)[~np.isnan(decodedshift - realshift)] / np.unique(realshift))
    shifterror.append(shifterrsbz)     
    shifterrornorm.append(shifterrsbznorm) 

shifterrorpooled = []
for err in shifterror:
    shifterrorpooled.append([b for a in err for b in a])

shifterrornormpooled = []
for err in shifterrornorm:
    shifterrornormpooled.append([b for a in err for b in a])


fig, axs = plt.subplots(2,3, sharey=True)
axs = axs.flatten()
shifts = np.unique(np.diff(simul2['XY shift'])*np.sqrt(2))[np.unique(np.diff(simul2['XY shift'])*np.sqrt(2))>0]
for idx, data in enumerate(shifterror):
    axs[idx].violinplot(data+[shifterrorpooled[idx]], showmedians=True, showextrema=False,  \
                                           quantiles=[[0.01,0.99] for a in range(len(data+[shifterrorpooled[idx]]))])
    hlp.plot_outliers_to_violinplot(data+[shifterrorpooled[idx]], pcutoff, axs[idx])   
    axs[idx].set_title('Number of filters=%i'%(np.unique(simul2['Number of filters'])[idx]))
    axs[idxl].set_xticks(np.arange(0,len(shifts)+2))
    xtcklabels = [0, *np.round(shifts,2), 'P']
    newlabels = hlp.violin_plot_xticklabels_add_nsimul(xtcklabels, data+[shifterrorpooled[idx]])
    axs[idx].set_xticklabels(newlabels)

fig.suptitle('Absolute shift decoding errors for filter populations of different size')   
axs[4].set_xlabel('Real shift magnitude [°]')
axs[0].set_ylabel('Decoding error [°]')
fig.delaxes(axs[-1])
plt.subplots_adjust(left=0.125, bottom=0.121, right=0.9, top=0.88, wspace=0.155, hspace=0.426)

fig, axs = plt.subplots(2,3, sharey=True)
axs = axs.flatten()
shifts = np.unique(np.diff(simul2['XY shift'])*np.sqrt(2))[np.unique(np.diff(simul2['XY shift'])*np.sqrt(2))>0]
for idx, data in enumerate(shifterrornorm):
    axs[idx].violinplot(data+[shifterrornormpooled[idx]], showmedians=True, showextrema=False,  \
                                      quantiles=[[0.01,0.99] for a in range(len(data+[shifterrornormpooled[idx]]))])
    hlp.plot_outliers_to_violinplot(data+[shifterrornormpooled[idx]], pcutoff, axs[idx])
    axs[idx].set_title('Number of filters=%i'%(np.unique(simul2['Number of filters'])[idx]))
    axs[idxl].set_xticks(np.arange(0,len(shifts)+2))
    xtcklabels = [0, *np.round(shifts,2), 'P']
    newlabels = hlp.violin_plot_xticklabels_add_nsimul(xtcklabels, data+[shifterrornormpooled[idx]])
    axs[idx].set_xticklabels(newlabels)

fig.suptitle('Relative shift decoding errors for filter populations of different size')   
axs[4].set_xlabel('Real shift magnitude [°]')
axs[0].set_ylabel('Decoding error')
fig.delaxes(axs[-1])
plt.subplots_adjust(left=0.13, bottom=0.126, right=0.898, top=0.88, wspace=0.086, hspace=0.438)

#get the number of isnans for each nfilters from simul2
for nf in np.unique(simul2['Number of filters']):
    decarray = np.array(simul2[simul2['Number of filters']==nf]['Decoded center X'])
    nnans = len(decarray[np.isnan(decarray)])
    print(nf, nnans)
    
df3 = r'\nsimul=1000_rsl=1_rdot=10_jsigma=4_gfsz=[20. 36. 60.]_gfsf=[0.1, 0.15, 0.2, 0.25]_' + \
      r'gfph=[  0  45  90 135 180 225 270 315]_nfilters=[ 18  50 128 392 968]_nimg=1'
simul3 = pd.read_csv(loadpath+df3)

#sort locerrs by idxs
sortedlocerrs1img = []
for nf in np.unique(simul3['Number of filters']):
    #stimulus centers
    cx = np.array(simul3[simul3['Number of filters']==nf]['Stimulus center X'])
    cy = np.array(simul3[simul3['Number of filters']==nf]['Stimulus center Y'])
    #decoded centers
    dx = np.array(simul3[simul3['Number of filters']==nf]['Decoded center X'])
    dy = np.array(simul3[simul3['Number of filters']==nf]['Decoded center Y'])
    locerr = np.sqrt((cx-dx)**2 +(cy-dy)**2)
    sortedlocerrs1img.append(locerr[~np.isnan(locerr)])    
sortedlocerrs1img.append(np.array([y for x in sortedlocerrs1img for y in x])) #to plot all together

fig, ax = plt.subplots(1,1)
ax.violinplot(sortedlocerrs1img, showmedians=True, showextrema=False, \
                                                  quantiles=[[0.01,0.99] for a in range(len(sortedlocerrs1img))])
hlp.plot_outliers_to_violinplot(sortedlocerrs1img, pcutoff, ax)
xtcklabels = [0, *np.unique(simul3['Number of filters']), 'Pooled']   
newlabels = hlp.violin_plot_xticklabels_add_nsimul(xtcklabels, sortedlocerrs1img)
ax.set_xticklabels(newlabels)
ax.set_xticklabels(newlabels)
ax.set_xlabel('Number of filters')
ax.set_ylabel('Decoding error [°]')
ax.set_title('Absolute location decoding error (distance)')
