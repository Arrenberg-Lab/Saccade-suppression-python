# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:54:06 2021

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
import os

#Violin plots for server simulations

#path for the simulation results
zp = r'simulation/analysis/zebrafish/test_stim_tdur_00850_shiftmag_340_rsl_05_startloc_-170_barwidth_5' #zebrafish path
mp = r'simulation/analysis/macaque/test_stim_tdur_00400_shiftmag_160_rsl_05_startloc_-80_barwidth_5' #macaque path

zsims = os.listdir(zp) #simulation files to be loaded for zebrafish
msims = os.listdir(mp) #simulation files to be loaded for macaque

fig, axs = plt.subplots(1,2)
cm = plt.cm.tab10

#zebrafish
nfs = []
for i, res in enumerate(zsims):
    with np.load(os.path.join(zp,res), allow_pickle=True) as dat:
        derr = dat['decerr']
        nf = int(res[18:24])
        nfs.append(nf)
        vlp = axs[0].violinplot(derr, positions=[i], quantiles=[0.01,0.99], showextrema=False, showmedians=True)
        for pc in vlp['bodies']:
            pc.set_facecolor('red')
        for pc in ['cmedians', 'cquantiles']:
            vlp[pc].set_edgecolor('red')
        
        #ax.plot(nf, np.median(derr), 'r*')

#tick labeling for zebrafish
axs[0].set_xticks(np.arange(len(nfs)))
axs[0].set_xticklabels(nfs)


#macaque
nfs = []
for i, res in enumerate(msims):
    with np.load(os.path.join(mp,res), allow_pickle=True) as dat:
        derr = dat['decerr']
        nf = int(res[18:24])
        nfs.append(nf)
        vlp = axs[1].violinplot(derr, positions=[i], quantiles=[0.01,0.99], showextrema=False, showmedians=True)
        #ax.plot(nf, np.median(derr), 'r*')
        for pc in vlp['bodies']:
            pc.set_facecolor('blue')
        for pc in ['cmedians', 'cquantiles']:
            vlp[pc].set_edgecolor('blue')

"""
# Make all the violin statistics marks red:
for partname in ('cbars','cmedians'):
    vp = vlp[partname]
    vp.set_edgecolor(cm(1))

# Make the violin body blue with a red border:
for vp in violin_parts['bodies']:
    vp.set_facecolor(cm(1))
    vp.set_edgecolor(cm(1))
"""


for ax in axs:
    ax.plot(ax.get_xlim(), (0,0), 'k-')

#tick labeling for macaque
axs[1].set_xticks(np.arange(len(nfs)))
axs[1].set_xticklabels(nfs)

#general formatting
fig.suptitle('Species specific motion decoding error')
axs[0].set_title('Zebrafish')
axs[1].set_title('Macaque')
axs[0].set_ylabel('Decoding error [°]')
axs[0].set_xlabel('Unit number', x=1)
plt.get_current_fig_manager().window.showMaximized()
plt.subplots_adjust(top=0.904, bottom=0.075, left=0.06, right=0.992, hspace=0.2, wspace=0.097)