# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:26:01 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt

#test random walk
def random_walk(dt,tl, dtnorm='mult', avg=0, std=1):
    """
    Random walk function

    Parameters
    ----------
    dt : float 
        Size of the time step
        
    tl : float
        Stimulus time length
        

    Returns
    -------
    stim : 1-D array
        Random walk array
    """
    t = np.linspace(0,tl,np.int(tl/dt)+1)
    stim = np.zeros(t.shape)
    noise = np.random.normal(avg*np.sqrt(dt), std, len(t))
    if dtnorm == 'mult':
        noise *= np.sqrt(dt)
    elif dtnorm == 'div':
        noise /= np.sqrt(dt)
    elif dtnorm == 'no':
        pass
    for i in range(1,len(t)):
        stim[i] = stim[i-1] + noise[i]
    return stim




figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 25,
           'xtick.labelsize' : 25,
           'ytick.labelsize' : 25,
           'legend.fontsize' : 25,
           'figure.titlesize' : 30,
           'image.cmap' : 'gray',
           'axes.formatter.limits' : [-7, 7]}
plt.style.use(figdict)

fig, axs = plt.subplots(1,3,sharex=True)
fig.suptitle('Example random walk noise scalings')
axs[0].set_ylabel('Variance (error bars SEM)')
axs[1].set_xlabel('dt')


tl = 10
dts = np.logspace(-5,0,10)
dtvarsm = []
dtvarsd = []
dtvarsn = []
for a in range(50):
    dtvarm = []
    dtvard = []
    dtvarn = []
    for dt in dts:
        stim1 = random_walk(dt, tl, dtnorm='mult')
        dtvarm.append(np.var(stim1))
        stim2 = random_walk(dt, tl, dtnorm='div')
        dtvard.append(np.var(stim2))
        stim3 = random_walk(dt, tl, dtnorm='no')
        dtvarn.append(np.var(stim3))
    dtvarsm.append(dtvarm)
    dtvarsd.append(dtvard)
    dtvarsn.append(dtvarn)
    print(a)

(dtvarsm, dtvarsd, dtvarsn) = [np.squeeze(a) for a in (dtvarsm, dtvarsd, dtvarsn)]

varsl = (dtvarsm, dtvarsd, dtvarsn)
titles=['multiplication', 'division', 'no scaling']
for i, ax in enumerate(axs):
    ax.bar(np.arange(varsl[i].shape[1]),np.mean(varsl[i],axis=0))
    ax.errorbar(np.arange(varsl[i].shape[1]),np.mean(varsl[i],axis=0), 
                yerr=np.sqrt(np.mean(varsl[i],axis=0)/len(varsl[i])), fmt='k.', capsize=5)
    ax.set_xticks(np.arange(0,varsl[i].shape[1]))
    ax.set_xticklabels(np.round(dts,5), rotation=90)
    ax.set_title(titles[i])
plt.get_current_fig_manager().window.showMaximized()
plt.pause(0.1)
plt.tight_layout()


fig1, axs1 = plt.subplots(1,3, sharex=True, sharey=True)
fig2, axs2 = plt.subplots(1,3, sharex=True, sharey=True)
fig3, axs3 = plt.subplots(1,3, sharex=True)

suptitles = ['Random walk simulation', 'Increment per each time step', 'Increment standard deviation distribution']

axs1[0].set_ylabel('Y [a.u.]')
axs1[1].set_xlabel('t [s]')
axs2[0].set_ylabel(r'$\frac{dY}{\sqrt{dt}}$ [a.u.]')
axs2[1].set_xlabel('t [s]')
axs3[0].set_ylabel('Probability density')
axs3[1].set_xlabel(r'std($\frac{dY}{\sqrt{dt}}$) [a.u.]')

tl = 10 #seconds
figs = [fig1, fig2, fig3]
for didx, dt in enumerate(dts[2::3]):
    t = np.linspace(0,tl,np.int(tl/dt+1))
    for aidx, axx in enumerate([axs1, axs2, axs3]):
        axx[didx].set_title('dt=%.5f' %(dt))
        figs[aidx].suptitle(suptitles[aidx])
    stds = []
    for a in range(50):
        stim1 = random_walk(dt, tl, dtnorm='mult')
        axs1[didx].plot(t,stim1)
        axs2[didx].plot(t[1:],np.diff(stim1)/np.sqrt(dt))
        #axs3[didx].plot(a,np.std(np.diff(stim1)/np.sqrt(dt)), '.')
        stds.append(np.std(np.diff(stim1)/np.sqrt(dt)))
    axs1[didx].plot([t[0], t[-1]], [0,0], 'k-', linewidth=2)
    axs2[didx].plot([t[0], t[-1]], [0,0], 'k-', linewidth=2)
    axs3[didx].hist(stds, bins=10, color='black', density=True)
    print(axs3[didx].get_ylim(), didx)
    axs3[didx].plot([np.mean(stds),np.mean(stds)], axs3[didx].get_ylim(), 'r-', linewidth=1)
    print(didx)

#drift case
trend = 1
std = 100
tl = 50 #seconds
dt = 0.001
stim = random_walk(dt, tl, avg=trend, std=std)
fig, axs = plt.subplots(1,2)
axs[0].plot(stim)
axs[1].plot(np.diff(stim)/np.sqrt(dt))
