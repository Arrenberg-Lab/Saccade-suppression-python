# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:24:41 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import random
from scipy.stats import reciprocal #log normal distribution
import matplotlib.pyplot as plt
import time

#Test the new settings for DoG RF placement
zf = True
mac = True

#Plot style: General figure parameters:
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 20,
           'ytick.labelsize' : 20,
           'legend.fontsize' : 20,
           'figure.titlesize' : 25,
           'image.cmap' : 'gray'}
plt.style.use(figdict)


#nfilters
nfilters = 10000

if zf == True:
    #Get the zebrafish size parameters
    zfszparams = []
    zfszparams.append(np.random.uniform(22.2837, 66.8511, np.round(nfilters*0.86).astype(int)))
    zfszparams.append(np.random.uniform(66.8511, 130.8141, np.round(nfilters*0.14).astype(int)))
    zfszparams = [a for b in zfszparams for a in b]
    zfszremoveidx = np.random.randint(0, len(zfszparams)-1, len(zfszparams)-nfilters)
    zfszparams = np.delete(zfszparams, zfszremoveidx)
    random.shuffle(zfszparams)
    
    #zebrafish sf distribution
    #This oversampling makes everything extremely slow
    zfsfdist = reciprocal(0.000001,1).rvs(nfilters*2)#for now oversample and hope you get enough subsampling power
                                                        #with adequate enough computation speed
                                        
    zfsfparams = np.zeros(nfilters)
    DoGzf = np.zeros(nfilters)
    
    start = time.process_time()
    print('Zebrafish loop started')
    for idx, sz in enumerate(zfszparams):
        #find the upper&lower limits of sf: Derivation in page 8 of Ibrahim's train of thoughs notebook
        fmin = 1/(2*sz)
        fmax = 1/sz #Previously 2/sz was wrong! see train of thoughts p 9 back for derivation
        #print(fmin,fmax)
        sfsubset = set(zfsfdist[(fmin<=zfsfdist) & (zfsfdist<=fmax)])
        #print(len(sfsubset))
        #sample the frequency selectivity
        fsmp = random.sample(sfsubset, 1)[0]
        zfsfparams[idx] = fsmp
        #check if gaussian or DoG
        if fsmp < 1/(np.sqrt(2)*sz): #Previously sqrt(2)/sz was WRONG.
            DoGzf[idx] = True
            #here will come the later stuff in generating the receptive field.
        else:
            DoGzf[idx] = False
            #later stuff for RF generation
    end = time.process_time()
    print('Whole iteration took %.1fs for %i iterations (zebrafish)'%(end-start, nfilters))       


#same business for macaque now
#CHANGE OF PLANS: for Macaque choose first the spatial frequency, then based on that initiate the size parameter since you choose
#size in macaque based on some uniform distribution (which is the fancy way of saying you have no idea about the distribution but
#some upper and lower size limits.)
if mac == True:
    #macaque size parameters
    #Update: start from 0 so that you can get filters with high spatial frequency selectivity (very small ones tho)
    #Update2: reciprocal (log uniform) distribution is used, lower limit is chosen based on the maximum sf value of 10
    #Update 3: all updates arer null and void. Now you first choose sf and based on that you determine the size.
    #choose first sf parameter:
    macsfs = np.array([0.5, 1.1, 2.2, 4.4, 10])
    macsfprobs = np.array([45, 30, 10, 20, 2]) / np.sum(np.array([45, 30, 10, 20, 2]))
    macsfparams = []
    start = time.process_time()
    print('Macaque loop started')
    for idx, sf in enumerate(macsfs):
        #sample for each probability interval from uniform distribution the appropriate number of units.
        if idx == len(macsfs)-1:    
            sfpar = np.random.uniform(sf-2, sf, np.ceil(nfilters*macsfprobs[idx]).astype(int)) #last intervl between 8-10  
        else:
            sfpar = np.random.uniform(sf, macsfs[idx+1], np.ceil(nfilters*macsfprobs[idx]).astype(int))
        macsfparams.append(list(sfpar))
    macsfparams = np.array([a for b in macsfparams for a in b])
    #take out random values to match nfilters
    macsfremoveidx = np.random.randint(0, len(macsfparams)-1, len(macsfparams)-nfilters)
    macsfparams = np.delete(macsfparams, macsfremoveidx)
    #shuffle the parameters.
    random.shuffle(macsfparams)
    
    #choose for each sf the size range and sample uniformly from there
    macszparams = np.zeros(nfilters)
    DoGmac = np.zeros(nfilters)
    for idx, sf in enumerate(macsfparams):
        #determine upper and lower boundaries for macaque RF size: Derivation in page 8 of Ibrahim's train of thoughs notebook
        smin = 1/(2*sf)
        smax = 1/sf #Corrected from 2/sf
        sz = np.random.uniform(smin, smax, 1)
        #print('smin=%.2f, smax=%.2f, sz=%.2f, sf=%.2f, sz_crit=%.2f' %(smin, smax, sz, sf, np.sqrt(2)/sf))
        #print('any value smaller than sz_crit turns into DoG')
        #SZ crit is TOO BIG for some reason
        macszparams[idx] = sz 
        #check if gaussian or DoG
        #Implement the required code to generate RF in the visual field.
        if sf < 1/(np.sqrt(2)*sz):
            DoGmac[idx] = True
            #here will come the later stuff in generating the receptive field.
        else:
            DoGmac[idx] = False
            #later stuff for RF generation
    end = time.process_time()
    print('Whole iteration took %.1fs for %i iterations (macaque)'%(end-start, nfilters))       
    
#check how the sf distribution looks like
cmap = plt.get_cmap('tab10')
sms = 1 #scatterplot marker size
jit = 0.05 #jitter extent
lms = 15 #legend marker scale
if zf == True and mac == True:
    #parameter distributions histogram
    macprevdist = np.random.uniform(0.5, 20, nfilters*10) #Oversample the previous distribution used for macaque RF size
    fig, axs = plt.subplots(1,2)
    axs[0].hist(zfsfdist, density=True, bins=np.int(len(zfsfdist)/(nfilters/20)), color='black', label='intended')
    axs[0].hist(zfsfparams, density=True, bins=np.int(len(zfsfparams)/(nfilters/20)), color='gray', label='achieved')
    axs[1].hist(macprevdist, density=True, bins=np.int(len(macprevdist)/(nfilters/20)), color='black', label='previous')
    axs[1].hist(macszparams, density=True, bins=np.int(len(zfsfparams)/(nfilters/20)), color='gray', label='new')
    axs[1].text(x=0.25, y=0.8, s='Maximum diameter : %.2f°' %np.max(macszparams), transform=axs[1].transAxes, fontsize=25)

    axs[0].set_title('Zebrafish RF spatial frequency distribution')
    axs[1].set_title('Macaque RF size (diameter) distribution')
    axs[0].set_xlabel(r'Spatial frequency [$\frac{cyc}{°}$]')
    axs[1].set_xlabel(r'RF diameter [°]')
    for ax in axs:
        ax.set_ylabel('Density')
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    
    #scatterplots
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(DoGzf+np.random.uniform(-jit,jit,len(DoGmac)), np.arange(1,len(DoGzf)+1), '.', color=cmap(0),
            label='Zebrafish', markersize=sms)
    ax.plot(DoGmac+0.4+np.random.uniform(-jit,jit,len(DoGmac)), np.arange(1,len(DoGmac)+1), '.', color=cmap(1), 
            label='Macaque', markersize=sms)
    ax.set_xticks([0.25, 1.25])
    ax.set_xlim(-0.5,2)
    ax.set_xticklabels(['Gauss', 'DoG'])
    ax.set_ylabel('Receptive field ID')
    ax.set_xlabel('Receptive field type')
    ax.legend(loc='best', markerscale=lms)
    ax.text(x=0.8, y=0.8, s='Macaque DoG : %.2f%%' %(100*np.sum(DoGmac)/len(DoGmac)), transform=ax.transAxes, fontsize=20)
    ax.text(x=0.8, y=0.75, s='Zebrafish DoG : %.2f%%' %(100*np.sum(DoGzf)/len(DoGzf)), transform=ax.transAxes, fontsize=20)

elif zf == False:
    #only macaque plots
    #histogram of parameter distribution    
    macprevdist = np.random.uniform(0.5, 20, nfilters*10) #Oversample the previous distribution used for macaque RF size
    fig, ax = plt.subplots()
    ax.hist(macprevdist, density=True, bins=np.int(len(macprevdist)/(nfilters/20)), color='black', label='previous')
    ax.hist(macszparams, density=True, bins=np.int(len(macszparams)/(nfilters/20)), color='gray', label='new')
    ax.set_title('Macaque RF size (diameter) distribution')
    ax.set_xlabel(r'RF diameter [°]')
    ax.set_ylabel('Density')
    ax.legend(loc='best')
    #scatterplot
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(DoGmac+np.random.uniform(-jit,jit,len(DoGmac)), np.arange(1,len(DoGmac)+1), '.', color=cmap(1), 
            label='Macaque', markersize=sms)
    ax.set_xlim(-0.5,1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Gauss', 'DoG'])
    ax.set_ylabel('Receptive field ID')
    ax.set_xlabel('Receptive field type')
    ax.legend(loc='best', markerscale=lms)
    ax.text(x=0.8, y=0.8, s='Macaque DoG : %.2f%%' %(100*np.sum(DoGmac)/len(DoGmac)), transform=ax.transAxes, fontsize=20)

elif mac == False:
    #only zf plots
    #parameter distributions histogram
    fig, ax = plt.subplots()
    ax.hist(zfsfdist, density=True, bins=np.int(len(zfsfdist)/200), color='black', label='intended')
    ax.hist(zfsfparams, density=True, bins=np.int(len(zfsfparams)/200), color='gray', label='achieved')
    ax.set_title('Zebrafish RF spatial frequency distribution')
    ax.set_xlabel(r'Spatial frequency [$\frac{cyc}{°}$]')
    ax.set_ylabel('Density')
    ax.legend(loc='best')
    
    #scatterplots
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(DoGzf+np.random.uniform(-jit,jit,len(DoGmac)), np.arange(1,len(DoGzf)+1), '.', color=cmap(0), 
            label='Zebrafish', markersize=sms)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.5,1.5)
    ax.set_xticklabels(['Gauss', 'DoG'])
    ax.set_ylabel('Receptive field ID')
    ax.set_xlabel('Receptive field type')
    ax.label(loc='best', markerscale=lms)
    ax.text(x=0.8, y=0.8, s='Zebrafish DoG : %.2f%%' %(100*np.sum(DoGzf)/len(DoGzf)), transform=ax.transAxes, fontsize=20)
else:
    print('Bruh why are you running this script if you don\'t wanna test any of the species???')
    



#I might have debugged xD xD xD YEA BABAYYYYYY.

