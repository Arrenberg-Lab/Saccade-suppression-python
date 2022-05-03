# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 19:07:53 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib import colors, ticker

#Plot style: General figure parameters:
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 15,
           'ytick.labelsize' : 15,
           'legend.fontsize' : 15,
           'figure.titlesize' : 25,
           'image.cmap' : 'gray'}
plt.style.use(figdict)




example = False #If True, example DoG is generated.

#Test the Fourier transform of DoG to see its filtering properties

#Expected: since Fourier transform is a linear operation, I think it should be possible to subtract the frequency domain signals
#of both gaussians. Since in frequency domain gaussian is also gaussian with inverse standard deviation, expected would be to see
#DOG to switch its center-surround polarity in the frequency domain. Lets get started!!!!

#Expectations are corroborated, with a difference that in fourier domain DoG has a minimum at zero.

#Try to come up with a way defining the peak of the DoG in the Fourier domain -> that is your frequency selectivity.

#I TRIED TO COMPARE FFT RESULTS WITH ANALYTICAL DERIVATIONS; WHICH DID NOT WORK AT ALL!
#18.08: For some reason DoG is working although rest (center and surround Gauss) is not working so well (I dunno why)

#Define some Gaussian function
def gauss(x,sigma):
    """
    Gaussian curve.

    Parameters
    ----------
    x : 1-D array
        Array over which Gaussian is calculated.
    sigma : float
        Standard deviation.

    Returns
    -------
    1-D array
        Gaussian curve.

    """
    dx = np.diff(x)[0]
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(x/sigma)**2)*dx

if example == True:
    #Generate Center and Surround Gaussians
    scent = 0.001 #center sigma
    ssur = 0.0011  #surround sigma
    x = np.linspace(-5*ssur, 5*ssur, 2**15, endpoint=False)
    cg = gauss(x, scent)
    sg = gauss(x,ssur)
    
    N = len(x)
    
    #FFT on Gaussians to see what is happening
    #Take only positive frequencies and multiply by 2
    fftc = fft(cg, n=N)[:N//2]
    frc = fftfreq(n=N, d=(np.diff(x))[0])[:N//2]
    ffts = fft(sg, n=N)[:N//2]
    frs = fftfreq(n=N, d=(np.diff(x))[0])[:N//2]
    fftdog = fft(cg-sg, n=N)[:N//2]
    frdog = fftfreq(n=len(cg-sg), d=(np.diff(x))[0])[:N//2]
    
    
    #sanity check
    fs = 100
    phi = np.pi
    sx = np.linspace(0, 10, 2**12, endpoint=False)
    sinw = np.sin(2*np.pi*sx*fs+phi)
    fftsin = fft(sinw, n=2**12)[:2**12//2]
    frsin = fftfreq(n=2**12, d=np.diff(sx)[0])[:2**12//2]
    
    fig, ax = plt.subplots()
    ax.plot(frsin, 2/N*np.abs(fftsin))
    
    fig, axs = plt.subplots(1,2)
    axs[0].plot(x,cg, label='center')
    axs[0].plot(x,sg, label='surround')
    axs[0].plot(x,cg-sg, label='DoG')
    axs[1].plot(frc, 2/N*np.abs(fftc))
    axs[1].plot(frs, 2/N*np.abs(ffts))
    axs[1].plot(frdog, 2/N*np.abs(fftdog))
    axs[1].set_xlim([0, 2/scent])
    axs[0].legend()
    axs[0].set_title('Time domain')
    axs[1].set_title('Frequency domain')
    
    #ANALYTICAL SOLUTION IS FOR 2D CASE ><
    analfr = np.sqrt((2*np.log(ssur**2/scent**2))/(ssur**2-scent**2)) #analytical solution for frequency peak
    numfr = np.array(frdog)[np.abs(fftdog) == np.max(np.abs(fftdog))] #numerical result
    print(analfr, numfr, analfr/numfr) #NUMBERS DO NOT MATCH, possibly there is an error in fftfreq.

else:
    verbose = True
    #Generate the Look-up table
    #Find the maximum and mimimum RF size values we have for the RF sets for generating the surround Gaussian std values.
    """
    surround gaussian std is 1/4 of the RF diameter (for now),
    stdg = sz/4 -> get max and min possible sz values, those define your interval of surround Gauss:
    Zebrafish: straightforward -> sz = [22.2837-130.8141]°
    Macaque: a little bit more complicated -> sf between 0.5-10 cyc/°, smin = 1/(2sf), smax=1/sf
             Thus: smallest possible s => 1/(2*10)= 0.05° (smin with biggest frequency), 
                   biggest possible s => 1/0.5 = 2° (smax with smallest frequency)
    ALL TOGETHER THE TOTAL interval is between 0.05-130.8141°
    Implement this!!!
    """ 
    #the step size between different surround gaussian std values for the scan:
    sts = 0.001 #step size
    fs = 10000 #sampling frequency in cyc/deg, half this is nyquist.
    
    zfszintv = [22.2837, 130.8141] #zebrafish rf diameter interval in °
    macsfintv = np.array([0.5, 10]) #macaque rf spatial frequency selectivity interval in cyc/°
    macszintv = np.flip(1/np.array([macsfintv[0],2*macsfintv[1]])) #macaque rf diameter interval 
    #combined RF size interval among both species:
    #To be on the safe side, just increase the upper limit of the interval by 5*step size
    szintvt = np.array([np.min(np.append(macszintv,zfszintv)), np.max(np.append(macszintv,zfszintv))+5*sts])
    #Now find the upper and lower limits for the surround Gauss std:
    #Super str8forward since stdsur = sz/2
    #surround Gauss array:
    stdsurarr = np.arange(*(szintvt / 2), sts).round(4)
    
    #For a given surround Gaussian, center Gaussian can only be within a limited range of values, since center area can be only
    #between 1/4 and 1/2 of the whole RF area.
    #Keep the interval big to be on the safe side, center Gaussian cannot be bigger than surround Gaussian, 
    #and initially go until zero.
    LUT = dict() #lookup table dictionary, first dictionaries have key surround std, within those dicts are 
                 #dictionires with key frequency tuning and value is center std
    check = False #sanity check which should be done only once
    for stds in stdsurarr:
        LUT[stds] = dict()            
        stdcarr = np.unique(np.arange(sts, stds, sts).round(4))
        x = np.arange(-5*stds,5*stds, 1/fs) #generate DoG between +-5 std of surround Gaussian
        asd = []
        for stdc in stdcarr:
            DoG = gauss(x,stdc) - gauss(x, stds) #DoG array
            #Here you can stop the iteration if DoG does not fulfill some criteria
            rfsz = stds * 2 #receptive field diameter
            #In negative part of DoG (i.e x<0), where it first switches from negative to positive gives you the radius of the center
            #Find that x value and take absolute of it, which is the center radius. DO this on the positive side to overestimate the
            #center radius to be on the safe side
            cr = x[DoG>=0][-1] #center radius (overestimated by max 10stds*fs)
            #Center area should be bigger than 1/4 of total area, and smaller than 1/2 of total area
            #In other words, r/sqrt(2) >= cr >= r/2
            ar = (cr**2/(rfsz/2)**2) #center and whole rf area ratio
            if ar < 0.25: 
                print('center std=%.3f, surround std=%.3f'%(stdc,stds))
                print(r'Center of the DoG is too small' + '\n' + 
                      r'Center radius : %.2f' %cr + '\n' + 'RF radius : %.2f' %(rfsz/2) + '\n' +  
                      r'Center/total area = %.3f' %(cr**2/(rfsz/2)**2) + '\n')  
                asd.append((cr**2/(rfsz/2)**2))
                continue
            elif ar > 0.5:
                print('center std=%.3f, surround std=%.3f'%(stdc,stds))
                print(r'Center of the DoG is too large' + '\n' + 
                      r'Center radius : %.2f' %cr + '\n' + 'RF radius : %.2f' %(rfsz/2) + '\n' +  
                      r'Center/total area = %.3f' %(cr**2/(rfsz/2)**2) + '\n')  
                asd.append((cr**2/(rfsz/2)**2))
                break
            
            #calculate the frequency tuning
            N = len(x)
            fftdog = 2/N * np.abs(fft(DoG, n=N)[:N//2])
            frdog = fftfreq(n=len(DoG), d=(np.diff(x))[0])[:N//2]
            
            #sanity check
            if check == False:
                sf = np.max(frdog)
                sx = np.arange(0, 1, 1/fs)
                sinw = np.sin(2*np.pi*sx*sf)
                fftsin = np.abs(fft(sinw, n=len(sx))[:len(sx)//2])
                frsin = fftfreq(n=len(sx), d=np.diff(sx)[0])[:len(sx)//2]

                if frsin[fftsin==np.max(fftsin)][0] <= sf-0-5:
                    print('Sanity check failed!')
                    print('intended=%.3f, achieved=%.3f' %(sf, frsin[fftsin==np.max(fftsin)][0]) + '\n')
                    break
                else:
                    print('Successful sanity check!')
                    check = True
                
            #find max frequency
            ftuning = np.round(frdog[fftdog==np.max(fftdog)][0], 4)
            if verbose == True:
                print('std center=%.3f, std surround=%.3f, max f=%.3f, max possible f=%.3f' %(stdc,stds,ftuning, frdog[-1]))
            #update the table
            try:
                LUT[stds][ftuning]
                if type(LUT[stds][ftuning]) != list:
                    cstdsc = [LUT[stds][ftuning]] #current center stds with given surround std and max frequency
                if stdc not in cstdsc:
                    cstdsc.append(stdc)
                    LUT[stds][ftuning] = cstdsc
            except:
                LUT[stds][ftuning] = stdc
    #save lookup table
    import pickle
    fname = '..\modelling\DOG_LUT_sts=%.3f_fs=%d_szint=%s'%(sts,fs,np.array([list(LUT.keys())[0],list(LUT.keys())[-1]])*2)
    with open(fname, 'wb') as file:      
        # A new file will be created
        pickle.dump(LUT, file)
    
    #do color plot 
    sc = [] #standard deviation of center
    ss = [] #standard deviation of surround
    ftun = []
    for key in LUT.keys():
        cdict = LUT[key]
        for fkey in cdict.keys():
            cs = cdict[fkey]
            if type(cs) == np.float64:
                sc.append(cs)
                ss.append(key)
                ftun.append(fkey)
            else:
                for c in cs:
                    sc.append(c)
                    ss.append(key)
                    ftun.append(fkey)
    
    perc = 99.5
    fig, ax = plt.subplots()
    sca = ax.scatter(sc,ss,c=ftun, cmap='jet', norm=colors.LogNorm(), vmax=np.percentile(ftun,perc))
    ax.set_title('DoG lookup-table results')
    ax.set_xlabel(r'$\gamma_{c}$')
    ax.set_ylabel(r'$\gamma_{s}$')
    cb = plt.colorbar(sca, extend='max', format=ticker.ScalarFormatter())
    cb.minorticks_off()
    cb.set_ticks(np.logspace(np.log10(np.min(ftun)), np.log10(np.percentile(ftun, perc)), 15))
    cb.set_label(r'Spatial frequency [cyc/°]')
    cb.ax.yaxis.set_label_position('left')
    plt.tight_layout()