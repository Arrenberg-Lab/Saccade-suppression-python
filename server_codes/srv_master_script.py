# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:44:26 2021

@author: Ibrahim Alperen Tunc
"""
import os 
import sys
import subprocess
import time
import numpy as np
#master script for the server simulations

#From generate RFs
seeds = np.arange(666,667) #int(sys.argv[1]) #666-676
animals = ['zebrafish', 'macaque'] #sys.argv[2] #'zebrafish' #sys.argv[3] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = 2*np.floor(np.logspace(np.log10(10),np.log10(72),5))**2 #int(sys.argv[3]) #50 #sys.argv[4] #50 #sys argv
rsl = 5 #int(sys.argv[4]) #5 #sys.argv[5]
jsigma = 4 #int(sys.argv[5]) #4

#From simulation
shiftmag = 340 #float(sys.argv[6]) #340 #sys.argv[2] #340
startloc = -170 #int(sys.argv[7]) #-170
bwidth = 5 #int(sys.argv[8]) #5

#subprocess business
print(os.getcwd())
for animal in animals[1:]:
    print('Animal : %s \n' %animal)
    for nfilter in nfilters[2:]:
        print('nfilter=%i \n'%nfilter)
        start = time.time()
        #first step - generate RFs
        ps = [] #processes list to wait until finished
        for seed in seeds:
            p = subprocess.Popen('python srv_generate_RFs.py %i %s %i %i %i'%(seed, animal, nfilter, rsl, jsigma))
            ps.append(p)
        #wait for each to end
        for pp in ps:
            pp.wait()        
        end = time.time()
        print('RF generation : %i parallel processes take %g seconds \n' %(len(seeds), end-start))
        
        start = time.time()
        #second step - simulate
        ps = [] #processes list to wait until finished
        for seed in seeds:
            p = subprocess.Popen('python srv_simulate_rfs_with_stimulus.py %i %s %i %i %i %i %i %i'%(seed, animal, nfilter,
                                                                                                     rsl, jsigma, shiftmag, 
                                                                                                     startloc, bwidth))
            ps.append(p)
        #wait for each to end
        for pp in ps:
            pp.wait()        
        end = time.time()
        print('Simulation : %i parallel processes take %g seconds \n' %(len(seeds), end-start))
        
        start = time.time()
        #last step - analysis
        ps = [] #processes list to wait until finished
        for seed in seeds:
            p = subprocess.Popen('python srv_motion_decoder.py %i %s %i %i %i %i %i %i'%(seed, animal, nfilter,  rsl, 
                                                                                         jsigma, shiftmag, startloc, bwidth))  
            ps.append(p)
        #wait for each to end
        for pp in ps:
            pp.wait()        
        end = time.time()
        print('Motion decoding : %i parallel processes take %g seconds \n' %(len(seeds), end-start))
            
        