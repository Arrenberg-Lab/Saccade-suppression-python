# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:08:35 2021

@author: Ibrahim Alperen Tunc
"""
import os 
import sys
import time
import subprocess
import numpy as np

#Run the simulations in the server

#From generate RFs
seeds = np.arange(666,676) #you will call this via pbs shell script at the same time
animals = ['zebrafish', 'macaque'] #sys.argv[2] #'zebrafish' #sys.argv[3] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = 2*np.floor(np.logspace(np.log10(5),np.log10(100),5))**2 #int(sys.argv[3]) #50 #sys.argv[4] #50 #sys argv
rsl = 5 #int(sys.argv[4]) #5 #sys.argv[5]
jsigma = 4 #int(sys.argv[5]) #4

#From simulation
shiftmag = 340 #float(sys.argv[6]) #340 #sys.argv[2] #340
startloc = -170 #int(sys.argv[7]) #-170
bwidth = 5 #int(sys.argv[8]) #5

#subprocess business
print(os.getcwd())

for seed in seeds:
    for nfilter in nfilters:
        for animal in animals:
            print('nfilter=%i, animal=%s, seed=%i \n'%(nfilter, animal, seed))
            start = time.time()
            #first step - generate RFs
            p = subprocess.Popen('python srv_generate_RFs.py %i %s %i %i %i'%(seed, animal, nfilter, rsl, jsigma))
            #wait for process to end
            p.wait()        
            end = time.time()
            print('RF generation : Process takes %g seconds \n' %(end-start))
            
            start = time.time()
            #second step - simulate
            p = subprocess.Popen('python srv_simulate_rfs_with_stimulus.py %i %s %i %i %i %i %i %i'%(seed, animal, nfilter,
                                                                                                         rsl, jsigma, shiftmag, 
                                                                                                         startloc, bwidth))
            #wait for process to end
            p.wait()        
            end = time.time()
            print('Simulation : Process takes %g seconds \n' %(end-start))
            
            start = time.time()
            #last step - analysis
            p = subprocess.Popen('python srv_motion_decoder.py %i %s %i %i %i %i %i %i'%(seed, animal, nfilter,  rsl, 
                                                                                             jsigma, shiftmag, startloc, bwidth))  
            #wait for process to end
            p.wait()        
            end = time.time()
            print('Motion decoding : Process takes %g seconds \n' %(end-start))
            
        