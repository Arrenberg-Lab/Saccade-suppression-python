# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:40:35 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
#from pathlib import Path
import sys
#sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import tarfile
import os
from io import BytesIO

#simulate the given RF set

#General parameters -> first argument in sys argv is rsl, second is shift magnitude
#From generate_RFs
seed = int(sys.argv[1]) #sys.argv[1]
animal = sys.argv[2] #'zebrafish' #sys.argv[3] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = int(sys.argv[3]) #sys.argv[4] #50 #sys argv
rsl = int(sys.argv[4]) #sys.argv[5]
jsigma = int(sys.argv[5]) #sys.argv[6]
#Required here
shiftmag = float(sys.argv[6]) #340 #sys.argv[2] #340
startloc = float(sys.argv[7]) #-170 #start location of the test stimulus in degrees
bwidth = int(sys.argv[8]) #bar width of the test stimulus in degrees

if animal == 'macaque':
    #if macaque, reduce the shift magnitude (along azimuth) by 180° and shift the start location by +90°
    #since shifts will be all the same and whole visual field, this is okay.
    shiftmag -= 180
    startloc += 90
    nfilters = int(np.ceil(np.sqrt(nfilters))**2) #choose the closest n^2 value
    
#Fixed parameters
test = True #if False, random dot stimulus is used, else a bar sweeps through. False is not yet implemented. 
#ensure shift is one pixel per frame -> frame per frame shift is determined by rsl
tdur = shiftmag/200*rsl*1000 / 10 #200 is fps (default value), 1000 is to convert the duration to ms. This gives a duration
                                 #which allows 1 pixel shift per frame for the given settings.
                                 #For time considerations, divide by 10 (i.e. 10 pixel shift per frame = 2 degrees)
shiftdir = 'right'

img = np.zeros(np.array([180,360])*rsl)

if test == True:
    stm = 'test'
else:
    stm == 'randomdot'

#save directory for all files
sdir = r'simulation/stimuli/%s/tdur_%05i_shiftmag_%3i_rsl_%02i_startloc_%i_barwidth_%i'%(stm, tdur, shiftmag, 
                                                                                         rsl, startloc, bwidth)

#save directory for the given RF set
fname = sdir + '/activities/%s/seed_%i_nfilters_%06i_jsigma_%i'%(animal, seed, nfilters, jsigma) 

try: 
    os.makedirs(fname)
except:
    pass

#stimulus
stim = hlp.cylindrical_random_pix_stimulus(rsl, shiftmag, tdur, maxelev=70)
stim.test_stimulus(bwidth, startloc)

#load the RFs tar file
fn = r'simulation/RFs/%s/seed_%i_nfilters_%06i_rsl_%02i_jsigma_%i.tar'%(animal,seed,nfilters,rsl,jsigma)
tf = tarfile.open(fn)
rfns = tf.getnames() #tarfile names

print('Running simulation for %s stimulus for animal %s, seed=%i, nfilters=%i'%(stm, animal, seed, nfilters))


#calculate the rf activities for the given stimulus.    
for i, n in enumerate(rfns):    
    print('Current RF number=%i, %.2f%% complete'%(i, i/(len(rfns))*100))

    #preallocate the activity array for the given RF and given stimulus frame
    rfacts = np.zeros(stim.frametot+1)
    rfacts2 = np.zeros(stim.frametot+1)
    rfactsnormed = np.zeros(stim.frametot+1)
    rfactsnormed2 = np.zeros(stim.frametot+1)
    
    #load the RF parameters
    array_file = BytesIO()
    array_file.write(tf.extractfile(n).read())
    array_file.seek(0)
    rfdat = np.load(array_file, allow_pickle=True)
    rfarr = rfdat['rfarr']
    rfarr2 = rfdat['rfarr2']
    rfdiam = rfdat['rfpars'][1]
    rfcent = rfdat['rfcent'] 
    rfcent2 = rfdat['rfcent2'] 
    for j in range(stim.frametot+1):    
        stimulus, __ , __= stim.move_stimulus(j, shiftdir) #shift in positive
        if j == 0:
            sname = '/frame%04i.npy'%(j)        
            np.save(sdir+sname, stimulus)        
        #Get the RF activity
        if animal == 'macaque':
            stimulus = stimulus[:,90*rsl:(360-90)*rsl] #very dirty trick: clip stimulus to macaque visual field
        rfacts[j] = np.sum(rfarr[rfarr!=0]*stimulus[rfarr!=0])
        rfacts2[j] = np.sum(rfarr2[rfarr2!=0]*stimulus[rfarr2!=0])
        rfactsnormed[j] = rfacts[j] / (np.pi*(rfdiam/2)**2) #normed to unit area
        rfactsnormed2[j] = rfacts2[j] / (np.pi*(rfdiam/2)**2) #normed to unit area
    
    #save the arrays
    np.savez(fname+'/'+n[-13:-4]+'_frame_%04i'%(j) + '.npz', rfacts=rfacts, rfacts2=rfacts2, 
                                                            rfactsnormed=rfactsnormed, rfactsnormed2=rfactsnormed2,
                                                            rfcent=rfcent, rfcent2=rfcent2)

#save all to a tar file
with tarfile.open(fname+'.tar', "w") as tar_handle:
        for root, dirs, files in os.walk(fname):
            for file in files:
                tar_handle.add(os.path.join(root, file))
                os.remove(os.path.join(root, file))
os.rmdir(fname)
print('Tar file is created.')    