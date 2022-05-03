# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:37:43 2022

@author: Ibrahim Alperen Tunc
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs_ as hlp
import tarfile
import os
from IPython import embed
import matplotlib.pyplot as plt 

#Generate for zebrafish (and later for macaque) as much RFs as possible and save them for future use
#REDUCE THE THING TO 1D -> LEAVE OUT ELEVATION

seed = 666
animal = 'zebrafish' #sys.argv[2] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = 500 #int(sys.argv[3]) #50 #sys argv -> Double the amount of RF for macaque
rsl = 5 #int(sys.argv[4])

np.random.seed(seed)

#Prepare species parameters
if animal == 'zebrafish':
    m = False #macaque is false
    speciesparams = hlp.generate_species_parameters(nfilters)
    params = speciesparams.zebrafish_updated()
    azilims = [-180, 180] #azimuth limits
    
    
elif animal == 'macaque': #needs update, macaque part not yet done, e.g. visual field has to be limited to +-90° azimuth
    m = True #macaque is True    
    nfilters = int(np.ceil(np.sqrt(nfilters))**2)    
    speciesparams = hlp.generate_species_parameters(nfilters)
    params = speciesparams.macaque_updated()
    azilims = [-90, 90] #azimuth limits

elelims = [-90, 90] #elevation limits the same for both species


params = np.delete(np.squeeze(params), 2, axis=1)
#NEW THING WITH RANDOMIZATION -> CHOOSE XYZ FIRST FROM RANDOM GAUSS.
xyz = np.random.multivariate_normal([0]*3, np.identity(3), nfilters).T
_, az, ele = hlp.coordinate_transformations.car2geo(xyz[0], xyz[1], xyz[2])
fltcenters = (np.array(list(zip(180+az, 90-ele)), dtype=float) * rsl).astype(int)

#phase shifted second set of filters with all the same parameters except with a phase shift as much as rf radius
#Just keep this in case, you might omit this and choose n random pairs for decoding algorithm.
fltcenters2 = fltcenters.copy()
for idx, param in enumerate(params):
    rds = param[1]*rsl/2 #RF radius in pixels
    fltcenters2[idx,0] += rds
if animal == 'zebrafish':
    #loop around the index (azimuth) if it exceeds current max index
    fltcenters2[:,0][fltcenters2[:,0]>=360*rsl] -= 360*rsl
else:
    #For macaque, if value exceeds +90 azimuth, set the second filter center location to +90 azimuth.
    fltcenters2[:,0][fltcenters2[:,0]>=180*rsl] = 180*rsl - 1 #-1 to get to the last index.

sdir = r'simulation/RFs/%s/seed_%i_nfilters_%06i_rsl_%02i'%(animal,seed,nfilters,rsl)

try:
    os.makedirs(sdir)
except:
    pass


#do at the same time coverage map
covm = np.zeros(np.array([180,360])*rsl) #coverage map for first RFs
covm2 = np.zeros(np.array([180,360])*rsl) #coverage map for phase shifted RFs

print('RF generation for %s, nfilters=%i, seed=%i'%(animal, nfilters, seed))
for i in range(len(params)):
    rfarr, rfcentcar = hlp.gaussian_rf_geog(params[i,0], params[i,1]/2, fltcenters[i,0], fltcenters[i,1], rsl,
                                            macaque=m)
    rfarr2, rfcentcar2 = hlp.gaussian_rf_geog(params[i,0], params[i,1]/2, fltcenters2[i,0], fltcenters2[i,1], rsl,
                                              macaque=m)
    
    covm[rfarr!=0] += 1
    covm2[rfarr2!=0] += 1

    
    fn = '/RF_%06i.npz'%i
    np.savez(sdir+fn, rfarr=rfarr, rfarr2=rfarr2, rfcent=fltcenters[i], rfcent2=fltcenters2[i],
                      rfcentcar=rfcentcar, rfcentcar2=rfcentcar2, rfpars=params[i, :])
    print('RF pairs %i is saved.'%i)    


"""
#save all to a tar file
with tarfile.open(sdir+'.tar', "w") as tar_handle: #try the gzip compression now.
        for root, dirs, files in os.walk(sdir):
            for file in files:
                tar_handle.add(os.path.join(root, file))
                os.remove(os.path.join(root, file))
os.rmdir(sdir)
print('Tar file is created.')
"""