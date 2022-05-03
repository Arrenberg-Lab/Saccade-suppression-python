# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:41:31 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
#from pathlib import Path
import sys
#sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import tarfile
import os
from IPython import embed

#generate RF set for the given settings and save them to a file.

#initial Reichardt like detector implementation
#General parameters -> some of those will be sys argv arguments
seed = int(sys.argv[1]) #666
animal = sys.argv[2] #'zebrafish' sys argv, can be zebrafish or macaque
nfilters = int(sys.argv[3]) #50 #sys argv -> Double the amount of RF for macaque
rsl = int(sys.argv[4])
jsigma = int(sys.argv[5])

np.random.seed(seed)

#Prepare species parameters
if animal == 'zebrafish':
    m = False #macaque is false
    speciesparams = hlp.generate_species_parameters(nfilters)
    params = speciesparams.zebrafish_updated()
    img = np.zeros(np.array([180,360])*rsl)
elif animal == 'macaque': #needs update, macaque part not yet done, e.g. visual field has to be limited to +-90° azimuth
    m = True #macaque is True    
    nfilters = int(np.ceil(np.sqrt(nfilters))**2)    
    speciesparams = hlp.generate_species_parameters(nfilters)
    params = speciesparams.macaque_updated()
    img = np.zeros(np.array([180,180])*rsl) #first dim elevation, second one azimuth


"""
#Code snnippet for RF parameter distributions
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2)
fig.suptitle('Parameter histograms')
nfilters = 3698 #50 #sys argv -> Double the amount of RF for macaque

axs[0,0].set_ylabel('Density')
axs[1,0].set_xlabel('Spatial frequency [cyc/°]')
axs[1,1].set_xlabel('RF diameter [°]')
axs[0,0].set_title('Spatial frequency')
axs[0,1].set_title('Size')


tits = ['Zebrafish', 'Macaque']
for i, an in enumerate(['zebrafish', 'macaque']):
    axx = fig.add_subplot(2,1,i+1)
    axx.set_title(tits[i], y=1.1)
    axx.axis('off')
    animal = an
    if an == 'zebrafish':
        speciesparams = hlp.generate_species_parameters(nfilters)
        params = speciesparams.zebrafish_updated()
    else:
         nfilters = int(np.ceil(np.sqrt(nfilters))**2)    
         speciesparams = hlp.generate_species_parameters(nfilters)
         params = speciesparams.macaque_updated()        
    sf = np.array(params)[:,0]
    sz = np.array(params)[:,1]
    axs[i,0].hist(sf, bins=100, density=True)    
    axs[i,1].hist(sz, bins=100, density=True)    
"""

#shuffle rf locations
__, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, params=params, 
                                                      macaque=m)


#phase shifted second set of filters with all the same parameters except with a phase shift as much as rf radius
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

params = np.squeeze(params)

#save directory
sdir = r'simulation/RFs/%s/seed_%i_nfilters_%06i_rsl_%02i_jsigma_%i'%(animal,seed,nfilters,rsl,jsigma)

try:
    os.makedirs(sdir)
except:
    pass

print('RF generation for %s, nfilters=%i, seed=%i'%(animal, nfilters, seed))
for i in range(len(params)):
    rfarr, rfcentcar = hlp.gaussian_rf_geog(params[i,0], params[i,1]/2, fltcenters[i,0], fltcenters[i,1], rsl,
                                            macaque=m)
    rfarr2, rfcentcar2 = hlp.gaussian_rf_geog(params[i,0], params[i,1]/2, fltcenters2[i,0], fltcenters2[i,1], rsl,
                                              macaque=m)
    fn = '/RF_%06i.npz'%i
    np.savez(sdir+fn, rfarr=rfarr, rfarr2=rfarr2, rfcent=fltcenters[i], rfcent2=fltcenters2[i],
                      rfcentcar=rfcentcar, rfcentcar2=rfcentcar2, rfpars=np.array([params[i,0], params[i,1]]))
    print('RF pairs %i is saved.'%i)    

#save all to a tar file
with tarfile.open(sdir+'.tar', "w") as tar_handle:
        for root, dirs, files in os.walk(sdir):
            for file in files:
                tar_handle.add(os.path.join(root, file))
                os.remove(os.path.join(root, file))
os.rmdir(sdir)
print('Tar file is created.')    