# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:43:50 2021

@author: Ibrahim Alperen Tunc
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 01:40:48 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import tarfile
import shutil
import zf_helper_funcs as hlp

#generate the stimulus, save it in a dedicated folder
sdir = r'../sensory_simuls/stimuli/stimulus_rsl_%02i_shiftmag_%03i'
try: 
    os.makedirs(sdir)
except:
    pass


#General parameters -> these will be sys argv soon
rsl = 5
shiftmag = 340
#ensure shift is one pixel per frame -> frame per frame shift is determined by rsl
tdur = shiftmag/200*rsl/100#*1000 #200 is fps (default value), 1000 is to convert the duration to ms. This gives a duration
                             #which allows 1 pixel shift per frame for the given settings.
shiftdir = 'right'
#zfparams = speciesparams.zebrafish()
img = np.zeros(np.array([180,360])*rsl)

#stimulus
stim = hlp.cylindrical_random_pix_stimulus(rsl, shiftmag, tdur, maxelev=70)
stim.test_stimulus(5, -170)

for j in range(stim.frametot+1):
    print('Current frame=%i, %.2f%% complete'%(j, j/(stim.frametot+1)*100))
    stimulus, __ , __= stim.move_stimulus(j, shiftdir) #shift in positive
    sname = '/frame%04i.npy'%(j)
    np.save(sdir+sname, stimulus)

#convert all to a tar file
with tarfile.open(sdir+'.tar.gz', "w:gz") as tar_handle:
    for root, dirs, files in os.walk(sdir):
        for file in files:
            tar_handle.add(os.path.join(root, file))
            os.remove(os.path.join(root, file)) 
print('Stimulus converted to tar file!')
shutil.rmtree(sdir)
print('Previous directory containing stimulus files is deleted!')

