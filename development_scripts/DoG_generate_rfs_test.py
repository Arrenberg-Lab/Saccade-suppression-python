# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:23:13 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp

#Develop the new (19.08, see TODO 16.08) RF set generation algorithm.
#General parameters
nfilters = 200 #10082
rsl = 5
jsigma = 4
nsimul = 10
tdur = 100
shiftmag = 20

#generate the RF parameters
speciesparams = hlp.generate_species_parameters(nfilters)
zfparams = speciesparams.zebrafish_updated()
img = np.zeros(np.array([180,360])*rsl)

__, fltcenters = hlp.florian_model_shuffle_parameters(nfilters, rsl, None, None, None, jsigma, img, params=zfparams)

zfparams = np.squeeze(zfparams)

#Code for making the receptive fields
#choose first value to begin with, then you will loop over all in the next step, finally you will check for macaque
#rfarr, rfcentcar = hlp.gaussian_rf_geog(zfparams[i,0], zfparams[i,1]/2, *fltcenters[i], rsl) this is the way you generate in function
#Recycle from gaussian_rf_geog
gtis = np.where((zfparams[:,0] < np.sqrt(2)/zfparams[:,1])==False)[0] #test index to get a Gaussian RF
for gti in gtis:
    (azidx,elevidx) = fltcenters[gti]
    (sf, sz) = zfparams[gti,:-1]
    maskparams = [*fltcenters[gti], zfparams[gti,1], rsl, 1]
    rfarr, rfcent, rfcentcar = hlp.crop_rf_geographic(*maskparams)
    
    #Generate first the Gaussian in the whole visual field
    gaz, gel = np.meshgrid(np.linspace(-180,180,np.int(360*rsl)), np.linspace(90,-90,np.int(180*rsl))) 
    #visual field in cartesian
    x, y, z = hlp.coordinate_transformations.geo2car(np.ones(gaz.shape), gaz, gel)
    #determine Gauss center, which is shifted in azimuth by as much as ph.
    #gauss center indices in geographic
    gcent = np.array([gaz[elevidx,azidx], gel[elevidx,azidx]])
    #gauss centers in cartesian
    gcentx = x[(gaz == gcent[0]) & (gel == gcent[1])]
    gcenty = y[(gaz == gcent[0]) & (gel == gcent[1])]
    gcentz = z[(gaz == gcent[0]) & (gel == gcent[1])]
    gcentcar = np.squeeze(np.array([gcentx, gcenty, gcentz]))
    
    #new part of the code: The profile is Gaussian depending on the ratio between RF size and spatial frequency selectivity:
    if sf < 1/(np.sqrt(2)*sz): #CORRECTED from sqrt(2)/sz
        #DoG: here will come the later stuff in generating the receptive field. This is the hard part so lets do Gauss first.
        #surround sigma: +-2 standard deviations cover the whole RF: 2stdsur = r <=> stdsur = sz/4
        stdsur = sz/4
        #determine center gauss: sf = sqrt(2ln(stdsur^2/stdcent^2)/(stdsur^2-stdcent^2))
        
        
        
    else:
        wl = 1/sf #wavelength
        #Adjust the Gaussian in such a way, that whole positive part of the preferred stimulus wavelength covers +-1.5 sigma
        #=> sigma*1.5 = wl/4 (wl/2 for positive part, standard deviation takes half width of the distribution hence wl/4).
        #i.e. stdg = wl/6
        stdg = wl/6 
        if stdg > 180:
            """
            if the standard deviation is bigger than 180°, the equation for stdcar does not properly work
            This is because sin(90) <= sin(90+n) for any n, meaning that a bigger angle than 180 leads to a yet smaller 
            (and in cases even to a negative!) standard deviation value. In this case, first find out how many full 180°s fit
            in the stdg (using // operator) and multiply the sine value by that. Then find the remainder and add it to stdcar
            using the sin equation.
            """
            numreps = (stdg//180) #floor division to find how many times the biggest value is added
            remainder = stdg - 180*numreps #remaining part to be added
            stdcar = (np.sin(np.deg2rad(180/2)) * 2) * numreps + (np.sin(np.deg2rad(remainder/2)) * 2)
        
        else:
            stdcar = np.sin(np.deg2rad(stdg/2)) * 2 #standard deviation in cartesian
            
        """
        Idea behind stdcar: Imagine 3D Gaussian on a sphere surface in its contour plots: each contour is a ring around the 
        sphere surface. One of these circles correspond to the deviation from center as much as one standard deviation.
        The radius of that circle equals to the standard deviation of the Gaussian, but we know its vaule in degrees in
        geographical coordinates. Thus, we can reduce this problem to the same problem we had, namely converting the radius
        of circle (projected on a sphere surface) in degrees from geographical coordinates to cartesian coordinates. 
        """
        pos = np.dstack((x,y,z)) #generate the array containing each point's xyz coordinates. 
                                 #1st dim along azimuth, 2nd along elevation and last dim xyz (in that order)
                                 
        rv = hlp.multivariate_normal(gcentcar ,np.diag(np.tile(stdcar,3))**2)
        gaussfield = rv.pdf(pos)
        gaussfield[rfarr==False] = 0
        gaussfield /= np.sum(gaussfield)  
        print("RF diameter: %.2f°" %sz)
        fig, ax = plt.subplots()
        ax.imshow(gaussfield, origin='lower', extent=[-180,180,-90,90], cmap='jet')
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        plt.tight_layout()        
        while True:
            if plt.waitforbuttonpress():
                plt.close()
                break
        a = input('c to continue, q to quit \n')
        if a == 'c':
            kill = False
        elif a == 'q':
            kill = True
        if kill == True:
            break