# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:49:22 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.stats import multivariate_normal

#Develop the Gaussian RF profile to DoG with SF selectivity.
#General parameters
rsl = 20
azidx = 180*rsl
elevidx = 90*rsl
sz = 10 #degrees
radiusfac = 1
sf = 10 #cycles per degree
ph = 0 #degrees, but keep at 0 for now.
maskparams = [azidx, elevidx, sz, rsl, radiusfac]
#generate the cirular mask
rfarr, rfcent, rfcentcar = hlp.crop_rf_geographic(*maskparams)
 
#Generate first the Gaussian in the whole visual field
#visual field in geographic
gaz, gel = np.meshgrid(np.linspace(-180,180,np.int(360*rsl)), np.linspace(90,-90,np.int(180*rsl))) 
#visual field in cartesian
x, y, z = hlp.coordinate_transformations.geo2car(np.ones(gaz.shape), gaz, gel)
#determine Gauss center, which is shifted in azimuth by as much as ph.
azidxg = azidx+ph*rsl #azimuth index for the multivariate gauss
#gauss center indices in geographic
gcent = np.array([gaz[elevidx,azidxg], gel[elevidx,azidxg]])
#gauss centers in cartesian
gcentx = x[(gaz == gcent[0]) & (gel == gcent[1])]
gcenty = y[(gaz == gcent[0]) & (gel == gcent[1])]
gcentz = z[(gaz == gcent[0]) & (gel == gcent[1])]
gcentcar = np.squeeze(np.array([gcentx, gcenty, gcentz]))
#convert the sf to corresponding gaussian 
wl = 1/sf #spatial frequency wavelength in degrees
stdg = wl/4 #set the standard deviation to wl/2, this is in degrees.
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
                         
rv = multivariate_normal(gcentcar ,np.diag(np.tile(stdcar,3))**2)
center = rv.pdf(pos)   
center /= np.sum(center)     

#Try with the DoG approach now
"""
1st possibility:
    For the spatial frequency selectivity, we want the surround to cover the whole negative parts of the wavelength.
2nd possibility:
    Center Gauss sigma determines spatial frequency selectivity. For surround, set sigma to 1/5 of the radius
"""
"""
#1st possibility
stdgsur = 3*stdg #surround standard deviation should cover 3/4 of the wavelength.
"""
#2nd possibility
stdgsur = sz/5 #5 std covers the whole RF surround

stdcarsur = np.sin(np.deg2rad(stdgsur/2)) * 2 #standard deviation in cartesian
surround = multivariate_normal(gcentcar ,np.diag(np.tile(stdcarsur,3))**2).pdf(pos)
surround /= np.sum(surround)
gaussfield = center-surround
filterimg = gaussfield.copy()
filterimg[rfarr==False] = 0

fig, ax = plt.subplots(1,1)
ax.imshow(filterimg, origin='lower', extent=[-180,180,-90,90], cmap='jet')
ax.set_xlabel('Azimuth [°]')
ax.set_ylabel('Elevation [°]')
ax.set_xticks(np.linspace(-180,180,9))
ax.set_yticks(np.linspace(-90,90,5))
ax.set_title('Example DoG receptive field, size=%.1f, sf=%.1f' %(sz, sf))
