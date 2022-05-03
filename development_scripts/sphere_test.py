# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:12:54 2021

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

#Test script for sphere and spherical projections

x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
z = np.linspace(-5,5,100)

xx,yy,zz = np.meshgrid(x,y,z)

#define the sphere initial parameters
cent = [0,0,0] #center coordinates in xyz
r = 1 #radius of the sphere

#sphere equation
sphmask = xx**2+yy**2+zz**2 <= r**2
xpoints = xx[sphmask]
ypoints = yy[sphmask]
zpoints = zz[sphmask]
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax.plot(xpoints,ypoints,zpoints, 'r.')
ax.set_title('Cartesian (initial)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#transform to geographic
r, az, el = hlp.coordinate_transformations.car2geo(xpoints,ypoints,zpoints)
ax = fig.add_subplot(132)
ax.plot(az,el, 'k.')
ax.set_title('Geographic')
ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')

#transform back to cartesian
xxx, yyy, zzz = hlp.coordinate_transformations.geo2car(r,az,el)
ax = fig.add_subplot(133, projection='3d')
ax.plot(xxx,yyy,zzz, 'r.')
ax.set_title('Cartesian (from geographic)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#SOLUTION IS EVEN SIMPLER! Choose a point in geographic (azimuth/elevation), convert to cartesian (r=1 so sphere) 
#and calculate circle coordinates there, then transform back to geographic. Point is you will have correspondence
#between cartesian-geographic and cylindirical.
rsl = 2 #pixel per degrees
geogarray = np.zeros(np.array([360,180])*rsl) #logical array of geographic coordinates
#geographic azimuth elevation coordinates
gaz, gel = np.meshgrid(np.linspace(-180,180,np.int(360*rsl)), np.linspace(90,-90,np.int(180*rsl))) 

r = np.ones(gaz.shape) #radius constant 1
cx, cy, cz = hlp.coordinate_transformations.geo2car(r, gaz, gel)#cartesian coordinates

"""
#test if this gives you sphere
cxf = cx.flatten()
cyf = cy.flatten()
czf = cz.flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(cxf,cyf,czf, '.')
#PERFECTO!
"""

grc = 90 #radius of the circle in degrees polar

circarr, cc, ccc = hlp.crop_rf_geographic(*(np.array((180,90)) * rsl), grc, rsl, radiusfac=1)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(circarr, extent=[-180,180,-90,90])
ax = fig.add_subplot(122, projection='3d')
ax.plot(cx.flatten(),cy.flatten(),cz.flatten(), 'k.', label='Whole field', alpha= 0.2)
ax.plot(cx[circarr],cy[circarr],cz[circarr], 'r.', label='Receptive field', alpha= 0.2)
ax.legend()

#cropping filter: first define the sine wave over all visual field, with the offset as much as the azimuth of the
#RF center (for now no orientation so no elevation offset) on top of which you implement the phase shift. Then
#you can use the mask (circarr) to crop the Gabor filter.
#some toy parameter values
ph = 0
sz = 40 #deg
sf = 0.1 #cyc/deg
elevidx, azidx = np.array((90,250)) * rsl
gccent = np.array([gaz[elevidx,azidx], gel[elevidx,azidx]]) #azimuth and elevation


#Generate first the rectangular Gabor in geographical coordinates
degrees = np.linspace(0,360, 360*rsl)
gaborsin = np.sin(2*np.pi * sf * degrees - gccent[0] + ph)
filterimg = np.tile(gaborsin,[180*rsl,1])

#Find the RF area
addidx = sz*rsl #index to be added to find the edge of the circle in geographic coordinates
#circle edge (one point shifted in elevation) in geographic coordinates
gcedge = np.array([gaz[elevidx,azidx], gel[elevidx+addidx,azidx]])
print(gccent,gcedge)
#find the center in cartesian
ccentx = cx[(gaz == gccent[0]) & (gel == gccent[1])]
ccenty = cy[(gaz == gccent[0]) & (gel == gccent[1])]
ccentz = cz[(gaz == gccent[0]) & (gel == gccent[1])]
#edge in cartesian
cedgex = cx[(gaz == gcedge[0]) & (gel == gcedge[1])]
cedgey = cy[(gaz == gcedge[0]) & (gel == gcedge[1])]
cedgez = cz[(gaz == gcedge[0]) & (gel == gcedge[1])]
#circle radius in cartesian
crc = np.sqrt((ccentx-cedgex)**2 + (ccenty-cedgey)**2 + (ccentz-cedgez)**2)

circarr = (cx-ccentx)**2 + (cy-ccenty)**2 + (cz-ccentz)**2 <= crc**2

filterimg[circarr==False] = 0

filterimg, __ = hlp.generate_RF_geographic(sf, sz, 45, 200, 50, rsl, radiusfac=1)

fig, ax = plt.subplots(1,1)
ax.imshow(filterimg, extent=[-180,180,-90,90])
ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')
ax.set_title('Azimuth %.1f Elevation %.1f' %(gccent[0], gccent[1]))


#Giulia Stimulus: first test if you messed up anything in implementation
cstim = hlp.cylindrical_random_pix_stimulus(shiftmag=20,tdur=70,rsl=rsl)

"""
fig, ax = plt.subplots()
plt.get_current_fig_manager().window.showMaximized()
for i in range(cstim.frametot):
    cf = cstim.move_stimulus(i, 'left')
    ax.imshow(cf, extent=[-168,168,-40,40])
    plt.pause(0.5)
#everything works as intended.
"""
#choose one frame to convert to spherical
exframe = cstim.move_stimulus(0, 'left')

"""
Maths behind cylindrical to spherical conversion: Azimuth stays the same. Radius is also known, based on that one 
can calculate the elevation angle using additionally z: elevation = arctan(z/r) since tan(elevation) = z/r.  
"""
azimutharr = np.linspace(-cstim.arenaaz, cstim.arenaaz, exframe.shape[1])
zarr = np.linspace(-cstim.maxz, cstim.maxz, exframe.shape[0])
azimuths, zs = np.meshgrid(azimutharr, zarr)
rs = np.ones(exframe.shape) * cstim.radius
elevations = np.rad2deg(np.arctan2(zs,rs))
maxelev = np.max(elevations)
#extend the final stimulus array to max azimuth elevation
finalarr = np.zeros(np.array([180,360])*rsl)
elidxs = rsl*np.array([np.int(90-maxelev), np.int(90+maxelev)]) #elevation indices for the stimulus
azidxs = rsl*np.array([np.int(180-cstim.arenaaz), np.int(180+cstim.arenaaz)]) #azimuth indices for the stimulus
finalarr[elidxs[0]:elidxs[1], azidxs[0]:azidxs[1]] = exframe
fig, ax = plt.subplots(1,1)
ax.imshow(finalarr, extent=[-180,180,-90,90])


#NOW YOU CAN GENERATE THE GABOR FILTERS!