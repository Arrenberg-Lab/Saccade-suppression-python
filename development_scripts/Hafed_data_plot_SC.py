# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:36:11 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import zf_helper_funcs as hlp
import pandas as pd
loadpath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data'

#Hafed data plot the SC area. 
#DATA FROM https://www.sciencedirect.com/science/article/pii/S0960982219306104#app2

data = pd.read_excel(loadpath+'\macaque_SC.xlsx') #all in mm (i.e. 1000 mikrons)
plt3d = plt.figure().gca(projection='3d')
plt3d.plot(data['X'],data['Y'],data['Z'], '.', markersize=1)

#convert data to microns
data *= 1000

#find the distance between minimum and maximum points between each dimensions. Take for each dimension the min and max,
#max - min is the distance (ez!)

distances = np.ptp(np.array(data), axis=0) #very cool function https://numpy.org/doc/stable/reference/generated/numpy.ptp.html
print(distances) #order in xyz (although irrelevant)    