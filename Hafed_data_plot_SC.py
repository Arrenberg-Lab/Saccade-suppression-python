# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:36:11 2021

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
import pandas as pd
loadpath = r'D:\ALPEREN\Tübingen NB\Semester 3\Arrenberg\git\codes\data'

#Hafed data plot the SC area.
data = pd.read_excel(loadpath+'\macaque_SC.xlsx')
plt3d = plt.figure().gca(projection='3d')
plt3d.plot(data['X'],data['Y'],data['Z'], '.', markersize=1)
