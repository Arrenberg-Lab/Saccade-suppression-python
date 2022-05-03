# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:36:20 2021

@author: Ibrahim Alperen Tunc
"""
import os 
import sys
import subprocess

#test the subprocess stuff
names = ['test', 'ibo', 'srv']
printnum = [5] * len(names)

for n,pn in zip(names,printnum):
    #This runs in parallel!
    subprocess.Popen("python test.py %s %i"%(n,pn))
    
