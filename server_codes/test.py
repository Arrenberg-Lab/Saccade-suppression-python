# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:46:11 2021

@author: Ibrahim Alperen Tunc
"""
import os
import sys
import time

#test script

name = sys.argv[1]
printnum = int(sys.argv[2])

fd = r'testfolder'
try: 
    os.makedirs(fd)
except:
    pass

with open(os.path.join(fd,'%s.txt'%name), 'w') as sys.stdout:
    for i in range(printnum):
        print(name)
    time.sleep(20)
    print('Finished')
    
