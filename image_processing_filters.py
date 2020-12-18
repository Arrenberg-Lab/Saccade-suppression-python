# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:21:37 2020

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import zf_helper_funcs as hlp
from zf_helper_funcs import rt
from imageio import imread
from skimage.color import rgb2gray

arr = imread(r'D:\ALPEREN\junk\IMG_9380.jpg') # 800x634x3 array
grarr = rgb2gray(arr)

#try a gabor kernel
gkernparams = {'gwl': 10,
               'theta': np.deg2rad(45),
               'sigma_x' : 10,
               'sigma_y' : 10,
               'offset' : np.deg2rad(0),
               'n_stds': 3}
kerngab, gaborr, gabori = hlp.gabor_image_filter(grarr, **gkernparams)

fig, axs = plt.subplots(1,3)
fig.suptitle('Gabor filtering of an image', size=25)
axs[0].imshow(np.real(kerngab), cmap='gray')
axs[1].imshow(gaborr, cmap='gray')
axs[2].imshow(grarr, cmap='gray')
axs[0].set_title('Gabor kernel')
axs[1].set_title('Filtered image')
axs[2].set_title('Raw image')

gkernparams['theta'] = np.deg2rad(0)
kerngab, gaborr, gabori = hlp.gabor_image_filter(grarr, **gkernparams)

fig, axs = plt.subplots(1,3)
fig.suptitle('Gabor filtering of an image', size=25)
axs[0].imshow(np.real(kerngab), cmap='gray')
axs[1].imshow(gaborr, cmap='gray')
axs[2].imshow(grarr, cmap='gray')
axs[0].set_title('Gabor kernel')
axs[1].set_title('Filtered image')
axs[2].set_title('Raw image')

#now try Gaussian
sigma = 5
gaussimg = hlp.gaussian_image_filter(grarr, sigma)
fig, axs = plt.subplots(1,2)
fig.suptitle('Gaussian filtering of an image', size=25)
axs[0].imshow(gaussimg, cmap='gray')
axs[1].imshow(grarr, cmap='gray')
axs[0].set_title('Filtered image')
axs[1].set_title('Raw image')