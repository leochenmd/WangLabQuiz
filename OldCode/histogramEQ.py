import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    hist, bins = np.histogram(image.flatten(), number_bins, normed=True)

    cdf = hist.cumsum() # cumulative distribution function
    #cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_normalized = cdf * hist.max()/ cdf.max() # normalize

    # use linear interpolation of cdf to find new pixel values
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = np.round(image_equalized).astype('uint8')

    return image_equalized.reshape(image.shape)

writer = sitk.ImageFileWriter()

T2dir = '/Users/Leo/Desktop/Temp/'
coredir = '/Users/Leo/Desktop/Temp/'
augmentdir = '/Users/Leo/Desktop/Temp/labels'


T2file = '8099139_T2.mha'
T2path = os.path.join(T2dir, T2file)
T2image = sitk.ReadImage(T2path, sitk.sitkFloat32)

origin = T2image.GetOrigin()
direction = T2image.GetDirection()
spacing = T2image.GetSpacing()

T2npy = sitk.GetArrayFromImage(T2image)

T2npynorm = image_histogram_equalization(T2npy)

normT2image = sitk.GetImageFromArray(T2npynorm)
normT2image.SetOrigin(origin)
normT2image.SetDirection(direction)
normT2image.SetSpacing(spacing)

writer.SetUseCompression(True)
writer.Execute(normT2image, os.path.join(T2dir, '8099139_T2_norm.mha'), True)
