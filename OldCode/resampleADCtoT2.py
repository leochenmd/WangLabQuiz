#########################
# Registers selected ADC and T2images
# 1. Read corresponding ADC as ITK
# 3. Rigid/rotation (6 DOF) transformation of DWI (moving) onto T2 (fixed)
# 4. Adaptive stochastic gradient descent
# 5. Linear interpolation (vs. bspline 3rd degree)
# 6. Save transformed DWI as numpy/ITK
# 7. The original coordinates will still be based on T2
# 8. Threshold based on transformation distance and/or min. mutual information

#%env SITK_SHOW_COMMAND '/Applications/Slicer.app/Contents/MacOS/Slicer'

import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk

from Leofunctions import removedashandspace
from Leofunctions import readxlslist


# Resample to 64 x 128 x 128
# pixel sizes 1 x 1 x 1.5

# set the direction to be the same as original scan
# set pixel spacing to be 1 x 1 x 1.5

# find the center of the scan

# make that the center of the new image
# calculate the origin (top left corner), make that the origin


hdpath = 'C:\\ProcessedFusionImages'
mhadir = os.path.join(hdpath, 'StanfordT2ADC')
postdir = os.path.join(hdpath, 'StanfordT2ADCvnet')  #StanfordT2ADCresampled

os.chdir(mhadir)

searchstr = '*T2.mha'
onlyT2files = [f for f in glob.glob(searchstr)]

#T2file = onlyT2files[0]
#if 0 == 0:
for T2file in onlyT2files:
    accession = T2file.split('_')[0]

    T2filepath = T2file
    ADCfilepath = os.path.join(mhadir, accession + '_ADC.mha')

    T2img = sitk.ReadImage(T2filepath, sitk.sitkFloat32)
    ADCimg = sitk.ReadImage(ADCfilepath, sitk.sitkFloat32)
    T2direction = T2img.GetDirection()

    T2center = T2img.TransformContinuousIndexToPhysicalPoint(np.array(T2img.GetSize()) / 2)


# Resample to 64 x 128 x 128, using T2 as the center
# pixel sizes 1 x 1 x 1.5

    fixed = sitk.Image(128, 128, 64, sitk.sitkFloat32)
    fixed.SetDirection(T2direction)
    fixed.SetSpacing([1, 1, 1.5])
    fixed.SetOrigin(T2center)
    topleft = fixed.TransformContinuousIndexToPhysicalPoint([-64, -64, -32])
    fixed.SetOrigin(topleft)


    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed)
    resample.SetInterpolator(sitk.sitkLinear)

    T2resampled = resample.Execute(T2img)
    ADCresampled = resample.Execute(ADCimg)

    writer = sitk.ImageFileWriter()
    writer.Execute(T2resampled, os.path.join(postdir, accession + '_T2resampled.mha'), True)
    writer.Execute(ADCresampled, os.path.join(postdir, accession + '_ADCresampled.mha'), True)
