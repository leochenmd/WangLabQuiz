import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt

#T2dir = '/data/prostate/postimages/PostImages2/'
#coredir = '/data/prostate/postimages/Labels/'

T2dir = '/Users/Leo/Desktop/Temp/labels'
coredir = '/Users/Leo/Desktop/Temp/'
augmentdir = '/Users/Leo/Desktop/Temp/labels'

    #T2file = accession + '_T2.mha'

T2file = '8099139_cores_r3_gAll.mha'

T2path = os.path.join(T2dir, T2file)

T2image = sitk.ReadImage(T2path, sitk.sitkFloat32)

print(T2image.GetDirection())
print(T2image.GetPixelIDTypeAsString())


npy = np.array(sitk.GetArrayFromImage(T2image))
print(np.max(npy))
print(np.min(npy))

npy.dump('/Users/Leo/Desktop/Temp/cores.pkl')
