# Loads segmentation dicom file

import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom

import re
import SimpleITK as sitk


segdir = 'C:\\ProcessedFusionImages\\stanfordsegmha'

def bbox_3D(img):

    z = np.any(img, axis=(1, 2))    #z
    c = np.any(img, axis=(0, 2))    #y
    r = np.any(img, axis=(0, 1))    #x

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    #x min max, y min max, z min max
    return rmin, rmax, cmin, cmax, zmin, zmax




outdata = []

os.chdir(segdir)

segfiles = glob.glob('*seg.mha')

for i, segfile in enumerate(segfiles):
    if 0 == 0:
        accession = segfile.split('_')[0]

        segmhapath = os.path.join(segdir, str(accession) + '_seg.mha')

        segmha = sitk.ReadImage(segmhapath, sitk.sitkUInt8)
        segnpy = sitk.GetArrayFromImage(segmha)

        [xmin, xmax, ymin, ymax, zmin, zmax] = bbox_3D(segnpy)

        width = xmax - xmin + 1
        height = ymax - ymin + 1
        slices = zmax - zmin + 1

        size = segmha.GetSize()

        outdata.append(dict(zip(['accession', 'size', 'boxwidth', 'boxheight', 'slices', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'], [accession, size, width, height, slices, xmin, xmax, ymin, ymax, zmin, zmax])))

        if i % 10 == 0:
            print(i)

outdf = pd.DataFrame(outdata)
outdf = outdf[['accession', 'size', 'boxwidth', 'boxheight', 'slices', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']]

outdf.to_excel('C:\\ProcessedFusionImages\\boundingbox.xls')
