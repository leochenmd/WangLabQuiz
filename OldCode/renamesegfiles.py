import numpy as np
import pandas as pd

import pydicom
import os
import glob
from shutil import copyfile


hdpath = '/volumes/Seagate Backup Plus Drive/prostateml/'
fusionimagesdir = os.path.join(hdpath, 'fusionimages')
segdir = os.path.join(hdpath, 'segdicoms')

os.chdir(fusionimagesdir)

datefolders = [f.path for f in os.scandir(fusionimagesdir) if f.is_dir()]

#datepath = datefolders[0]
#if 0 == 0:
for datepath in datefolders:
    os.chdir(datepath)
    searchstr = 'SEG*'
    onlysegfiles = [f for f in glob.glob(searchstr)]

    for segfile in onlysegfiles:
        try:
            segdcm = pydicom.read_file(segfile)
            date = segdcm.StudyDate
            accession = segdcm.AccessionNumber

            outfile = accession + '_' + date + '.seg'
            outpath = os.path.join(segdir, outfile)

            copyfile(segfile, outpath)
        except:
            print('error reading file in ' + datepath)
