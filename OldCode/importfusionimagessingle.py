#########################################
# Loads 'FusionImages' directory
# 1. Read all series in directory
# 2. Saves .xls file containing labels, whether its T2/ADC/DWI


import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk


#hdpath = 'C:\\'
fusionimagesdir = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML/PATDBASE/10174019/20150331/090506'
#os.path.join(hdpath, 'FusionImages')

writexls = True
writemha = True

outdir = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML' #os.path.join(hdpath, 'ProcessedFusionImages')
xlspath = os.path.join(outdir, 'fusionimagesdata.xls')
errorpath = os.path.join(outdir, 'fusionimageserror.xls')
OUTPUT_DIR = os.path.join(outdir, 'Images')


def labelseries(seriesname):
    # Processes seriesname and identifies it as T2, DWI, or ADC
    # returns 'error' if no T2, DWI, or ADC strings are found
    T2strings = ['T2']
    DWIstrings = ['DWI']
    ADCstrings = ['ADC', 'APPARENT DIFFUSION']
    DCEstrings = ['DCE', 'DYNAMIC CONTRAST']

    isT2 = isDWI = isADC = isDCE = False

    seriesname = seriesname.upper()

    for T2string in T2strings:
        p = re.compile(T2string)
        if re.search(p, seriesname):
            isT2 = True

    for DWIstring in DWIstrings:
        p = re.compile(DWIstring)
        if re.search(p, seriesname):
            isDWI = True

    for ADCstring in ADCstrings:
        p = re.compile(ADCstring)
        if re.search(p, seriesname):
            isADC = True

    for DCEstring in DCEstrings:
        p = re.compile(DCEstring)
        if re.search(p, seriesname):
            isDCE = True

    if isDCE:
        return 'DCE'
    elif isADC:
        return 'ADC'
    elif isDWI:
        return 'DWI'
    elif isT2:
        return 'T2'
    else:
        return 'error'

def removeslash(string):
    string = string.replace('/', '')
    string = string.replace('\\', '')

    return string

def removedashandspace(string):
    string = string.replace(' ', '')
    string = string.replace('-', '')
    string = string.replace('_', '')

    return string


# In 'FusionImages', next sublevel is [date]
# Read all dicoms in [date] folder and see how many different studies:
# group into
#    a. patients and
#      b. sequences (T2 vs ADC vs DWI etc)
# write into xls file [xlspath]

accession_list = []
mrdate_list = []
seriesname_list = []
mrfiles_list = []
dimensions_list = []
numslices_list = []
serieslabel_list = []
mhanames_list = []
pixelspacing_list = []

errormsgs = []


#datefolders = [f.path for f in os.scandir(fusionimagesdir) if f.is_dir()]

imageReader = sitk.ImageSeriesReader()
writer = sitk.ImageFileWriter()

#for i, datepath in enumerate(datefolders):
if 1 == 1:
    datepath = fusionimagesdir
    mhacounter = 0

    print(datepath)

    seriesIDs = imageReader.GetGDCMSeriesIDs(datepath)

    for series in seriesIDs:
    #if 1 == 1:
    #    series = seriesIDs[1]

        if 1 == 1:
        #try:
            bvaluedf = []

            seriesfilenames = imageReader.GetGDCMSeriesFileNames(datepath, series)

            imageReader.SetFileNames(seriesfilenames)
            print(seriesfilenames[0])
            print(len(seriesfilenames))
            image = imageReader.Execute()
            size = image.GetSize()

            for i, file in enumerate(seriesfilenames):
                ds = pydicom.read_file(seriesfilenames[i])
                value = ds[0x43, 0x1039].value
                bvaluedf.append(value)

            bvaluedf = pd.DataFrame(bvaluedf)
            bvaluedf.to_excel('C:\\ProcessedFusionImages\\bvalues.xls')

            #print(ds)

            seriesname = ds.SeriesDescription.encode('ascii', 'ignore').decode()
            serieslabel = labelseries(seriesname)

            print(serieslabel)
            if serieslabel != 'DCE':
                pixelspacing = str(ds.PixelSpacing)
                accessionnum = removedashandspace(str(ds.AccessionNumber))
                mrdate = str(ds.StudyDate)
                firstpos = ds.ImagePositionPatient
                imgorientationpatient = ds.ImageOrientationPatient


                mhaname = accessionnum + '_' + mrdate + '_' + str(mhacounter)
                mhacounter = mhacounter + 1

                print(mhaname)

                accession_list.append(accessionnum)
                mrdate_list.append(mrdate)
                seriesname_list.append(seriesname)
                mrfiles_list.append(seriesfilenames)
                dimensions_list.append(size)
                numslices_list.append(len(seriesfilenames))
                serieslabel_list.append(serieslabel)
                mhanames_list.append(mhaname)
                pixelspacing_list.append(pixelspacing)

                # read the images into mha file
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(seriesfilenames)
                image = sitk.Cast(reader.Execute(), sitk.sitkFloat32)

                if writemha == True:
                    writer.Execute(image, os.path.join(OUTPUT_DIR, mhaname + ".mha"), True)

        #except:
        #    print('Error in reading ' + str(datepath) + '/' + str(series))
        #    errormsgs.append('Error in reading ' + str(series))

outDF = pd.DataFrame([accession_list, mrdate_list, seriesname_list, serieslabel_list, numslices_list, dimensions_list, pixelspacing_list, mhanames_list])
outDF = outDF.transpose()
outDF.columns = ['accession', 'mrdate', 'seriesname', 'serieslabel', 'numslices', 'dimensions', 'pixelspacing', 'mhafilename']

if writexls == True:
    outDF.to_excel(xlspath, sheet_name='Sheet1')

    errorDF = pd.DataFrame(data = errormsgs)
    errorDF.to_excel(errorpath)
