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


hdpath = 'C:\\'
fusionimagesdir = os.path.join(hdpath, 'FusionImages')

writexls = True
writemha = True

outdir = os.path.join(hdpath, 'ProcessedFusionImages')
xlspath = os.path.join(outdir, 'fusionimagesdataDWI.xls')
errorpath = os.path.join(outdir, 'fusionimageserrorDWI.xls')
OUTPUT_DIR = os.path.join(outdir, 'DWIonly')


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
bvalue_list = []

outdata = []

errormsgs = []


datefolders = [f.path for f in os.scandir(fusionimagesdir) if f.is_dir()]

imageReader = sitk.ImageSeriesReader()
writer = sitk.ImageFileWriter()

processedcounter = 0

for i, datepath in enumerate(datefolders):
#if 1 == 1:
#    datepath = datefolders[100]
    mhacounter = 0

    #print(datepath)

    seriesIDs = imageReader.GetGDCMSeriesIDs(datepath)

    for series in seriesIDs:
    #if 1 == 1:
    #    series = seriesIDs[1]

    #    if 1 == 1:
        try:
            bvalues = []

            seriesfilenames = imageReader.GetGDCMSeriesFileNames(datepath, series)

            imageReader.SetFileNames(seriesfilenames)

            image = imageReader.Execute()
            size = image.GetSize()

            ds = pydicom.read_file(seriesfilenames[0])
            seriesname = ds.SeriesDescription.encode('ascii', 'ignore').decode()
            serieslabel = labelseries(seriesname)

            if serieslabel == 'DWI':
                processedcounter += 1

                pixelspacing = str(ds.PixelSpacing)
                accessionnum = removedashandspace(str(ds.AccessionNumber))
                slicethickness = ds.SliceThickness
                mrdate = str(ds.StudyDate)

                for i, file in enumerate(seriesfilenames):
                    ds = pydicom.read_file(seriesfilenames[i])
                    value = ds[0x43, 0x1039].value
                    bvalues.append(value[0])

                #sortedindex = numpy.argsort(bvalues)
                uniquebvalues = list(set(bvalues))
                bvaluesdf = pd.DataFrame(bvalues)

                for bvalue in uniquebvalues:
                    bvaluedf = bvaluesdf[bvaluesdf.iloc[:,0] == bvalue]

                    shortbvalue = int(str(bvalue)[-5:])
                    fileindices = bvaluedf.index.values
                    filenames = [seriesfilenames[i] for i in fileindices]

                    mhaname = accessionnum + '_' + mrdate + '_DWI_' + str(mhacounter) + '_b' + str(shortbvalue)

                    #print(mhaname)

                    # read the images into mha file
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(filenames)
                    image = sitk.Cast(reader.Execute(), sitk.sitkFloat32)

                    if writemha == True:
                        writer.Execute(image, os.path.join(OUTPUT_DIR, mhaname + ".mha"), True)


                    outdata.append(dict(zip(['accession', 'mrdate', 'seriesname', 'serieslabel', 'bvalue', 'numslices', 'dimensions', 'pixelspacing', 'slicethickness', 'mhafilename'], [accessionnum, mrdate, seriesname, serieslabel, shortbvalue, len(filenames), size, pixelspacing, slicethickness, mhaname])))

#                    accession_list.append(accessionnum)
#                    mrdate_list.append(mrdate)
#                    seriesname_list.append(seriesname)
#                    mrfiles_list.append(seriesfilenames)
#                    dimensions_list.append(size)
#                    numslices_list.append(len(seriesfilenames))
#                    serieslabel_list.append(serieslabel)
#                    mhanames_list.append(mhaname)
#                    pixelspacing_list.append(pixelspacing)
#                    bvalue_list.append(shortbvalue)

                mhacounter = mhacounter + 1

                if processedcounter % 50 == 0:
                    print(processedcounter)
                    outDF = pd.DataFrame(outdata)
                    outDF = outDF[['accession', 'mrdate', 'seriesname', 'serieslabel', 'bvalue', 'numslices', 'dimensions', 'pixelspacing', 'slicethickness', 'mhafilename']]


                    if writexls == True:
                        outDF.to_excel(xlspath, sheet_name='Sheet1')
                        errorDF = pd.DataFrame(data = errormsgs)
                        errorDF.to_excel(errorpath)
                        print('written to ' + xlspath)

        except:
            print('Error in reading ' + str(datepath) + '/' + str(series))
            errormsgs.append(dict(zip(['series', 'datepath'], [str(series), str(datepath)])))

#outDF = pd.DataFrame(data = [accession_list, mrdate_list, seriesname_list, serieslabel_list, bvalue_list, numslices_list, dimensions_list, pixelspacing_list, mhanames_list])
#outDF = outDF.transpose()
#outDF.columns = ['accession', 'mrdate', 'seriesname', 'serieslabel', 'bvalue', 'numslices', 'dimensions', 'pixelspacing', 'mhafilename']

outDF = pd.DataFrame(outdata)
outDF = outDF[['accession', 'mrdate', 'seriesname', 'serieslabel', 'bvalue', 'numslices', 'dimensions', 'pixelspacing', 'slicethickness', 'mhafilename']]


if writexls == True:
    outDF.to_excel(xlspath, sheet_name='Sheet1')
    errorDF = pd.DataFrame(data = errormsgs)
    errorDF.to_excel(errorpath)
