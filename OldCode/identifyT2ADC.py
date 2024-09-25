#########################################
# Loads xls file in 'FusionImages' directory
# Identifies which T2 and ADC sequence to use
# Outputs xls file with the .mha file name for the chosen T2/ADC series
#
# If there is only one T2 or ADC, it will pick that one
# If there are multiple T2s, it will pick the one that has "prostate"
#
# If there are multiple ADCs, it will pick the one that is closest in shape
#    to the T2 (i.e. closest number of slices, closest pixel spacing)


import os
import pandas as pd
import numpy as np

hdpath = 'C:\\'
fusionimagesdir = os.path.join(hdpath, 'FusionImages')

outdir = os.path.join(hdpath, 'ProcessedFusionImages')
xlspath = os.path.join(outdir, 'fusionimagesdata.xls')
linkedxlspath = os.path.join(outdir, 'linkedT2ADC.xls')
#errorpath = os.path.join(hdpath, 'fusionimageserrors.xls')
#coregisterdir = os.path.join(hdpath, 'CoregisteredImages')

df = pd.read_excel(xlspath)
df['containsProstate'] = np.where(df['seriesname'].str.contains('prostate', case=False, na=False), 'T', 'F')
df['containsSmFOV'] = np.where(df['seriesname'].str.contains('smfov', case=False, na=False), 'T', 'F')

accession_list = []
errormsg_list = []
T2count_list = []
ADCcount_list = []
T2numslices_list = []
ADCnumslices_list = []
T2pixelspacing_list = []
ADCpixelspacing_list = []
T2mhafilename_list = []
ADCmhafilename_list = []

uniqueaccessionslist = list(set(df.accession.values[:]))
# goes through every unique accession number

for accession in uniqueaccessionslist:
    accessiondf = df[df.accession == accession]

    errormsg = ''
    T2mhafilename = ADCmhafilename = ''
    T2numslices = T2pixelspacing = ''
    ADCnumslices = ADCpixelspacing = ''

    T2count = accessiondf[accessiondf.serieslabel == 'T2'].shape[0]
    ADCcount = accessiondf[accessiondf.serieslabel == 'ADC'].shape[0]

    if T2count == 0:
        errormsg = 'No T2 series found'
        print(str(accession) + ' does not have T2 series; skipped')
    elif ADCcount == 0:
        errormsg = 'No ADC series found'
        print(str(accession) + ' does not have ADC series; skipped')
    else:
        if T2count == 1:
            T2mhafilename = accessiondf[accessiondf.serieslabel == 'T2'].mhafilename.values[0]
            T2numslices = accessiondf[accessiondf.serieslabel == 'T2'].numslices.values[0]
            T2pixelspacing = accessiondf[accessiondf.serieslabel == 'T2'].pixelspacing.values[0]
        elif T2count > 1:
            T2df = accessiondf[accessiondf.serieslabel == 'T2']
            T2prostatedf = T2df[T2df.containsProstate == 'T']

            if T2prostatedf.shape[0] == 1:
                T2mhafilename = T2prostatedf.mhafilename.values[0]
                T2numslices = T2prostatedf.numslices.values[0]
                T2pixelspacing = T2prostatedf.pixelspacing.values[0]
            else:
                T2prostate2df = T2df[T2df.containsSmFOV == 'T']
                if T2prostate2df.shape[0] == 1:
                    T2mhafilename = T2prostate2df.mhafilename.values[0]
                    T2numslices = T2prostate2df.numslices.values[0]
                    T2pixelspacing = T2prostate2df.pixelspacing.values[0]
                else:
                    errormsg = 'multiple T2s with _prostate_'

        if errormsg == '':
            if ADCcount == 1:
                # if only one ADC series, then uses that one
                ADCmhafilename = accessiondf[accessiondf.serieslabel == 'ADC'].mhafilename.values[0]
                ADCnumslices = accessiondf[accessiondf.serieslabel == 'ADC'].numslices.values[0]
                ADCpixelspacing = accessiondf[accessiondf.serieslabel == 'ADC'].pixelspacing.values[0]
            elif ADCcount > 1:
                # otherwise, uses ADC series closest in dimension/pixel spacing to chosen T2
                ADCdf = accessiondf[accessiondf.serieslabel == 'ADC']

                minslicedifference = abs(T2numslices - ADCdf.numslices.values[0])
                minindex = 0

                slices = ADCdf.numslices.values[:]
                names = ADCdf.mhafilename.values[:]
                spacings = ADCdf.pixelspacing.values[:]
                for i, slice in enumerate(slices):
                    if abs(T2numslices - slices[i]) < minslicedifference:
                        minindex = i
                        minslicedifference = abs(T2numslices - ADCdf.numslices.values[i])

                ADCmhafilename = names[i]
                ADCnumslices = slices[i]
                ADCpixelspacing = spacings[i]

    accession_list.append(accession)
    errormsg_list.append(errormsg)
    T2count_list.append(T2count)
    ADCcount_list.append(ADCcount)
    T2numslices_list.append(T2numslices)
    ADCnumslices_list.append(ADCnumslices)
    T2pixelspacing_list.append(T2pixelspacing)
    ADCpixelspacing_list.append(ADCpixelspacing)
    T2mhafilename_list.append(T2mhafilename)
    ADCmhafilename_list.append(ADCmhafilename)

outDF = pd.DataFrame(data = [accession_list, errormsg_list,  T2count_list, ADCcount_list, T2numslices_list, ADCnumslices_list, T2pixelspacing_list, ADCpixelspacing_list,
    T2mhafilename_list, ADCmhafilename_list])
outDF = outDF.transpose()
outDF.columns = ['accession', 'errormsg', 'T2count', 'ADCcount', 'T2numslices', 'ADCnumslices', 'T2pixelspacing', 'ADCpixelspacing', 'T2mhafile', 'ADCmhafile']

outDF.to_excel(linkedxlspath, sheet_name='Sheet1')
