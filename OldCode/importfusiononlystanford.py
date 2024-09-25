### Goes through excel file and only picks Stanford scans

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
processedfusionimages = os.path.join(hdpath, 'ProcessedFusionImages')
xlspath = os.path.join(processedfusionimages, 'fusionimagesdata.xls')
onlystanfordpath = os.path.join(processedfusionimages, 'fusionimagesonlystanford.xls')
noerrorspath = os.path.join(processedfusionimages, 'fusionimagesonlystanfordnoerrors.xls')
phasecounterpath = os.path.join(processedfusionimages, 'fusionimagesphasecounter.xls')

savexls = True

def isStanford(institution):
    if 'ccsbmr' in institution.lower() or 'stanford' in institution.lower():
        return True
    else:
        return False

def isError(string):
    if 'error' in string:
        return True
    else:
        return False


df = pd.read_excel(xlspath)

# add smFOV into series label
for i, row in df.iterrows():
    if row.serieslabel == 'T2':
        if 'smFOV' in row.seriesname:
            df.loc[i, 'serieslabel'] = 'T2 smFOV'

df.to_excel(os.path.join(processedfusionimages, 'fusionimagesdatasmFOV.xls'))


institutionsdf = df.institution
onlystanfordindex = institutionsdf.apply(isStanford)
onlystanforddf = df[onlystanfordindex == True]
if savexls == True:
    onlystanforddf.to_excel(onlystanfordpath)

errordf = onlystanforddf.serieslabel
noerrorindex = errordf.apply(isError)
noerrordf = onlystanforddf[noerrorindex == False]
if savexls == True:
    noerrordf.to_excel(noerrorspath)


##### count the number of T2, DWI, and ADC sequences per MRN
accessions = list(set(noerrordf.accession))
accessions.remove(np.nan)

phasecounter = []

for accession in accessions:
#if 0 == 0:
#   accession = accessions[0]

    accessiondf = noerrordf[noerrordf.accession == str(accession)]

    numT2 = sum(accessiondf.serieslabel == 'T2')
    numT2smFOV = sum(accessiondf.serieslabel == 'T2 smFOV')
    numADC = sum(accessiondf.serieslabel == 'ADC')
    numDWI = sum(accessiondf.serieslabel == 'DWI')

    phasecounter.append(dict(zip(['accession', 'numT2', 'numT2smFOV', 'numADC', 'numDWI'], [accession, numT2 + numT2smFOV, numT2smFOV, numADC, numDWI])))

phasecounterdf = pd.DataFrame(phasecounter)

phasecounterdf = phasecounterdf[['accession', 'numT2', 'numT2smFOV', 'numADC', 'numDWI']]
phasecounterdf.to_excel(phasecounterpath)
