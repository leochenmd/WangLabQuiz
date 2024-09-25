### Pseudocode
### 1. Loads data (MR mha, cores mha)
### 2. For each slice, plot the cores onto MR based on pathology
###       note imshow draws row, then column (DICOM pixel_array values are coded row, column)

import numpy as np
import pandas as pd
import os
import pydicom
import re
import math

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

# Color RGBs
colorbenign = [0, 255, 0] #green aka '#00FF00'
color6 = [255, 255, 0] #yellow aka '#FFFF00'
color7 = [255, 131, 0] # orange aka '#FF8300'
color8plus = [255, 0, 0] #red aka '#FF0000'



##############################
# STOPPED WRITING HERE
##############################

imagesdir = 'E:/NPYimages'
datadir = 'E:/patdbdata'
#imagesdir = 'C:/Users/profuse/Documents/NPYimages'
#datadir = 'C:/Users/profuse/Documents/patdbdata'
savedir = 'C:/Users/profuse/Documents/plotswcores/'
#coredatapath = 'C:/Users/profuse/Documents/plotswcores/CorePath2019Jan.csv'
coredatapath = 'C:/Users/profuse/Documents/plotswcores/ArtemisBiopsyDB-PatientLevelSingleRo_DATA_2019-01-09_2022.csv'
#'C:/Users/profuse/Documents/plotswcores/ArtemisBiopsyDB-PatientLevelSingleRo_DATA_2018-10-03_2116.csv'

yesplot = True
savefigs = False
noplotbenign = False
powerfactor = 0.7
rad = 3

singlecore = True
singletargetcore = 102


def categorizeGleason(pathscore):
# categorizes gleason score
    if pathscore == 6:
        return 1
    elif pathscore >= 7:
        return 2
    else:
        return 0


def gleasontocolor(catscore):
    if catscore == 0:
        return '#00FF00'
    elif catscore == 1:
        return '#FFA500'
    else:
        return '#FF0000'
#######################################################################
# ##########    END FUNCTIONS      ####################################
#######################################################################

### LOAD DATA
### 1. Read CSV file
# 'accession', 'mrn', 'biopsydate', 'mrdate', 'version', 'errormsgs',
# 'numcores', 'rowspacing', 'colspacing', 'zspacing', 'rowdim', 'coldim', 'zdim'
data = pd.read_csv(datadir + '/patdbdata.csv', dtype = {'accession':str, 'mrn':str, 'biopsydate':str})


#numrows = np.shape(data)[0]

### 2. Go through accession numbers and load the MR numpy
# for accessionindex, accessionnum in enumerate(data.accession):



#### CHANGE ACCESSION NUMBER HERE ####    ###############################################################
#### CHANGE ACCESSION NUMBER HERE ####

accessionnum = str(8251660)

#### CHANGE ACCESSION NUMBER HERE ####
#### CHANGE ACCESSION NUMBER HERE ####



accessionindex = data.accession == accessionnum

accessiondata = data.loc[accessionindex, ]

accessionnum = accessiondata.accession.values[0]
mrn = accessiondata.mrn.values[0]
date = accessiondata.biopsydate.values[0]
version = accessiondata.version.values[0]


print('accession number: ' + accessionnum)
print('MRN: ' + mrn)
print('Date: ' + date)
print('version: ' + version)

if version[0:3] == '1.4':
    usetransformedcores = True
    print('Using transformed cores...')
elif float(date) <= 20170407:
    usetransformedcores = True
    print('Using transformed cores... (on or prior to 2017-04-07)')
else:
    usetransformedcores = False

imagespath = imagesdir + '/' + accessionnum + '_' + mrn + '_' + date
accessionpath = datadir + '/' + accessionnum + '_' + mrn + '_' + date

# read MR numpy
mrnumpy = np.load(imagespath + '.npy')

# read corenames, corelocations, and DICOM img position/orientation
imgpositionpatientarr = np.load(accessionpath + '_imgpospat.npy')
imgorientationpatient = np.load(accessionpath + '_imgoripat.npy')
corenames = np.load(accessionpath + '_corenames.npy')
print(corenames)

# if biopsy done on or before 2017-04-07, use transformed cores (version 1.4)
if usetransformedcores == False:
    coretips = np.load(accessionpath + '_coretips.npy')
    corebots = np.load(accessionpath + '_corebots.npy')
else:
    coretips = np.load(accessionpath + '_coretipsnew.npy')
    corebots = np.load(accessionpath + '_corebotsnew.npy')


# load dimensions
[rowdim, coldim, zdim] = np.shape(mrnumpy)
rowspacing = accessiondata.rowspacing.values[0]
colspacing = accessiondata.colspacing.values[0]


### 4. Read the pathology
coredata = pd.read_csv(coredatapath, dtype = {'mrn':str})
mrn = str(int(mrn))
numsyscores = coredata[coredata.mrn == mrn]['number_of_systematic_cores'].values[0]

#systematic cores
for i in range(12):
    coredata['syscorepath' + str(i+1)] = coredata['path' + str(i+1) + '_pri'] + coredata['path' + str(i+1) + '_sec']

systematics = []
for i in range(12):
    systematics.append(categorizeGleason(coredata[coredata.mrn == mrn]['syscorepath' + str(i+1)].values[0]))

for i in range(4):
    systematics.append(categorizeExtra(coredata[coredata.mrn == mrn]['extra' + str(i + 13)].values[0]))

print(systematics)

#targeted cores
for i in range(5):
    for j in range(5):
        coredata['roicore' + str(i+1) + 'path' + str(j+1)] = coredata['target_' + str(i+1) + '_core_' + str(j+1)+ '_primary'] + coredata['target_' + str(i+1) + '_core_' + str(j+1)+ '_secondary']

targeted = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        targeted[i, j] = categorizeGleason(coredata[coredata.mrn == mrn]['roicore' + str(i+1) + 'path' + str(j+1)].values[0])


print(targeted)
# associate with names:
finalcorepaths = []

targetedcounter = 0
for i in range(len(corenames)):
    if corenames[i] < numsyscores + 1:
        # fill systematics
            finalcorepaths.append(systematics[math.floor(corenames[i] - 1)])
    else:
        # fill targeted
        if targetedcounter == 0:
            firsttargeted = math.floor(corenames[i])

        indexi = math.floor(corenames[i] - firsttargeted)
        indexj = round(math.modf(corenames[i] - firsttargeted)[0] * 10)

        if indexj > 5:
            continue

        print(str(indexi) + '_' + str(indexj))
        finalcorepaths.append(targeted[indexi, indexj - 1])

        targetedcounter += 1

finalcorepaths = np.array(finalcorepaths)
print(finalcorepaths)

### 3. Go through each slice, and calculate pixel locations for each core
pixellist = []

for zslicei in range(zdim):
    imgpositionpatient = imgpositionpatientarr[zslicei]

    for corei in range(len(corenames)):
        singular = False
        coretip = coretips[corei]
        corebot = corebots[corei]

        try:
            [col, row, t] = corepixelsolver(imgpositionpatient, imgorientationpatient, corebot, coretip, rowspacing, colspacing)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                singular = True
                print('Warning: singular matrix in core ' + str([corenames[corei]]) + '  slice ' + str(zslicei))
#        print('z slice: ' + str(zslicei) + '  (col, row): ' + str(col) + ', ' + str(row) + '    t: ' + str(t))
        if (0 <= col and col <= coldim) and (0 <= row and row <= rowdim) and (0 <= t and t <= 1):
            if singular == False:
                pixellist.append([corenames[corei], row, col, zslicei, finalcorepaths[corei]])


# voxel (i,j) is pixel_array(j,i)
# This is because DICOM is saved row, column... imshow also plots row, column
# Circles are drawn column i, row j
pixeldf = pd.DataFrame(pixellist, columns = ['corename', 'row', 'col', 'zslice', 'catpath'])


# only targeted cores
#if onlytargeted == True:
#    targetedindex = (pixeldf.corename > firsttarget)
#    pixeldf = pixeldf.loc[targetedindex]

#print(pixeldf)

mrnumpycol = np.zeros((rowdim, coldim, 3, zdim))





### PLOT CORES ON MR SLICES ###
for c in range(3):
    # increase brightness, normalize pixel intensity 0 to 1, incorporate 3 color layer
    mrnumpycol[:,:,c,:] =  np.power((mrnumpy / np.max(mrnumpy)), powerfactor)

if savefigs == True:
    try:
        os.mkdir(savedir + '/%s_%s/' %(accessionnum, mrn))
    except:
        print()

for zslicei in range(zdim):
    patches = []
    colors = []

    criteria = (pixeldf['zslice'] == zslicei)    #  plots for current zslice
    pixelsubdf = pixeldf[criteria]

    for corei, row in pixelsubdf.iterrows():
        xpix = pixelsubdf.col[corei]
        ypix = pixelsubdf.row[corei]

        if noplotbenign:
            if pixelsubdf.catpath[corei] != 0:
                patches.append(Circle([xpix, ypix], radius=rad)) # plot circle at x, y on slice z
                colors.append(gleasontocolor(pixelsubdf.catpath[corei]))
        else:
            patches.append(Circle([xpix, ypix], radius=rad)) # plot circle at x, y on slice z
            colors.append(gleasontocolor(pixelsubdf.catpath[corei]))

    if yesplot == True:
        fig, axes = plt.subplots()
        axes.axis('off')
        axes.imshow(mrnumpycol[:,:,:, zslicei])
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.set_frame_on(False)

        our_cmap = ListedColormap(colors)
        p = PatchCollection(patches, cmap=our_cmap)
        p.set_array(np.arange(len(patches)))
        axes.add_collection(p)
        if savefigs == True:
            plt.savefig(savedir + '/%s_%s/FusionBxMapAllCores%0d.png' %(accessionnum, mrn, zslicei), bbox_inches="tight", pad_inches=-0.1)
        plt.show()
        plt.close()
