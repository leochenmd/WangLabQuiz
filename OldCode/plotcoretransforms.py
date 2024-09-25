import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt

def corepixelsolver(imgpositionpatient, imgorientationpatient, corebot, coretip, rowspacing, colspacing):
###    Solves for intersection between MR slice (plane) and core (line)
###    Returns 'col' (column index), 'row' (row index),
###        and 't' (dummy parametrization variable, core lies in t = [0, 1]
###
###    a. plane is parametrization using ImagePositionPatient and
###       and direction vectors from ImageOrientationPatient
###    b. core is parametrization using coretip and corebot
###
###   Equation: xvec * c + yvec * r + imgpospat = corebot + (coretip - corebot) * t
###   Rearranging gives:
###      xvec * c + yvec * r - (coretip - corebot) * t = corebot - imgpospat
###   This is represented as Amatrix * (c, r, t) = b
    xvec = np.array(imgorientationpatient[0:3])
    yvec = np.array(imgorientationpatient[3:6])
    colspacing = np.array(colspacing)
    rowspacing = np.array(rowspacing)
    corebot = np.array(corebot)
    coretip = np.array(coretip)
    imgpositionpatient = np.array(imgpositionpatient)

    xvec = xvec * colspacing   # colspacing is 2nd element of dicom.pixelspacing
    yvec = yvec * rowspacing   # rowspacing is 1st element of dicom.pixelspacing

    Amatrix = np.zeros((3,3))
    Amatrix[:, 0] = xvec
    Amatrix[:, 1] = yvec
    Amatrix[:, 2] = corebot - coretip

    b = corebot - imgpositionpatient


    [col, row, t] = np.linalg.solve(Amatrix, b)

    return [col, row, t]

# 2. Generates mha files labeled with core locations

# Calculates the core locations and plots the core tracts on T2 space
#  rad = radius of the core tract
#  cutoff = gleason score cutoff (i.e. cutoff of 6 means Gleason >= 6 is cancer)
# 6-10 for cancer
# 1 for benign
# 0 for unknown
# Outputs as .mha file ([accession]_label_cutoff_rad.mha)

# For overlaps, it takes the majority vote
#   can try to incorporate nearest neighbor later

rootdir = '/Users/Leo/Desktop/Temp/augmented'
os.chdir(rootdir)
outdir = '/Users/Leo/Desktop/Temp/Labels'

mrpathwlocpath = '/users/leo/desktop/temp/MRpathwcorelocs3.csv'

mrpathlocdf = pd.read_csv(mrpathwlocpath, dtype=object, index_col=0)

# variables
#  core radius (distance around pixel that is labeled)
#  path score >= threshold is considered cancer (6 for any ca, 7 for clin significant)
#  method to resolve overlapping cores
#     1. max takes the maximum gleason score
#   To be implemented
#     2. majority takes the majority vote (ties broken by greater pathology)
#     3. nearest neighbor takes pathology score of nearest neighboring core

coreradius = 3         #in mm
paththreshold = 6
percentthreshold = 20
method = 'max'         # max, majority, nearest neighbor (to be implemented)


for file in glob.glob('*T2_flip1*'):
    file2 = file.split('.')[0]

    print(file)

    accession = file.split('_')[0]

    try:
        accessiondf = mrpathlocdf[mrpathlocdf.MRaccession == accession]
        mrn = accessiondf.mrn.values[0]
        date = accessiondf.biopsydate.values[0]
       #version = accessiondf.version.values[0]

    except:
        print('accession not found in csv file: ' + accession)
#        continue


    # read T2 image file
    T2filename = file
    T2filepath = os.path.join(rootdir, T2filename)
    T2image = sitk.ReadImage(T2filepath, sitk.sitkUInt8)

    # sitk image coordinates are (col, row, z)
    origin = T2image.GetOrigin()
    direction = T2image.GetDirection()
    spacing = T2image.GetSpacing()


    # the numpy is (z, row, col)
    T2npy = sitk.GetArrayFromImage(T2image)
    corelocnpy = -1*np.ones(np.shape(T2npy))

    imgorientationpatient = direction[0:6]
    [colspacing, rowspacing] = spacing[0:2]


    # labels the 3D MR space with gleason score (0, 6, 7, 8, etc)
    # goes through slice by slice
    for zslicei in range(np.shape(corelocnpy)[0]):
        imgpositionpatient = T2image.TransformIndexToPhysicalPoint([0, 0, zslicei])

        # first pass: solve for all of the points in each slice
        rowlist = []
        collist = []
        pathlist = []
        percentlist = []
        numpoints = 0

        for i, row in accessiondf.iterrows():
            path = int(row.Total)

            if path == 0:
                percent = 100
            else:
                try:
                    percent = int(row.PercentCore)
                    #corelength = float(row.CoreLength)
                except:
                    percent = 0
                    continue

            corebot = [float(row.corebotx), float(row.coreboty), float(row.corebotz)]
            coretip = [float(row.coretipx), float(row.coretipy), float(row.coretipz)]


            [col, row, t] =  corepixelsolver(imgpositionpatient, imgorientationpatient, corebot, coretip, rowspacing, colspacing)

            if 0 <= t <= 1:      # can modify this to only include central __% of core
                                 # i.e. 60% would be 0.2 <= t <= 0.8
                rowlist.append(row)
                collist.append(col)
                pathlist.append(path)
                percentlist.append(percent)


        # second pass: label the 3D MR space with gleason scores
        # current method to address overlaps is max path score
        # reminder: the npy is [z, row, col]
        for r in range(np.shape(corelocnpy)[1]):
            for c in range(np.shape(corelocnpy)[2]):

                for i in range(len(rowlist)):
                    corerow = rowlist[i]
                    corecol = collist[i]
                    path = pathlist[i]
                    percent = percentlist[i]

                    distance = math.sqrt((rowspacing*(corerow - r))**2 + (colspacing*(corecol-c))**2)
                    if distance < coreradius:    # if the current pixel is in the space of the core
                        # if method = 'max'    takes the maximum
                        if corelocnpy[zslicei, r, c] < path and percent >= percentthreshold:
                            corelocnpy[zslicei, r, c] = path

    # convert -1 to 0, and 0 to 1 (to facilitate storage as UInt8)
    corelocnpy[corelocnpy == 0] = 1
    corelocnpy[corelocnpy == -1] = 0
    corelocnpy = corelocnpy.astype('uint8')

    # generate the final 3D npy and write to file
    writer = sitk.ImageFileWriter()

    img = sitk.GetImageFromArray(corelocnpy)
    img.SetDirection(direction)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    outpath = os.path.join(outdir, file2 + '_cores_r' + str(coreradius) + '_gAll.mha')
    writer.Execute(img, outpath, True)


   # corelocnpy[corelocnpy >= paththreshold] = 1
   # img = sitk.GetImageFromArray(corelocnpy)
   # img.SetDirection(direction)
   # img.SetSpacing(spacing)
   # img.SetOrigin(origin)

   # outpath = os.path.join(outdir, accession + '_cores_r' + str(coreradius) + '_g' + str(paththreshold) + '.mha')
   # sitk.WriteImage(img, outpath)

    print('image saved')
    print('---')
    #for zslicei in range(np.shape(corelocnpy)[0]):
    #    plt.figure()
    #    plt.imshow(T2npy[zslicei,:, :])
    #    plt.figure()
    #    plt.imshow(corelocnpy[zslicei, :, :])

print('finished')
