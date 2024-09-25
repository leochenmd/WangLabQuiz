import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt


def rotateImage(original, physicalcenter, anglex, angley, anglez):
    # returns the 'rotated' 3d image about the physical center that is resampled
    #  based on the original image
    # original is the original image
    #   angle z is pitch / tilt along the superior/inferior axis (i.e trendelenburg)
    #   angle y is yaw / rotating the body like a propeller blade
    #   angle x is roll / twisting the body like a rolling pin, turning in dance
    interpolator = sitk.sitkLinear
    radx = anglex * math.pi / 180
    rady = angley * math.pi / 180
    radz = anglez * math.pi / 180

    center = np.array(physicalcenter)
    origin = np.array(original.GetOrigin())

    transform = sitk.Euler3DTransform()
    transform.SetCenter(physicalcenter)
    transform.SetRotation(radz, rady, radx)    # note the order is z, y, x

    unitvecs = np.transpose(np.reshape(original.GetDirection(), (-1, 3)))
    matrix = np.reshape(transform.GetMatrix(), (-1, 3))
    inverse = np.linalg.inv(matrix)

    # the transform matrix is actually mapping backwards: post to pre
    # therefore the forward transformation is the inverse matrix
    transformedunitvecs = inverse @ unitvecs   # new i, j, k are columns
    newdirection = transformedunitvecs.flatten('F')    # flatten by column

    neworigin = (matrix @ (origin - center)) + center

    rotatedImage = sitk.Resample(original, original, transform, interpolator)
    rotatedImage.SetDirection(newdirection)
    rotatedImage.SetOrigin(neworigin)

    return rotatedImage


def rotateCores(original, physicalcenter, anglex, angley, anglez):
    # returns the 'rotated' 3d image about the physical center that is resampled
    #  based on the original image
    # original is the original image
    #   angle z is pitch / tilt along the superior/inferior axis (i.e trendelenburg)
    #   angle y is yaw / rotating the body like a propeller blade
    #   angle x is roll / twisting the body like a rolling pin, turning in dance
    interpolator = sitk.sitkNearestNeighbor
    radx = anglex * math.pi / 180
    rady = angley * math.pi / 180
    radz = anglez * math.pi / 180

    center = np.array(physicalcenter)
    origin = np.array(original.GetOrigin())

    transform = sitk.Euler3DTransform()
    transform.SetCenter(physicalcenter)
    transform.SetRotation(radz, rady, radx)    # note the order is z, y, x

    unitvecs = np.transpose(np.reshape(original.GetDirection(), (-1, 3)))
    matrix = np.reshape(transform.GetMatrix(), (-1, 3))
    inverse = np.linalg.inv(matrix)

    # the transform matrix is actually mapping backwards: post to pre
    # therefore the forward transformation is the inverse matrix
    transformedunitvecs = inverse @ unitvecs   # new i, j, k are columns
    newdirection = transformedunitvecs.flatten('F')    # flatten by column

    neworigin = (matrix @ (origin - center)) + center

    rotatedImage = sitk.Resample(original, original, transform, interpolator)
    rotatedImage.SetDirection(newdirection)
    rotatedImage.SetOrigin(neworigin)

    return rotatedImage


def flipImage(original):
    flipped = sitk.Flip(original, [True, False, False])

    return flipped


def histEQimage(img):
    # Normalizes mha image from 0 (min) to 255 (max)
    # Note: this function returns the CDF and the bins so that the same
    #       CDF and bins can be used to normalize the augmented/rotated images

    image = sitk.GetArrayFromImage(img)

    # generate image histogram
    hist, bins = np.histogram(image.flatten(), 256, normed=True)

    cdf = hist.cumsum() # cumulative distribution function
    cdf_normalized = cdf * hist.max()/ cdf.max() # normalize

    # assign normalized intensity values to pixels
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = np.round(image_equalized).astype('uint8')

    image_equalized = image_equalized.reshape(image.shape)

    origin = img.GetOrigin()
    direction = img.GetDirection()
    spacing = img.GetSpacing()

    newimg = sitk.GetImageFromArray(image_equalized)
    newimg.SetOrigin(origin)
    newimg.SetDirection(direction)
    newimg.SetSpacing(spacing)

    return newimg, bins, cdf

def histEQrotatedimage(img, bins, cdf):
    # Note: this function requires the CDF and the bins generated from 'histEQimage'

    image = sitk.GetArrayFromImage(img)

    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = np.round(image_equalized).astype('uint8')

    image_equalized = image_equalized.reshape(image.shape)

    origin = img.GetOrigin()
    direction = img.GetDirection()
    spacing = img.GetSpacing()

    newimg = sitk.GetImageFromArray(image_equalized)
    newimg.SetOrigin(origin)
    newimg.SetDirection(direction)
    newimg.SetSpacing(spacing)

    return newimg


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


# 1. Reads all T2 and ADC sequences and generates augmentations, then normalizes them

#### SET ANGLES OF AUGMENTATIONS HERE ####
angles = [1, 2, 3, 4, 5]

T2dir = '/data/prostate/postimages/PostImages2/'   # also ADC dir
augmentdir = coredir = '/data/prostate/postimages/Augmented/'

os.chdir(T2dir)

writer = sitk.ImageFileWriter()
counter = 0
total = len(glob.glob('*ADCresampled.mha'))

#for ADCfile in glob.glob('*ADCresampled.mha'):
#    accession = ADCfile.split('_')[0]

if 0 == 0:
    accession = '8246567'

    ADCfile = accession + '_ADCresampled.mha'
    T2file = accession + '_T2.mha'

    T2path = os.path.join(T2dir, T2file)
    ADCpath = os.path.join(T2dir, ADCfile)

    T2image = sitk.ReadImage(T2path, sitk.sitkFloat32)
    ADCimage = sitk.ReadImage(ADCpath, sitk.sitkFloat32)

    # generates normalized images of originals
    normT2, T2bins, T2cdf = histEQimage(T2image)
    normADC, ADCbins, ADCcdf = histEQimage(ADCimage)

    normT2path = os.path.join(augmentdir, accession + '_T2.mha')
    normADCpath = os.path.join(augmentdir, accession + '_ADC.mha')
    writer.Execute(normT2, normT2path, True)
    writer.Execute(normADC, normADCpath, True)


    # finds the center of the MR scan in physical coordinates
    shape = np.array(T2image.GetSize())
    pixelcenter = (shape / 2).astype(int).tolist()
    physicalcenter = T2image.TransformIndexToPhysicalPoint(pixelcenter)


    # generates and saves rotations
    # when angles y and z = 0, only twists body like rolling pin
    angley = 0
    anglez = 0


    # flip, no rotation
    T2flipped = flipImage(normT2)
    ADCflipped = flipImage(normADC)
    T2flipname = accession + '_T2_flip1.mha'
    ADCflipname = accession + '_ADC_flip1.mha'
    writer.Execute(T2flipped, os.path.join(augmentdir, T2flipname), True)
    writer.Execute(ADCflipped, os.path.join(augmentdir, ADCflipname), True)

    # positive rotations (natural turn, left shoulder forward right shoulder back)
    for angle in angles:
        anglex = angle

        rotatedT2 = rotateImage(T2image, physicalcenter, anglex, angley, anglez) # rotate
        rotatedT2 = histEQrotatedimage(rotatedT2, T2bins, T2cdf)                 # normalize
        rotatedADC = rotateImage(ADCimage, physicalcenter, anglex, angley, anglez)
        rotatedADC = histEQrotatedimage(rotatedADC, ADCbins, ADCcdf)

        # file name is [accession]_T2_flip#_rotateX##.mha
        # e.g. 8099139_T2_flip1_rotateR03.mha
        #   flip 0 is not flipped, flip 1 is flipped
        #   rotate X##   L05 is rotated left arm back, right arm forward
        #                R10 is rotated 10 deg (left arm forward, right arm back)
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        rotatedT2name = accession + '_T2_flip0_rotateR' + anglename + '.mha'
        rotatedADCname = accession + '_ADC_flip0_rotateR' + anglename + '.mha'

        writer.Execute(rotatedT2, os.path.join(augmentdir, rotatedT2name), True)
        writer.Execute(rotatedADC, os.path.join(augmentdir, rotatedADCname), True)

        # positive rotation + flip
        T2flipped = flipImage(rotatedT2)
        ADCflipped = flipImage(rotatedADC)
        T2flipname = accession + '_T2_flip1_rotateL' + anglename + '.mha'
        ADCflipname = accession + '_ADC_flip1_rotateL' + anglename + '.mha'
        writer.Execute(T2flipped, os.path.join(augmentdir, T2flipname), True)
        writer.Execute(ADCflipped, os.path.join(augmentdir, ADCflipname), True)

    # negative rotations (reverse turn, right shoulder forward left shoulder back)
        anglex = -angle

        rotatedT2 = rotateImage(T2image, physicalcenter, anglex, angley, anglez) # rotate
        rotatedT2 = histEQrotatedimage(rotatedT2, T2bins, T2cdf)                 # normalize
        rotatedADC = rotateImage(ADCimage, physicalcenter, anglex, angley, anglez)
        rotatedADC = histEQrotatedimage(rotatedADC, ADCbins, ADCcdf)

        # file name is [accession]_T2_flip#_rotateX##.mha
        # e.g. 8099139_T2_flip1_rotateR03.mha
        #   flip 0 is not flipped, flip 1 is flipped
        #   rotate X##   L05 is rotated clockewise (left arm back, right arm forward)
        #                R10 is rotated counterclockwise 10 deg (left arm forward, right arm back)
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        rotatedT2name = accession + '_T2_flip0_rotateL' + anglename + '.mha'
        rotatedADCname = accession + '_ADC_flip0_rotateL' + anglename + '.mha'

        writer.Execute(rotatedT2, os.path.join(augmentdir, rotatedT2name), True)
        writer.Execute(rotatedADC, os.path.join(augmentdir, rotatedADCname), True)

        # negative rotation + flip
        T2flipped = flipImage(rotatedT2)
        ADCflipped = flipImage(rotatedADC)
        T2flipname = accession + '_T2_flip1_rotateR' + anglename + '.mha'
        ADCflipname = accession + '_ADC_flip1_rotateR' + anglename + '.mha'
        writer.Execute(T2flipped, os.path.join(augmentdir, T2flipname), True)
        writer.Execute(ADCflipped, os.path.join(augmentdir, ADCflipname), True)

    # progress bar
    counter = counter + 1
    if (counter % 5) == 0:
        print('processed ' + str(counter) + ' of ' + str(total))

print('done!')





# 2. Generates mha file labeled with core locations

# Calculates the core locations and plots the core tracts on T2 space
#  rad = radius of the core tract
#  cutoff = gleason score cutoff (i.e. cutoff of 6 means Gleason >= 6 is cancer)
# 6-10 for cancer
# 1 for benign
# 0 for unknown
# Outputs as .mha file ([accession]_label_cutoff_rad.mha)

# For overlaps, it takes the majority vote
#   can try to incorporate nearest neighbor later

angles = [1, 2, 3, 4, 5]

rootdir = '/data/prostate/postimages/Augmented/'
os.chdir(rootdir)
outdir = '/data/prostate/postimages/AugmentedLabels/'

mrpathwlocpath = '/data/prostate/MRpathwcorelocs5.csv'

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
usepaththreshold = True

percentthreshold = 0
method = 'max'         # max, majority, nearest neighbor (to be implemented)


counter = 0
total = len(glob.glob('*T2.mha'))

#for file in glob.glob('*T2.mha'):
if 1 == 1:
    file = '8246567_T2.mha'
    file2 = file.split('.')[0]
    file2 = file2.replace('T2', 'cores')

    accession = file.split('_')[0]
    print(accession)

    try:
        accessiondf = mrpathlocdf[mrpathlocdf.MRaccession == accession]
        mrn = accessiondf.mrn.values[0]
        date = accessiondf.biopsydate.values[0]
       #version = accessiondf.version.values[0]

    except:
        print('accession not found in csv file: ' + accession)
    #    continue


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
                #    continue

            corebot = [float(row.corebotx), float(row.coreboty), float(row.corebotz)]
            coretip = [float(row.coretipx), float(row.coretipy), float(row.coretipz)]


            [col, row, t] =  corepixelsolver(imgpositionpatient, imgorientationpatient, corebot, coretip, rowspacing, colspacing)

            #if (.5 - percent/200) <= t <= (.5 + percent/200):
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


    corelocnpy[corelocnpy == 0] = 1
    corelocnpy[corelocnpy == -1] = 0
    #if usepaththreshold == True:
    #    corelocnpy[corelocnpy >= paththreshold] = 1

    corelocnpy = corelocnpy.astype('uint8')

    # generate the final 3D npy and write to file
    writer = sitk.ImageFileWriter()

    img = sitk.GetImageFromArray(corelocnpy)
    img.SetDirection(direction)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    outpath = os.path.join(outdir, file2 + '_r' + str(coreradius) + '_gAll.mha')
    writer.Execute(img, outpath, True)
    print(outpath)


   # corelocnpy[corelocnpy >= paththreshold] = 1
   # img = sitk.GetImageFromArray(corelocnpy)
   # img.SetDirection(direction)
   # img.SetSpacing(spacing)
   # img.SetOrigin(origin)

   # outpath = os.path.join(outdir, accession + '_cores_r' + str(coreradius) + '_g' + str(paththreshold) + '.mha')
   # sitk.WriteImage(img, outpath)



    # 2b. Generates the core locations for augmentations
    #     Uses nearest neighbor interpolation
    corefile = accession + '_cores_r3_gAll.mha'
    corefile2 = accession + '_cores_r3_gAll'

    corepath = os.path.join(outdir, corefile)

    coreimage = img   #sitk.ReadImage(corepath, sitk.sitkUInt8)

    # finds the center of the MR scan in physical coordinates
    shape = np.array(coreimage.GetSize())
    pixelcenter = (shape / 2).astype(int).tolist()
    physicalcenter = coreimage.TransformIndexToPhysicalPoint(pixelcenter)


    # generates and saves rotations
    # when angles y and z = 0, only twists body like rolling pin
    angley = 0
    anglez = 0


    # flip, no rotation
    coreflipped = flipImage(coreimage)
    coreflipname = corefile2 + '_flip1.mha'

    writer.Execute(coreflipped, os.path.join(outdir, coreflipname), True)

    # positive rotations (natural turn, left shoulder forward right shoulder back)
    for angle in angles:
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        anglex = angle

        rotatedcores = rotateCores(coreimage, physicalcenter, anglex, angley, anglez) # rotate
        rotatedcoresname = corefile2 + '_flip0_rotateR' + anglename + '.mha'
        writer.Execute(rotatedcores, os.path.join(outdir, rotatedcoresname), True)

        # positive rotation + flip
        coresflipped = flipImage(rotatedcores)
        coresflipname = corefile2 + '_flip1_rotateL' + anglename + '.mha'
        writer.Execute(coresflipped, os.path.join(outdir, coresflipname), True)


    # negative rotations (reverse turn, right shoulder forward left shoulder back)
        anglex = -angle

        rotatedcores = rotateCores(coreimage, physicalcenter, anglex, angley, anglez) # rotate
        rotatedcoresname = corefile2 + '_flip0_rotateL' + anglename + '.mha'
        writer.Execute(rotatedcores, os.path.join(outdir, rotatedcoresname), True)

         # negative rotation + flip
        coresflipped = flipImage(rotatedcores)
        coresflipname = corefile2 + '_flip1_rotateR' + anglename + '.mha'
        writer.Execute(coresflipped, os.path.join(outdir, coresflipname), True)



    # progress bar
    counter = counter + 1
    if (counter % 5) == 0:
        print('processed ' + str(counter) + ' of ' + str(total))

print('done!')
