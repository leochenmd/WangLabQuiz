import numpy as np
import os
import pandas as pd
import random
import SimpleITK as sitk
import math
import pickle


def bbox_2D(img):
    c = np.any(img, axis=1)    #y
    r = np.any(img, axis=0)    #x

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]

    #x min max, y min max, z min max
    return rmin, rmax, cmin, cmax


def bbox_3D(img):
    # returns 3D bounding box of image
    z = np.any(img, axis=(1, 2))    #z
    c = np.any(img, axis=(0, 2))    #y
    r = np.any(img, axis=(0, 1))    #x

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    #x min max, y min max, z min max
    return [rmin, rmax, cmin, cmax, zmin, zmax]


def rotate3DImage(original, anglex, angley, anglez, interpolate='linear'):
    # returns the 'rotated' 3d image about the physical center that is resampled
    #  based on the original image
    #  original is the original image
    #   angle z is pitch / tilt along the superior/inferior axis (i.e trendelenburg)
    #   angle y is yaw / rotating the body like a propeller blade, like break dancing
    #   angle x is roll / twisting the body like a rolling pin, turning in dance

    if interpolate == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolate == 'NN':
        interpolator = sitk.sitkNearestNeighbor

    radx = anglex * math.pi / 180
    rady = angley * math.pi / 180
    radz = anglez * math.pi / 180

    origin = np.array(original.GetOrigin())
    pixelcenter = np.array(original.GetSize()) / 2.
    physicalcenter = original.TransformContinuousIndexToPhysicalPoint(pixelcenter)

    transform = sitk.Euler3DTransform()
    transform.SetCenter(physicalcenter)
    transform.SetRotation(radz, rady, radx)    # note the order is z, y, x

################
    # uncomment the following if you want pre and post-transform to match on MR space
    #unitvecs = np.transpose(np.reshape(original.GetDirection(), (-1, 3)))
    #matrix = np.reshape(transform.GetMatrix(), (-1, 3))
    #inverse = np.linalg.inv(matrix)


    # the transform matrix is actually mapping backwards: post to pre
    # therefore the forward transformation is the inverse matrix
    #transformedunitvecs = inverse @ unitvecs   # new i, j, k are columns
    #newdirection = transformedunitvecs.flatten('F')    # flatten by column

    #neworigin = (matrix @ (origin - physicalcenter)) + physicalcenter
################

    if interpolate == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolate == 'nn':
        interpolator = sitk.sitkNearestNeighbor

    rotatedImage = sitk.Resample(original, original, transform, interpolator)
    # uncomment if you want pre and post-transform to have the same MR space
    #rotatedImage.SetDirection(newdirection)
    #rotatedImage.SetOrigin(neworigin)


    return rotatedImage


def rotate2Dslice(img, angle, pivot, interpolate='spline'):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]

    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False, mode=interpolate)

    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def flipImage(original):
    # flips 3D mha/nifti image (reverses x indices)
    flipped = sitk.Flip(original, [True, False, False])
    return flipped

def flipSlice(original):
    # flips 2D NUMPY slice (reverses 'x-axis' indices)
    flipped = np.flipud(original)  #np.fliplr(original)
    return flipped




def cropImageUsingThreshold(image, threshold, xshift, yshift):
    # crops image around bounding box (makes square)
    # shifted by xshift and yshift (data augmentation purposes)
    # if you just want to crop threshold > k, set xshift = yshift = 0

    # load image
    npy = sitk.GetArrayFromImage(image)

    # GET METADATA
    direction = image.GetDirection()
    spacing = image.GetSpacing()

    # CALCULATE BOUNDING BOX OF HEAD (removes black space)
    mask = npy > threshold

    [xmin, xmax, ymin, ymax, zmin, zmax] = bbox_3D(mask)

    # check to make sure shifts do not extend outside boundaries of image
    if xmin + xshift < 0 or xmax + xshift > npy.shape[2]:
        xshift = 0

    if ymin + yshift < 0 or ymax + yshift > npy.shape[1]:
        yshift = 0


    # CROP IMAGE
    newnpy = npy[zmin:zmax, (ymin+yshift):(ymax+yshift), (xmin+xshift):(xmax+xshift)]

    newimage = sitk.GetImageFromArray(newnpy)
    topleft = [int(xmin), int(ymin), int(zmin)]
    neworigin = image.TransformIndexToPhysicalPoint(topleft)

    newimage.SetOrigin(neworigin)
    newimage.SetDirection(direction)
    newimage.SetSpacing(spacing)

    return newimage



def crop2DSlice(npy, size, xshift, yshift):
    # crops axial slice in npy format to new size (makes square)
    # shifted by xshift and yshift (data augmentation purposes)
    # it does not check if cropped 'size' is greater than the original image size
    #   or pad with 0s if this is the case

    # images are [row, col] = [y, x]
    width = npy.shape[1]
    height = npy.shape[0]

    xmin = (width - size) / 2
    xmax = width - xmin

    ymin = (height - size) / 2
    ymax = height - ymin

    # check to make sure shifts do not extend outside boundaries of image
    if xmin + xshift < 0 or xmax + xshift > npy.shape[1]:
        xshift = 0

    if ymin + yshift < 0 or ymax + yshift > npy.shape[0]:
        yshift = 0


    # CROP IMAGE
    newnpy = npy[(ymin+yshift):(ymax+yshift), (xmin+xshift):(xmax+xshift)]

    return newnpy


def crop3DImage(image, xysize, xshift, yshift):
    # crops image to 'xysize' (makes square) around the center
    # shifted by xshift and yshift (data augmentation purposes)

    # load image
    npy = sitk.GetArrayFromImage(image)

    # GET METADATA
    direction = image.GetDirection()
    spacing = image.GetSpacing()

    # images are [z, row, col] = [z, y, x]
    width = npy.shape[2]
    height = npy.shape[1]
    ## depth = npy.shape[0]

    xmin = (width - size) / 2
    xmax = width - xmin

    ymin = (height - size) / 2
    ymax = height - ymin

    zmin = 0        # can adjust this if you want to crop z dimension as well


    # check to make sure shifts do not extend outside boundaries of image
    if xmin + xshift < 0 or xmax + xshift > npy.shape[2]:
        xshift = 0

    if ymin + yshift < 0 or ymax + yshift > npy.shape[1]:
        yshift = 0


    # CROP IMAGE
    newnpy = npy[:, (ymin+yshift):(ymax+yshift), (xmin+xshift):(xmax+xshift)]

    newimage = sitk.GetImageFromArray(newnpy)
    topleft = [int(xmin), int(ymin), int(zmin)]
    neworigin = image.TransformIndexToPhysicalPoint(topleft)

    newimage.SetOrigin(neworigin)
    newimage.SetDirection(direction)
    newimage.SetSpacing(spacing)

    return newimage


def square3DImage(image):
    # makes 3D image square in XY (pads with 0s)
    [numcols, numrows, numslices] = image.GetSize()
    npy = sitk.GetArrayFromImage(image)

    if numcols < numrows:    #pad columns
        numzerostopad = numrows - numcols
        leftpad = int(numzerostopad / 2)
        rightpad = numzerostopad - leftpad

        newnpy = np.concatenate((np.zeros([numslices, numrows, leftpad]), npy, np.zeros([numslices, numrows, rightpad])), axis=2)

        topleft = [-leftpad, 0, 0]
        neworigin = image.TransformIndexToPhysicalPoint(topleft)

    elif numrows <= numcols:  #pad rows
        numzerostopad = numcols - numrows
        toppad = int(numzerostopad / 2)
        botpad = numzerostopad - toppad

        newnpy = np.concatenate((np.zeros([numslices, toppad, numcols]), npy, np.zeros([numslices, botpad, numcols])), axis=1)

        topleft = [0, -toppad, 0]
        neworigin = image.TransformIndexToPhysicalPoint(topleft)

    paddedimg = sitk.GetImageFromArray(newnpy)
    paddedimg.SetOrigin(neworigin)
    paddedimg.SetDirection(image.GetDirection())
    paddedimg.SetSpacing(image.GetSpacing())

    return paddedimg


def resampleImage(image, finalsize, interpolation='linear'):
    # resamples square image to finalsize
    size = image.GetSize()
    numslices = size[2]
    squaresize = size[1]

    # RESAMPLE TO finalsize x finalsize
    finalnpy = np.zeros([numslices, finalsize, finalsize])
    reference = sitk.GetImageFromArray(finalnpy)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())

    spacing = image.GetSpacing()
    newspacing = np.zeros(3)
    newspacing[0:2] = (squaresize - 1) * np.array(spacing[0:2]) / (finalsize - 1)
    newspacing[2] = spacing[2]
    reference.SetSpacing(newspacing)

    # MAKING RESAMPLING FILTER
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    if interpolation == 'linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'NN':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # RESAMPLE TO finalsize x finalsize x n
    resampledimg = resample.Execute(image)

    return resampledimg


def projectImage(reference, moving, interpolate = 'linear'):
    # projects moving image onto reference image space
    # use  interpolate = 'NN' for segmentation masks

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    if interpolate == 'linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolate == 'NN':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    resampledimg = resample.Execute(moving)

    return resampledimg


def removedashandspace(string):
    string = string.replace(' ', '')
    string = string.replace('-', '')
    string = string.replace('_', '')

    return string

def padzeros(string, length):
    string = str(string)

    while len(string) < length:
        string = '0' + string

    return string

def findDuplicates(l):
    return list(set([x for x in l if l.count(x) > 1]))

def parseplane(direction):
    try:
        #direction2 = direction.replace('(', '')
        #direction2 = direction2.replace(')', '')
        #direction3 = direction2.split(',')
        directionnpy = np.array(direction).astype('float')
        absdirectionnpy = np.absolute(directionnpy)

        #axial is ijk
        axialvecs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

        #sagittal is ikj
        sagittalvecs = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0])

        #coronal is kij
        coronalvecs = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])

        axialdp = np.dot(absdirectionnpy, axialvecs) / 3
        coronaldp = np.dot(absdirectionnpy, coronalvecs) / 3
        sagittaldp = np.dot(absdirectionnpy, sagittalvecs) / 3

        if np.absolute(axialdp) > .8:
            parsed = 'axial'
            sign = np.sign(np.dot(directionnpy, axialvecs))
        elif np.absolute(coronaldp) > .8:
            parsed = 'coronal'
            sign = np.sign(np.dot(directionnpy, coronalvecs))
        elif np.absolute(sagittaldp) > .8:
            parsed = 'sagittal'
            sign = np.sign(np.dot(directionnpy, sagittalvecs))
        else:
            parsed = 'unknown'
            sign = 'unknown'

        if sign == 1:
            sign = 'positive'
        elif sign == -1:
            sign = 'negative'

        return([parsed, sign])
    except:
        return(['unknown', 'unknown'])


# Checks to see if the spacing between dicom slices is the same
def isGoodDcmSeries(dcmpaths):
    ds = pydicom.read_file(dcmpaths[0])
    pixelspacing = ds.PixelSpacing
    firstpos = ds.ImagePositionPatient
    imgorientationpatient = ds.ImageOrientationPatient
    xvec = imgorientationpatient[0:3]
    yvec = imgorientationpatient[3:6]
    zvec = np.cross(xvec, yvec)

    ### figure out order of DICOM files
    ### arrange based on z coord of imagepositionpatient, negative to positive
    normdistances = []

    for i, dcmpath in enumerate(dcmpaths):
        ds = pydicom.read_file(dcmpath)
        imgpositionpatient = np.array(ds.ImagePositionPatient)
        normdistances.append(np.dot(zvec, imgpositionpatient))

    # if distances are not approximately equal (< 10^-3 tolerance),
        # abort and output error MISSING SLICES
    sorteddistances = sorted(normdistances)
    distancediffs = []
    for i, distance in enumerate(sorteddistances):
        if i != 0:
            distancediffs.append(sorteddistances[i] - sorteddistances[i-1])
    if abs(min(distancediffs) - max(distancediffs)) > 1e-3:
        return False
    else:
        return True
