import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
from scipy import ndimage



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
    elif interpolate == 'bspline':
        interpolator = sitk.sitkBSpline
    elif interpolate == 'nn':
        interpolator = sitk.sitkNearestNeighbor

    rotatedImage = sitk.Resample(original, original, transform, interpolator)
    # uncomment if you want pre and post-transform to have the same MR space
    #rotatedImage.SetDirection(newdirection)
    #rotatedImage.SetOrigin(neworigin)


    return rotatedImage

def rotate2DImage(original, angle, interpolate='linear'):
    # returns the 'rotated' 2d image about the physical center that is resampled
    #  based on the original image
    #  original is the original image
    #   angle x is counterclockwise in degrees

    if interpolate == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolate == 'NN':
        interpolator = sitk.sitkNearestNeighbor
    elif interpolate == 'bspline':
        interpolator = sitk.sitkBSpline

    radx = angle * math.pi / 180

    origin = np.array(original.GetOrigin())
    pixelcenter = np.array(original.GetSize()) / 2.
    physicalcenter = original.TransformContinuousIndexToPhysicalPoint(pixelcenter)

    transform = sitk.Euler2DTransform()
    transform.SetCenter(physicalcenter)
    transform.SetAngle(radx)

    rotatedImage = sitk.Resample(original, original, transform, interpolator)

    return rotatedImage


def rotate2Dslice(npy, angle, interpolate='linear'):
    img = sitk.GetImageFromArray(npy)
    rotatedimg = rotate2DImage(img, angle, interpolate)
    rotatednpy = sitk.GetArrayFromImage(rotatedimg)
    return rotatednpy

def flipImage(original):
    # flips 3D mha/nifti image (reverses x indices)
    flipped = sitk.Flip(original, [True, False, False])
    return flipped

def flipSlice(original):
    # flips 2D NUMPY slice (reverses 'x-axis' indices)
    flipped = np.flipud(original)  #np.fliplr(original)
    return flipped




def cropImageUsingThreshold(image, threshold, xshift, yshift):
    # crops 3D image around bounding box (makes square)
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

    xmin = int((width - size) / 2)
    xmax = width - xmin

    ymin = int((height - size) / 2)
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

    xmin = (width - xysize) / 2
    xmax = width - xmin

    ymin = (height - xysize) / 2
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
    elif interpolation == 'bspline':
        resample.SetInterpolator(sitk.sitkBSpline)

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
    elif interpolate == 'bspline':
        resample.SetInterpolator(sitk.sitkBSpline)

    resampledimg = resample.Execute(moving)

    return resampledimg



def rotateAndCrop3Drandom(image, mask, minangle, maxangle, minshift, maxshift, xysize, interpolate='linear'):
    # rotates an image and mask, then crops
    # rotation angle is random float between minagle and maxangle
    # x and y shifts are integers between minshift and maxshift
    # interpolation = 'linear', 'NN', or 'bspline'

    # rotate image
    angle = random.uniform(minangle, maxangle)

    rotatedImage = rotate3DImage(image, angle, 0, 0, interpolate)
    rotatedMask = rotate3DImage(mask, angle, 0, 0, interpolate='NN')

    # crop image
    xshift = random.randint(minshift, maxshift)
    yshift = random.randint(minshift, maxshift)


    rotatedAndCroppedImage = crop3DImage(rotatedImage, xysize, xshift, yshift)
    rotatedAndCroppedMask = crop3DImage(rotatedMask, xysize, xshift, yshift)

    return [rotatedAndCroppedImage, rotatedAndCroppedMask]


def rotateAndCrop2Drandom(image, mask, minangle, maxangle, minshift, maxshift, xysize, interpolate='linear'):
    # rotates an image, then crops
    # rotation angle is random float between minagle and maxangle
    # x and y shifts are integers between minshift and maxshift
    # interpolation = 'linear', 'NN', or 'bspline'

    # rotate image
    angle = random.uniform(minangle, maxangle)
    rotatedImage = rotate2Dslice(image, angle, interpolate)
    rotatedMask = rotate2Dslice(mask, angle, interpolate='NN')

    # crop image
    xshift = random.randint(minshift, maxshift)
    yshift = random.randint(minshift, maxshift)

    rotatedAndCroppedImage = crop2DSlice(rotatedImage, xysize, xshift, yshift)
    rotatedAndCroppedMask = crop2DSlice(rotatedMask, xysize, xshift, yshift)

    return [rotatedAndCroppedImage, rotatedAndCroppedMask]


## On the fly
imgpath = '/Users/Leo/Documents/temp/001_0100_0_slice_016_img.png'
maskpath = '/Users/Leo/Documents/temp/001_0100_0_slice_016_mask.png'

minangle = -10   # degrees
maxangle = 10
minshift = -10   # pixels
maxshift = 10
xysize = 128     # pixels

img = sitk.ReadImage(imgpath)
imgnpy = sitk.GetArrayFromImage(img)

mask = sitk.ReadImage(maskpath)
masknpy = sitk.GetArrayFromImage(mask)

[newimage, newmask] = rotateAndCrop2Drandom(imgnpy, masknpy, minangle, maxangle, minshift, maxshift, xysize, interpolate='linear')
