import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt

#T2dir = '/data/prostate/postimages/PostImages2/'
#coredir = '/data/prostate/postimages/Labels/'

T2dir = '/Users/Leo/Desktop/Temp/'
coredir = '/Users/Leo/Desktop/Temp/'
augmentdir = '/Users/Leo/Desktop/Temp/augmented'


os.chdir(T2dir)

paththreshold = 'All'
coreradius = 3

# transform constats
angles = [10]


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

    unitvecs = np.reshape(original.GetDirection(), (-1, 3))
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


reader = sitk.ImageFileReader()
writer = sitk.ImageFileWriter()


if 0 == 0:
    accession = '8099139'

# run through resampled ADC files
#for ADCfile in glob.glob('*ADCresampled.mha'):
#    accession = ADCfile.split('_')[0]

    # read files into images
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
