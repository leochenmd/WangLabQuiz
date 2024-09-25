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
angles = [5, 10]


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
    print(unitvecs)
    matrix = np.reshape(transform.GetMatrix(), (-1, 3))
    inverse = np.linalg.inv(matrix)


    # the transform matrix is actually mapping backwards: post to pre
    # therefore the forward transformation is the inverse matrix
    transformedunitvecs = inverse @ unitvecs   # new i, j, k are columns
    print(transformedunitvecs)
    newdirection = transformedunitvecs.flatten('F')    # flatten by column

    print(newdirection)
    neworigin = (matrix @ (origin - center)) + center

    rotatedImage = sitk.Resample(original, original, transform, interpolator)
    rotatedImage.SetDirection(newdirection)
    rotatedImage.SetOrigin(neworigin)

    return rotatedImage




def flipImage(original):
    flipped = sitk.Flip(original, [True, False, False])

    return flipped


writer = sitk.ImageFileWriter()


if 0 == 0:
    accession = '8099139'

    ADCfile = accession + '_ADCresampled.mha'
    T2file = accession + '_T2.mha'

    T2path = os.path.join(T2dir, T2file)
    ADCpath = os.path.join(T2dir, ADCfile)

    T2image = sitk.ReadImage(T2path, sitk.sitkFloat32)
    ADCimage = sitk.ReadImage(ADCpath, sitk.sitkFloat32)

    origin = T2image.GetOrigin()
    direction = T2image.GetDirection()
    spacing = T2image.GetSpacing()

    # finds the center of the MR scan in physical coordinates
    shape = np.array(T2image.GetSize())
    pixelcenter = (shape / 2).astype(int).tolist()
    physicalcenter = T2image.TransformIndexToPhysicalPoint(pixelcenter)


    # generates and saves rotations
    # when angles y and z = 0, only twists body like rolling pin
    angley = 0
    anglez = 0

    T2flipname = accession + '_T2flip1.mha'

    anglex = 15

    rotatedT2 = rotateImage(T2image, physicalcenter, anglex, angley, anglez)
    flipT2 = flipImage(T2image)
    print(T2image.GetDirection())
    print(flipT2.GetDirection())
    print(rotatedT2.GetDirection())

#    rotatedADC = rotateImage(ADCimage, physicalcenter, anglex, angley, anglez)

    #print(rotatedT2.GetSize())
    #writer.Execute(rotatedT2, 'rotatedT2_15.mha', True)
