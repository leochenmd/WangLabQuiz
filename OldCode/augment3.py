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

def flipslice(original):
    # flips 2D slice (reverses x indices)
    flipped = np.flipud(original)  #np.fliplr(original)
    return flipped

def cropimage(original, mask):


    return croppedimage

def squareimage(original):



#To generate slice:
########Rotation xy plane, randomly (360 deg?)
#Crop
#-Vary threshold of bounding box
#-add offset by a few pixels to cropping bounding box
######## Flip +/-

#modify generalized dice loss to be per volume, not per slice
