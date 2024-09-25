import re
import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import filecmp
import os
import shutil

##################################
# Assortment of methods/functions

def removeslash(string):
    string = string.replace('/', '')
    string = string.replace('\\', '')

    return string

def removedashandspace(string):
    string = string.replace(' ', '')
    string = string.replace('-', '')
    string = string.replace('_', '')

    return string

def readxlslist(xlsstring):
    xlsstring = xlsstring.replace('(', '')
    xlsstring = xlsstring.replace('[', '')
    xlsstring = xlsstring.replace(']', '')
    xlsstring = xlsstring.replace('\"', '')
    xlsstring = xlsstring.replace('\'', '')
    xlsstring = xlsstring.replace(')', '')
    xlsstring = xlsstring.split(', ')

    return xlsstring

def labelseries(seriesname):
    # Processes seriesname and identifies it as T2, DWI, or ADC
    # returns 'error' if no T2, DWI, or ADC strings are found
    T2strings = ['T2']
    DWIstrings = ['DWI']
    ADCstrings = ['ADC', 'APPARENT DIFFUSION']
    DCEstrings = ['DCE', 'DYNAMIC CONTRAST']

    isT2 = isDWI = isADC = isDCE = False

    seriesname = seriesname.upper()

    for T2string in T2strings:
        p = re.compile(T2string)
        if re.search(p, seriesname):
            isT2 = True

    for DWIstring in DWIstrings:
        p = re.compile(DWIstring)
        if re.search(p, seriesname):
            isDWI = True

    for ADCstring in ADCstrings:
        p = re.compile(ADCstring)
        if re.search(p, seriesname):
            isADC = True

    for DCEstring in DCEstrings:
        p = re.compile(DCEstring)
        if re.search(p, seriesname):
            isDCE = True

    if isDCE:
        return 'DCE'
    elif isADC:
        return 'ADC'
    elif isDWI:
        return 'DWI'
    elif isT2:
        return 'T2'
    else:
        return 'error'

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

##############################################



def increment_filename(filename, marker="-"):
    """Appends a counter to a filename, or increments an existing counter."""
    basename, fileext = os.path.splitext(filename)

    # If there isn't a counter already, then append one
    if marker not in basename:
        components = [basename, 1, fileext]

    # If it looks like there might be a counter, then try to coerce it to an
    # integer and increment it. If that fails, then just append a new counter.
    else:
        base, counter = basename.rsplit(marker, 1)
        try:
            new_counter = int(counter) + 1
            components = [base, new_counter, fileext]
        except ValueError:
            components = [base, 1, fileext]

    # Drop in the marker before the counter
    components.insert(1, marker)

    new_filename = "%s%s%d%s" % tuple(components)
    return new_filename

def copyfile(src, dst):
    """Copies a file from path src to path dst.

    If a file already exists at dst, it will not be overwritten, but:

     * If it is the same as the source file, do nothing
     * If it is different to the source file, pick a new name for the copy that
       is distinct and unused, then copy the file there.

    Returns the path to the copy.
    """
    if not os.path.exists(src):
        raise ValueError("Source file does not exist: {}".format(src))

    # Create a folder for dst if one does not already exist
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    # Keep trying to copy the file until it works
    while True:

        # If there is no file of the same name at the destination path, copy
        # to the destination
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            return dst

        # If the namesake is the same as the source file, then we don't need to
        # do anything else
        if filecmp.cmp(src, dst):
            return dst

        # There is a namesake which is different to the source file, so pick a
        # new destination path
        dst = increment_filename(dst)

    return dst
