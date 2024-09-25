import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import SimpleITK as sitk
import math

import pickle

from collections import defaultdict
from scipy.ndimage.interpolation import rotate
from torch.autograd import Variable

def rotateImage(original, anglex, angley, anglez, interpolate='linear'):
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
    pixelcenter = np.array(sitk.GetSize(original)) / 2.
    physicalcenter = sitk.TransformContinuousIndexToPhysicalPoint(pixelcenter)

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
    neworigin = (matrix @ (origin - physicalcenter)) + physicalcenter

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

def bbox_3D(img):
    z = np.any(img, axis=(1, 2))    #z
    c = np.any(img, axis=(0, 2))    #y
    r = np.any(img, axis=(0, 1))    #x

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    #x min max, y min max, z min max
    return [rmin, rmax, cmin, cmax, zmin, zmax]

def cropImage(image, threshold, xshift, yshift):
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
    topleft = [int(xmin+xshift), int(ymin+yshift), zmin]
    neworigin = image.TransformIndexToPhysicalPoint(topleft)

    newimage.SetOrigin(neworigin)
    newimage.SetDirection(direction)
    newimage.SetSpacing(spacing)

    return newimage

######################
# PAD WITH zeros
# RESAMPLE TO 256 x 256

def squareImage(image):
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

    # CHECKING MY WORK!
    #referencecenter = fixed.TransformContinuousIndexToPhysicalPoint([512/2., 512/2., 0])
    # reference center (-5.521988951588327, -28.465328686857873, -62.08827209472656)
    #oldcenter = paddedimg.TransformContinuousIndexToPhysicalPoint([squaresize/2., squaresize/2., 0])
    # old center (-5.542247489094734, -28.48558722436428, -62.08827209472656)


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
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    if interpolate == 'linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolate == 'NN':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    resampledimg = resample.Execute(moving)

    return resampledimg


#########################


class Dataset(data.Dataset):
    def __init__(self, imgpaths, segpaths, args, augment=False, train=False):
        super().__init__()
        self.args = args
        self.train = train
        self.augment = augment

        self.imgpaths = imgpaths
        self.segpaths = segpaths


    def __getitem__(self, index, slice=0):
        imgpath = self.imgpaths[index]
        segpath = self.segpaths[index]

        img = sitk.ReadImage(imgpath)
        seg = sitk.ReadImage(segpath)

        if self.train:     # training set, randomly select slice
            # transform:
            if self.augment:
                # 1. rotate 3D image
                anglex = random.randint(0, 23) * 15  # rotate randomly, 15 degree increments from 0 to 345 degrees
                angley = anglez = 0    #just rolling pin rotation

                img = rotateImage(original, anglex, angley, anglez, 'linear')
                seg = rotateImage(original, anglex, angley, anglez, 'nn')

                # 2. crop with bounding box (shift crop box by a couple of pixels)
                threshold = 100
                xshift = random.randint(-5, 5)
                yshift = random.randint(-5, 5)

                img = cropImage(img, threshold, xshift, yshift)

                # 3. make square and resample to 256 x 256
                img = squareImage(img)
                img = resampleImage(img, 256)

                # 4. resample mask to final image
                seg = projectImage(img, seg, interpolate='NN')


            # calculate bounding box
            segbbox = bbox_3D(seg)       #[xmin, xmax, ymin, ymax, zmin, zmax]
            imgbbox = bbox_3D(img)

            # get random z slice, preferentially select for tumor slices
            tumorprob = 0.5

            zslicelist = range(imgbbox[4], imgbbox[5] + 1)   # goes from z min to z max
            tumorslices = range(segbbox[4], segbbox[5] + 1)
            nontumorslices = zslicelist - tumorslices

            # calculate volume of 0 pixels and 1 pixels from transformed seg
            numonepixels = np.sum(seg[zslicelist] == 1)
            numzeropixels = np.sum(seg[zslicelist] == 0)

            if random.random <= tumorprob:   # select slice with tumor
                slice = random.choice(tumorslices)
            else:
                slice = random.choice(nontumorslices)

        # get slice, divide by mean and std dev
        slice = self.slice

        imgslice = img[slice]
        segslice = seg[slice]

        imgslice = (imgslice - mean) / stdev
        #mean 374.91597254605796
        #std 823.4467560330842


        label_vol = segslice
        input_vol = np.stack([imgslice], axis=0)

        input_vol_tensor = torch.FloatTensor(input_vol)
        label_vol_tensor = torch.FloatTensor(label_vol)

        return input_vol_tensor, label_vol_tensor, prefix

    def __len__(self):
        return len(self.paths)



def load_data(args):
    #group = args.group
    T2dir = args.image_dir
    labeldir = args.label_dir

    print('Loading data...')


    ################################################
    # GET THIS FROM GENERATE TRAIN VAL TEST
    ################################################

    with open('trainvaltest.pickle', 'rb') as handle:
        [train_accessions, valid_accessions, test_accessions] = pickle.load(handle)

    def append_slices(accessions):
        dataset = []

        # file format: 0077_1_mb_stanford_0_slice_079_T2_img.npy
        npyfiles = [f for f in os.listdir(args.image_dir) if 'img' in f and f.split('_')[0] in accessions]

        prefixes = [f.split('_T2')[0] for f in npyfiles]
        prefixes.sort()

        for prefix in prefixes:
            dataset.append(prefix)

        return dataset



    train_names = append_slices(train_accessions)
    valid_names = append_slices(valid_accessions)
    test_names = append_slices(test_accessions)


    train_dataset = Dataset(train_names, args, train=True)
    valid_dataset = Dataset(valid_names, args)
    test_dataset = Dataset(test_names, args)


    print('num train slices: ', len(train_dataset))
    print('num valid slices: ', len(valid_dataset))
    print('num test slices: ', len(test_dataset))

    print('test accessions: ', test_accessions)


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print('done loading data!')



    return train_loader, valid_loader, test_loader    #, train_names, valid_names, test_names
