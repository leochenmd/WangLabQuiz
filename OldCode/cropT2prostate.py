##############################################################
# Leo Chen, 3/10/2019
#
##############################
# CROPS AROUND PROSTATE (80%)
# RESAMPLE TO 128 x 128 x n
#    (for 2D models)
##############################

# imports
import numpy as np
import pandas as pd
import os
import glob
import pydicom
import SimpleITK as sitk


# functions
def bbox_3D(img):

    z = np.any(img, axis=(1, 2))    #z
    c = np.any(img, axis=(0, 2))    #y
    r = np.any(img, axis=(0, 1))    #x

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    #x min max, y min max, z min max
    return rmin, rmax, cmin, cmax, zmin, zmax

###############################
# FINAL SIZE OF RESAMPLING
percentprostate = 0.8
finalsize = 128
widthprostate = int(percentprostate * 128)
###############################

# directories
T2dir = 'C:\\ProcessedFusionImages\\stanfordT2'
segdir = 'C:\\ProcessedFusionImages\\segmha'
#      out
T2croppeddir = 'C:\\ProcessedFusionImages\\stanfordT2cropped2D'
xlspath = 'C:\\ProcessedFusionImages\\croppedT2metadata.xls'

# INITIALIZE SIMPLE ITK READER/WRITER
writer = sitk.ImageFileWriter()

outdata = []


# identify/count number of patients with T2 and segmentations
os.chdir = segdir
segfiles = [f for f in os.listdir(segdir) if os.path.isfile(os.path.join(segdir, f)) and 'seg' in f]
segaccessions = [f.split('_')[0] for f in segfiles]
T2files = [f for f in os.listdir(T2dir) if os.path.isfile(os.path.join(T2dir, f))]
T2accessions = [f.split('_')[0] for f in T2files]

commonaccessions = list(set(segaccessions) & set(T2accessions))

print(len(commonaccessions))
print(len(T2accessions))


# for each T2
for i, accession in enumerate(commonaccessions):

    T2file = os.path.join(T2dir, str(accession) + '_T2.mha')
    segfile = os.path.join(segdir, str(accession) + '_seg.mha')

    # LOAD T2 and SEGMENTATION
    T2img = sitk.ReadImage(T2file)
    T2npy = sitk.GetArrayFromImage(T2img)
    size = T2img.GetSize()
    direction = T2img.GetDirection()
    spacing = T2img.GetSpacing()

    segimg = sitk.ReadImage(segfile)
    segnpy = sitk.GetArrayFromImage(segimg)

    # CALCULATE BOUNDING BOX
    [xmin, xmax, ymin, ymax, zmin, zmax] = bbox_3D(segnpy)

    width = xmax - xmin
    height = ymax - ymin

    largerdim = np.max([width, height])
    croppedwidth = croppedheight = int(largerdim / percentprostate)

    cropxmin = int((xmin+xmax - croppedwidth)/2)
    cropxmax = int((xmin+xmax + croppedwidth)/2)
    cropymin = int((ymin+ymax - croppedheight)/2)
    cropymax = int((ymin+ymax + croppedheight)/2)

    cropxmin = max([cropxmin, 0])
    cropxmax = min([cropxmax, size[0]])
    cropymin = max([cropymin, 0])
    cropymax = min([cropymax, size[0]])

    # CROP TO PROSTATE
    newnpy = T2npy[zmin:zmax, cropymin:cropymax, cropxmin:cropxmax]

    croppedimage = sitk.GetImageFromArray(newnpy)

    # calculate new origin
    topleft = [int(cropxmin), int(cropymin), int(zmin)]
    neworigin = T2img.TransformIndexToPhysicalPoint(topleft)

    croppedimage.SetOrigin(neworigin)
    croppedimage.SetDirection(direction)
    croppedimage.SetSpacing(spacing)


    # RESAMPLE TO FINAL SIZE (128 x 128)
    finalnpy = np.zeros([croppedimage.GetSize()[2], finalsize, finalsize])
    reference = sitk.GetImageFromArray(finalnpy)
    reference.SetOrigin(neworigin)
    reference.SetDirection(direction)

    newspacing = np.zeros([3])
    newspacing[0:2] = (croppedwidth - 1) * np.array(spacing[0:2]) / (finalsize-1)
    newspacing[2] = spacing[2]
    reference.SetSpacing(newspacing)


    # MAKING RESAMPLING FILTERS
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetInterpolator(sitk.sitkLinear)

    # nearest neighbor interpolation for segmentation mask
    resampleNN = sitk.ResampleImageFilter()
    resampleNN.SetReferenceImage(reference)
    resampleNN.SetInterpolator(sitk.sitkNearestNeighbor)


    # RESAMPLE TO 512 x 512 x n
    resampledimage = resample.Execute(croppedimage)
    resampledseg = resampleNN.Execute(segimg)


    # WRITE IMAGE AND SEGMENTATION
    writer.Execute(resampledimage, os.path.join(T2croppeddir, str(accession) + '_cropped_T2.nii'), True)
    writer.Execute(resampledseg, os.path.join(T2croppeddir, str(accession) + '_cropped_seg.nii'), True)


    # APPEND METADATA
    outdata.append(dict(zip(['accession', 'originalsize', 'originalspacing', 'croppedsize', 'croppedspacing'], [accession, size, spacing, resampledimage.GetSize(), resampledimage.GetSpacing()])))

    # display counter and periodic saving of metadata
    if i % 20 == 0:
        print(str(i) + ' of ' + str(len(commonaccessions)))

        outDF = pd.DataFrame(outdata)
        outDF = outDF[['accession', 'originalsize', 'originalspacing', 'croppedsize', 'croppedspacing']]
        outDF.to_excel(xlspath)


# save final xls file
outDF = pd.DataFrame(outdata)
outDF = outDF[['accession', 'originalsize', 'originalspacing', 'croppedsize', 'croppedspacing']]
outDF.to_excel(xlspath)

print('done!')
