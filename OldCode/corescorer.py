import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt


# records the maximum probability for each core
# goes through all pixels that the core goes through


#######################
# FUNCTIONS #
#######################
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


##################################
# START CODE #
##################################
rootdir = '/data/prostate/postimages/Augmented/'
os.chdir(rootdir)
outdir = '/data/prostate/postimages/AugmentedLabelsPercent/'

# open csv file with core + accession data
mrpathwlocpath = '/data/prostate/MRpathwcorelocs3.csv'
mrpathlocdf = pd.read_csv(mrpathwlocpath, dtype=object, index_col=0)


counter = 0
for batch in loader:
    accession = loader.dataset.paths[counter][0]
    print(accession)
    counter += 1

    vol, label = batch
    if loader.dataset.use_gpu:
        vol = vol.cuda()
        label = label.cuda()
    vol = Variable(vol)
    label = Variable(label)

    logit = model.forward(vol)

    probs = torch.sigmoid(logit).cpu().detach().numpy()




#For each accession number in list
for accession in accessions:
    accessiondf = mrpathlocdf[mrpathlocdf.MRaccession == accession]
    mrn = accessiondf.mrn.values[0]
    date = accessiondf.biopsydate.values[0]

    # Load the model prediction MHA or numpy
    predictionnpy = 0  ############

    T2filename = accession + '_T2.mha'
    T2filepath = os.path.join(rootdir, T2filename)
    T2image = sitk.ReadImage(T2filepath, sitk.sitkUInt8)

    for i, row in accessiondf.iterrows():
        maxprob = 0

        corebot = [float(row.corebotx), float(row.coreboty), float(row.corebotz)]
        coretip = [float(row.coretipx), float(row.coretipy), float(row.coretipz)]

        # npy is z, y, x
        for zslicei in range(np.shape(corelocnpy)[0]):  #######
            imgpositionpatient = T2image.TransformIndexToPhysicalPoint([0, 0, zslicei])

            # first pass: solve for all of the points in each slice

            [col, row, t] =  corepixelsolver(imgpositionpatient, imgorientationpatient, corebot, coretip, rowspacing, colspacing)

            if 0 <= t <= 1:      # can modify this to only include central __% of core
                                 # i.e. 60% would be 0.2 <= t <= 0.8
                rowlist.append(row)
                collist.append(col)
                pathlist.append(path)
                percentlist.append(percent)

                #############


        For each slice in MHA file
        If core goes through slice
          Record predicted probability
          # may need to implement core radius here?
          if maxprob < pixelprob
            maxprob = pixelprob

      Store maximum probability for the core

Save as new file



  # calculate ROC curves on core level
  # cancer = 6+ or 7+
  # if prob < threshold, then it is benign
  # if prob > threshold, then it is cancerous
