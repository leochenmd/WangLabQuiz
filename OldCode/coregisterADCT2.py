#########################
# Registers selected ADC and T2images
# 1. Read corresponding ADC as ITK
# 3. Rigid/rotation (6 DOF) transformation of DWI (moving) onto T2 (fixed)
# 4. Adaptive stochastic gradient descent
# 5. Linear interpolation (vs. bspline 3rd degree)
# 6. Save transformed DWI as numpy/ITK
# 7. The original coordinates will still be based on T2
# 8. Threshold based on transformation distance and/or min. mutual information

#%env SITK_SHOW_COMMAND '/Applications/Slicer.app/Contents/MacOS/Slicer'

import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk

from Leofunctions import removedashandspace
from Leofunctions import readxlslist


def save_transform_and_image(transform, fixed_image, moving_image, outputfile_prefix):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid and save the
    result to file.

    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
        outputfile_prefix (string): transform is written to outputfile_prefix.tfm and resampled image is written to
                                    outputfile_prefix.mha.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)

    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)
    sitk.WriteImage(resample.Execute(moving_image), outputfile_prefix+'.mha')
    sitk.WriteTransform(transform, outputfile_prefix+'.tfm')


hdpath = 'C:\\ProcessedFusionImages'
mhadir = os.path.join(hdpath, 'PreImages')
postdir = os.path.join(hdpath, 'PostImages2')
linkedxlspath = os.path.join(hdpath, 'linkedT2ADC.xls')
coregisterxlspath = os.path.join(hdpath, 'coregister.xls')
errorpath = os.path.join(hdpath, 'coregerrors.csv')
#errorpath = os.path.join(hdpath, 'fusionimageserrors.xls')
#coregisterdir = os.path.join(hdpath, 'CoregisteredImages')

df = pd.read_excel(linkedxlspath)

accession_list = []
anglex_list = []
angley_list = []
anglez_list = []
deltax_list = []
deltay_list = []
deltaz_list = []
metric_list = []
distance_list = []
stop_list = []
errors_list = []

for i, row in df.iterrows():
    accession = row.accession

    try:
        if 0==0:
            print(accession)
            print(str(row.errormsg))
            if str(row.errormsg) == 'nan':
                #accession = row.accession.values[0]

                T2filepath = os.path.join(mhadir, row.T2mhafile)
                ADCfilepath = os.path.join(mhadir, row.ADCmhafile)

                fixed = sitk.ReadImage(T2filepath + '.mha', sitk.sitkFloat32)
                moving = sitk.ReadImage(ADCfilepath + '.mha', sitk.sitkFloat32)

                initialTx = sitk.CenteredTransformInitializer(fixed,
                                moving, sitk.Euler3DTransform(),
                                sitk.CenteredTransformInitializerFilter.GEOMETRY)

                registration_method = sitk.ImageRegistrationMethod()

                # Similarity metric settings
                #registration_method.SetShrinkFactorsPerLevel([1])
                #registration_method.SetSmoothingSigmasPerLevel([2,2,1])
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 25)
                registration_method.SetMetricSamplingPercentage(.1, sitk.sitkWallClock)
                registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)

                # Optimizer settings
                registration_method.SetOptimizerAsGradientDescent(learningRate=1,
                        numberOfIterations=100, convergenceMinimumValue=1e-6,
                        convergenceWindowSize=10)
                registration_method.SetOptimizerScalesFromPhysicalShift()

                registration_method.SetInitialTransform(initialTx)
                registration_method.SetInterpolator(sitk.sitkLinear)


                final_transform = registration_method.Execute(fixed, moving)

                metricvalue = registration_method.GetMetricValue()
                stopcondition = registration_method.GetOptimizerStopConditionDescription()
                parameters = final_transform.GetParameters()
                anglex = parameters[0]
                angley = parameters[1]
                anglez = parameters[2]
                deltax = parameters[3]
                deltay = parameters[4]
                deltaz = parameters[5]
                distance = math.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2)

                save_transform_and_image(final_transform, fixed, moving, os.path.join(postdir, str(accession) + '_ADCpost'))
                sitk.WriteImage(fixed, os.path.join(postdir, str(accession) + "_T2.mha"))
                sitk.WriteImage(moving, os.path.join(postdir, str(accession)+ "_ADCpre.mha"))

                outdata = np.array([anglex, angley, anglez, deltax, deltay, deltaz, metricvalue, stopcondition])
                np.save(os.path.join(postdir, str(accession) + '_transform.npy'), outdata)

                accession_list.append(accession)
                anglex_list.append(anglex)
                angley_list.append(angley)
                anglez_list.append(anglez)
                deltax_list.append(deltax)
                deltay_list.append(deltay)
                deltaz_list.append(deltaz)
                distance_list.append(distance)
                metric_list.append(metricvalue)
                stop_list.append(stopcondition)
    except:
        errors_list.append(accession)
        print('Error for accession ' + accession)

outDF = pd.DataFrame(data = [accession_list, anglex_list, angley_list, anglez_list,
                        deltax_list, deltay_list, deltaz_list, distance_list,
                        metric_list, stop_list])
outDF = outDF.transpose()
outDF.columns = ['accession', 'anglex', 'angley', 'anglez',
                'deltax', 'deltay', 'deltaz', 'distance',
                'metric', 'stopcondition']

outDF.to_excel(coregisterxlspath, sheet_name='Sheet1')

errorDF = pd.DataFrame(data = errors_list)
errorDF.to_csv()
