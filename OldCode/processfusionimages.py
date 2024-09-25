#########################
# processes fusion images
# 2. Read corresponding ADC as ITK
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

xlsdir = '/users/Leo/documents/Stanford Sensitive/python/BiopsyML/'
os.chdir(xlsdir)

from methods import removedashandspace
from methods import readxlslist


hdpath = '/volumes/Seagate Backup Plus Drive/prostateml/'
fusionimagesdir = os.path.join(hdpath, 'fusionimages')
npyimagesdir = os.path.join(hdpath, 'npyimages')
patdbdatadir = os.path.join(hdpath, 'patdbdata')


xlspath = os.path.join(xlsdir, 'fusionimagesdata.xls')
OUTPUT_DIR = os.path.join(xlsdir, 'output')



def command_iteration(method) :
    if (method.GetOptimizerIteration() == 0):
        print("\tLevel: {0}".format(method.GetCurrentLevel()))
        print("\tScales: {0}".format(method.GetOptimizerScales()))
    print("#{0}".format(method.GetOptimizerIteration()))
    print("\tMetric Value: {0:10.5f}".format( method.GetMetricValue()))
    print("\tLearningRate: {0:10.5f}".format(method.GetOptimizerLearningRate()))
    if (method.GetOptimizerConvergenceValue() != sys.float_info.max):
        print("\tConvergence Value: {0:.5e}".format(method.GetOptimizerConvergenceValue()))


def command_multiresolution_iteration(method):
    print("\tStop Condition: {0}".format(method.GetOptimizerStopConditionDescription()))
    print("============= Resolution Change =============")


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


dimension = 3

df = pd.read_excel(xlspath)

#print(df.head(0))
#accesion, mrdate, seriesname, numslices, mrfiles

uniqueaccessions = list(set(df.accession))

#for accesionnum in uniqueaccessions:
if 0 == 0:
    accessionnum = uniqueaccessions[1]

    accessiondf = df[df.accession == accessionnum]

    multipleT2s = multipleDWIs = multipleADCs = False

    if accessiondf[accessiondf.serieslabel == 'T2'].shape[0] > 1:
        print('WARNING: Multiple T2 series detected')
        multipleT2s = True
    if accessiondf[accessiondf.serieslabel == 'DWI'].shape[0] > 1:
        multipleDWIs = True
        print('WARNING: Multiple DWI series detected')
    if accessiondf[accessiondf.serieslabel == 'ADC'].shape[0] > 1:
        multipleADCs = True
        print('WARNING: Multiple ADC series detected')

    if not (multipleT2s and multipleDWIs and multipleADCs):
        T2files = str(accessiondf[accessiondf.serieslabel == 'T2'].mrfiles.values[:])
        DWIfiles = str(accessiondf[accessiondf.serieslabel == 'DWI'].mrfiles.values[:])
        ADCfiles = str(accessiondf[accessiondf.serieslabel == 'ADC'].mrfiles.values[:])

        T2files = readxlslist(T2files)
        DWIfiles = readxlslist(DWIfiles)
        ADCfiles = readxlslist(ADCfiles)

        T2reader = sitk.ImageSeriesReader()
        DWIreader = sitk.ImageSeriesReader()
        ADCreader = sitk.ImageSeriesReader()

        T2reader.SetFileNames(T2files)
        DWIreader.SetFileNames(DWIfiles)
        ADCreader.SetFileNames(ADCfiles)

        T2image = T2reader.Execute()
        DWIimage = DWIreader.Execute()
        ADCimage = ADCreader.Execute()

        fixed = sitk.Cast(T2image, sitk.sitkFloat32)
        moving = sitk.Cast(ADCimage, sitk.sitkFloat32)

        #initialTx = sitk.TranslationTransform(3)

        initialTx = sitk.CenteredTransformInitializer(fixed,
                                                      moving,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Save images to file and view overlap using external viewer.
        sitk.WriteImage(fixed, os.path.join(OUTPUT_DIR, str(accessionnum) + "_T2.mha"))
        sitk.WriteImage(moving, os.path.join(OUTPUT_DIR, str(accessionnum)+ "_ADCpre.mha"))

        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetShrinkFactorsPerLevel([3,2,1])
        registration_method.SetSmoothingSigmasPerLevel([2,1,1])
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=10)
        registration_method.MetricUseFixedImageGradientFilterOff()
        #registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        #registration_method.SetMetricSamplingPercentage(0.2)



        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-5, convergenceWindowSize=5)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        registration_method.SetInitialTransform(initialTx)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Setup for the multi-resolution framework.

        #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initialTx)

        # Connect all of the observers so that we can perform plotting during registration.
        #registration_method.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(registration_method) )
        #registration_method.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(registration_method) )

        final_transform = registration_method.Execute(fixed, moving)

        #reg.SetMetricSamplingStrategy
        print(final_transform)
        metricvalue = registration_method.GetMetricValue()
        stopcondition = registration_method.GetOptimizerStopConditionDescription()
        parameters = final_transform.GetParameters()
        anglex = parameters[0]
        angley = parameters[1]
        anglez = parameters[2]
        deltax = parameters[3]
        deltay = parameters[4]
        deltaz = parameters[5]

        #moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        save_transform_and_image(final_transform, fixed, moving, os.path.join(OUTPUT_DIR, str(accessionnum) + '_ADCpost'))

        #interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_resampled));
    #label as T2, ADC, and DWI sequeces


        print(final_transform.GetParameters())
