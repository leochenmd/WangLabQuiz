import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk

##### OUTPUTS INDIVIDUAL SLICES OF SCANS

## FORMAT IS ACCESSION_SLICE_T2.npy

T2DWIADCdir = '/data/prostate/stanfordAllCropped_128'
segdir = '/data/prostate/stanfordAllCropped_128'
lesionlabeldir = '/data/prostate/stanfordLesionsCropped_128'

outdir = '/data/prostate/stanfordAllCropped_128_slices'
accessions = [f.split('_')[0] for f in os.listdir(lesionlabeldir)]

for accession in accessions:
    T2path = os.path.join(T2DWIADCdir, accession + '_croppedT2.nii')
    DWIpath = os.path.join(T2DWIADCdir, accession + '_cropped_DWI_b1200.nii')
    ADCpath = os.path.join(T2DWIADCdir, accession + '_cropped_ADC.nii')
    label_path = os.path.join(lesionlabeldir, accession + '_cropped_label.nii')

    T2img = sitk.ReadImage(T2path)
    DWIimg = sitk.ReadImage(DWIpath)
    ADCimg = sitk.ReadImage(ADCpath)
    labelimg = sitk.ReadImage(label_path)

    T2npy = sitk.GetArrayFromImage(T2img)
    DWInpy = sitk.GetArrayFromImage(DWIimg)
    ADCnpy = sitk.GetArrayFromImage(ADCimg)
    labelnpy = sitk.GetArrayFromImage(labelimg)

    for i in range(T2npy.shape[0]):
        T2slice = T2npy[i]
        DWIslice = DWInpy[i]
        ADCslice = ADCnpy[i]
        labelslice = labelnpy[i]

        if i < 10:
            sliceindex = '0' + str(i)
        else:
            sliceindex = str(i)

        T2name = os.path.join(outdir, accession + '_' + sliceindex + '_T2.npy')
        DWIname = os.path.join(outdir, accession + '_' + sliceindex + '_DWI.npy')
        ADCname = os.path.join(outdir, accession + '_' + sliceindex + '_ADC.npy')
        labelname = os.path.join(outdir, accession + '_' + sliceindex + '_label.npy')

        np.save(T2name, T2slice)
        np.save(DWIname, DWIslice)
        np.save(ADCname, ADCslice)
        np.save(labelname, labelslice)
