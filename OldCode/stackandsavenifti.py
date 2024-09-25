#### IMPORTS ###
import numpy as np
import argparse
import os
import SimpleITK as sitk

#######################
### LOAD MODEL HERE ###
#######################


###############################
# SAVES SLICES AS NPYs

outdir = '/data/prostate/stanfordAllCropped_128_slices_pred/0314_unet_crop128_04'
#######

try:
    os.mkdir(outdir)
except:
    print('dir exists')

# load test cases
loader = test_loader


for i, batch in enumerate(loader):
    if 0 == 0:
        vol, label, accession = batch
        if loader.dataset.args.gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)

        logit = model.forward(vol)

        probs = torch.sigmoid(logit).cpu().detach().numpy()

        preds = np.zeros_like(probs)
        #preds[probs >= .9] = 1          # thresholding vs.
        preds = probs                    # actual probability


        for batch_item in range(preds.shape[0]):
            # saves each prediction slice as npy
            p = preds[batch_item][0]
            predslicepath = os.path.join(outdir, str(accession[batch_item]) + '_pred.npy')
            np.save(predslicepath, p)

print('done! saving prediction slices as npy')


##########################################
# STACKS PREDICTION SLICES
#  saves as mha/nifti files with metadata
#
predslicesdir = '/data/prostate/stanfordAllCropped_128_slices_pred/0314_unet_crop128_04'
T2croppeddir = '/data/prostate/stanfordAllCropped_128'

preddir = '/data/prostate/stanfordAllCropped_128_pred/0314_unet_crop128_04'   #final dir to save pred stacks
#####################################

try:
    os.mkdir(preddir)
except:
    print('dest dir exists')

writer = sitk.ImageFileWriter()

accessions = list(set([f.split('_')[0] for f in os.listdir(predslicesdir)]))
accessions.sort()

os.chdir(predslicesdir)

for accession in accessions:
    if 1 == 1:   # can change accession = ### to test one case
        # GET ALL SLICES FOR ONE PATIENT ACCESSION
        slices = [f for f in os.listdir(predslicesdir) if accession in f]
        slices.sort()

        prednpy = []

        # STACKS ALL OF THE NPY SLICES
        for i, file in enumerate(slices):
            if i == 0:
                prednpy = np.load(slices[0])
            else:
                npyslice = np.load(file)
                prednpy = np.dstack([prednpy, npyslice])   # append slice

        prednpy2 = np.transpose(prednpy, (2, 0, 1))   # (col, row, z) to (row, col z)

        # FINDS ORIGINAL T2/seg file (to get metadata)
        T2path = os.path.join(T2croppeddir, accession + '_croppedT2.nii')
        T2img = sitk.ReadImage(T2path)
        origin = T2img.GetOrigin()
        spacing = T2img.GetSpacing()
        direction = T2img.GetDirection()

        # Converts prediction npy stack to 3D image, assigns metadata
        predimg = sitk.GetImageFromArray(prednpy2)
        predimg.SetOrigin(origin)
        predimg.SetSpacing(spacing)
        predimg.SetDirection(direction)

        predpath = os.path.join(preddir, accession + '_pred.nii')

        # Write image to path
        writer.Execute(predimg, predpath, True)
