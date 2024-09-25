import numpy as np
import os
import pandas as pd
import random
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.utils.data as data

from collections import defaultdict
from scipy.ndimage.interpolation import rotate
from torch.autograd import Variable

#T2_MEAN = 442.69894
#T2_STDDEV = 442.15298
T2_MEAN = 128.1895
T2_STDDEV = 72.66089

#ADC_MEAN = 586.25195
#ADC_STDDEV = 1561.6196
ADC_MEAN = 81.00176
ADC_STDDEV = 83.06546

class Dataset(data.Dataset):
    def __init__(self, exams, args, train=False):
        super().__init__()
        self.args = args
        self.train = train

        self.paths = [] # (accession, transformation)

        for accession in exams:
            self.paths.append((accession, ''))
            if self.train:
                for flip in range(2):
                    if flip == 1:
                        self.paths.append((accession, '_flip1'))
                    for direction in ['L', 'R']:
                        for amount in range(1, 6):
                            self.paths.append((accession, '_flip%d_rotate%s%02d' % (flip, direction, amount)))

#        means = []
#        variances = []
#        for a in exams:
#            path = AUG_IMAGE_DIR + a + '_T2.mha'
#            sitk_vol = sitk.ReadImage(path, sitk.sitkFloat32)
#            vol = sitk.GetArrayFromImage(sitk_vol)
#            means.append(np.mean(vol))
#            variances.append(np.var(vol))
#
#        print(np.mean(means), np.sqrt(np.mean(variances)))

    def __getitem__(self, index):

        accession, transformation = self.paths[index]
        t2_path = self.args.image_dir + accession + '_T2' + transformation + '.mha'
        adc_path = self.args.image_dir + accession + '_ADC' + transformation + '.mha'
        label_path = self.args.label_dir + accession + '_cores_r3_gAll' + transformation + '.mha'

        def crop_middle(vol):
            NUM_SLICES = self.args.num_slices
            CROP_DIM = self.args.crop_dim

            z, x, y = vol.shape
            z0 = int((z - NUM_SLICES) / 2.)
            x0 = int((x - CROP_DIM) / 2.)
            y0 = int((y - CROP_DIM) / 2.)
            return vol[z0:z0+NUM_SLICES, x0:x0+CROP_DIM, y0:y0+CROP_DIM]

        t2_sitk_vol = sitk.ReadImage(t2_path, sitk.sitkUInt8)
        t2_vol = sitk.GetArrayFromImage(t2_sitk_vol)
        t2_vol = crop_middle(t2_vol)

        t2_vol = (t2_vol - T2_MEAN) / T2_STDDEV

        adc_sitk_vol = sitk.ReadImage(adc_path, sitk.sitkUInt8)
        adc_vol = sitk.GetArrayFromImage(adc_sitk_vol)
        adc_vol = crop_middle(adc_vol)

        adc_vol = (adc_vol - ADC_MEAN) / ADC_STDDEV

        input_vol = np.stack([t2_vol, adc_vol], axis=0)

        label_sitk_vol = sitk.ReadImage(label_path, sitk.sitkUInt8)
        label_vol = sitk.GetArrayFromImage(label_sitk_vol)
        label_vol = crop_middle(label_vol)

        label_vol[label_vol < 6] = 0
        label_vol[label_vol >= 6] = 1

#         label_vol[label_vol < 0] = 0
#         label_vol /= 100

        input_vol_tensor = torch.FloatTensor(input_vol)
        label_vol_tensor = torch.FloatTensor(label_vol)

        return input_vol_tensor, label_vol_tensor

    def __len__(self):
        return len(self.paths)

def load_data(args):

    print('Loading data...')

    good_exams = [file_name[:file_name.index('_')] for file_name in os.listdir(args.label_dir) if 'r3_gAll' in file_name]

    print('total exams:', len(good_exams))

    random.seed(42)

    random.shuffle(good_exams)

    train_cutoff = int(.6*len(good_exams))
    valid_cutoff = int(.8*len(good_exams))

    train_exams = good_exams[:train_cutoff]
    valid_exams = good_exams[train_cutoff:valid_cutoff]
    test_exams  = good_exams[valid_cutoff:]

    print('train exams:', len(train_exams))
    print('valid exams:', len(valid_exams))
    print('test exams:' , len(test_exams))

    train_dataset = Dataset(train_exams, args, train=True)
    valid_dataset = Dataset(valid_exams, args)
    test_dataset = Dataset(test_exams, args)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader
