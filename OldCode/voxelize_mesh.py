# Loads segmentation dicom file and extracts mesh
# Saves mesh as .stl file (can read with trimesh package)

# updated 3/1/2020

# Imports
import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
from pydicom.tag import Tag
import re
import SimpleITK as sitk
import trimesh
import scipy
#import mesh_to_sdf
from mesh_to_sdf import mesh_to_sdf

# mesh file
meshfile = '10174019_7789442_seg.stl'
meshdir = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML/Images/'
meshpath = os.path.join(meshdir, meshfile)

# T2 mha file
mhapath = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML/Images/7789442_20150206_0.mha'

# out file
segmhafile = '10174019_7789442_seg.mha'

# load files
mesh = trimesh.load(meshpath)
T2img = sitk.ReadImage(mhapath)
origin = T2img.GetOrigin()
direction = T2img.GetDirection()
spacing = T2img.GetSpacing()


T2npy = sitk.GetArrayFromImage(T2img)

segnpy = np.zeros(T2npy.shape)

# generate the 3 x n array of all voxels in the T2
print(T2npy.shape)
zindex = 0
yindex = 0
xindex = 0

query_points = []
indices = []
for zindex in range(T2npy.shape[0]):
    for yindex in range(T2npy.shape[1]):
        for xindex in range(T2npy.shape[2]):
            [xcoord, ycoord, zcoord] = T2img.TransformIndexToPhysicalPoint([xindex, yindex, zindex])
            query_points.append([xcoord, ycoord, zcoord])
            indices.append([xindex, yindex, zindex])

query_points = np.array(query_points)
indices = np.array(indices)
#print(query_points)

sdfs = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=1000000, normal_sample_count=11)

for i, sdf in enumerate(sdfs):
    [xindex, yindex, zindex] = indices[i, :]
    if sdfs[i] <= 0:
        segnpy[zindex, yindex, xindex] = 1

segimg = sitk.GetImageFromArray(segnpy)
segimg.SetOrigin(origin)
segimg.SetDirection(direction)
segimg.SetSpacing(spacing)

writer = sitk.ImageFileWriter()
writer.Execute(segimg, os.path.join(meshdir, segmhafile), True)
