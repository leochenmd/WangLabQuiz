# Loads segmentation dicom file
# updated 12/9/2019

import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
from pydicom.tag import Tag
import re
import SimpleITK as sitk


segdir = '/users/leo/desktop/temp/20180709'
filepath = 'SEG1.3.6.1.4.1.30323.238170551979660.20180813.085037671'

segpath = os.path.join(segdir, filepath)

seg =  pydicom.read_file(segpath)

print(seg)

#print(seg)

points = seg.SurfaceSequence[0].SurfacePointsSequence[0].PointCoordinatesData
print('============')
tripoints = seg.SurfaceSequence[0].SurfaceMeshPrimitivesSequence[0].TrianglePointIndexList

#print(tripoints)
xpoints = points[0::3]
ypoints = points[1::3]
zpoints = points[2::3]

tripointslist = []
reallist = []
for tripoint in tripoints:
    tripointslist.append(tripoint)

tripointslist2 = tripointslist[1:]
tripointslist2 = tripointslist2[:-1]
#tripointslist2 = tripointslist

#print(tripointslist2)
ones = []
zeros = []
for i in range(int(len(tripointslist2)/2)):
    one = tripointslist2[2*i]
    zero = tripointslist2[2*i+1]
    ones.append(tripointslist2[2*i])
    zeros.append(tripointslist2[2*i+1])

    number = (256 * one) + zero
    reallist.append(number)


hexdf = pd.DataFrame([ones, zeros])
hexdf = hexdf.transpose()

print(hexdf)

print(max(reallist))

points2df = pd.DataFrame(reallist)

points2df.to_excel('/users/leo/desktop/temp/points.xls')


print('============')

pointsdf = pd.DataFrame([xpoints, ypoints, zpoints])
pointsdf = pointsdf.transpose()

print(np.shape(pointsdf)[0])



#0 is Profuse segmentation of prostate
#1 is FusionWS segmentation of lesion L-1


#triangletag = Tag(0x66,0x23)
#pointcloudtag = Tag(0x66, 0x16)

#print(seg)
#print(seg[0x66, 0x16].value)
