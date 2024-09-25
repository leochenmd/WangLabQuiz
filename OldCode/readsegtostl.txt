# Mines patdbdata/segdir and processes all of the SEG files
# saves them as individual .stl files
#   seg = prostate
#   les1, les2, les3 = lesion 1, lesion 2, lesion 3 ...

# Modified 3/24/2020

import numpy as np
import pandas as pd

import pydicom
import os
import SimpleITK as sitk

import trimesh
import scipy
from mesh_to_sdf import mesh_to_sdf


def transformtoindex(image, pointx, pointy, pointz):
    [z, y, x] = image.TransformPhysicalPointToContinuousIndex([pointx, pointy, pointz])
    return pd.Series([z, y, x])

patdbasedir = 'E:\\PATDBASE'
imagesdir = 'E:\\NPYimages'
datadir = 'E:\\patdbdata'
segdir = 'C:\\ProcessedFusionImages\\segdicoms'
T2dir = 'C:\\ProcessedFusionImages\\PostImages2'

writesegstl = False   ### change to 'True' if you want to save the files

hdpath = 'C:\\'
### This will mine fusion images
# NEED TO WRITE SOMETHING THAT MINES PATDBASE
processedfusionimagesdir = os.path.join(hdpath, 'processedfusionimages')
segcsvpath = os.path.join(processedfusionimagesdir, 'segstlnames.csv')

segdir = os.path.join(processedfusionimagesdir, 'segdicoms')
segstldir = os.path.join(processedfusionimagesdir, 'segstl')

os.chdir(segdir)
errorcounter = 0
processedsegfiles = 0
nummhanotfound = 0

outdata = []

### uncomment below and comment for loop to debug individual file
#segfile = os.path.join(segdir, '12154718_20180711.seg')
#if 1 == 1:

for segfile in os.listdir(segdir):
    accession = segfile.split('_')[0]
    date = segfile.split('_')[1]

    ### output file names
    segstlpath = os.path.join(segstldir, str(accession) + '_seg.stl')
    lesionstlpath = os.path.join(segstldir, str(accession) + '_lesion')

    lesioncounter = 0
    try:
        segdcm = pydicom.read_file(segfile)
        dicomerror = False
    except:
        print('Cannot read seg file for ' + str(accession))
        dicomerror = True

    if dicomerror == False:
        print('---------')
        print(accession)
        mhafound = False

        for i, sequence in enumerate(segdcm.SurfaceSequence):
            isLesion = False
            isProstate = False
            numPixels = 0

            name = sequence.SurfaceComments
            points = sequence.SurfacePointsSequence[0].PointCoordinatesData

            if 'lesion' in name:
                isLesion = True
                lesioncounter += 1
            elif 'prostate' in name:
                isProstate = True
            else:
                print('ERROR: neither lesion nor prostate in name!')
                errorcounter += 1

            # reads the triangle mesh from dicom

            # x, y, z, coordinates of points
            points = seg[0x66, 0x2][ROIindex][0x66, 0x11][0][0x66, 0x16].value
            #points = seg.SurfaceSequence[ROIindex].SurfacePointsSequence[0].PointCoordinatesData
            xpoints = points[0::3]  #points, x coord
            ypoints = points[1::3]  #points, y coord
            zpoints = points[2::3]  #points, z coord

            # triangles (faces)
            tripoints = seg.SurfaceSequence[ROIindex].SurfaceMeshPrimitivesSequence[0].TrianglePointIndexList
            tripointslist = []
            for tripoint in tripoints:
                tripointslist.append(tripoint)

            ones = []
            zeros = []
            reallist = []
            for i in range(int(len(tripointslist)/2)):
                one = tripointslist[2*i]
                zero = tripointslist[2*i+1]
                ones.append(tripointslist[2*i])
                zeros.append(tripointslist[2*i+1])

                number = (256 * zero) + one
                reallist.append(number)

            triangleindices = np.reshape(reallist, (int(len(tripointslist)/6), 3))

            # face normals
            normals = []
            for i, triangleindex in enumerate(range(len(triangleindices))):
                p1 = points[triangleindices[triangleindex, 0]]
                p2 = points[triangleindices[triangleindex, 1]]
                p3 = points[triangleindices[triangleindex, 2]]

                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)


            normals = np.array(normals)
            #print(normals)

            # mesh is points, faces, normals
            mesh = trimesh.Trimesh(points, triangleindices, normals)
            #print('watertight?', mesh.is_watertight)
            #mesh.show()

            if isProstate:
                filename = segstlpath
                prostateOrLesion = 'prostate'

                mesh.export(filename)
            elif isLesion:
                lesionpath = lesionstlpath + str(lesioncounter) + '.stl'
                prostateOrLesion = 'lesion'
                filename = lesionpath

                mesh.export(filename)
            else:
                filename = 'ERROR'
                prostateOrLesion = 'ERROR'

            outdata.append(dict(zip(['accession', 'prostateOrLesion', 'commentsname', 'filename', 'lesioncounter', 'numPixels'], [accession, prostateOrLesion, name, filename, lesioncounter, numPixels])))

    processedsegfiles += 1




outdf = pd.DataFrame(outdata)
outdf = outdf[['accession', 'prostateOrLesion', 'lesioncounter', 'numPixels', 'commentsname', 'filename']]
outdf.to_csv(segcsvpath)
print('Finished!')
print('Processed a total of seg files: ' + str(processedsegfiles))
#print('T2 mha not found for: ' + str(nummhanotfound))
print('Number of seg files without lesion or prostate in name: ' + str(errorcounter))
