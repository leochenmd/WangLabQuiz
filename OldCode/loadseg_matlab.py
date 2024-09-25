# Loads segmentation dicom file and extracts mesh
# Saves mesh as .stl file (can read with trimesh package)

# creatd 12/9/2019
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
#from mesh_to_sdf import mesh_to_voxels



# import file path
segdir = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML/PATDBASE/10174019/20150331/090506'
MRdir = segdir
filepath = 'seg'

# save mesh as
meshfile = '10174019_7789442_seg.stl'
meshdir = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML/Images/'
meshpath = os.path.join(meshdir, meshfile)

# mha path
#mhapath = '/Users/Leo/Documents/Stanford Sensitive/Python/BiopsyML/Images/7789442_20150206_0.mha'

######################################################
### PROCESS MR DIRECTORY #############################
######################################################
def processdicoms(mrdir):
    #processes MR folder and returns:
    #assessionnum, mrdate, slices, imgorientationpatient, imgpositionpatientarr, pixelspacing,
    #   zspacing, xyzdim, errormsg
    #if savenpy == True, then saves to npydir
    accessionnum = ''
    mrdate = ''
    slices = []
    imgorientationpatient = []
    imgpositionpatientarr = []
    pixelspacing = 0
    zspacing = 0
    xyzdim = []
    errormsgs = ''


    # finds all the MR files   labeled 'MR##'
    searchstr = mrdir + '/MR*'
    onlymrfiles = [f for f in glob.glob(searchstr)]

    if len(onlymrfiles) == 0:
        errormsgs = errormsgs + 'No MR files in folder; '
    else:
    # get imageorientationpatient, pixelspacing, accessionnum, mrdate from first slice
        try:
            ds = pydicom.read_file(onlymrfiles[0])
            pixelspacing = ds.PixelSpacing
            accessionnum = removedashandspace(ds.AccessionNumber)
            mrdate = ds.StudyDate
            firstpos = ds.ImagePositionPatient
            imgorientationpatient = ds.ImageOrientationPatient
            xvec = imgorientationpatient[0:3]
            yvec = imgorientationpatient[3:6]
            zvec = np.cross(xvec, yvec)

            ### figure out order of DICOM files
            ### arrange based on z coord of imagepositionpatient, negative to positive
            normdistances = []
            for i, mrfile in enumerate(onlymrfiles):
                ds = pydicom.read_file(mrfile)
                imgpositionpatient = np.array(ds.ImagePositionPatient)
                normdistances.append(np.dot(zvec, imgpositionpatient))

            # if distances are not approximately equal (< 10^-3 tolerance),
                # abort and output error MISSING SLICES
            sorteddistances = sorted(normdistances)
            distancediffs = []
            for i, distance in enumerate(sorteddistances):
                if i != 0:
                    distancediffs.append(sorteddistances[i] - sorteddistances[i-1])
            if abs(min(distancediffs) - max(distancediffs)) > 1e-3:
                errormsgs = 'MISSING SLICES (distances not equal); '
            else:    #if distances are equal to tolerance 1e-3
                # sort dicom files into sortedmris
                sortedi = sorted(range(len(normdistances)), key=lambda k: normdistances[k])
                sortedmris = []
                sortednormdistances = sorted(normdistances)
                for i, mrfile in enumerate(onlymrfiles):
                    sortedmris.append(onlymrfiles[sortedi[i]])

                ### go through dicom files in order to generate what we are looking for
                ##    From DICOMs, get the accession number, MRN, image orientation patient
                ###      image position patient, xy pixel spacing, slice thickness, xyz dimensions
                ###      and 3D numpy matrix with pixel intensities
                for i, mrfile in enumerate(sortedmris):
                    ds = pydicom.read_file(mrfile)
                    imgpositionpatientarr.append(ds.ImagePositionPatient)
                    slices.append(np.array(ds.pixel_array))

                #print(imgpositionpatientarr)

                mindistance = sortednormdistances[0]
                minimgpositionpatient = imgpositionpatientarr[0]
                zspacing = sortednormdistances[1] - sortednormdistances[0]

                # output np file with pixel array

                slices = np.stack(slices, axis=2)
                xyzdim = np.shape(slices)
        except:
            errormsgs = errormsgs + 'CANNOT READ DICOM; '

    return([accessionnum, mrdate, slices, imgorientationpatient, imgpositionpatientarr, pixelspacing, zspacing, xyzdim, errormsgs])



########################################
# PROCESS SEGMENTATION FILE
########################################
segpath = os.path.join(segdir, filepath)


#read file
seg =  pydicom.read_file(segpath)

#print(seg)

# includes prostate segmentation
numROIs = seg[0x66, 0x1].value

ROIindex = 0

# for each ROI:
#for ROIindex in range(numROIs):
if ROIindex == 0:
    #print(seg[0x66, 0x2][ROIindex][])
    ROInumber = seg[0x66, 0x2][ROIindex][0x66, 0x3].value
    ROIname = seg[0x66, 0x2][ROIindex][0x66, 0x4].value
    # number of points in ROI
    numPoints = seg[0x66, 0x2][ROIindex][0x66, 0x11][0][0x66, 0x15].value

    # x, y, z, coordinates of points
    points = seg[0x66, 0x2][ROIindex][0x66, 0x11][0][0x66, 0x16].value
    #points = seg.SurfaceSequence[ROIindex].SurfacePointsSequence[0].PointCoordinatesData
    xpoints = points[0::3]  #points, x coord
    ypoints = points[1::3]  #points, y coord
    zpoints = points[2::3]  #points, z coord

    points = np.reshape(points, (int(len(points)/3), 3))

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


    print('ROI index, ROI number, ROI name: ', ROInumber, ROIname)
    print('vertices of 5+1 th triangle', triangleindices[5])
    print('coords of 10+1 th point', xpoints[10], ypoints[10], zpoints[10])

    # if ROI is prostate

    # else if ROI is lesion


    print(np.shape(triangleindices))
    print(np.shape(points))

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
    mesh.export(meshpath)
