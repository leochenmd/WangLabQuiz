import numpy as np
import pandas as pd

from pyntcloud import PyntCloud
import pydicom
import os
from scipy.ndimage.morphology  import binary_fill_holes
from skimage.morphology import convex_hull_object
import SimpleITK as sitk
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def transformtoindex(image, pointx, pointy, pointz):
    [z, y, x] = image.TransformPhysicalPointToContinuousIndex([pointx, pointy, pointz])
    return pd.Series([z, y, x])

patdbasedir = 'E:\\PATDBASE'
imagesdir = 'E:\\NPYimages'
datadir = 'E:\\patdbdata'
segdir = 'C:\\ProcessedFusionImages\\segdicoms'
T2dir = 'C:\\ProcessedFusionImages\\PostImages2'

writecsv = False

hdpath = 'C:\\'
processedfusionimagesdir = os.path.join(hdpath, 'processedfusionimages')
segcsvpath = os.path.join(processedfusionimagesdir, 'segnames.csv')

segdir = os.path.join(processedfusionimagesdir, 'segdicoms')
segmhadir = os.path.join(processedfusionimagesdir, 'segmha')

os.chdir(segmhadir)
errorcounter = 0
processedsegfiles = 0

outdata = []

searchstr = '_seg.mha'
onlysegfiles = [f for f in glob.glob(searchstr)]

if 1 == 1:
    accession = str(8246567)
    segfile = accession + '_seg.mha'

#for segfile in onlysegfiles:
    accession = segfile.split('_')[0]
    segmhapath = os.path.join(segmhadir, str(accession) + '_seg.mha')
    lesionmhapath = os.path.join(segmhadir, str(accession) + '_lesion')

    maxlesioncounter = 0
    breaker = False

    # create blank npy file with same dimensions
    segmha = sitk.ReadImage(segpath, sitk.sitkUInt8)
    segdim = np.shape(sitk.GetArrayFromImage(segmha))
    lesionsnpy = zeros(segdim)

    # figure out the number of lesions for the patient
    while breaker == False:
        maxlesioncounter += 1
        fname = lesionmhapath + str(maxlesioncounter) + '.mha'
        if os.path.isfile(fname) == False:
            breaker = True

    if maxlesioncounter == 1:       # no lesions


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

        # load T2 file for imagepositionpatient and imageorientationpatient
        try:
            T2file = accession + '_T2.mha'
            T2path = os.path.join(T2dir, T2file)
            T2mha = sitk.ReadImage(T2path, sitk.sitkFloat32)
            mhafound = True
        except:
            print('mha file not found for accession ' + str(accession))
            nummhanotfound += 1

        if mhafound == True:
            origin = T2mha.GetOrigin()
            direction = T2mha.GetDirection()
            spacing = T2mha.GetSpacing()
            T2dim = np.shape(sitk.GetArrayFromImage(T2mha))

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

                # compute the segmentation using convex hulls
                points_x = points[0::3]
                points_y = points[1::3]
                points_z = points[2::3]
                pointsdf = pd.DataFrame({'x':points_x,'y':points_y,'z':points_z})

                pixelsdf = pointsdf.apply(lambda x: transformtoindex(T2mha, x['x'], x['y'], x['z']), axis=1)
                pixelsdf.columns = ['x', 'y', 'z']

                cloud = PyntCloud(pixelsdf)
                convex_hull_id = cloud.add_structure("convex_hull")
                convex_hull = cloud.structures[convex_hull_id]
                cloud.mesh = convex_hull.get_mesh()

                mesh_cloud = cloud.get_sample("mesh_random", n=1000000, as_PyntCloud=True)
                mesh_cloud.to_file("mesh_cloud.npz")

                voxel_cloud = np.load('mesh_cloud.npz')['points']

                voxels = np.zeros(T2dim) # parameterize
                for x in voxel_cloud:
                    try:
                        voxels[int(round(x[2])),int(round(x[1])),int(round(x[0]))] = 1
                    except:
                        print('voxel error in ' + str(accession))

                voxels = binary_fill_holes(voxels)
                voxels = voxels.astype('uint8')
                numPixels = np.sum(voxels)

                T2segmha = sitk.GetImageFromArray(voxels)
                T2segmha.SetOrigin(origin)
                T2segmha.SetDirection(direction)
                T2segmha.SetSpacing(spacing)

                writer = sitk.ImageFileWriter()

                if isProstate:
                    writer.Execute(T2segmha, segmhapath, True)
                    filename = segmhapath
                    prostateOrLesion = 'prostate'
                elif isLesion:
                    lesionpath = lesionmhapath + str(lesioncounter) + '.mha'
                    prostateOrLesion = 'lesion'
                    filename = lesionpath
                    writer.Execute(T2segmha, lesionpath, True)
                else:
                    filename = 'ERROR'
                    prostateOrLesion = 'ERROR'

                outdata.append(dict(zip(['accession', 'prostateOrLesion', 'commentsname', 'filename', 'lesioncounter', 'numPixels'], [accession, prostateOrLesion, name, filename, lesioncounter, numPixels])))

        processedsegfiles += 1



        #if processedsegfiles >= 3:
    #        break


if writecsv == True:
    outdf = pd.DataFrame(outdata)
    outdf = outdf[['accession', 'prostateOrLesion', 'lesioncounter', 'numPixels', 'commentsname', 'filename']]

    outdf.to_csv(segcsvpath)
    print('CSV written to ' + str(segcsvpath))

print('Finished!')
print('Processed a total of seg files: ' + str(processedsegfiles))
print('T2 mha not found for: ' + str(nummhanotfound))
print('Number of seg files without lesion or prostate in name: ' + str(errorcounter))
