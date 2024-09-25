# for each accession
#   load the cores
#   calculate the core locations per slice
#   save the data with accession, core num, (x y z) pixel, (x y z) MR location, path
#
# for each accession
#   load the core data
#   for each lesion mha file
#      for each core
#         if the core locations intersect with lesion
#            label lesion with maximum intersecting pathology
#            delete core from database


# 1. Calculate core locations and save cores
# Calculates the core locations and plots the core tracts on T2 space
#  rad = radius of the core tract
#  cutoff = gleason score cutoff (i.e. cutoff of 6 means Gleason >= 6 is cancer)
# 1 for cancer
# 0 for benign
# -1 for unknown
# Outputs as .mha file ([accession]_label_cutoff_rad.mha)

# For overlaps, it takes the majority vote
#   can try to incorporate nearest neighbor later

rootdir = '/data/prostate/postimages/Augmented/'
os.chdir(rootdir)
outdir = '/data/prostate/postimages/AugmentedLabels/'

mrpathwlocpath = '/data/prostate/MRpathwcorelocs3.csv'

mrpathlocdf = pd.read_csv(mrpathwlocpath, dtype=object, index_col=0)

outdata = {}


#for file in glob.glob('*T2*'):
if 0 == 0:
    file = '8099139_T2.mha'
    file2 = file.split('.')[0]

    print(file)

    accession = file.split('_')[0]

    try:
        accessiondf = mrpathlocdf[mrpathlocdf.MRaccession == accession]
        mrn = accessiondf.mrn.values[0]
        date = accessiondf.biopsydate.values[0]
       #version = accessiondf.version.values[0]

    except:
        print('accession not found in csv file: ' + accession)
#        continue


    # read T2 image file
    T2filename = '8099139_T2.mha' # accession + '_T2.mha'

    T2filepath = os.path.join(rootdir, T2filename)

    T2image = sitk.ReadImage(T2filepath, sitk.sitkFloat32)

    # sitk image coordinates are (col, row, z)
    origin = T2image.GetOrigin()
    direction = T2image.GetDirection()
    spacing = T2image.GetSpacing()


    # the numpy is (z, row, col)
    T2npy = sitk.GetArrayFromImage(T2image)
    corelocnpy = -1*np.ones(np.shape(T2npy))

    imgorientationpatient = direction[0:6]
    [colspacing, rowspacing] = spacing[0:2]


    # labels the 3D MR space with gleason score (0, 6, 7, 8, etc)
    # goes through slice by slice
    for zslicei in range(np.shape(corelocnpy)[0]):
        imgpositionpatient = T2image.TransformIndexToPhysicalPoint([0, 0, zslicei])

        # first pass: solve for all of the points in each slice
        rowlist = []
        collist = []
        pathlist = []
        percentlist = []
        numpoints = 0

        for i, row in accessiondf.iterrows():
            path = int(row.Total)
            name = row.CoreName
            sysortar = row.SysOrTar
            length = row.CoreLength


            if path == 0:
                percent = 100
            else:
                try:
                    percent = int(row.PercentCore)
                    #corelength = float(row.CoreLength)
                except:
                    percent = 0
                    continue

            corebot = [float(row.corebotx), float(row.coreboty), float(row.corebotz)]
            coretip = [float(row.coretipx), float(row.coretipy), float(row.coretipz)]


            [col, row, t] =  corepixelsolver(imgpositionpatient, imgorientationpatient, corebot, coretip, rowspacing, colspacing)

            if 0 <= t <= 1:      # can modify this to only include central __% of core
                                 # i.e. 60% would be 0.2 <= t <= 0.8
                [mrx, mry, mrz] = T2image.TransformIndexToPhysicalPoint([row, col, zslicei])

                outdata.append(dict(zip(['accession', 'corename', 'path', 'percent', 'length', 'row', 'col', 'zslice', 'x', 'y', 'z'], [accession, corename, path, percent, length, row, col, zslicei, x, y, z])

print('finished')
