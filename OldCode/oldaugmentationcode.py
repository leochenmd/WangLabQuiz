#### OLD CODE #####


# 1. Normalize all T2 and ADC sequences
#     reads all T2 and resampled ADC images in 'T2dir'
#     uses histogram normalization, casts to unsigned 8-bit images [0-255]
#     saves to 'normdir'

T2dir = '/data/prostate/postimages/PostImages2/'   # also ADC dir
normdir = '/data/prostate/postimages/Normalized/'

os.chdir(T2dir)

writer = sitk.ImageFileWriter()
counter = 0
total = len(glob.glob('*ADCresampled.mha'))

#### OLD CODE #####

for ADCfile in glob.glob('*ADCresampled.mha'):
    accession = ADCfile.split('_')[0]

    # read files into images
    ADCfile = accession + '_ADCresampled.mha'
    T2file = accession + '_T2.mha'
    T2path = os.path.join(T2dir, T2file)
    ADCpath = os.path.join(T2dir, ADCfile)

    T2image = sitk.ReadImage(T2path, sitk.sitkFloat32)
    ADCimage = sitk.ReadImage(ADCpath, sitk.sitkFloat32)

    T2npy = sitk.GetArrayFromImage(T2image)
    ADCnpy = sitk.GetArrayFromImage(ADCimage)
  #### OLD CODE #####
    origin = T2image.GetOrigin()
    direction = T2image.GetDirection()
    spacing = T2image.GetSpacing()

    normT2npy = histEQimage(T2npy)
    normADCnpy = histEQimage(ADCnpy)

    normT2image = sitk.GetImageFromArray(normT2npy)
    normT2image.SetOrigin(origin)
    normT2image.SetDirection(direction)
    normT2image.SetSpacing(spacing)
   #### OLD CODE #####
    normADCimage = sitk.GetImageFromArray(normADCnpy)
    normADCimage.SetOrigin(origin)
    normADCimage.SetDirection(direction)
    normADCimage.SetSpacing(spacing)

    ADCfile = accession + '_ADC.mha'

    writer.Execute(normT2image, os.path.join(normdir, T2file), True)   # True compresses image
    sitk.WriteImage(normADCimage, os.path.join(normdir, ADCfile), True)  # True compresses image

    print('Normalized ' + accession)

    counter = counter + 1

    if counter % 10 == 0:
        print('done with ' + str(counter) + ' of ' + str(total))

        #### OLD CODE #####


print('----------')
print('Completed!')

    #### OLD CODE #####


    # 2. Generate augmentations
#    Rotations, flips
#

# define angles in degrees
angles = [2, 4]

T2dir = '/data/prostate/postimages/Normalized/'   # also ADC dir
augmentdir = coredir = '/data/prostate/postimages/Augmented/'

os.chdir(T2dir)

writer = sitk.ImageFileWriter()



# run through resampled ADC files
for ADCfile in glob.glob('*ADC.mha'):
    accession = ADCfile.split('_')[0]

    # read files into images
    ADCfile = accession + '_ADC.mha'
    T2file = accession + '_T2.mha'

    T2path = os.path.join(T2dir, T2file)
    ADCpath = os.path.join(T2dir, ADCfile)

    T2image = sitk.ReadImage(T2path, sitk.sitkUInt8)
    ADCimage = sitk.ReadImage(ADCpath, sitk.sitkUInt8)

    # finds the center of the MR scan in physical coordinates
    shape = np.array(T2image.GetSize())
    pixelcenter = (shape / 2).astype(int).tolist()
    physicalcenter = T2image.TransformIndexToPhysicalPoint(pixelcenter)


    # generates and saves rotations
    # when angles y and z = 0, only twists body like rolling pin
    angley = 0
    anglez = 0

    # positive rotations (natural turn, left shoulder forward right shoulder back)
    for angle in angles:
        anglex = angle

        rotatedT2 = rotateImage(T2image, physicalcenter, anglex, angley, anglez)
        rotatedADC = rotateImage(ADCimage, physicalcenter, anglex, angley, anglez)

        # file name is [accession]_T2_flip#_rotateX##.mha
        # e.g. 8099139_T2_flip1_rotateR03.mha
        #   flip 0 is not flipped, flip 1 is flipped
        #   rotate X##   L05 is rotated left arm back, right arm forward
        #                R10 is rotated 10 deg (left arm forward, right arm back)
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        rotatedT2name = accession + '_T2_flip0_rotateR' + anglename + '.mha'
        rotatedADCname = accession + '_ADC_flip0_rotateR' + anglename + '.mha'

        writer.Execute(rotatedT2, os.path.join(augmentdir, rotatedT2name), True)
        writer.Execute(rotatedADC, os.path.join(augmentdir, rotatedADCname), True)

    # negative rotations (reverse turn, right shoulder forward left shoulder back)
        anglex = -angle

        rotatedT2 = rotateImage(T2image, physicalcenter, anglex, angley, anglez)
        rotatedADC = rotateImage(ADCimage, physicalcenter, anglex, angley, anglez)

        # file name is [accession]_T2_flip#_rotateX##.mha
        # e.g. 8099139_T2_flip1_rotateR03.mha
        #   flip 0 is not flipped, flip 1 is flipped
        #   rotate X##   L05 is rotated clockewise (left arm back, right arm forward)
        #                R10 is rotated counterclockwise 10 deg (left arm forward, right arm back)
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        rotatedT2name = accession + '_T2_flip0_rotateL' + anglename + '.mha'
        rotatedADCname = accession + '_ADC_flip0_rotateL' + anglename + '.mha'

        writer.Execute(rotatedT2, os.path.join(augmentdir, rotatedT2name), True)
        writer.Execute(rotatedADC, os.path.join(augmentdir, rotatedADCname), True)


    # flip
    T2flipped = flipImage(T2image)
    ADCflipped = flipImage(ADCimage)

    T2flipname = accession + '_T2_flip1.mha'
    ADCflipname = accession + '_ADC_flip1.mha'

    writer.Execute(T2flipped, os.path.join(augmentdir, T2flipname), True)
    writer.Execute(ADCflipped, os.path.join(augmentdir, ADCflipname), True)


    # flip and rotate positive (natural turn to right, left shoulder forward)
    for angle in angles:
        anglex = angle

        rotatedT2 = rotateImage(T2flipped, physicalcenter, anglex, angley, anglez)
        rotatedADC = rotateImage(ADCflipped, physicalcenter, anglex, angley, anglez)

        # file name is [accession]_T2_flip#_rotateX##.mha
        # e.g. 8099139_T2_flip1_rotateR03.mha
        #   flip 0 is not flipped, flip 1 is flipped
        #   rotate X##   L05 is rotated left arm back, right arm forward
        #                R10 is rotated 10 deg (left arm forward, right arm back)
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        rotatedT2name = accession + '_T2_flip1_rotateR' + anglename + '.mha'
        rotatedADCname = accession + '_ADC_flip1_rotateR' + anglename + '.mha'

        writer.Execute(rotatedT2, os.path.join(augmentdir, rotatedT2name), True)
        writer.Execute(rotatedADC, os.path.join(augmentdir, rotatedADCname), True)


    # flip and rotate negative (reverse turn to left, right shoulder forward)
        anglex = -angle

        rotatedT2 = rotateImage(T2flipped, physicalcenter, anglex, angley, anglez)
        rotatedADC = rotateImage(ADCflipped, physicalcenter, anglex, angley, anglez)

        # file name is [accession]_T2_flip#_rotateX##.mha
        # e.g. 8099139_T2_flip1_rotateR03.mha
        #   flip 0 is not flipped, flip 1 is flipped
        #   rotate X##   L05 is rotated left arm back, right arm forward
        #                R10 is rotated 10 deg (left arm forward, right arm back)
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        rotatedT2name = accession + '_T2_flip1_rotateL' + anglename + '.mha'
        rotatedADCname = accession + '_ADC_flip1_rotateL' + anglename + '.mha'

        writer.Execute(rotatedT2, os.path.join(augmentdir, rotatedT2name), True)
        writer.Execute(rotatedADC, os.path.join(augmentdir, rotatedADCname), True)

    print(accession + ' augmented')

print('Done!')



# 2b. Generates the mha files for the non flipped scans

outdir = '/data/prostate/postimages/AugmentedLabels/'
os.chdir(outdir)


for file in glob.glob('*T2_flip1*'):
    filepath = os.path.join(outdir, file)

    coreimage = sitk.ReadImage(filepath, sitk.sitkUInt8)
    flippedcoreimage = flipImage(coreimage)

    writer = sitk.ImageFileWriter()
    flippedfilename = file.replace('flip1', 'flip0')

    if 'flip0_cores' in flippedfilename:
        flippedfilename = flippedfilename.replace('flip0_cores', 'cores')

    outpath = os.path.join(outdir, flippedfilename)

    writer.Execute(flippedcoreimage, flippedfilename, True)

    print(flippedfilename)

print('done with generating cores for non-flipped')







counter = 0

if 0 == 0:
    accession = '9190612'

    corefile = accession + '_cores_r3_gAll.mha'
    corefile2 = accession + '_cores_r3_gAll'

    corepath = os.path.join(outdir, corefile)

    coreimage = sitk.ReadImage(corepath, sitk.sitkUInt8)

    # finds the center of the MR scan in physical coordinates
    shape = np.array(coreimage.GetSize())
    pixelcenter = (shape / 2).astype(int).tolist()
    physicalcenter = coreimage.TransformIndexToPhysicalPoint(pixelcenter)


    # generates and saves rotations
    # when angles y and z = 0, only twists body like rolling pin
    angley = 0
    anglez = 0


    # flip, no rotation
    coreflipped = flipImage(coreimage)
    coreflipname = corefile2 + '_flip1.mha'

    writer.Execute(coreflipped, os.path.join(outdir, coreflipname), True)

    # positive rotations (natural turn, left shoulder forward right shoulder back)
    for angle in angles:
        if abs(angle) < 10:
            anglename = '0' + str(angle)
        else:
            anglename = str(angle)

        anglex = angle

        rotatedcores = rotateCores(coreimage, physicalcenter, anglex, angley, anglez) # rotate
        rotatedcoresname = corefile2 + '_flip0_rotateR' + anglename + '.mha'
        writer.Execute(rotatedcores, os.path.join(outdir, rotatedcoresname), True)

        # positive rotation + flip
        coresflipped = flipImage(rotatedcores)
        coresflipname = corefile2 + '_flip1_rotateL' + anglename + '.mha'
        writer.Execute(coresflipped, os.path.join(outdir, coresflipname), True)


    # negative rotations (reverse turn, right shoulder forward left shoulder back)
        anglex = -angle

        rotatedcores = rotateCores(coreimage, physicalcenter, anglex, angley, anglez) # rotate
        rotatedcoresname = corefile2 + '_flip0_rotateL' + anglename + '.mha'
        writer.Execute(rotatedcores, os.path.join(outdir, rotatedcoresname), True)

         # negative rotation + flip
        coresflipped = flipImage(rotatedcores)
        coresflipname = corefile2 + '_flip1_rotateR' + anglename + '.mha'
        writer.Execute(coresflipped, os.path.join(outdir, coresflipname), True)



    # progress bar
    counter = counter + 1
    if (counter % 5) == 0:
        print('processed ' + str(counter) + ' of ' + str(total))

print('done!')
