# Adds the core locations onto MRpathdata
# Output file is MRpathwcorelocs.csv


mrpathdatacsv = '/data/prostate/MRpathdata2.csv'
mrpathdf = pd.read_csv(mrpathdatacsv, dtype=object, index_col=0)
mrpathdf['coretipx'] = 'NaN'
mrpathdf['coretipy'] = 'NaN'
mrpathdf['coretipz'] = 'NaN'
mrpathdf['corebotx'] = 'NaN'
mrpathdf['coreboty'] = 'NaN'
mrpathdf['corebotz'] = 'NaN'


usetransformedcores = False
datadir = '/data/prostate/patdbdata/'

rootdir = '/data/prostate/postimages/PostImages2/'
outdir = '/data/prostate/postimages/Labels/'

os.chdir(rootdir)

for file in glob.glob('*T2.mha'):
#if 0 == 0:
#    file = '11576254_T2.mha'
    print(file)       #8757032_T2.mha

    targetdifference = 0
    yestargets = False
    change15to14 = False

    accession = file.split('_')[0]

    try:
        accessiondf = mrpathdf[mrpathdf.MRaccession == accession]
        mrn = accessiondf.mrn.values[0]
        date = accessiondf.biopsydate.values[0]
        version = accessiondf.version.values[0]

    except:
        print('loading error for accession: ' + accession)
        continue

    print(accession + '  ' + mrn + '  ' + date)

    if version[0:3] == '1.4':
        usetransformedcores = True
        print('Using transformed cores...')
    elif float(date) <= 20170407:
        usetransformedcores = True
        print('Using transformed cores... (on or prior to 2017-04-07)')
    else:
        usetransformedcores = False


    # get the biopsy core coordinates
    accessionpath = datadir + '/' + accession + '_' + mrn + '_' + date
    corenames = np.load(accessionpath + '_corenames.npy')
    print(corenames)
    # check for duplicate core names (e.g. 101.1, 101.1, 101.1, 101.2)
    # if duplicates, then skip
    if len(set(corenames)) != len(corenames):
        print('Duplicate core names found')
        continue



    if usetransformedcores == False:
        coretips = np.load(accessionpath + '_coretips.npy')
        corebots = np.load(accessionpath + '_corebots.npy')
    else:
        coretips = np.load(accessionpath + '_coretipsnew.npy')
        corebots = np.load(accessionpath + '_corebotsnew.npy')


    # split into targeted and systematic cores
    systematicdf = accessiondf[accessiondf.SysOrTar == 'systematic']
    targeteddf = accessiondf[accessiondf.SysOrTar == 'targeted']
    numsyscores = np.shape(accessiondf[accessiondf.SysOrTar == 'systematic'])[0]


    # adds the core coordinates for targeted cores
    if np.shape(targeteddf)[0] > 0:     # checks if targeted cores exist
        yestargets = True

        # check if corenames (.npy) has 101.1, 101.2
        # converts them to the first targeted core (ie 17 if there are 16 sys cores)
        if float(corenames[-1]) > 100:
            firsttarget = targeteddf.CoreName.values[0]   #this corresponds to 101.1
            firsttarget = str(firsttarget).split('-')[0]
            targetdifference = 101 - int(firsttarget)
            print('101 changed')

        for i, row in targeteddf.iterrows():
            try:
                name = str(row.CoreName)
                name = name.replace('-', '.') if '-' in name else name + '.1'
                name = float(name) + targetdifference

                nameindex = list(corenames).index(name)

                [corebotx, coreboty, corebotz] = corebots[nameindex, :]
                [coretipx, coretipy, coretipz] = coretips[nameindex, :]

                mrpathdf.loc[i,'corebotx'] = corebotx
                mrpathdf.loc[i,'coreboty'] = coreboty
                mrpathdf.loc[i,'corebotz'] = corebotz
                mrpathdf.loc[i,'coretipx'] = coretipx
                mrpathdf.loc[i,'coretipy'] = coretipy
                mrpathdf.loc[i,'coretipz'] = coretipz

            except:
                print('core location not found for ' + str(name))
                continue

    # get core locations for systematic cores

    # for patients with 14 systematic cores, I named them 1, 2, 3, ... 11, 12, 13, 15
    # 14 is skipped (to help differentiate between left and right)
    # will change 15 to 14
    if np.shape(systematicdf)[0] == 14:
        rowindex = systematicdf.index[systematicdf.CoreName == '15'].tolist()

        if len(rowindex) == 1:
            mrpathdf.loc[mrpathdf.CoreName == '15', 'CoreName'] = 14
            change15to14 = True

        print('15 changed to 14')

    for i, row in systematicdf.iterrows():
        try:
            name = row.CoreName
            name = name.replace('-', '.') if '-' in name else name + '.1'
            name = float(name)

            if change15to14:
                if name == 15.1:
                    name = 14.1


            nameindex = list(corenames).index(name)

            [corebotx, coreboty, corebotz] = corebots[nameindex, :]
            [coretipx, coretipy, coretipz] = coretips[nameindex, :]

            mrpathdf.loc[i,'corebotx'] = corebotx
            mrpathdf.loc[i,'coreboty'] = coreboty
            mrpathdf.loc[i,'corebotz'] = corebotz
            mrpathdf.loc[i,'coretipx'] = coretipx
            mrpathdf.loc[i,'coretipy'] = coretipy
            mrpathdf.loc[i,'coretipz'] = coretipz
        except:
            print('core location not found for ' + str(name))
            continue


print('Finished!')
