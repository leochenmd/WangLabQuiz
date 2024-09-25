### Adds pathology data to coresXML file

import os
import numpy as np
import pandas as pd
import re
import datetime

debugmode = False

writeMRpathaccessions = False
writeMRpathdata = True

maxdaydifference = 30

mrdir = 'C:\\ProcessedFusionImages\\StanfordT2ADCDWI'

corepathcsv = 'C:\\ProcessedFusionImages\\path\\parsed.csv'
corelocxls = 'E:\\patdbdata\\coresXML20190330.csv'
redcapcsv = 'E:\\patdbdata\\ArtemisBiopsyDB-LegendLeo_DATA_2019-03-31_0155.csv'

consolidatedcsv = 'E:\\patdbdata\\corelocspath.csv'

corepathdf = pd.read_csv(corepathcsv, dtype=object)
corelocdf = pd.read_csv(corelocxls, dtype=object)
redcapdf = pd.read_csv(redcapcsv, dtype=object)

corelocdf["studyID"] = ''
corelocdf["pathaccession"] = ''
corelocdf["pathdate"] = ''
corelocdf["patherror"] = ''


'''
PSEUDOCODE

find all MR accession numbers in scans folder
separate into 2 groups:
    1 group with core location data
    1 group without core location data (patdbase issue)

Process group with core location data
For each accession:
    find corresponding MRN, study ID, and path accession
    (manually check duplicate MRNs)

    add path data to core location csv
        (includes path corenames, error messages, gleason, percent, and core length)

3rd group with core location but without path data (parser issue)
'''



# 1. Find all MR accession numbers in stanford scan folder
T2files = [f for f in os.listdir(mrdir) if os.path.isfile(os.path.join(mrdir, f)) and 'T2' in f]
stanfordMRaccessions = list(set([f.split('_')[0] for f in T2files]))

coreMRaccessions = list(set(list(corelocdf.MRaccession)))

commonaccessions = list(set(stanfordMRaccessions) & set(coreMRaccessions))
commonaccessions.sort()

print('number of stanford MRs w/ T2/DWI/ADC: ', len(stanfordMRaccessions))
print('number of MRs with core locations: ', len(coreMRaccessions))
print('number of stanford MRs w/ core locations: ', len(commonaccessions))
      # 522 stanford scans with core locations

# make new dataframe that only has common accessions
commonaccessionsdf = corelocdf[corelocdf.MRaccession.isin(commonaccessions)]

# 2. For each MR accession:
#       find corresponding MRN, study ID, and path accession in redcapcsv
#        (manually check duplicate MRNs)
#       write into two new columns 'redcapStudyID' and 'pathAccession'

for i, mraccession in enumerate(commonaccessions):
    if 10 == 10:
        accessiondf = commonaccessionsdf[commonaccessionsdf.MRaccession == mraccession]

        if len(accessiondf) == 0:
            continue

        mrn = accessiondf.mrn.values[0]
        biopsydate = accessiondf.biopsydate.values[0]
        mrdate = accessiondf.MRdate.values[0]
        pathdateerror = ''

        yyyy = biopsydate[0:4]
        mm = biopsydate[4:6]
        dd = biopsydate[6:8]
        biopsydate = datetime.date(int(yyyy), int(mm), int(dd))

        pathaccession = pathdate = ''
        pathdf = corepathdf
        matchingredcapdf = redcapdf[redcapdf.mrn == mrn]

        studyIDs = list(matchingredcapdf.study_id.values)
        pathaccessions = list(matchingredcapdf.path_accession.values)
        studyDates = list(matchingredcapdf.date_enrolled.values)

        if len(studyIDs) == 0:
            continue

        elif len(studyIDs) == 1:         # if only one matching study ID/path accession to MRN:
            studyID = studyIDs[0]
            pathaccession = pathaccessions[0]

            pathdatedf = corepathdf[corepathdf.Accession == pathaccession]

            if len(pathdatedf) > 0:
                pathreportdate = pathdatedf['PathDate'].values[0]
                pathdate = pd.to_datetime(pathreportdate).date()
            else:
                pathdateerror = 'NO PATH REPORT FOUND'

        elif len(studyIDs) > 1:
            for j, studyID in enumerate(studyIDs):
                pathaccession = pathaccessions[j]

                #if more than one path accession, then find the closest one
                pathdatedf = corepathdf[corepathdf.Accession == pathaccession]

                if len(pathdatedf) > 0:
                    pathreportdate = pathdatedf['PathDate'].values[0]
                    pathdate = pd.to_datetime(pathreportdate).date()
                else:
                    pathdateerror = 'NO PATH REPORT FOUND'
                    continue

                dif = abs((biopsydate - pathdate).days)

                if j == 0:
                    minDif = dif
                    minIndex = j

                if dif < minDif:
                    minDif = dif
                    minIndex = j

            studyID = studyIDs[minIndex]
            pathaccession = pathaccessions[minIndex]

        # check if the closest path report date/visit xml biopsydate are more than X days apart
        try:
            if abs(biopsydate - pathdate).days <= maxdaydifference:
                pathdate = str(pathdate.strftime('%Y%m%d')).replace('-', '')
            elif abs(biopsydate - pathdate).days > maxdaydifference:
                studyID = ''
                pathaccession = ''
                pathdateerror = 'PATH DATE AND BIOPSY DATE OUT OF MAX RANGE'
        except:
                studyID = ''
                pathaccession = ''
                pathdateerror = 'MISSING BIOPSY DATE OR PATH REPORT DATE'

        if debugmode:
            print('---------------------')
            print('mr accession: ', mraccession)
            print('study ID: ', studyID, '       mrn: ', mrn)
            print('visit.xml biopsy date: ', biopsydate)
            print('path report date: ', pathdate, '    error: ', pathdateerror)

        commonaccessionsdf.loc[commonaccessionsdf.MRaccession == mraccession, 'studyID'] = studyID
        commonaccessionsdf.loc[commonaccessionsdf.MRaccession == mraccession, 'pathaccession'] = pathaccession
        commonaccessionsdf.loc[commonaccessionsdf.MRaccession == mraccession, 'pathdate'] = pathdate
        commonaccessionsdf.loc[commonaccessionsdf.MRaccession == mraccession, 'patherror'] = pathdateerror

commonaccessionsdf = commonaccessionsdf[['studyID', 'MRaccession', 'pathaccession', 'mrn', 'MRdate', 'biopsydate', 'pathdate', 'patherror', 'version', 'originalcorename', 'corename', 'coretipX', 'coretipY', 'coretipZ', 'corebotX', 'corebotY', 'corebotZ']]

numaccessions = len(list(set(list(commonaccessionsdf[commonaccessionsdf.patherror == ''].MRaccession.values))))
print(numaccessions)

commonaccessionsdf.to_csv(consolidatedcsv)





'''
        if 'SHS' in pathaccession:
            mrdf.at[i, "studyID"] = studyID
            mrdf.at[i, "pathaccession"] = pathaccession
            pathdate = str(pathdate).replace('-','')
            #pathdate = datetime.datetime.strptime(str(pathdate), "%m/%d/%Y").strftime("%Y%m%d")
            mrdf.at[i, "pathdate"] = pathdate

mrdf = mrdf[mrdf.pathaccession != '']
mrdf = mrdf[mrdf.errormsgs.isnull()]

mrdf.rename(columns={'accession':'MRaccession'}, inplace=True)
cols = ['studyID', 'pathaccession', 'MRaccession', 'mrn', 'mrdate', 'biopsydate', 'pathdate', 'version', 'errormsgs', 'numcores', 'rowspacing', 'colspacing', 'zspacing', 'rowdim', 'coldim', 'zdim']
mrdf = mrdf[cols]

if writeMRpathaccessions == True:
    mrdf.to_csv(os.path.join(csvdir, 'MRpathaccessions2.csv'))


#Populate path report with MR data, including study ID and accession
newcols = ['studyID', 'MRaccession', 'mrn', 'biopsydate', 'mrdate', 'version',
    'numcores', 'rowspacing', 'colspacing', 'zspacing', 'rowdim', 'coldim', 'zdim']
pathdf = pathdf.reindex(columns = pathdf.columns.tolist() + newcols)
pathdf[newcols] = pathdf[newcols].astype(str)
pathdf.rename(columns={'Accession':'pathaccession'}, inplace=True)


for i,pathrow in pathdf.iterrows():
    pathaccession = pathrow['pathaccession']
    mrrow = mrdf[mrdf['pathaccession'] == pathaccession]

    if mrrow.shape[0] == 1:
        for j in range(len(newcols)):
            pathdf.at[i, newcols[j]] = str(mrrow[newcols[j]].values[0])
    elif mrrow.shape[0] > 1:
        print('duplicates for ' + str(pathaccession))
    else:
        for j in range(len(newcols)):
            pathdf.at[i, newcols[j]] = ''

pathdf = pathdf[pathdf.studyID != '']

finalcols = ['studyID',	'MRaccession', 'mrn', 'mrdate', 'biopsydate', 'PathDate',
    'pathaccession', 'version', 'numcores', 'CoreName', 'SysOrTar',	'PIRADS',
	'CoreLocation', 'Primary', 'Secondary', 'Total', 'PercentCore', 'CoreLength',
    'PercentPattern4', 'PercentPattern5', 'Other', 'rowspacing', 'colspacing',
    'zspacing', 'rowdim', 'coldim', 'zdim']

pathdf = pathdf[finalcols]

if writeMRpathdata == True:
    pathdf.to_csv(os.path.join(csvdir, 'MRpathdata6.csv'))
'''
