### consolidates pathology data 'parsedOutput.csv'
###    with MR files/data 'patdbdata.csv'
###    to form new file with all data 'consolidateddata.csv'
### uses redcap database as reference key (linking mrn/MR accession/path accession)

import os
import numpy as np
import pandas as pd
import re
import datetime

writeMRpathaccessions = False
writeMRpathdata = True
maxdaydifference = 30

csvdir = '/users/Leo/documents/Stanford Sensitive/python/BiopsyML/'
pathcsv = os.path.join(csvdir, 'parsedOutput20181113.csv')
mrcsv = os.path.join (csvdir, 'patdbdata20181019.csv')
redcapcsv = os.path.join(csvdir, 'RedcapDB20190107.csv')

pathdf = pd.read_csv(pathcsv)
mrdf = pd.read_csv(mrcsv)
redcapdf = pd.read_csv(redcapcsv)

mrdf["studyID"] = ''
mrdf["pathaccession"] = ''
mrdf["pathdate"] = ''


# for each MR accession number, find corresponding study ID in redcapcsv
#    find also the corresponding path accession
#    write into two new columns 'redcapStudyID' and 'pathAccession'
for i,row in mrdf.iterrows():
    mrn = str(row["mrn"])
    mraccession = str(row["accession"])
    biopsydate = str(row["biopsydate"])
    mrdate = str(row["mrdate"])
    pathaccession = ''
    yyyy = biopsydate[0:4]
    mm = biopsydate[4:6]
    dd = biopsydate[6:8]
    biopsydate = datetime.date(int(yyyy), int(mm), int(dd))

    matchingredcapdf = redcapdf[redcapdf.mrn == mrn]

    if matchingredcapdf.shape[0] == 1:
        #if only one matching MRN, then take that study ID
        # if pathreport date and biopsy date within 1 month
        studyID = str(matchingredcapdf.study_id.values[0])
        pathaccession = str(matchingredcapdf.path_accession.values[0])

        pathrow = pathdf[pathdf['Accession'] == pathaccession]

        if pathrow.shape[0] > 0:
            pathreportdate = pathrow['PathDate'].values[0]
            pathdate = pd.to_datetime(pathreportdate).date()

            if abs((biopsydate - pathdate).days) > maxdaydifference:
                studyID = ''
                pathaccession = ''
        else:
            studyID = ''
            pathaccession = ''

    elif matchingredcapdf.shape[0] > 1:
        #if more than one MRN, then find the closest one (within the year)
        #studyDates = matchingredcapdf['date_enrolled'].values[:]

        #studyDate = pd.to_datetime(studyDates[0]).date()
        pathaccessions = matchingredcapdf.path_accession.values.tolist()
        firstworkingaccession = False

        for j, accession in enumerate(pathaccessions):
            #studyDate =  pd.to_datetime(date).date()
            pathrow = pathdf[pathdf['Accession'] == pathaccessions[j]]

            if pathrow.shape[0] > 0:
                pathreportdate = pathrow['PathDate'].values[0]
                pathdate = pd.to_datetime(pathreportdate).date()

                if firstworkingaccession == False:
                    minDif = abs((biopsydate - pathdate).days)
                    indexj = j
                    firstworkingaccession == True

                dif = abs((biopsydate - pathdate).days)
                if dif < minDif:
                    minDif = dif
                    indexj = j
            else:
                continue

        if firstworkingaccession == True:
            studyID = str(matchingredcapdf['study_id'].values[indexj])
            pathaccession = str(matchingredcapdf['path_accession'].values[indexj])

            if abs(biopsydate - pathdate).days > maxdaydifference:
                studyID = ''
                pathaccession = ''

    else:
        continue

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
