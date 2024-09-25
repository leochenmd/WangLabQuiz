import numpy as np
import pandas as pd
import re
from decimal import Decimal

# PROFUSE PC
#file = 'C:\\Users\\profuse\\Box\\CASE-036150_Richard_Fan\\pathology.xlsx'
#outfile = 'C:\\ProcessedFusionImages\\path\\parsed.csv'


# LEO MAC
file = '/Users/Leo/Documents/Stanford Sensitive/ProstateData/biopsypath/pathology20190329.xlsx'
outfile = '/Users/Leo/Documents/Stanford Sensitive/ProstateData/biopsypath/parsed.csv'

df = pd.read_excel(file, sheet_name='Export Worksheet')

writecsv = True
includeerrormsgs = True

##################################################################
# Captures block of text between SPECIMEN SUBMITTED and DIAGNOSIS
# This block of text states all of the cores collected and anatomical locations
def getblocks(report):
    specimenblock = 'ERROR'
    diagnosisblock = 'ERROR'
    grossblock = 'ERROR'
    errormsgs = ''

    if report.count('ADDENDUM REPORT') > 0:
        errormsgs = errormsgs + 'ADDENDUM REPORT; '
        # If addendum report, will return 'ERROR: ADDENDUM REPORT')
    else:
        specimensubmittedstr = 'SPECIMEN SUBMITTED:'
        diagnosismicrostr = 'DIAGNOSIS (MICROSCOPIC):'
        diagnosisstr = 'DIAGNOSIS:'
        clindiagnosisstr = 'CLINICAL DIAGNOSIS:'
        operationstr = 'OPERATION:'
        grossdescriptionstr = 'GROSS DESCRIPTION'

        specimensubmittedindex = report.find(specimensubmittedstr)

        erroritems = [clindiagnosisstr]
        errorphrase = False

        if report.count(diagnosismicrostr) == 1:
            diagnosisindex = report.find(diagnosismicrostr)
            specimenblock = report[(specimensubmittedindex + len(specimensubmittedstr)+1):diagnosisindex]
        elif report.count(diagnosismicrostr) > 1:
            errormsgs = errormsgs + 'MULTIPLE _DIAGNOSIS (MICROSCOPIC)_; '
            errorphrase = True

        #####################
        # If "DIAGNOSIS MICROSCOPIC" is not in the file, then it will find
        # the first "DIAGNOSIS:" string that is not "CLINICAL DIAGNOSIS:"
        elif report.count(diagnosismicrostr) == 0:
            if report.count(diagnosisstr) == 0:
                errormsgs = errormsgs + 'ERROR: NO _DIAGNOSIS:_; '
                errorphrase = True
            elif report.count(diagnosisstr) == 1:    #if only one "DIAGNOSIS:", then that's the one
                diagnosisindex = report.find(diagnosisstr)

            elif report.count(diagnosisstr) > 1:     #if multiple "DIAGNOSIS", find the one that is not "CLINICAL DIAGNOSIS"
                diagnosisindices = [match.start() for match in re.finditer(diagnosisstr, report)]
                clindiagnosisindex = report.find(clindiagnosisstr)

                if abs(diagnosisindices[0] - clindiagnosisindex) > 10 or clindiagnosisindex == -1:
                    diagnosisindex = diagnosisindices[0]
                else:
                    diagnosisindex = diagnosisindices[1]

                for phrase in erroritems:
                    if diagnosisindex > report.find(phrase):
                        errorphrase = True
                        errormsgs = errormsgs + 'POSSIBLE TRUNCATED SPECIMEN BLOCK; '

            if errorphrase is False:
                specimenblock = report[(specimensubmittedindex + len(specimensubmittedstr)+1):diagnosisindex]


                ##############################################################
                # pathology block should be between 'DIAGNOSIS (MICROSCOPIC) and 'OPERATION'/'CLINICAL'
                ##############################################################
        if report.find(grossdescriptionstr):
            enddiagnosisindex = report.find(grossdescriptionstr)
        else:
            if report.find(operationstr):
                enddiagnosisindex = report.find(operationstr)
            else:
                errorphrase = True
                errormsgs = errormsgs + 'ERROR: No _GROSS DESCRIPTION_or _OPERATION_; '

        if errorphrase is False:
            diagnosisblock = report[diagnosisindex:enddiagnosisindex]
            grossblock = report[enddiagnosisindex: ]

    return [specimenblock, diagnosisblock, grossblock, errormsgs]


##############################################
# accepts line in 'systematiclines'
# returns [coreindex, errormsg]
#   coreindex is core number (1 LLB, 2 LLM, 3 LLA, etc)
def getsystematiccoreindex(line):
    errormsg = ''
    coreindex = -1      #returns -1 if there is an error

    left = line.count('LEFT') > 0
    right = line.count('RIGHT') > 0
    lateral = line.count('LATERAL') > 0
    medial = line.count('MEDIAL') > 0
    base = line.count('BASE') > 0
    mid = line.count('MID') > 0
    apex = line.count('APEX') > 0
    anterior = line.count('ANTERIOR') > 0 or line.count('ANT') > 0




    # Filters out errors
    if left and right:
        errormsg = 'ERROR: LEFT AND RIGHT in line ' + str(i + 1)
    elif (lateral and medial):
        errormsg = 'ERROR: LATERAL AND MEDIAL in line ' + str(i + 1)
    elif (base and mid) or (base and apex) or (mid and apex):
        errormsg = 'ERROR: BASE/MID/APEX in line ' + str(i + 1)

    if len(errormsg) == 0:
        # Finds corresponding core
        if not anterior:
            if left and lateral and base:
                coreindex = 1
            elif left and lateral and mid:
                coreindex = 2
            elif left and lateral and apex:
                coreindex = 3
            elif left and (medial or not lateral) and base:  # if medial is missing
                coreindex = 4                                # and lateral is not there
            elif left and (medial or not lateral) and mid:
                coreindex = 5
            elif left and (medial or not lateral) and apex:
                coreindex = 6
            elif right and (medial or not lateral) and base:
                coreindex = 7
            elif right and (medial or not lateral) and mid:
                coreindex = 8
            elif right and (medial or not lateral) and apex:
                coreindex = 9
            elif right and lateral and base:
                coreindex = 10
            elif right and lateral and mid:
                coreindex = 11
            elif right and lateral and apex:
                coreindex = 12
        elif left and anterior:
            if not(base or apex):       # 'left anterior' means 14 cores
                coreindex = 13
            elif base:
                coreindex = 13          # 'left anterior base' means 16 cores
            elif apex:
                coreindex = 14
        elif right and anterior:
            if not(base or apex):
                coreindex = 15
            elif base:
                coreindex = 15
            elif apex:
                coreindex = 16

    return [coreindex, errormsg]

###############################################
# parses percent pattern [num] from line
# searches for 'PATTERN [num]'
def parsepattern(line, num):
    percentpattern = ''

    if num == '3':
        if re.search(r'\d{1,2}\s*\%\s*PATTERN\s*3', line):
            string = re.search(r'\d{1,2}\s*\%\s*PATTERN\s*3', line).group()
            percentpattern = re.match(r'\d{1,2}', string).group()
    elif num == '4':
        if re.search(r'\d{1,2}\s*\%\s*PATTERN\s*4', line):
            string = re.search(r'\d{1,2}\s*\%\s*PATTERN\s*4', line).group()
            percentpattern = re.match(r'\d{1,2}', string).group()
    elif num == '5':
        if re.search(r'\d{1,2}\s*\%\s*PATTERN\s*5', line):
            string = re.search(r'\d{1,2}\s*\%\s*PATTERN\s*5', line).group()
            percentpattern = re.match(r'\d{1,2}', string).group()

    return percentpattern

###############################################
# parses gleason score from line
# searches for '# + # =' (ignores whitespace)
def parsegleason(line):
    primary = 0
    secondary = 0
    total = 0

    if re.search(r'[3-5]\s*\+\s*[3-5]\s*=\s*\d', line):
        score = re.search(r'[3-5]\s*\+\s*[3-5]\s*=\s*\d', line).group()
        numberlist = re.findall(r'\d', score)

        primary = numberlist[0]
        secondary = numberlist[1]
        total = numberlist[2]
    return [primary, secondary, total]

def parsepercentcore(line):
    percent = ''
    if re.search(r'\d*%\s*(?:\w+\W+){0,4}(CORE|TISSUE)', line, re.IGNORECASE):
        string = re.search(r'\d*%\s*(?:\w+\W+){0,4}(CORE|TISSUE)', line, re.IGNORECASE).group()
        # of TOTAL CORE
        # of ONE CORE
        # LESS THAN 5% OF THE CORE
        # APPROXIMATE 50% OF THE CORE
        # 50% OF SUBMITTED TISSUE / CORE
        # 5% OF THE TISSUE

        percent = re.match(r'\d*', string).group()
    return percent

def getaccession(report):
    accession = ''
    if re.search(r'SHS-\d{2}-\d{5}', report):
        accession =  re.search(r'SHS-\d{2}-\d{5}', report).group()
    return accession


######################################
# PARSES GROSS DESCRIPTION FOR LENGTHS OF CORES
# RETURNS LIST OF LENGTHS
def parsecorelength(line):
    lengths = []
    dim = 0     #dimension of #x#x# (1 = # cm, 2 = #x# cm, 3 = #x#x# cm)
    #####
    # searches first for #.# x #.# x #.# cm
    # if not found, then searches for #.# x #.# # cm
    # if also not found, then searches for #.# cm length

    str = re.findall(r'\d.\d\W*x\W*\d.\d\W*x\W*\d.\d\W*cm', line)
    if len(str) > 0:
        dim = 3
    else:
        str = re.findall(r'\d.\d\W*x\W*\d.\d\W*cm', line)
        if len(str) > 0:
            dim = 2
        else:
            str = re.findall(r'\d.\d\s*(?:\w+\W+){0,3}length', line)

            if len(str) > 0:
                dim = 1
            else:
                dim = 0
                erorrmsgs = errormsgs + 'ERROR: NO # cm found'


    if dim == 1:
        for core in str:
            lengths.append(re.match('\d.\d', core).group())
    elif dim == 2 or dim == 3:
        for core in str:
            nums = re.findall('\d.\d', core)
            if dim == 2:
                lengths.append(max(nums[0], nums[1]))
            if dim == 3:
                lengths.append(max(nums[0], nums[1], nums[2]))

    return [lengths, dim, errormsgs]

#############################################
#
def indextolocation(coreindex):
    location = ''

    if coreindex < 1:
        return 'Error: coreindex < 1'
    else:
        if (1 <= coreindex and coreindex <= 6) or (13 <= coreindex and coreindex <= 14):
            location = location + 'left '
        else:
            location = location + 'right '

        if (1 <= coreindex and coreindex <= 3) or (10 <= coreindex and coreindex <= 12):
            location = location + 'lateral '
        elif (4 <= coreindex and coreindex <= 9):
            location = location + 'medial '

        if (1 <= coreindex and coreindex <= 12) and (coreindex + 2) %  3 == 0:
            location = location + 'base'
        elif (1 <= coreindex and coreindex <= 12) and (coreindex + 1) %  3 == 0:
            location = location + 'mid'
        elif (1 <= coreindex and coreindex <= 12) and coreindex %  3 == 0:
            location = location + 'apex'

        if coreindex == 13 or coreindex == 15:
            location = location + '(anterior) '
        elif coreindex == 14 or coreindex == 16:
            location = location + 'posterior '

        if 13 <= coreindex and coreindex <= 16:
            location = location + 'apex '

        return location

def getsecondary(line):
    secondarychar = ''
    if re.search(r'Inflammation', line, re.IGNORECASE):
        secondarychar = secondarychar + 'Inflammation '
    if re.search(r'\sPIN\s', line, re.IGNORECASE) or re.search(r'PROSTATIC\s*INTRAEPITHELIAL\s*NEOPLASIA', line, re.IGNORECASE):
        secondarychar = secondarychar + 'PIN '
    if re.search(r'SMALL\s*ACINAR\s*PROLIFERATION', line, re.IGNORECASE):
        secondarychar = secondarychar + 'ASAP '

        #RADIATION

    return secondarychar

def gettargetedcorename(line):
    corename = ''
    if re.search(r'\d\d\s*-\s*\d', line):
        corename = re.search(r'\d\d\s*-\s*\d', line).group()
        corename = corename.replace(' ', '')

    return(corename)

def getPIRADS(line):
    PIRADS = ''
    if re.search(r'PIRADS\s*\d', line):
        string = re.search(r'PIRADS\s*\d', line).group()
        PIRADS = re.search(r'\d', string).group()
    return PIRADS

def gettargetedcoreloc(line):
    corelocation = ''
    left = right = False
    base = mid = apex = False
    posterior = anterior = False
    peripheral = transition = central = False

    if re.search(r'right', line, re.IGNORECASE):
        right = True
    if re.search(r'left', line, re.IGNORECASE):
        left = True

    if re.search(r'base', line, re.IGNORECASE):
        base = True
    if re.search(r'mid', line, re.IGNORECASE):
        mid = True
    if re.search(r'apex', line, re.IGNORECASE):
        apex = True

    if re.search(r'posterior', line, re.IGNORECASE):
        posterior = True
    if re.search(r'anterior', line, re.IGNORECASE):
        anterior = True

    if re.search(r'peripheral', line, re.IGNORECASE):
        peripheral = True
    if re.search(r'pz', line, re.IGNORECASE):
        peripheral = True
    if re.search(r'transition', line, re.IGNORECASE):
        transition = True
    if re.search(r'tz', line, re.IGNORECASE):
        transition = True
    if re.search(r'central', line, re.IGNORECASE):
        central = True

    if right:
        corelocation = corelocation + 'right '
    if left:
        corelocation = corelocation + 'left '
    if base:
        corelocation = corelocation + 'base '
    if mid:
        corelocation = corelocation + 'mid '
    if apex:
        corelocation = corelocation + 'apex '
    if anterior:
        corelocation = corelocation + 'anterior '
    if posterior:
        corelocation = corelocation + 'posterior '
    if peripheral:
        corelocation = corelocation + 'peripheral zone'
    if transition or central:
        corelocation = corelocation + 'transition zone'

    return corelocation
###########################################################
# Splits the pathology report into 3 blocks:
# specimenblock has the names of the cores
# diagnosisblock has the pathology of the cores
# grossblock has the lengths of the cores

parseddf = []

for index, row in df.iterrows():
    if index < 25:
        errormsgs = ''
        chartID = df['MRN'][index].astype(str)
        date = df['PROC_START_TIME'][index]

        report = df['REPORT'][index]
        [specimenblock, diagnosisblock, grossblock, blockerrormsgs] = getblocks(report)
        errormsgs = errormsgs + blockerrormsgs

        accession = df['ACCESSION_NUMBER'][index]
        #accession = getaccession(report)
        if accession == '':
            errormsgs = errormsgs + 'No accession num; '

        print(report.encode('utf-8'))

        ########
        # Check for errors

        if len(errormsgs) != 0:
            primary = secondary = total = coreindex = corelength = -1
            PIRADS = systar = percentcore = pattern4 = pattern5 = corelocation = secondarychar = ''
            parseddf.append(dict(ChartID=chartID, Accession=accession, SysOrTar=systar, PIRADS=PIRADS,
                    CoreName=coreindex, CoreLocation=corelocation, Primary=primary, Secondary=secondary,
                    PercentCore=percentcore, CoreLength=corelength, PercentPattern4=pattern4,
                    PercentPattern5=pattern5, Other=secondarychar, ErrorMsg=errormsgs))
            #print(errormsgs)
            #continue


        ###########################################################
        # Parses _specimenblock_ into individual lines
        # and stores them in an array _specimenlines_

        specimenlines = specimenblock.splitlines()
        #specimenlines = re.split(r'([A-Z]+[.])', specimenblock)

        endlineindex = 0

        # This loop finds the last line '[A-Z].' and truncates the lines after that
        # e.g. this gets rid of "SUBMITTED..." in
        #     "L. PROSTATE, 12-RIGHT LATERAL APEX BIOPSY
        #      SUBMITTED ICD9 CODE:  R97.2"
        for line in specimenlines:
            if re.match(r'[A-Z]+[.]', line):
                endlineindex += 1
            else:
                continue

        specimenlines = specimenlines[0:endlineindex]

        ####################################
        # Splits specimenlines into systematiclines and targetedlines
        # It looks for '##-1' or '## - 1' (ignores whitespace)
        firsttargetindex = -1    #-1 if no target indexes
        for i, line in enumerate(specimenlines):
            if re.search(r'\d{1,2}\s*-\s*1', line):
                firsttargetindex = i
                break

        if firsttargetindex == -1:
            systematiclines = specimenlines
            targetedlines = []
        else:
            systematiclines = specimenlines[0:firsttargetindex]
            targetedlines = specimenlines[firsttargetindex: ]


        ######################################################
        # Identifies which core is which in systematiclines
        # and stores it in systematiclinesindices
        # 1: Left lateral base      # 2: Left lateral mid   # 3: Left lateral apex
        # 4: Left (medial) base     # 5: Left (medial) mid  # 6: Left (medial) apex
        # 7: Right (medial) base    # 8: Right (medial) mid # 9: Right (medial) apex
        # 10: Right lateral base    # 11: Right lateral mid # 12: Right lateral apex
        # 13: Left anterior (base)      if 14 core, then left anterior
        # 14: Left anterior (apex)       blank if 14 core biopsy
        # 15: Right anterior  vs. base      14 vs. 16 cores
        # 16: Right anterior (apex)     blank if 14 core biopsy

        systematicindices = -np.int_(np.ones(len(systematiclines)))

        for i, line in enumerate(systematiclines):
            [systematicindices[i], getsysindexerrormsg] = getsystematiccoreindex(line)
            errormsgs = errormsgs + getsysindexerrormsg

        numsyscores = len(systematiclines)

        #print('Index: ' + str(index))
        #print('Chart ID: ' + str(df['CHART_ID'][index]))
        #print('Accesion no: ' + getaccession(report))
        #print('Error msgs ' + str(errormsgs))

        #print('Number of systematic cores: ' + str(numsyscores))

    #    print('first systematic line: ' + systematiclines[0])
    #    print('last systematic line: ' + systematiclines[-1])
        #print('core numbers: ' + str(systematicindices))


                        # Ignores text after PathologistName/PathologistName
                        # Splits using delimeter [word] / [word]
        diagnosisblock2 = re.split(r'\w+\/\w+', diagnosisblock)[0]

                        # Splits using delimiter 'A.' or other [letter] period
        diagnosislines = re.split(r'\s[A-Z]\.\s*', diagnosisblock2)[1:]  # ignores first element

        systematicdiagnosislines = diagnosislines[0:numsyscores]
        targeteddiagnosislines = diagnosislines[numsyscores:numsyscores+len(targetedlines)]

        print(str(targeteddiagnosislines).encode('utf-8'))
        #print('Number of diagnosis lines: ' + str(len(diagnosislines)))
        #print('Number of systematic lines: ' + str(len(systematicdiagnosislines)))
        #print('Number of targeted lines: ' + str(len(targetedlines)))

        #print('==================')
    #    print('Last systematic diagnosis line (with pathology) :')
    #    print(systematicdiagnosislines[-1])
    #    if len(targeteddiagnosislines) > 0:
    #        print('First targeted diagnosis line (with pathology): ')
    #        print(targeteddiagnosislines[0])
    #        print('Last targeted diagnosis line (with pathology): ')
    #        print(targeteddiagnosislines[-1])
    #    print('==================')


        #####################################
        grosslines = re.split(r'(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|twelveth|[A-Z]*[a-z]*teenth|twentieth|\d\dth|\d\drd|\d\dst|\d\dnd)(?:\W+\w+){0,5}?\W+label[a-z]{0,2}',
                        grossblock)
        grosslines = grosslines[1:]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # can add 'first/second/...' back in by doing re.findall(...)
        # and then concatenating the two strings

    #    print(grosslines)

        corelengths = []
        for i, line in enumerate(grosslines):
            [corelengthlist, dimension, grosserrormsgs] = parsecorelength(line)

            if len(corelengthlist) > 1:
                errormsgs = errormsgs + 'Multiple core lengths detected in core ' + str(i+1) + '; '
                corelengths.append(str(corelengthlist))
                continue
            elif len(corelengthlist) == 0:
                errormsgs = errormsgs + 'No core length detected for core ' + str(i+1) + '; '
                corelengths.append('-1')
            else:
                corelengths.append(corelengthlist[0])

        if len(systematicindices) != len(set(systematicindices)):
            errormsgs = errormsgs + 'DUPLICATE SYSTEMATIC CORES FOUND '

        #print(corelengths)
        #print(errormsgs)

        if len(chartID) == 6:
            chartID = '00' + chartID
        elif len(chartID) == 7:
            chartID = '0' + chartID

        if len(errormsgs) > 0:
            primary = secondary = total = coreindex = corelength = -1
            PIRADS = date = systar = percentcore = pattern4 = pattern5 = corelocation = secondarychar = ''
            parseddf.append(dict(Index=index, ChartID=chartID, PathDate=date, Accession=accession, SysOrTar=systar, PIRADS=PIRADS,
                    CoreName=coreindex, CoreLocation=corelocation, Primary=primary, Secondary=secondary,
                    PercentCore=percentcore, CoreLength=corelength, PercentPattern4=pattern4,
                    PercentPattern5=pattern5, Other=secondarychar, ErrorMsg=errormsgs))
        else:
            # gets information from systematic cores:
            numsyscores = len(systematicdiagnosislines)

            for i, line in enumerate(systematicdiagnosislines):
                [primary, secondary, total] = parsegleason(line)
                coreindex = systematicindices[i]
                corelocation = indextolocation(coreindex)
                percentcore = parsepercentcore(line)
                pattern4 = parsepattern(line, '4')
                pattern5 = parsepattern(line, '5')


                # CORRECT FOR 14 CORE BIOPSIES
                if int(numsyscores) == 14 and coreindex == 15:
                    coreindex = 14

                try:
                    corelength = corelengths[i]
                except:
                    errormsgs = errormsgs + 'MISSING CORE LENGTH for SYS CORE ' + str(i + 1) + '; '
                PIRADS = ''
                systar = 'systematic'
                secondarychar = getsecondary(line)

                parseddf.append(dict(Index=index, ChartID=chartID, PathDate=date, Accession=accession, SysOrTar=systar, PIRADS=PIRADS,
                        CoreName=coreindex, CoreLocation=corelocation, NumSysCores=numsyscores, Primary=primary, Secondary=secondary,
                        Total=total, PercentCore=percentcore, CoreLength=corelength, PercentPattern4=pattern4,
                        PercentPattern5=pattern5, Other=secondarychar, ErrorMsg=errormsgs))

            for i, line in enumerate(targeteddiagnosislines):
                [primary, secondary, total] = parsegleason(line)
                coreindex = gettargetedcorename(line)
                coreindex2 = gettargetedcorename(targetedlines[i])

                if len(coreindex) == 0 and len(coreindex2) > 0:
                    print('=0 ' + accession + '  ' + coreindex + '  ' + coreindex2)
                    print(line)
                    print(i)
                    print(targetedlines)
                    coreindex = coreindex2

                elif len(coreindex) > 0 and len(coreindex2) > 0:
                    if coreindex != coreindex2:
                        errormsgs = errormsgs + 'INCONSISTENT TARGETED CORE NAME ' + str (i+1)
                        print('>0 ' + accession + '  ' + coreindex + '  ' + coreindex2)
                        print(line.encode('utf-8'))
                        print(i)
                        print(targetedlines)
                        #coreindex = coreindex2 = ''

                corelocation = gettargetedcoreloc(line)
                percentcore = parsepercentcore(line)
                pattern4 = parsepattern(line, '4')
                pattern5 = parsepattern(line, '5')
                try:
                    corelength = corelengths[i+numsyscores]
                except:
                    errormsgs = errormsgs + 'MISSING CORE LENGTH for TARGET CORE ' + str(i + 1) + '; '

                PIRADS = getPIRADS(line)
                systar = 'targeted'
                secondarychar = getsecondary(line)

                parseddf.append(dict(Index=index, ChartID=chartID, PathDate=date, Accession=accession, SysOrTar=systar, PIRADS=PIRADS,
                CoreName=coreindex, CoreLocation=corelocation, NumSysCores=numsyscores, Primary=primary, Secondary=secondary,
                Total=total, PercentCore=percentcore, CoreLength=corelength, PercentPattern4=pattern4,
                PercentPattern5=pattern5, Other=secondarychar, ErrorMsg=errormsgs))



cols = ['Index', 'ChartID', 'PathDate', 'Accession', 'SysOrTar', 'ErrorMsg', 'NumSysCores', 'CoreName', 'PIRADS', 'CoreLocation', \
            'Primary', 'Secondary', 'Total', 'PercentCore', 'CoreLength', \
            'PercentPattern4', 'PercentPattern5', 'Other']
parseddf = pd.DataFrame(parseddf, columns=cols)

if includeerrormsgs == False:
    parseddf = parseddf[parseddf.ErrorMsg == '']

if writecsv == True:
    parseddf.to_csv(outfile)
