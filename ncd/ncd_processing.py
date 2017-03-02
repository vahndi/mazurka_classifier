import pandas as pd
import os, pickle, copy, time
from datetime import datetime

from paths import NCDpath, getFileNames, runHistoryPath, CRPpath
from regexps import reNCDfilename, reNCDfnRoot
from other import rcut



def deleteCRPPickleFiles():
    
    for fn in os.listdir(CRPpath):
        if fn.endswith('.pkl'):
            os.remove(NCDpath + fn)


def deleteNCDPickleFiles():
    
    for fn in os.listdir(NCDpath):
        if fn.endswith('.pkl'):
            os.remove(NCDpath + fn)


def convertNCDs(lstNCDfn_NCDdict, dataFrameFileName = None):
    
    #TODO: implement this
    dataFrameFileName = rcut(dataFrameFileName, '.pkl.res') + '.pkl.res'    
    
    # Load new NCD files
    lstNCDs = []
    iNCD = 0
    print 'Converting NCDs...'
    for NCDfn_NCDdict in lstNCDfn_NCDdict:
        try:
            NCDdict = NCDfn_NCDdict[1]
            NCDdict['FileName'] = NCDfn_NCDdict[0].rstrip('.pkl')
            m = reNCDfnRoot.search(NCDfn_NCDdict[0])
            NCDdict['Piece 1 Id'] = m.group(1)
            NCDdict['Piece 1 Performance Id'] = m.group(2)
            NCDdict['Piece 2 Id'] = m.group(3)
            NCDdict['Piece 2 Performance Id'] = m.group(4)
            NCDdict['CRP Method'] = m.group(5)
            NCDdict['CRP Dimension'] = float(m.group(6))
            NCDdict['CRP Time Delay'] = float(m.group(7))
            NCDdict['CRP Neighbourhood Size'] = float(m.group(8))
            NCDdict['Downsample Factor'] = m.group(9)
            NCDdict['Feature'] = m.group(10)
            NCDdict['Sequence Length'] = m.group(11)
            NCDdict['NCD DateTime'] = datetime.now()
            lstNCDs.append(copy.deepcopy(NCDdict))
        except:
            print 'Error converting NCD!!!'
            
        iNCD += 1
        if float(iNCD) / 10000 == int(float(iNCD) / 10000):
            print 'Processing NCD #%i' %iNCD
            
    print 'Creating dataframe from results'
    dfNewNCDs = pd.DataFrame(lstNCDs)
    
    # Check for existing NCD dataframe
    if dataFrameFileName is not None:
        if os.path.exists(NCDpath + dataFrameFileName):
            # Read old NCDs dataframe and concatenate new NCDs
            print 'Reading existing results dataframe...'
            dfOldNCDs = pd.read_pickle(NCDpath + dataFrameFileName)
            dfAllNCDs = pd.concat([dfOldNCDs, dfNewNCDs], ignore_index=True)
        else:
            dfAllNCDs = dfNewNCDs
    
        # Save file
        print 'Saving results dataframe %s...' %dataFrameFileName
        dfAllNCDs.to_pickle(NCDpath + dataFrameFileName)
    
    return dfAllNCDs


def convertNCDfiles(dataFrameFileName):
    '''
    Converts NCD results files in the NCD folder into a pandas dataframe
    If the dataframe already exists with old results then the new results are appended
    '''
    
    dataFrameFileName = rcut(dataFrameFileName, '.pkl.res') + '.pkl.res'    
    
    # Load new NCD files
    NCDfiles = [fn for fn in getFileNames(NCDpath, endsWith = '.pkl') if reNCDfilename.search(fn)]
    print 'Total number of files: %i' % len(NCDfiles)
    lstNCDs = []
    iFile = 0
    
    print 'Reading files...'
    for NCDfile in NCDfiles:
        try:
            NCDfileDict = pickle.load(open(NCDpath + NCDfile, 'rb'))
            NCDfileDict['FileName'] = NCDfile.rstrip('.pkl')
            m = reNCDfilename.search(NCDfile)
            NCDfileDict['Piece 1 Id'] = m.group(1)
            NCDfileDict['Piece 1 Performance Id'] = m.group(2)
            NCDfileDict['Piece 2 Id'] = m.group(3)
            NCDfileDict['Piece 2 Performance Id'] = m.group(4)
            NCDfileDict['CRP Method'] = m.group(5)
            NCDfileDict['CRP Dimension'] = float(m.group(6))
            NCDfileDict['CRP Time Delay'] = float(m.group(7))
            NCDfileDict['CRP Neighbourhood Size'] = float(m.group(8))
            NCDfileDict['Downsample Factor'] = m.group(9)
            NCDfileDict['Feature'] = m.group(10)
            NCDfileDict['Sequence Length'] = m.group(11)
            NCDfileDict['File DateTime'] = time.ctime(os.path.getmtime(NCDpath + NCDfile))
            
            
            lstNCDs.append(copy.deepcopy(NCDfileDict))
        except:
            print 'Error reading file: %s' % NCDfile
            
        iFile += 1
        if float(iFile) / 10000 == int(float(iFile) / 10000):
            print 'Processing file #%i' %iFile
            
    print 'Creating dataframe from results files'
    dfNewNCDs = pd.DataFrame(lstNCDs)
    
    # Check for existing NCD dataframe
    if os.path.exists(NCDpath + dataFrameFileName):
        # Read old NCDs dataframe and concatenate new NCDs
        print 'Reading existing results dataframe...'
        dfOldNCDs = pd.read_pickle(NCDpath + dataFrameFileName)
        dfAllNCDs = pd.concat([dfOldNCDs, dfNewNCDs], ignore_index=True)
    else:
        dfAllNCDs = dfNewNCDs

    # Save file
    print 'Saving results dataframe %s...' %dataFrameFileName
    dfAllNCDs.to_pickle(NCDpath + dataFrameFileName)
    
    # Delete old NCD files
    print 'Deleting old results files...'
    for NCDfile in NCDfiles:
        os.remove(NCDpath + NCDfile)
    
    return dfAllNCDs


def getRunHistoryDataFrame():
    '''
    Loads the history of runs and returns a dataframe of the settings
    '''
    lstRunHistory = []
    runHistoryFiles = getFileNames(runHistoryPath, endsWith = '.pkl', orderAlphabetically = True)
    for rhFile in runHistoryFiles:
        lstRunHistory.append(pickle.load(open(runHistoryPath + rhFile, 'rb')))

    df = pd.DataFrame(lstRunHistory)
    return df


def getNCDresults(subFolder = '', featureNames = None, downSampleFactors = None, 
                  methods = None, dimensions = None, timeDelays = None, neighbourhoodSizes = None,
                  numFilesPerFolder = None, sequenceLengths = None):
    '''
    Loads NCD results from results dataframes in the NCD folder
    If you want to only select some results then for each parameter you want to filter,
    include a list of the values you want to keep
    '''
    if subFolder != '':
        subFolder = subFolder.rstrip('/') + '/'
    runHistoryFiles = getFileNames(runHistoryPath + subFolder, endsWith = '.pkl', orderAlphabetically = True)
    resultsDataFrames = []
    for rhFile in runHistoryFiles:

        # Load history file
        runDict = pickle.load(open(runHistoryPath + subFolder + rhFile, 'rb'))
        useFile = True

        # Check filters
        if featureNames is not None:
            if runDict['featureName'] not in featureNames:
                useFile = False
        if downSampleFactors is not None:
            if runDict['downSampleFactor'] not in downSampleFactors:
                useFile = False
        if methods is not None:
            if runDict['method'] not in methods:
                useFile = False
        if dimensions is not None:
            if runDict['dimension'] not in dimensions:
                useFile = False
        if timeDelays is not None:
            if runDict['timeDelay'] not in timeDelays:
                useFile = False
        if neighbourhoodSizes is not None:
            if runDict['neighbourhoodSize'] not in neighbourhoodSizes:
                useFile = False
        if numFilesPerFolder is not None:
            if runDict['numFilesPerFolder'] not in numFilesPerFolder:
                useFile = False
        if sequenceLengths is not None:
            if runDict['sequenceLength'] not in sequenceLengths:
                useFile = False

        # Load results file
        if useFile:
            # Read dataframe and append results
            print 'Reading %s...' % (rhFile + '.res')
            resultsDataFrames.append(pd.read_pickle(NCDpath + subFolder + rhFile + '.res'))

    # Create and return dataframe of all results
    print 'Creating results dataframe'
    dfAll = pd.concat(resultsDataFrames)
    return dfAll
