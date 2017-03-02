from paths import CRPpath, mazurkasPath, NCDpath, getFolderNames, getFileNames, getCRPfilename, getNCDfilename
from regexs import reCRPfilename
from ncd import memoryNCD
import os, copy, pickle
import numpy as np
from crp_v4 import crp_v4
import multiprocessing as mp
from time import sleep

### Settings ###

numProcesses = 8
featureNames = ['chroma', 'mfcc']
CRPmethods = ['maxnorm', 'euclidean', 'minnorm', 'rr', 'fan', 'nrmnorm']
CRPdimensions = 1 + np.arange(7)
CRPtimeDelays = 1 + np.arange(5)

#################


featuresDict= {'mfcc': {'folder prefix': 'qm-mfcc-standard',
                        'file suffix': '_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv'},
               'chroma': {'folder prefix': 'qm-chromagram_standard',
                          'file suffix': '_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv'}}


def getFeatureFolderPathAndName(piecePath, featureName):
    '''
    Returns the path and name of the folder containing specific features within the
    folder for a specific piece
    '''
    featureFolderPrefix = featuresDict[featureName]['folder prefix']
    featureFolders = [fldr for fldr in getFolderNames(piecePath) if featureFolderPrefix in fldr]
    assert len(featureFolders) == 1, 'number of feature folders must be 1 (it is %i)' % len(featureFolders)
    featuresFolder = featureFolders[0]
    featuresPath = piecePath + featuresFolder + '/'
    
    return featuresPath, featuresFolder


def getFeatureFileDict(piecesPath, pieceFolder, featuresPath, featureName):
    '''
    Returns a dictionary of all the performances of a given piece
    The dictionary keys are the filenames with the standard suffix for that feature
    removed.
    The dictionary values are new dictionaries with keys and associated values for
    FileName, FilePath and PieceId
    '''
    featureFileDict = {}
    featureFileSuffix = featuresDict[featureName]['file suffix']
    featureFileNames = getFileNames(featuresPath, featureFileSuffix)
    featureFilePaths = [featuresPath + fName for fName in featureFileNames]
    featureFileIds = [fName.rstrip(featureFileSuffix) for fName in featureFileNames]
    for i in np.arange(len(featureFileIds)):
        featureFileDict[featureFileIds[i]] = {}
        featureFileDict[featureFileIds[i]]['FileName'] = featureFileNames[i]
        featureFileDict[featureFileIds[i]]['FilePath'] = featureFilePaths[i]
        featureFileDict[featureFileIds[i]]['PieceId'] = pieceFolder
        
    return featureFileDict

    
def getFeatureFileDictAllFolders(rootPath, folderNames, featureName):
    '''
    Returns featureFileDicts for the given feature for all performances of pieces
    in the given folder names. See getFeatureFileDict for details of the featureFileDict
    '''
    featureFileDict = {}
    for pieceFolder in folderNames:
        featuresPath, featuresFolder = getFeatureFolderPathAndName(rootPath + pieceFolder + '/', featureName)
        ffDict = copy.deepcopy(getFeatureFileDict(rootPath, pieceFolder, featuresPath, featureName))
        featureFileDict.update(ffDict)
        
    return featureFileDict


def createCRPfile(featureFilePath, pieceId, performanceId, method, dimension, timeDelay):
    '''
    Calculates a CRP and saves a file for the given file and CRP settings
    '''
    outputFileName = getCRPfilename(pieceId, performanceId, method, dimension, timeDelay)
    outputFilePath = CRPpath + outputFileName
    runningFilePath = CRPpath + outputFileName + '.txt'
    if not (os.path.exists(outputFilePath + '.npy') or os.path.exists(runningFilePath)):
        # print 'Creating CRP file %s' % outputFileName
        fRunning = open(runningFilePath, 'w')
        fRunning.close()
        x = np.genfromtxt(featureFilePath, delimiter = ',')
        y = crp_v4(x, dimension = dimension, timeDelay = timeDelay, method = method)
        np.save(outputFilePath, y)
    else:
        pass
        # print 'File %s already exists' % outputFileName


def loadCRPfiles():
    '''
    Loads all CRP files into memory
    '''
    CRPfiles = sorted(getFileNames(CRPpath, '.npy'))
    CRPs = []
    for CRPfile in CRPfiles:
        fCRP = open(CRPpath + CRPfile, 'rb')
        CRPdata = fCRP.read()
        fCRP.close()
        CRPs.append(CRPdata)
        
    return CRPfiles, CRPs


def createNCDfile(CRPfilename1, CRPfilename2, CRP1, CRP2, existingNCDsDataFrame = None):
    '''
    Calculates the NCD between two CRPs and saves a results file
    '''
    m1 = reCRPfilename.search(CRPfilename1)
    m2 = reCRPfilename.search(CRPfilename2)
    ncdFn = getNCDfilename(m1.group(1), m1.group(2), m2.group(1), m2.group(2), 
                           m1.group(3), float(m1.group(4)), float(m1.group(5)))
    ncdFilePath = NCDpath + ncdFn + '.pkl'
    runningFilePath = NCDpath + ncdFn + '.txt'
    ncdExists = False
    if os.path.exists(ncdFilePath) or os.path.exists(runningFilePath):
        ncdExists = True
    if existingNCDsDataFrame is not None:
        if ncdFn in existingNCDsDataFrame['FileName']:
            ncdExists = True
    if not ncdExists:
        # print i1, i2
        # print 'Creating NCD file %s' % ncdFn
        fRunning = open(runningFilePath, 'w')
        fRunning.close()
        ncd = memoryNCD(CRP1, CRP2)
        pickle.dump(ncd, open(ncdFilePath, 'wb'))
        try:
            os.remove(runningFilePath)
        except:
            pass


def createNCDfiles(existingNCDsDataFrame = None):

    mazurkaIds = getFolderNames(mazurkasPath)

    # For each time delay
    for timeDelay in CRPtimeDelays:
        print 'Time Delay: %s' % str(timeDelay)
        # For each dimension
        for dimension in CRPdimensions:
            print '\tDimension: %s' % str(dimension)
            # For each method
            for method in CRPmethods:
                print '\t\tMethod: %s' % method
                # For each feature
                for featureName in featureNames:
                    print '\t\t\tFeature: %s' % featureName
                    # Get performances from each pair of folders (N.B. this only does the first pair at the moment)
                    featureFileDict = getFeatureFileDictAllFolders(mazurkasPath, mazurkaIds[:2], featureName)
                    # Create CRPs for all files
                    print 'Creating CRP files'
                    for featureFileId in featureFileDict.keys():
                        filePath = featureFileDict[featureFileId]['FilePath']
                        pieceId = featureFileDict[featureFileId]['PieceId']
                        createCRPfile(filePath, pieceId, featureFileId, method, dimension, timeDelay)
                    # Load CRP files into memory
                    CRPfiles, CRPs = loadCRPfiles()
                    numCRPfiles = len(CRPfiles)
                    # Create NCDs for all pairs of CRPs
                    print 'Creating NCD files'
                    for i1 in np.arange(numCRPfiles - 1):
                        for i2 in np.arange(i1 + 1, numCRPfiles):
                            createNCDfile(CRPfiles[i1], CRPfiles[i2], CRPs[i1], CRPs[i2])
                    # Delete CRP files
                    print 'Deleting CRP files'
                    for CRPfile in CRPfiles:
                        if os.path.exists(CRPpath + CRPfile):
                            try:
                                os.remove(CRPpath + CRPfile)
                            except:
                                pass
                            
    fFinished = open(NCDpath + 'finished.txt', 'w')
    fFinished.close()

    
processList = []
for i in range(numProcesses):
    p = mp.Process(target = createNCDfiles) 
    p.start()
    processList.append(p)

    
while not os.path.exists(NCDpath + 'finished.txt'):
    sleep(10)
    pass
    
sleep(5)
    
for p in processList:
    p.join()
