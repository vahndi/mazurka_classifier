from shared import *
from regexs import reCRPfilename
from ncd import memoryNCD
import os, copy, pickle
import numpy as np
from crp_v4 import crp_v4
import multiprocessing as mp
from time import sleep
from multiprocessing import Pool
import pandas as pd
import multiprocessing

# v2: This version compares all files in 2 folders

### Settings ###

numProcesses = 8
featureNames = ['chroma']
CRPmethods = ['maxnorm', 'euclidean', 'minnorm', 'rr', 'fan', 'nrmnorm']
CRPdimensions = [1, 2, 3, 4, 5, 6, 7]
CRPtimeDelays = [1, 2, 3, 4, 5]

#################


def getFeatureFolderPathAndName(piecePath, featureName):
    '''
    Returns the path and name of the folder containing specific features within the
    folder for a specific piece
    '''
    featureFolderPrefix = featureFileProps.folderPrefix[featureName]
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
    featureFileSuffix = featureFileProps.fileSuffix[featureName]
    featureFileNames = getFileNames(featuresPath, featureFileSuffix)
    featureFilePaths = [featuresPath + fName for fName in featureFileNames]
    featureFileIds = [fName.rstrip(featureFileSuffix) for fName in featureFileNames]
    for i in np.arange(len(featureFileIds)):
        fileName = featureFileNames[i]
        filePath = featureFilePaths[i]
        pieceId = pieceFolder
        featureFileDict[featureFileIds[i]] = featureFileProps(fileName, filePath, pieceId)
        
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


def createCRPfile(crpProps):
    '''
    Calculates a CRP and saves a file for the given file and CRP settings
    Returns True if successful or False if there was an error
    '''
    outputFilePath = CRPpath + crpProps.getFileName()
    try:
        x = np.genfromtxt(crpProps.featureFilePath, delimiter = ',')
        y = crp_v4(x, dimension = crpProps.dimension, timeDelay = crpProps.timeDelay, method = crpProps.method)
        np.save(outputFilePath, y)
        return True
    except:
        return False


def multi_createCRPfile(args):
    '''
    Run wrapper for createCRPfile()
    '''
    return createCRPfile(*args)
   

def loadCRPfiles(lstCRPprops):
    '''
    Loads all CRP files into memory for the given list of CRP properties
    '''
    CRPfiles = {}
    for CRPprops in lstCRPprops:
        fCRP = open(CRPpath + CRPprops.getFileName() + '.npy', 'rb')
        CRPdata = fCRP.read()
        fCRP.close()
        CRPprops.featureFilePath = None
        CRPfiles[CRPprops.toTuple(False)] = CRPdata
        
    return CRPfiles


def NCDexists(ncdFilename, existingNCDs = None):
    '''
    Returns true if the NCD exists either as a file or a record in the set of existing NCDs, if given
    '''
    if existingNCDs is not None:
        if ncdFilename in existingNCDs:
            return True
    if os.path.exists(NCDpath + ncdFilename + '.pkl'):
        return True

    return False


def createNCDfile(NCDfilename, CRP1, CRP2):
    '''
    Calculates the NCD between two CRPs and saves a results file
    Returns True if successful or False if there was an error
    '''
    ncdFilePath = NCDpath + NCDfilename + '.pkl'
    try:
        ncd = memoryNCD(CRP1, CRP2)
        pickle.dump(ncd, open(ncdFilePath, 'wb'))
        return True
    except:
        return False


def multi_createNCDfile(args):
    '''
    Run wrapper for createNCDfile()
    '''
    return createNCDfile(*args)


def createNCDfiles(existingNCDs):

    pool = Pool(numProcesses)
    mazurkaIds = getFolderNames(mazurkasPath)
    if existingNCDs is not None:
        existingNCDs = set(existingNCDs) # makes checking faster

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
                    # Create list of required NCD files
                    requiredNCDs = []
                    featureFileIds = featureFileDict.keys()
                    numFeatureFiles = len(featureFileIds)
                    print 'Checking for NCD files...'
                    for f1 in np.arange(numFeatureFiles - 1):
                        featureFilePath1 = featureFileDict[featureFileIds[f1]].filePath
                        pc1Id = featureFileDict[featureFileIds[f1]].pieceId
                        pc1pfId = featureFileIds[f1]
                        for f2 in np.arange(f1 + 1, numFeatureFiles):
                            featureFilePath2 = featureFileDict[featureFileIds[f2]].filePath
                            pc2Id = featureFileDict[featureFileIds[f2]].pieceId
                            pc2pfId = featureFileIds[f2]
                            ncdProps = NCDprops(pc1Id, pc1pfId, pc2Id, pc2pfId, 
                                                method, dimension, timeDelay,
                                                featureName, featureFilePath1, featureFilePath2)
                            if not NCDexists(ncdProps.getFileName(), existingNCDs = existingNCDs):
                                requiredNCDs.append(ncdProps)
                    print 'Number of NCD files missing for combination: %i' % len(requiredNCDs)
                    # Create Required CRPs for NCD files
                    if len(requiredNCDs) > 0:
                        # Create CRP files and save to the CRPs folder
                        print 'Calculating # of required CRP files'
                        requiredCRPs = []
                        sourceCRPs = []
                        for requiredNCD in requiredNCDs:
                            crp1 = requiredNCD.getCRP1()
                            crp2 = requiredNCD.getCRP2()
                            sourceCRPs.append(crp1)
                            sourceCRPs.append(crp2)
                            if not crp1.hasExistingFile():
                                requiredCRPs.append(crp1)
                            if not crp2.hasExistingFile():
                                requiredCRPs.append(crp2)
                        requiredCRPs = CRPprops.uniqueCRPprops(requiredCRPs)
                        sourceCRPs = CRPprops.uniqueCRPprops(sourceCRPs)
                        numRequiredCRPs = len(requiredCRPs)
                        print 'Creating %i required CRP files' % numRequiredCRPs
                        if numRequiredCRPs > 0:
                            CRPargList = []
                            for crp in requiredCRPs:
                                CRPargList.append((crp,)) # the comma is to make it a tuple
                            pool.map(multi_createCRPfile, CRPargList)
                        # Load CRP files into memory
                        print 'Loading %i CRP files' % len(sourceCRPs)
                        CRPfiles = loadCRPfiles(sourceCRPs)
                        # Create NCD files
                        numNCDs = len(requiredNCDs)
                        print 'Creating %i NCD files' % numNCDs
                        NCDindex = 0                         
                        while NCDindex < numNCDs:
                            NCDargList = []
                            for iNCD in np.arange(NCDindex, min(NCDindex + 100, numNCDs)):
                                requiredNCD = requiredNCDs[iNCD]
                                NCDfn = requiredNCD.getFileName()
                                CRPtuple1 = requiredNCD.getCRP1().toTuple(False)
                                CRPtuple2 = requiredNCD.getCRP2().toTuple(False)
                                NCDargList.append((NCDfn, CRPfiles[CRPtuple1], CRPfiles[CRPtuple2]))
                            pool.map(multi_createNCDfile, NCDargList)
                            NCDindex += 100
                            print NCDindex
                        # Delete CRP files
                        print 'Deleting CRP files'
                        for CRPfilename in getFileNames(CRPpath, '.npy'):
                            try:
                                os.remove(CRPpath + CRPfilename)
                            except:
                                pass



# Main
existingNCDs = None
resultsFn = NCDpath + 'NCDs.pkl.res'
if os.path.exists(resultsFn):
    dfExistingNCDs = pd.read_pickle(resultsFn)
    existingNCDs = list(dfExistingNCDs['FileName'])

createNCDfiles(existingNCDs)

