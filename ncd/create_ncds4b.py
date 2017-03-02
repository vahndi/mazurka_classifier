from shared import *
from ncd import memoryNCD
import os, copy, pickle
import numpy as np
from crp_v4 import crp_v4
from scipy import signal
from math import ceil


# v4: This version allows for changing the length of the CRPs before finding the NCD
# v4b: this subversion is used to visualise results at each stage of the process

def downSample(array2d, factor):
    '''
    Used to downsample a feature vector before calculating its CRP
    '''
    numRows, numCols = array2d.shape
    numNewRows = ceil(float(numRows) / factor)
    newArray = np.zeros([numNewRows, numCols])
    newRow = 0
    for rowStart in range(0, numRows, factor):
        newArray[newRow] = array2d[rowStart:min(rowStart + factor, numRows), :].mean(axis = 0)
        newRow += 1
        
    return newArray


def downSampleCRP(CRP, numFrames):
    '''
    Used to downsample a CRP before calculating the NCD between two CRPs for fixed
    # frames implementation. Use None for numFrames to return the original CRP
    '''
    if numFrames == 'var':
        return CRP
    else:
        # Resample along each axis
        CRP_ds1 = signal.resample(CRP, numFrames, axis = 0)
        CRP_ds2 = signal.resample(CRP_ds1, numFrames, axis = 1)
        
        # Convert back to zero or one
        CRP_ds3 = np.array(CRP_ds2 + 0.5, dtype = np.uint8)
        return CRP_ds3


def getFeatureFolderPathAndName(piecePath, featureName):
    '''
    Returns the path and name of the folder containing specific features within the
    folder for a specific piece
    '''
    if 'CENS' in featureName:
        return piecePath, ''
    else:
        featureFolderPrefix = featureFileProps.folderPrefix[featureName]
        featureFolders = [fldr for fldr in getFolderNames(piecePath) if featureFolderPrefix in fldr]
        assert len(featureFolders) == 1, 'number of feature folders must be 1 (it is %i)' % len(featureFolders)
        featuresFolder = featureFolders[0]
        featuresPath = piecePath + featuresFolder + '/'
        
        return featuresPath, featuresFolder


def getFeatureFileDict(piecesPath, pieceFolder, featuresPath, featureName, numFiles = None):
    '''
    Returns a dictionary of all the performances of a given piece
    The dictionary keys are the filenames with the standard suffix for that feature
    removed.
    The dictionary values are new dictionaries with keys and associated values for
    FileName, FilePath and PieceId
    If numFiles is None then all files will be returned, 
        otherwise return the first numFiles files alphabetically
    '''
    featureFileDict = {}
    featureFileSuffix = featureFileProps.fileSuffix[featureName]
    featureFileNames = sorted(getFileNames(featuresPath, featureFileSuffix))
    if numFiles is not None:
        featureFileNames = featureFileNames[: numFiles]
    featureFilePaths = [featuresPath + fName for fName in featureFileNames]
    featureFileIds = [fName.rstrip(featureFileSuffix) for fName in featureFileNames]
    for i in np.arange(len(featureFileIds)):
        fileName = featureFileNames[i]
        filePath = featureFilePaths[i]
        pieceId = pieceFolder
        featureFileDict[featureFileIds[i]] = featureFileProps(fileName, filePath, pieceId)
        
    return featureFileDict

    
def getFeatureFileDictAllFolders(rootPath, folderNames, featureName, numFilesPerFolder = None):
    '''
    Returns featureFileDicts for the given feature for multiple performances of pieces
    in the given folder names. See getFeatureFileDict for details of the featureFileDict
    If numFilesPerFolder is None then all files will be returned from each folder,
        otherwise numFilesPerFolder files will be returned from each folder
    '''
    featureFileDict = {}
    for pieceFolder in folderNames:
        featuresPath, featuresFolder = getFeatureFolderPathAndName(rootPath + pieceFolder + '/', featureName)
        ffDict = copy.deepcopy(getFeatureFileDict(rootPath, pieceFolder, featuresPath, featureName, numFilesPerFolder))
        featureFileDict.update(ffDict)
        
    return featureFileDict


def createCRPfile(crpProps):
    '''
    Calculates a CRP and saves a file for the given file and CRP settings
    Returns True if successful or False if there was an error
    '''
    outputFilePath = CRPpath + crpProps.getFileName()
    try:
        # Load file
        x = np.genfromtxt(crpProps.featureFilePath, delimiter = ',')
        # Downsample
        x_ds = downSample(x, crpProps.downSampleFactor)
        # Calculate CRP
        y = crp_v4(x_ds, 
                   dimension = crpProps.dimension, timeDelay = crpProps.timeDelay, method = crpProps.method,
                   neighbourhoodSize = crpProps.neighbourhoodSize)
        # Normalise Length
        y = downSampleCRP(y, crpProps.sequenceLength)
        # Save CRP
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


def createNCDfiles(existingNCDs, processPool,
                   featureName, downSampleFactor,
                   timeDelay, dimension, method, neighbourhoodSize, 
                   numFilesPerFolder, sequenceLength):
    '''
    Inputs:
        :existingNCDs: a list of existing NCD files in order to avoid duplication
        :processPool: a pool of multiprocessing processes to use for running the script
        :featureName: the name of the feature e.g 'chroma', 'mfcc'
        :downSampleFactor: the factor to use in downsampling the original signals before creating CRPs
        :timeDelay: the time delay to use in creating the CRPs
        :method: the method to use in creating the CRPs
        :neighbourhoodSize: the neighbourhood size to use in creating the CRPs
        :numFilesPerFolder: the number of performances of each piece to use - set to None to use all performances
        :sequenceLength: fixed sequence length to normalise CRPs to (use 'var' for variable length)
    '''
    
    mazurkasPath = featureFileProps.rootPath[featureName]
    
    mazurkaIds = getFolderNames(mazurkasPath)[:2] # only use 2 folders
    if existingNCDs is not None:
        existingNCDs = set(existingNCDs) # makes checking faster

    print 'Time Delay: %s' % str(timeDelay)
    print 'Dimension: %s' % str(dimension)
    print 'Neighbourhood Size: %s' % str(neighbourhoodSize)
    print 'Method: %s' % method
    print 'Feature: %s' % featureName
    print 'DownSampling Factor: %s' % str(downSampleFactor)
    print 'Sequence Length: %s' % str(sequenceLength)

    # Get performances from folders
    featureFileDict = getFeatureFileDictAllFolders(mazurkasPath, mazurkaIds, featureName, numFilesPerFolder)

    # Create list of required NCD files
    requiredNCDs = []
    featureFileIds = featureFileDict.keys()
    numFeatureFiles = len(featureFileIds)
    print 'Checking for existing NCD files...'
    for f1 in np.arange(numFeatureFiles - 1):
        featureFilePath1 = featureFileDict[featureFileIds[f1]].filePath
        pc1Id = featureFileDict[featureFileIds[f1]].pieceId
        pc1pfId = featureFileIds[f1]
        for f2 in np.arange(f1, numFeatureFiles):
            featureFilePath2 = featureFileDict[featureFileIds[f2]].filePath
            pc2Id = featureFileDict[featureFileIds[f2]].pieceId
            pc2pfId = featureFileIds[f2]
            ncdProps = NCDprops(pc1Id, pc1pfId, pc2Id, pc2pfId, 
                                method, dimension, timeDelay,
                                neighbourhoodSize, downSampleFactor, sequenceLength,
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
            processPool.map(multi_createCRPfile, CRPargList)

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
            processPool.map(multi_createNCDfile, NCDargList)
            NCDindex += 100
            print NCDindex

        # Delete CRP files
#        print 'Deleting CRP files'
#        for CRPfilename in getFileNames(CRPpath, '.npy'):
#            try:
#                os.remove(CRPpath + CRPfilename)
#            except:
#                pass
