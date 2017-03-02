from __future__ import division

from ncd import memoryNCD
from FeatureFileProps import FeatureFileProps
from CRPprops import CRPprops
from NCDprops import NCDprops
from paths import getFolderNames, getFileNames, CRPpath, NCDpath
from crp_v4 import crp_v4

import os, copy, pickle
import numpy as np
from scipy import signal
from math import ceil


# v5: This version allows for an additional preprocessing step on input features 
# to apply the weights of a neural network before calculating CRPs


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
    # frames implementation. Use 'var' for numFrames to return the original CRP
    N.B this implementation has been found to sometimes return 2 instead of 1 (0.1% of the time)!!
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

        
def downSampleCRPmax(CRP, numFrames):
    '''
    Used to downsample a CRP before calculating the NCD between two CRPs for fixed
    # frames implementation. Use 'var' for numFrames to return the original CRP.
    Uses a max filter to favour ones over zeros
    '''
    if numFrames == 'var':
        return CRP
    else:
        newCRP = np.zeros([numFrames, numFrames], dtype = np.uint8)
        filterSize = (int(np.ceil(float(CRP.shape[0]) / numFrames)), int(np.ceil(float(CRP.shape[1]) / numFrames)))
        rowArr = np.linspace(0, CRP.shape[0] - filterSize[0], numFrames, dtype= np.int)
        colArr = np.linspace(0, CRP.shape[1] - filterSize[1], numFrames, dtype= np.int)
        for iRow in range(numFrames):
            for iCol in range(numFrames):
                newCRP[iRow, iCol] = np.max(CRP[rowArr[iRow]: rowArr[iRow] + filterSize[0], 
                                                colArr[iCol]: colArr[iCol] + filterSize[1]])
        
        return newCRP


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
    featureFileSuffix = FeatureFileProps.fileSuffix[featureName]
    featureFileNames = sorted(getFileNames(featuresPath, featureFileSuffix, True))
    if numFiles is not None:
        featureFileNames = featureFileNames[: numFiles]
    featureFilePaths = [featuresPath + fName for fName in featureFileNames]
    featureFileIds = [fName.rstrip(featureFileSuffix) for fName in featureFileNames]
    for i in np.arange(len(featureFileIds)):
        fileName = featureFileNames[i]
        filePath = featureFilePaths[i]
        pieceId = pieceFolder
        featureFileDict[featureFileIds[i]] = FeatureFileProps(fileName, filePath, pieceId)
        
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
        featuresPath = FeatureFileProps.getFeatureFolderPath(rootPath + pieceFolder + '/', featureName)
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
        # Transform by weight matrix and add bias
        if crpProps.weightMatrix is not None:
            x_t = np.expand_dims(x[:,0], 1)
            x_data = x[:, 1:]
            x_data = (x_data + crpProps.featureOffset) * crpProps.featureScaling
            xmat_transformed = np.matrix(x_data) * crpProps.weightMatrix
            xarr_transformed = np.array(xmat_transformed) + np.tile(crpProps.biases.transpose(), [x_data.shape[0], 1])
            x_transformed = np.append(x_t, xarr_transformed, axis = 1)
        else:
            x_transformed = x
        # Downsample
        x_ds = downSample(x_transformed, crpProps.downSampleFactor)
        # Calculate CRP
        y = crp_v4(x_ds, dimension = crpProps.dimension, timeDelay = crpProps.timeDelay, 
                   method = crpProps.method, neighbourhoodSize = crpProps.neighbourhoodSize)
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
    for crpProps in lstCRPprops:
        try:
            fCRP = open(CRPpath + crpProps.getFileName() + '.npy', 'rb')
            CRPdata = fCRP.read()
            fCRP.close()
            crpProps.featureFilePath = None
            CRPfiles[crpProps.toTuple(False)] = CRPdata
        except:
            pass
        
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
                   numFilesPerFolder, sequenceLength,
                   weightMatrix = None, biases = None, featureOffset = 0.0, featureScaling = 1.0):
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
        :weightMatrix: a matrix of weights (inputFeatureLength rows x outputFeatureLength columns) 
                       to transform the input feature files with before calculating the CRPs
    '''
    
    mazurkasPath = FeatureFileProps.rootPath[featureName]
    
    mazurkaIds = getFolderNames(mazurkasPath, True)[:20]
    if existingNCDs is not None:
        existingNCDs = set(existingNCDs) # makes checking faster

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
                crp.weightMatrix = weightMatrix
                crp.biases = biases
                crp.featureOffset = featureOffset
                crp.featureScaling = featureScaling
                CRPargList.append((crp,))
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
                try:
                    NCDargList.append((NCDfn, CRPfiles[CRPtuple1], CRPfiles[CRPtuple2]))
                except:
                    pass
            if NCDargList:
                processPool.map(multi_createNCDfile, NCDargList)
            NCDindex += 100
            print '\r%i...' %NCDindex,

        # Delete CRP files
        print 'Deleting CRP files'
        for CRPfilename in getFileNames(CRPpath, '.npy', True):
            try:
                os.remove(CRPpath + CRPfilename)
            except:
                pass
