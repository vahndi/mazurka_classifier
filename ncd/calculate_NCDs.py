from __future__ import division

import numpy as np

from FeatureFileProps import FeatureFileProps
from paths import getFolderNames
from FeatureFileProps import FeatureFileProps as FFP
from MAPhelpers import calculateRequiredNCDs, calculateRequiredCRPs
from ncd import memoryNCD


# Revision History
# ----------------


### As create_ncds{i}.py ###

# v1: Initial version

# v2: This version compares all files in 2 folders

# v3: This version compares a specified number of files from each folder

# v4: This version allows for changing the length of the CRPs before finding the NCD

# v5: This version allows for an additional preprocessing step on input features 
#     to apply the weights of a neural network before calculating CRPs

# v6: This version allows for stacking of horizontal features to a certain number
#     of time steps before multiplication by the weight matrix to incorporate 
#     temporal effects from the feature learning

# v7: Added support for using an integer subset of features from the feature files

# v8: Moved a lot of the functions out to different files


### As calculate_NCDs.py ###

# v1 Doing everything in memory now - only use for training runs


def calculateCRP(crpProps):
    
    crpProps.calculateCRP()
    return crpProps


def multi_calculateCRPs(args):
    '''
    Run wrapper for calculateCRP()
    '''
    return calculateCRP(*args)
   

def calculateNCD(NCDfn, CRP1, CRP2): 
                       
    ncd = memoryNCD(CRP1, CRP2)
    return (NCDfn, ncd)
    

def multi_calculateNCDs(args):
    '''
    Run wrapper for calculateNCD()
    '''
    return calculateNCD(*args)


def calculateNCDs(processPool,
                  featureName, numFeatures, 
                  downSampleFactor, timeDelay, dimension, method, neighbourhoodSize, 
                  numFolders, numFilesPerFolder, sequenceLength,
                  weightMatrix = None, biases = None, featureOffset = 0.0, featureScaling = 1.0, timeStacking = None,
                  featureFileDict = None, pieceIds = None):
    '''
    Inputs:
        :processPool: a pool of multiprocessing processes to use for running the script
        :featureName: the name of the feature e.g 'chroma', 'mfcc'
        :downSampleFactor: the factor to use in downsampling the original signals before creating CRPs
        :timeDelay: the time delay to use in creating the CRPs
        :method: the method to use in creating the CRPs
        :neighbourhoodSize: the neighbourhood size to use in creating the CRPs
        :numFilesPerFolder: the number of performances of each piece to use - set to None to use all performances
        :sequenceLength: fixed sequence length to normalise CRPs to (use 'var' for variable length)

    Feature Transformation Inputs (optional - specify all or none):
        :weightMatrix: a matrix of weights (inputFeatureLength rows x outputFeatureLength columns) 
                       to transform the input feature files with before calculating the CRPs
        :biases: a matrix of biases to add to the transformed features
        :featureOffset: the offset to add to the features before scaling
        :featureScaling: the scaling to apply to the features
        :timeStacking: how much to stack the features by horizontally
        
    Precalculated Inputs:
        :featureFileDict: a pre-loaded or -calculated featureFileDict
        :pieceIds: The names of the pieces
    '''

    # Get performances from folders or use ones in memory
    piecesPath = FeatureFileProps.getRootPath(featureName)  
    if pieceIds is None:
        pieceIds = getFolderNames(piecesPath, contains = 'mazurka', orderAlphabetically = True)[:numFolders]
    if featureFileDict is None:
        print 'Loading feature file dict'
        featureFileDict = FFP.loadFeatureFileDictAllFolders(piecesPath, pieceIds, featureName, numFilesPerFolder)
    
    # Create list of required NCDs
    requiredNCDs = calculateRequiredNCDs(featureFileDict, method, dimension, 
                                         timeDelay, neighbourhoodSize, downSampleFactor, 
                                         sequenceLength, featureName)

    # Create Required CRPs for NCD files
    if len(requiredNCDs) > 0:

        # Calculate required CRPs from featureFileDict
        requiredCRPs = calculateRequiredCRPs(requiredNCDs)
        numRequiredCRPs = len(requiredCRPs)
        
        print 'Creating %i required CRPs' % numRequiredCRPs
        if numRequiredCRPs > 0:
            CRPargList = []
            for crp in requiredCRPs:
                # assign CRP properties
                crp.weightMatrix = weightMatrix
                crp.biases = biases
                crp.featureOffset = featureOffset
                crp.featureScaling = featureScaling
                crp.timeStacking = timeStacking
                crp.numFeatures = numFeatures
                # find feature file data for CRP from featureFileDict
                # currently matches on path for a file that does not exist - not pretty
                for featureFileId in featureFileDict.keys():
                    featureFileProps = featureFileDict[featureFileId]
                    if featureFileProps.pieceId == crp.pieceId and \
                       featureFileProps.performanceId == crp.performanceId:
                        crp.featureFileData = featureFileProps.featureFileData
                CRPargList.append((crp, ))
            CRPs = processPool.map(multi_calculateCRPs, CRPargList)

            # Calculate NCDs
            numNCDs = len(requiredNCDs)
            print 'Creating %i NCDs' % numNCDs
            NCDindex = 0             
            NCDs = []            
            while NCDindex < numNCDs:
                NCDargList = []
                for iNCD in np.arange(NCDindex, min(NCDindex + 100, numNCDs)):
                    requiredNCD = requiredNCDs[iNCD]
                    NCDfn = requiredNCD.getFileName()
                    crp1 = requiredNCD.getCRP1()
                    crp2 = requiredNCD.getCRP2()
                    CRPdata1 = None
                    CRPdata2 = None
                    for crp in CRPs:
                        if crp.pieceId == crp1.pieceId and crp.performanceId == crp1.performanceId:
                            CRPdata1 = crp.CRPdata
                        if crp.pieceId == crp2.pieceId and crp.performanceId == crp2.performanceId:
                            CRPdata2 = crp.CRPdata
                    if CRPdata1 is None or CRPdata2 is None:
                        return None
                    NCDargList.append((NCDfn, CRPdata1.tostring(), CRPdata2.tostring()))
                if NCDargList:
                    NCDs.extend(processPool.map(multi_calculateNCDs, NCDargList))
                NCDindex += 100
                print '\r%i...' % NCDindex,
            
            return NCDs

    return None