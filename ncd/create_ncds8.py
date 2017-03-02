from __future__ import division

import numpy as np

from FeatureFileProps import FeatureFileProps
from CRPprops import CRPprops
from NCDprops import NCDprops
from paths import getFolderNames
from FeatureFileProps import FeatureFileProps as FFP
from MAPhelpers import cleanCRPfolder



# Revision History
# ----------------

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



def multi_createCRPfile(crpProps):
    '''
    Run wrapper for createCRPfile()
    '''
    return CRPprops.createCRPfile()
   

def multi_createNCDfile(args):
    '''
    Run wrapper for createNCDfile()
    '''
    return NCDprops.createNCDfile(*args)


def createNCDfiles(existingNCDs, processPool,
                   featureName, numFeatures, 
                   downSampleFactor, timeDelay, dimension, method, neighbourhoodSize, 
                   numFolders, numFilesPerFolder, sequenceLength,
                   weightMatrix = None, biases = None, featureOffset = 0.0, featureScaling = 1.0, timeStacking = None):
    '''
    Inputs:
        :existingNCDs: a list of existing NCD files in order to avoid duplication (legacy - set to None)
        :processPool: a pool of multiprocessing processes to use for running the script
        :featureName: the name of the feature e.g 'chroma', 'mfcc'
        :downSampleFactor: the factor to use in downsampling the original signals before creating CRPs
        :timeDelay: the time delay to use in creating the CRPs
        :method: the method to use in creating the CRPs
        :neighbourhoodSize: the neighbourhood size to use in creating the CRPs
        :numFilesPerFolder: the number of performances of each piece to use - set to None to use all performances
        :sequenceLength: fixed sequence length to normalise CRPs to (use 'var' for variable length)

    Feature Transformation Inputs (optional - all or none):
        :weightMatrix: a matrix of weights (inputFeatureLength rows x outputFeatureLength columns) 
                       to transform the input feature files with before calculating the CRPs
        :biases: a matrix of biases to add to the transformed features
        :featureOffset: the offset to add to the features before scaling
        :featureScaling: the scaling to apply to the features
        :timeStacking: how much to stack the features by horizontally
    '''
    
    mazurkasPath = FeatureFileProps.getRootPath(featureName)
    mazurkaIds = getFolderNames(mazurkasPath, contains = 'mazurka', orderAlphabetically = True)[:numFolders]
    
    if existingNCDs is not None:
        existingNCDs = set(existingNCDs) # makes checking faster

    # Get performances from folders
    featureFileDict = FFP.getFeatureFileDictAllFolders(mazurkasPath, mazurkaIds, featureName, numFilesPerFolder)
    
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
            if not NCDprops.NCDfileExists(ncdProps.getFileName(), existingNCDs = existingNCDs):
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
                crp.timeStacking = timeStacking
                crp.numFeatures = numFeatures
                CRPargList.append((crp,))
            processPool.map(multi_createCRPfile, CRPargList)

        # Load CRP files into memory
        print 'Loading %i CRP files' % len(sourceCRPs)
        CRPfiles = CRPprops.loadCRPfiles(sourceCRPs)

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
        cleanCRPfolder()
