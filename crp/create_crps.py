from paths import CRPpath, mazurkasPath
import os
import numpy as np
from crp_v4 import crp_v4
from other import rcut



featureNames = ['mfcc', 'chroma']
CRPmethods = ['maxnorm', 'euclidean', 'minnorm', 'rr', 'fan', 'nrmnorm']
CRPdimensions = 1 + np.arange(7)
CRPtimeDelays = 1 + np.arange(5)




featuresDict= {'mfcc': {'folder prefix': 'qm-mfcc-standard',
                        'file suffix': '_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv'},
               'chroma': {'folder prefix': 'qm-chromagram_standard',
                          'file suffix': '_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv'}}



def getFolderNames(inputPath):
    
    return [folder for folder in os.listdir(inputPath) if os.path.isdir(inputPath + folder)]
    

def getFileNames(inputPath):
    
    return [fName for fName in os.listdir(inputPath) if os.path.isfile(inputPath + fName)]


def getFeatureFolderPathAndName(piecePath, featureName):
    
    featureFolderPrefix = featuresDict[featureName]['folder prefix']
    featureFolders = [fldr for fldr in getFolderNames(piecePath) if featureFolderPrefix in fldr]
    assert len(featureFolders) == 1
    featuresFolder = featureFolders[0]
    featuresPath = mazurkaPath + featuresFolder + '/'
    
    return featuresPath, featuresFolder


def getFeatureFileDict(featuresPath, featureName):
    
    featureFileDict = {}
    featureFileSuffix = featuresDict[featureName]['file suffix']
    featureFileNames = [fName for fName in getFileNames(featuresPath) if fName.endswith(featureFileSuffix)]
    featureFilePaths = [featuresPath + fName for fName in featureFileNames]
    featureFileIds = [rcut(fName, featureFileSuffix) for fName in featureFileNames]
    for i in np.arange(len(featureFileIds)):
        featureFileDict[featureFileIds[i]] = {}
        featureFileDict[featureFileIds[i]]['FileName'] = featureFileNames[i]
        featureFileDict[featureFileIds[i]]['FilePath'] = featureFilePaths[i]
    
    return featureFileDict


def getCRPfilename(pieceId, performanceId, method, dimension, timeDelay):

    crpFileName = 'CRP_pcId_%s_pfId_%s_mthd_%s_dimn_%s_tdly_%s' \
                  % (mId, featureFileId, method, str(dimension), str(timeDelay))

    return crpFileName


def createCRPfile(pieceId, performanceId, method, dimension, timeDelay):

    outputFileName = getCRPfilename(mId, featureFileId, method, dimension, timeDelay)
    outputFilePath = CRPpath + outputFileName
    runningFilePath = CRPpath + outputFileName + '.txt'
    if not (os.path.exists(outputFilePath + '.npy') or os.path.exists(runningFilePath)):
        print 'Creating CRP file %s' % outputFileName
        fRunning = open(runningFilePath, 'w')
        fRunning.close()
        x = np.genfromtxt(featureFileDict[featureFileId]['FilePath'], delimiter = ',')
        y = crp_v4(x, dimension = dimension, timeDelay = timeDelay, method = method)
        np.save(outputFilePath, y)
        os.remove(runningFilePath)
    else:
        print 'File %s already exists' % outputFileName

        
mazurkaIds = getFolderNames(mazurkasPath)

# For each piece
for mId in mazurkaIds:

    print '\nMazurka Id:', mId
    mazurkaPath = mazurkasPath + mId + '/'
    mIdFolders = getFolderNames(mazurkaPath)
    
    # For each feature
    for featureName in featuresDict.keys():
        featuresPath, featuresFolder = getFeatureFolderPathAndName(mazurkaPath, featureName)
        print '\tFeatures folder (%s): %s' %(featureName, featuresFolder)
        print '\tFeature files:'
        featureFileDict = getFeatureFileDict(featuresPath, featureName)
        
        # For each performance
        for featureFileId in featureFileDict.keys():
            print '\t\tId: %s' % featureFileId
            
            # For each time delay
            for timeDelay in CRPtimeDelays:

                # For each dimension
                for dimension in CRPdimensions:

                    # For each method
                    for method in CRPmethods:

                        