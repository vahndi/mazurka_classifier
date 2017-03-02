# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:18:22 2015

@author: Vahndi
"""

import os
from featureConverter import featuresToFENS
from featureAnalysis import loadAndTransformFeatureFile
from paths import getFileNames, getFolderNames, getFENSrootPath
from FeatureFileProps import FeatureFileProps as FFP
from NNs import get_NN_NCD_params



def createFENSfeatures(featureName, numFolders, numFilesPerFolder,
                       FENSwindowLength, FENSdownsampling,
                       NNtype, learningRate, learningRateBoostFactor, corruptionLevel, timeStacking,
                       numOriginalFeatures, numNewFeatures, batchSize, frequencyStandardisation,
                       saveTransformedFeatures = False, 
                       NNnumFolders = None, NNnumFilesPerFolder = None,
                       FENStransformationFunction = None,
                       FENSnormalisationThreshold = 0.001, 
                       FENSquantisationSteps = [0.4, 0.2, 0.1, 0.05], 
                       FENSquantisationWeights = [1, 1, 1, 1]):
    
    # Initialise
    FENSfeatures = []
    
    # Get the folders (performances)
    piecesPath = FFP.rootPath[featureName]
    piecesFolders = getFolderNames(piecesPath, 
                                   contains = 'mazurka',
                                   orderAlphabetically = True) # added the contains parameter to avoid the new powerspectrum folder
    if numFolders is not None:
        piecesFolders = piecesFolders[: numFolders]
    
    # Load weights and biases        
    weightMatrix, biases, featureOffset, featureScaling = get_NN_NCD_params(
                                                            NNtype, featureName, learningRate, learningRateBoostFactor,
                                                            corruptionLevel, numOriginalFeatures, numNewFeatures, batchSize, 
                                                            freqStd = frequencyStandardisation, NNnumFolders = NNnumFolders, 
                                                            NNnumFilesPerFolder = NNnumFilesPerFolder,
                                                            NNtimeStacking = timeStacking)
                                          
    # Create root path for FENS features
    FENSrootPath = getFENSrootPath(featureName, NNtype, numOriginalFeatures, numNewFeatures, FENSdownsampling, FENSwindowLength)
    if not os.path.exists(FENSrootPath):
        print 'Creating Root Path for FENS files...'
        os.makedirs(FENSrootPath)

    # For each piece
    for piecesFolder in piecesFolders:
    
        # Create folder for FENS files if it doesn't exist
        FENSpath = FENSrootPath + piecesFolder + '/'
        if os.path.exists(FENSpath):
            print 'FENS folder already exists...'
        else:
            print 'making FENS folder for %s' % piecesFolder
            os.makedirs(FENSpath)

        # Get performances of the piece
        featuresPath = FFP.getFeatureFolderPath(piecesPath + piecesFolder, featureName)
        performances = getFileNames(featuresPath, orderAlphabetically = True, endsWith =  '.csv')
        if numFilesPerFolder is not None:
            performances = performances[: numFilesPerFolder]
        print 'found %i performances' % len(performances)
        
        # Create CENS files of each performance
        for performance in performances:
            print 'converting %s' %performance
            # Load feature file and transform
            dfFeatures = loadAndTransformFeatureFile(featuresPath + performance, 
                                                     featureOffset, featureScaling, timeStacking,
                                                     weightMatrix, biases)
            # Save transformed features, if selected
            if saveTransformedFeatures:
                FeaturesFn = FENSpath + performance.split('_')[0] + '_%s-%s-%iv-%ih_TransformedFeatures.pkl' \
                                                             % (featureName, NNtype, numOriginalFeatures, numNewFeatures)
                dfFeatures.to_pickle(FeaturesFn)

            # Convert to FENS features
            dfFENS = featuresToFENS(dfFeatures, numNewFeatures, 
                                    downSampleFactor = FENSdownsampling, 
                                    downSampleWindowLength = FENSwindowLength,
                                    transformationFunction = FENStransformationFunction,
                                    normalisationThreshold = FENSnormalisationThreshold,
                                    quantisationSteps = FENSquantisationSteps,
                                    quantisationWeights = FENSquantisationWeights)
            FENSfn = FENSpath + performance.split('_')[0] + '_%s-%s-%iv-%ih-FENS_dsf%i_dswl%i.csv' \
                                                             % (featureName, NNtype, numOriginalFeatures, 
                                                                numNewFeatures, FENSdownsampling, FENSwindowLength)
            FENSfeatures.append((FENSfn, dfFENS))
        
    return FENSfeatures



def createFENSfiles(featureName, numFolders, numFilesPerFolder,
                    FENSwindowLengths, FENSdownsamplings, 
                    NNtype, learningRate, learningRateBoostFactor, corruptionLevel, timeStacking,
                    numOriginalFeatures, numNewFeatures, batchSize, frequencyStandardisation,
                    saveTransformedFeatures = False,
                    NNnumFolders = None, NNnumFilesPerFolder = None,
                    FENStransformationFunction = None,
                    FENSnormalisationThreshold = 0.001, 
                    FENSquantisationSteps = [0.4, 0.2, 0.1, 0.05], 
                    FENSquantisationWeights = [1, 1, 1, 1]):
    
    FENSfeatures = createFENSfeatures(featureName, numFolders, numFilesPerFolder,
                   FENSwindowLengths, FENSdownsamplings, 
                   NNtype, learningRate, learningRateBoostFactor, corruptionLevel, timeStacking,
                   numOriginalFeatures, numNewFeatures, batchSize, frequencyStandardisation,
                   saveTransformedFeatures, NNnumFolders, NNnumFilesPerFolder,
                   FENStransformationFunction, FENSnormalisationThreshold, FENSquantisationSteps, FENSquantisationWeights)
                   
    for ff in FENSfeatures:
        FENSfn = ff[0]
        dfFENS = ff[1]
        dfFENS.to_csv(FENSfn, header = False)



if __name__ == '__main__':
    
    #################################### Settings ############################
    
    featureName = 'logfreqspec'
    numFolders = 20 # set to None to use all folders
    numFilesPerFolder = None # set to None to use all files    
    
    # weights
    NNtype = 'dA'
    numOriginalFeatures = 128
    numNewFeatures = 12
    batchSize = 40
    corruptionLevel = 0
    learningRate = 0.05
    learningRateBoostFactor = 1.5
    timeStacking = 1
    frequencyStandardisation = False
    NNnumFolders = 20
    NNnumFilesPerFolder = 5
    
    # single FENS settings
    normalisationThreshold = 0.0
    quantisationSteps = [0.8097468642655186, 0.6570640034464664, 0.3455049357267789, 0.11655355670485129]
    quantisationWeights = [1.0366803456948817, 1.0142465637185185, 1.0018409772501293, 1.0446440641452273]
    transformationFunction = None
    
    # looped FENS settings
    FENSwindowLength = 41
    FENSdownsampling = 10

    saveTransformedFeatures = True
    
    ##########################################################################
    
    
    createFENSfiles(featureName, numFolders, numFilesPerFolder,
                    FENSwindowLength, FENSdownsampling,
                    NNtype, learningRate, learningRateBoostFactor, corruptionLevel, timeStacking,
                    numOriginalFeatures, numNewFeatures, batchSize, frequencyStandardisation,
                    saveTransformedFeatures, 
                    NNnumFolders, NNnumFilesPerFolder,
                    FENStransformationFunction = transformationFunction,
                    FENSnormalisationThreshold = normalisationThreshold,
                    FENSquantisationSteps = quantisationSteps,
                    FENSquantisationWeights = quantisationWeights)
