# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 14:48:41 2015

@author: Vahndi
"""

import os
import pandas as pd
import numpy as np

from featureConverter import transformFeatures
from paths import getFileNames, getFolderNames
from FeatureFileProps import FeatureFileProps as FFP
from other import rcut



def loadFeatureFile(filePath):
    
    x = np.genfromtxt(filePath, delimiter = ',')
    dfFeatures = pd.DataFrame(x)
    dfFeatures.set_index(0, inplace = True)
    return dfFeatures
    

def loadAndTransformFeatureFile(filePath, featureOffset, featureScaling, timeStacking, 
                                weightMatrix, biasesMatrix):
    '''
    loads a feature file from disk, transforms by some weights if specified, and
    return a dataframe of the results
    '''
    x = np.genfromtxt(filePath, delimiter = ',')
    
    # Multiply by weights and add biases
    x_transformed = transformFeatures(x, weightMatrix, biasesMatrix,
                                      featureOffset, featureScaling, timeStacking)
    dfFeatures = pd.DataFrame(x_transformed)
    dfFeatures.set_index(0, inplace = True)
    
    return dfFeatures


def getFeatureFilesDetails(featureNames):
    '''
    Get a list of the files for a particular feature and their details
    '''
    fileDetails = []
   
    for featureName in featureNames:
        # Get names of feature folders
        rootPath = FFP.getRootPath(featureName)
        featuresPath = FFP.getFeatureFolderPath(rootPath, featureName)
        pieceFolders = getFolderNames(featuresPath, 
                                      orderAlphabetically = True)
        # Iterate over pieces
        for pieceFolder in pieceFolders:
            performanceFiles = getFileNames(featuresPath + pieceFolder, 
                                            endsWith = '.csv', 
                                            orderAlphabetically = True)
            # Iterate over performances
            for performanceFile in performanceFiles:
                fileDetails.append({'Feature': featureName,
                                    'Piece': pieceFolder,
                                    'Performance': rcut(performanceFile, FFP.fileSuffix[featureName]),
                                    'Filename': performanceFile})
        
    df = pd.DataFrame(fileDetails)
    df.to_csv('Feature File Details.csv')


def getFeatureValuesDataFrame(featureName, 
                              numFolders, numFilesPerFolder, 
                              featureFileType = '.csv'):
    '''
    Returns a dataframe of the feature values for a specified number of performances 
    '''
    # Get names of feature folders
    rootPath = FFP.getRootPath(featureName)
    pieceFolders = getFolderNames(rootPath, contains = 'mazurka', 
                                            orderAlphabetically = True)

    featureDataFrames = []
    # Iterate over pieces
    for pieceFolder in pieceFolders:
        print 'processing folder: %s' %pieceFolder
        featuresPath = FFP.getFeatureFolderPath(rootPath + pieceFolder, featureName)
        performanceFiles = getFileNames(featuresPath, 
                                        endsWith = featureFileType, 
                                        orderAlphabetically = True)[: numFilesPerFolder]
        # Iterate over performances
        for performanceFile in performanceFiles:
            print '\tprocessing file: %s' % performanceFile
            featureFn = os.path.join(featuresPath, performanceFile)
            if featureFileType == '.csv':
                featureDataFrames.append(pd.read_csv(featureFn, header = None, index_col = 0))
            elif featureFileType == '.pkl':
                featureDataFrames.append(pd.read_pickle(featureFn))
                
    dfAllPerformances = pd.concat(featureDataFrames, ignore_index = True)    
    return dfAllPerformances  


def featureHistogram(featureName, 
                     numFolders, numFilesPerFolder, 
                     featureFileType = '.csv', numBins = 100):
    '''
    Returns a numpy histograme of the feature values for a specified number of performances 
    '''
    dfAllPerformances = getFeatureValuesDataFrame(featureName, 
                                                  numFolders, numFilesPerFolder, 
                                                  featureFileType = featureFileType)
    mat = dfAllPerformances.as_matrix()
    hist = np.histogram(mat, bins = numBins)
    
    return hist


def featureFrequenciesHistogram(featureName, 
                                numFolders, numFilesPerFolder, 
                                featureFileType = '.csv', numBins = 100):
    '''
    Returns an overall and feature-wise dict of  numpy histograms of the 
    feature values for a specified number of performances 
    '''
    dfAllPerformances = getFeatureValuesDataFrame(featureName,
                                                  numFolders, numFilesPerFolder, 
                                                  featureFileType = featureFileType)
    mat = dfAllPerformances.as_matrix()
    hist = {}
    hist['All'] = np.histogram(mat, bins = numBins)
    
    for col in dfAllPerformances.columns:
        hist[col] = np.histogram(dfAllPerformances[col], bins = numBins)
    
    return hist


def getFeatureFrequenciesDataFrame(featureName, weightMatrix, biasesMatrix,
                                   featureOffset, featureScaling, NNtimeStacking, 
                                   numFolders, numFilesPerFolder):
                                       
    # Get the folders (performances)
    piecesPath = FFP.getRootPath(featureName)
    piecesFolders = getFolderNames(piecesPath, 
                                   contains = 'mazurka',
                                   orderAlphabetically = True)[:20] # added the contains parameter to avoid the new powerspectrum folder
    if numFolders is not None:
        piecesFolders = piecesFolders[: numFolders]
    
    # For each piece
    featureDataFrames = []
    for piecesFolder in piecesFolders:
        # Get performances of the piece
        featuresPath = FFP.getFeatureFolderPath(piecesPath + piecesFolder, featureName)
        performances = getFileNames(featuresPath, orderAlphabetically = True, endsWith =  '.csv')
        if numFilesPerFolder is not None:
            performances = performances[: numFilesPerFolder]
        pf = 0
        for performance in performances:
            pf += 1
            print 'Transforming Features %i' % pf
            # Load feature file and transform
            dfTransformedFeatures = loadAndTransformFeatureFile(featuresPath + performance, 
                                                                featureOffset, featureScaling, NNtimeStacking,
                                                                weightMatrix, biasesMatrix)
            featureDataFrames.append(dfTransformedFeatures)
    
    # Calculate Histogram of the transformed features
    dfAllPerformances = pd.concat(featureDataFrames, ignore_index = True)
    
    return dfAllPerformances
    

def transformedFeatureFrequenciesHistogram(featureName, weightMatrix, biasesMatrix,
                                           featureOffset, featureScaling, NNtimeStacking, 
                                           numFolders, numFilesPerFolder, numBins = 100):
    '''
    Calculate histograms of transformed features for  a specified number of 
    performances
    '''
    
    dfTransformedFeatures = getFeatureFrequenciesDataFrame(featureName, weightMatrix, biasesMatrix,
                                                           featureOffset, featureScaling, NNtimeStacking, 
                                                           numFolders, numFilesPerFolder)
    
    mat = dfTransformedFeatures.as_matrix()
    hist = {}
    hist['All'] = np.histogram(mat, bins = numBins)
    for col in dfTransformedFeatures.columns:
        hist[col] = np.histogram(dfTransformedFeatures[col], bins = numBins)
    
    return hist



#def transformedFeatureFrequenciesStatistics(featureName, weightMatrix, biasesMatrix,
#                                           featureOffset, featureScaling, NNtimeStacking, 
#                                           numFolders, numFilesPerFolder):
#
#    stats = {}
#    
#    stats['Histogram'] = 