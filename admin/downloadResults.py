# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:57:21 2015

@author: vahndi
"""

import os, shutil, pickle

from paths import NCDpath, copyFiles, transferPath, getFolderNames, NNpath, createPath, newFeaturesPath
from featureAnalysis import featureFrequenciesHistogram, transformedFeatureFrequenciesHistogram
from NNs import get_NN_WeightsAndBiases, get_NN_featureOffsetAndScaling


def download_MAP_csv_results(resultsIndex = None):
    
    if resultsIndex is not None:
        MAPpath = '%srun%i/' % (NCDpath, resultsIndex)
        copyFiles(MAPpath, transferPath, filesThatEndWith = '.csv')
    else:        
        for folder in getFolderNames(NCDpath, startsWith = 'run', 
                                     orderAlphabetically = True):
            folderIndex = int(folder[3:])
            download_MAP_csv_results(folderIndex)


def download_NN_training_results(NNtype = 'dA'):
    
    trainingPath = '%straining_results/%s/' % (NNpath, NNtype)
    resultsFolders = getFolderNames(trainingPath, orderAlphabetically = True)
    outputPath = transferPath + 'NN training results/'
    createPath(outputPath)
    for folder in resultsFolders:
        resultsPath = trainingPath + folder + '/'
        resultsFn = resultsPath + 'training_records.pkl'
        if os.path.exists(resultsFn):
            shutil.copyfile(resultsFn, outputPath + folder + '.pkl')


def download_Feature_histograms(featureNames = ['chroma', 'constantQspec', 'mfcc', 'logfreqspec', 
                                                'CENS_dsf2_wl41', 'logfreqspec-dA-256v-12h-FENS_dsf2_dswl41'], 
                                transformedFeatureNames = ['logfreqspec-dA-256v-12h-FENS_dsf2_dswl41']):
    
    resultsPath = transferPath + 'feature histograms/'
    createPath(resultsPath)
    
    for featureName in featureNames:
        featureHist = featureFrequenciesHistogram(featureName, 20, 5)
        pickle.dump(featureHist, open(resultsPath + featureName + '_frequency_histograms.pkl', 'wb'))
    
    for featureName in transformedFeatureNames:
        featureHist = featureFrequenciesHistogram(featureName, 20, 5, featureFileType = '.pkl')
        pickle.dump(featureHist, open(resultsPath + featureName + '_transformed_features_frequency_histograms.pkl', 'wb'))


def download_TransformedFeature_histograms(featureName, NNtype, numVisible, numHidden,
                                           numFolders, numFilesPerFolder,
                                           batchSize, 
                                           learningRate, learningRateBoostFactor, 
                                           corruptionLevel, timeStacking, 
                                           frequencyStandardisation):

    # Load weights, biases, feature offset and feature scaling
    weights, biases = get_NN_WeightsAndBiases(NNtype, featureName, 
                                              batchSize, learningRate, learningRateBoostFactor, 
                                              corruptionLevel, timeStacking, frequencyStandardisation,
                                              numVisible, numHidden)
                                                  
    featureOffset, featureScaling = get_NN_featureOffsetAndScaling(featureName,
                                                                   numFolders, numFilesPerFolder, timeStacking,
                                                                   numVisible, frequencyStandardisation)
    
    # Get histogram of transformed features
    hist = transformedFeatureFrequenciesHistogram(featureName, weights, biases,
                                                  featureOffset, featureScaling, timeStacking, 
                                                  numFolders, numFilesPerFolder, numBins = 100)
    
    # Save histogram
    resultsPath = transferPath + 'feature histograms/'
    createPath(resultsPath)
    pickle.dump(hist, open(resultsPath + featureName + 
                           '_crpt_%i_nin_%i_nhdn_%i_basz_%i_lnrt_%0.2f' \
                           % (corruptionLevel, numVisible, numHidden, batchSize, learningRate) + 
                           '_transformed_features_frequency_histograms.pkl', 'wb'))


def downloadAllFeatureHistograms():
    '''
    downloads histograms of all features in the new_features folder
    '''
    newFeatureNames = getFolderNames(newFeaturesPath, orderAlphabetically = True)
    download_Feature_histograms(featureNames = newFeatureNames, transformedFeatureNames = [])
    
    
# Download transformed histograms for different corruption levels
#for corruptionLevel in range(10, 80, 10):
#
#    print corruptionLevel
#    
#    download_TransformedFeature_histograms(featureName = 'logfreqspec',
#                                           NNtype = 'dA', 
#                                           numVisible = 256, 
#                                           numHidden = 12, 
#                                           numFolders = 20, 
#                                           numFilesPerFolder = 5,
#                                           batchSize = 40, 
#                                           learningRate = 0.05, 
#                                           learningRateBoostFactor = 1.5, 
#                                           corruptionLevel = corruptionLevel,
#                                           timeStacking = 1, 
#                                           frequencyStandardisation = True)

## Download authored feature histograms for logfreqspecFENS
#for downsampling in [1, 2, 4, 8, 12]:
#    for windowLength in [10, 20, 30, 41, 50, 60 ,70]:
#        print downsampling, windowLength
#        featureName = 'logfreqspec-dA-256v-12h-FENS_dsf%i_dswl%i' % (downsampling, windowLength)
#        download_Feature_histograms(featureNames = [featureName])
        