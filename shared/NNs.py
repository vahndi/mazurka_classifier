# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 14:06:51 2015

@author: Vahndi
"""

import pickle
import numpy as np
from paths import NNtrainingPath, getWeightsPath
from FeatureFileProps import FeatureFileProps as FFP


def get_NN_WeightsAndBiases(NNtype, featureName, numFolders, numFilesPerFolder,
                            batchSize, learningRate, learningRateBoostFactor, 
                            corruptionLevel, timeStacking, frequencyStandardisation,
                            numFeatures, numHidden):
    
    weightsPath = getWeightsPath(NNtype, featureName, numFolders, numFilesPerFolder, 
                                 batchSize = batchSize,
                                 learningRate = learningRate, learningRateBoostFactor = learningRateBoostFactor,
                                 timeStacking = timeStacking, numFeatures = numFeatures, 
                                 frequencyStandardisation = frequencyStandardisation)
                                 
    strLnRt = format(learningRate, '.2f').rstrip('0').rstrip('.')
    weightsFn = 'weights_corruption_%i_nin_%i_nhdn_%i_basz_%i_lnrt_%s.csv' \
                % (corruptionLevel, numFeatures * timeStacking, numHidden, batchSize, strLnRt)
    weightsMatrix = np.matrix(np.loadtxt(weightsPath + weightsFn, delimiter = ',')).transpose()
    biasesFn = 'biases_corruption_%i_nin_%i_nhdn_%i_basz_%i_lnrt_%s.csv' \
                % (corruptionLevel, numFeatures * timeStacking, numHidden, batchSize, strLnRt)
    biasesMatrix = np.matrix(np.loadtxt(weightsPath + biasesFn, delimiter = ',')).transpose()
    
    return weightsMatrix, biasesMatrix
    

def get_NN_featureOffsetAndScaling(featureName,
                                   NNnumFolders, NNnumFilesPerFolder, NNtimeStacking,
                                   numVisible, freqStd):
    
    if freqStd:
        standardisationDict = pickle.load(
            open(NNtrainingPath + 'feature_%s_numPieces_%i_numPerformances_%i_numFeatures_%i_timeStacking_%i_freqStd_True_standardisationValues.pkl.gz' 
                 % (featureName, NNnumFolders, NNnumFilesPerFolder, numVisible, NNtimeStacking), 'rb'))
        featureOffset = - standardisationDict['Min Value']
        featureScaling = 1 / (standardisationDict['Max Value'] - 
                                standardisationDict['Min Value'])
    else:
        featureOffset = FFP.featureOffset[featureName]
        featureScaling = FFP.featureScaling[featureName]
        
    return featureOffset, featureScaling

        
def get_NN_NCD_params(NNtype, featureName, learningRate, learningRateBoostFactor,
                      corruptionLevel, numVisible, numHidden, batchSize, 
                      freqStd = False, NNnumFolders = None, NNnumFilesPerFolder = None,
                      NNtimeStacking = None):
    
    if NNtype is None:
        return None, None, None, None
    else:
        weights, biases = get_NN_WeightsAndBiases(NNtype, featureName, NNnumFolders, NNnumFilesPerFolder, 
                                                  batchSize, learningRate, 
                                                  learningRateBoostFactor, corruptionLevel, 
                                                  NNtimeStacking, freqStd, numVisible, numHidden)
        
        featureOffset, featureScaling = get_NN_featureOffsetAndScaling(featureName,
                                                                       NNnumFolders, NNnumFilesPerFolder, NNtimeStacking,
                                                                       numVisible, freqStd)
    
        return weights, biases, featureOffset, featureScaling