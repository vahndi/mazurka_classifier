# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:02:54 2015

@author: Vahndi
"""

from paths import NNtrainingPath, NNpath
from dA_vm3 import test_dA
import os

############################## Settings ##############################

#featureNames = ['CENS_dsf2_wl41', 'chroma', 'constantQspec', 'logfreqspec', 'CENS_dsf10_wl41']
featureNames = ['chroma']

numPieces = 20
numPerformances = 5

numFeatures = 12
numHiddens = [12]
batchSize = 40
learningRate = 0.05
learningRateBoostFactor = 1.5
timeStacking = 1 # set to None for non-stacked features
frequencyStandardisation = False

######################################################################


for featureName in featureNames:
    trainingFn = 'feature_' + featureName + '_numPieces_%i_numPerformances_%i' % (numPieces, numPerformances) 
    trainingFn += '_numFeatures_%i' % numFeatures
    if timeStacking is not None:
        trainingFn += '_timeStacking_%i' % timeStacking
    if frequencyStandardisation:
        trainingFn += '_freqStd_True'
    trainingFn += '.pkl.gz'
    print 'Training Filename: %s' % trainingFn
    outputPath = NNpath + 'training_results/dA/numFolders_%i_numFiles_%i/%s_bs_%i_lr_%0.2f_lrbf_%0.2f' \
                 % (numPieces, numPerformances, featureName, batchSize, learningRate, learningRateBoostFactor)
    if timeStacking is not None:
        outputPath += '_ts_%i' % timeStacking
    outputPath += '_nf_%i' % numFeatures
    if frequencyStandardisation:
        outputPath += '_freqStd'
    outputPath += '/'
    print 'Output Path: %s' %outputPath
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for numHidden in numHiddens:
        test_dA(NNtrainingPath + trainingFn, 
                numFeatures = numFeatures, 
                numHidden = numHidden, 
                maxCorruptionLevel = 0.7,
                learning_rate = learningRate,
                batch_size = batchSize,
                learningRateBoostFactor = learningRateBoostFactor,
                output_folder = outputPath)
