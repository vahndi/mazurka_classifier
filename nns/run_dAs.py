# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:02:54 2015

@author: Vahndi
"""

from paths import NNtrainingPath, NNpath
from dA_vm3 import test_dA
import os

############### Settings ###############

#featureNames = ['CENS_dsf2_wl41', 'chroma']
featureNames = ['constantQspec']
 
numFeatures = 48
#numHiddens = [int(round(0.25 * numFeatures)), 
#              int(round(0.5 * numFeatures)), 
#              int(round(0.75 * numFeatures)), 
#              numFeatures]
numHiddens = [12, 24, 36, 48]
batchSizes = [40]
learningRates = [0.1]

########################################


for featureName in featureNames:
    trainingFn = 'feature_' + featureName + '_numPieces_20_numPerformances_5_timeStacking_10.pkl.gz'
    outputPath = NNpath + 'training_results/dA/' + featureName + '/'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for learningRate in learningRates[::-1]:
        for batchSize in batchSizes:
            for numHidden in numHiddens:
                test_dA(NNtrainingPath + trainingFn, 
                        numFeatures = numFeatures, 
                        numHidden = numHidden, 
                        learning_rate = learningRate,
                        batch_size= batchSize,
                        output_folder = outputPath)
