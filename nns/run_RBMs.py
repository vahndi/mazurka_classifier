# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:02:54 2015

@author: Vahndi
"""

from paths import NNtrainingPath, NNpath
from rbm_vm import test_rbm
import os
from logistic_sgd import load_data

############### Settings ###############

featureNames = ['constantQspec']

numPieces = 20
numPerformances = 5

numFeatures = 48
numHiddens = [12, 24, 36, 48]
batchSize = 20
learningRate = 0.05
learningRateBoostFactor = 1.5
timeStacking = None # set to None for non-stacked features
Ks = [5, 10, 15, 20, 25]

########################################


for featureName in featureNames:
    
    # Create training file name
    trainingFn = 'feature_%s_numPieces_%i_numPerformances_%i' \
                % (featureName, numPieces, numPerformances) 
    if timeStacking is not None:
        trainingFn += '_timeStacking_%i' % timeStacking
    trainingFn += '.pkl.gz'
    trainingData = load_data(NNtrainingPath + trainingFn)
        
    for k in Ks:

        # Create output path
        outputPath = NNpath + 'training_results/RBM/%s_bs_%i_lr_%0.2f_lrbf_%0.2f' \
                              % (featureName, batchSize, learningRate, learningRateBoostFactor)
        if timeStacking is not None:
            outputPath += '_ts_%i' % timeStacking
        outputPath += '/'
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
            
        # Do training
        for numHidden in numHiddens:
            test_rbm(trainingData, 
                     n_visible = numFeatures, 
                     n_hidden = numHidden, 
                     learning_rate = learningRate,
                     batch_size = batchSize,
                     learningRateBoostFactor = learningRateBoostFactor,
                     output_folder = outputPath,
                     k = k)
