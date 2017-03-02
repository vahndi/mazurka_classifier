# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 15:09:41 2015

@author: Vahndi
"""

import pickle
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from copy import deepcopy

from paths import runHistoryPath, getWeightsPath
from create_ncds7 import createNCDfiles
from ncd_processing import convertNCDfiles
from getResults2 import getDataFrameMAPresult
from MAPhelpers import moveResultsFiles, get_NN_NCD_params, cleanCRPfolder, cleanNCDfolder
from adjusters import ArrayAdjuster

################################### Settings ##################################

subFolder = 'run47'
numProcesses = 8
numFolders = 20
numFilesPerFolder = 5 # Set to None to compare all files in each folder


# Feature Settings
featureName = 'CENS_dsf2_wl41'
numFeatures = 12
numHidden = 12
downSampleFactor = 2

NNtype = 'dA'
batchSize = 40
corruptionLevel = 70
learningRate = 0.05
learningRateBoostFactor = 1.5
timeStacking = None
freqStd = False


# CRP settings
CRPmethod = 'fan'
CRPdimension = 4
CRPneighbourhoodSize = 0.2
CRPtimeDelay = 7

# NCD settings
NCDsequenceLength = 1100

# Fine Tuners
weightsAdjustment = 0.01
biasesAdjustment = 0.01

# Stop Criteria
stopRunningAt = datetime(2015, 9, 4, 18)
maxUnsuccessfulFinetunes = 50


###############################################################################


# Load initial weights
weightsPath = getWeightsPath(NNtype, featureName, batchSize = batchSize,
                             learningRate = learningRate, learningRateBoostFactor = learningRateBoostFactor,
                             timeStacking = timeStacking, numFeatures = numFeatures, frequencyStandardisation = freqStd)
weightMatrix, biases, featureOffset, featureScaling = get_NN_NCD_params(
    weightsPath, featureName, learningRate, corruptionLevel, numFeatures,
    numHidden, batchSize,
    freqStd, numFolders, numFilesPerFolder, timeStacking)

origWeightMatrix = weightMatrix
origBiases = biases


# Initialise
cleanCRPfolder()
cleanNCDfolder()
existingNCDs = None
processPool = Pool(numProcesses)

numRuns = 0
iteration = 0


currentDateTime = datetime.now()

settingsList = []

while currentDateTime < stopRunningAt:
    
    # Initialise Iteration
    iteration += 1
    bestMAPresult = 0
    weightsAdjuster = ArrayAdjuster(origWeightMatrix)
    biasesAdjuster = ArrayAdjuster(origBiases)
    weightMatrix = weightsAdjuster.getCurrentArray()
    biases = biasesAdjuster.getCurrentArray()
    
    # Loop while maxUnsuccessfulFinetunes is not exceeded
    while (weightsAdjuster.getNumUnsuccessulAdjustments() < maxUnsuccessfulFinetunes 
           and currentDateTime < stopRunningAt):

        createNCDfiles(existingNCDs, processPool,
                       featureName, numFeatures, 
                       downSampleFactor, CRPtimeDelay, 
                       CRPdimension, CRPmethod, CRPneighbourhoodSize, 
                       numFolders, numFilesPerFolder, NCDsequenceLength, 
                       weightMatrix, biases, featureOffset, featureScaling, timeStacking)
                       
        # create and save a record of the run settings
        runDict = {'CRP Dimension': CRPdimension,
                    'CRP Method': CRPmethod,
                    'CRP Neighbourhood Size': CRPneighbourhoodSize,
                    'CRP Time Delay': CRPtimeDelay,
                    'Feature DownSample Factor': downSampleFactor,
                    'Feature Name': featureName,
                    'NCD Sequence Length': NCDsequenceLength,
                    'NN Time Stacking': timeStacking,
                    'NN Type': NNtype,
                    'Run Name': subFolder,
                    'dA Batch Size': batchSize,
                    'dA Corruption Level': corruptionLevel,
                    'dA Learning Rate': learningRate,
                    'dA Num Hidden Units': numHidden,
                    'dA Learning Rate Boost Factor': learningRateBoostFactor}
    
        # Save run settings
        runTime = str(datetime.now()).replace(':', '-')
        pickle.dump(runDict, open(runHistoryPath + runTime + '.pkl', 'wb'))
        
        # Convert NCD files into a dataframe
        dfNCDs = convertNCDfiles(dataFrameFileName = runTime)
    
        # Create subfolders and move results files into them
        NCDdest = moveResultsFiles(subFolder)
        
        # Get the overall MAP of the run and add to the setting
        MAPresult = getDataFrameMAPresult(dfNCDs)
        if MAPresult is not None:
            print 'Mean Average Precision: %0.3f\n' % MAPresult
            if MAPresult > bestMAPresult:
                print 'MAP increased by %0.3f' % (MAPresult - bestMAPresult)
                weightsAdjuster.keepAdjustedArray()
                biasesAdjuster.keepAdjustedArray()
                bestWeights = weightMatrix
                bestBiases = biases
                pickle.dump(bestWeights, open(NCDdest + subFolder + '_best_weights_%i.pkl' % iteration, 'wb'))
                pickle.dump(bestBiases, open(NCDdest + subFolder + '_best_biases_%i.pkl' % iteration, 'wb'))
                bestMAPresult = MAPresult
            else:
                print 'MAP did not increase' % (MAPresult - bestMAPresult)
                weightsAdjuster.keepCurrentArray()
                biasesAdjuster.keepCurrentArray()
        else:
            print 'No MAP result found!'
            weightsAdjuster.keepCurrentArray()
            biasesAdjuster.keepCurrentArray()
        
        # Save the settings and results
        settings = deepcopy(runDict)
        settings['Run Index'] = numRuns
        settings['Iteration'] = iteration
        settings['Mean Average Precision'] = MAPresult
        settings['Weight Adjustment Mean'] = weightsAdjuster.getAdjustmentMean()
        settings['Biases Adjustment Mean'] = biasesAdjuster.getAdjustmentMean()
        settings['Weight Adjustment StDev'] = weightsAdjuster.getAdjustmentStandardDeviation()
        settings['Biases Adjustment StDev'] = biasesAdjuster.getAdjustmentStandardDeviation()
        settingsList.append(settings)
        df = pd.DataFrame(settingsList)
        df.to_csv(NCDdest + subFolder + '_' + str(iteration) + '.csv')
        
        # Update the weights and biases matrices
        weightMatrix = weightsAdjuster.getAdjustedArray(maxAdjustment = weightsAdjustment)
        biases = biasesAdjuster.getAdjustedArray(maxAdjustment = biasesAdjustment)
        
        # Increment the number of runs
        numRuns += 1
        
        print '\nNumber of runs: %i\nIteration: %i\nBest Result: %0.3f\n\n' \
                                          %(numRuns, iteration, bestMAPresult)
        
        # Prepare for next iteration
        cleanCRPfolder()
        cleanNCDfolder()
        currentDateTime = datetime.now()
        