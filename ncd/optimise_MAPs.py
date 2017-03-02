# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""

# This version works with create_ncds5.py i.e. after support for feature transformations
# by neural net weights and works specifically with denoising AutoEncoders

import pickle
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

from paths import runHistoryPath, getWeightsPath
from optimiser import Optimiser
from create_ncds7 import createNCDfiles
from ncd_processing import convertNCDfiles, deleteNCDPickleFiles
from getResults2 import getDataFrameMAPresult
from MAPhelpers import moveResultsFiles
from NNs import get_NN_NCD_params

# This script was formally know as run_ncds6.py but has been renamed to reflect
# the type of run that it does i.e. an 'optimisation' over parameters to produce
# the best MAP

################################### Settings ##################################

numProcesses = 10
subFolder = 'run49'

# Single settings
featureName = 'logfreqspec-dA-256v-12h-FENS_dsf8_dswl30'
CRPmethod = 'fan'

numFolders = 20
numFilesPerFolder = 5 # Set to None to compare all files in each folder

# Feature Settings
numFeatures = 12

# CRP settings
CRPdimensions = [1, 2, 3, 4, 5]
CRPneighbourhoodSizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
CRPtimeDelays = [1, 2, 3, 4, 5, 6, 7]

# NCD settings
sequenceLengths = [300, 500, 700, 900, 1100]

# Neural Network settings
NNtype = None
learningRate = 0.05
learningRateBoostFactor = 1.5
batchSize = 40
downSampleFactors = [1, 2, 4, 8]
numVisible = 256
numHiddens = [3, 6, 9, 12, 15, 18, 21, 24]
corruptionLevels = [0, 10, 20, 30, 40, 50, 60, 70]
timeStacking = 1
freqStd = True

stopRunningAt = datetime(2015, 9, 5, 23)

###############################################################################


# Get path to weights files
if NNtype is not None:
    weightsPath = getWeightsPath(NNtype, featureName, batchSize = batchSize,
                                 learningRate = learningRate, learningRateBoostFactor = learningRateBoostFactor,
                                 timeStacking = timeStacking, numFeatures = numFeatures, frequencyStandardisation = freqStd)

# Create a dictionary of settings to optimise over
settingsDict = {'Dimension': CRPdimensions,
                'DownSample Factor': downSampleFactors,
                'Neighbourhood Size': CRPneighbourhoodSizes,
                'Sequence Length': sequenceLengths,
                'Time Delay': CRPtimeDelays}
if NNtype is not None:
    settingsDict['dA Num Hidden Units'] = numHiddens
    settingsDict['dA Corruption Level'] = corruptionLevels
   
             
numRuns = 0
deleteNCDPickleFiles()
existingNCDs = None
processPool = Pool(numProcesses)
iteration = 0

opt = Optimiser(settingsDict, 
                oldResultsDataFrame = None, 
                resultsColumn = 'Mean Average Precision',
                noImprovementStoppingRounds = None)

currentDateTime = datetime.now()


while currentDateTime < stopRunningAt:
    
    nextSettings = True
    iteration += 1

    while nextSettings is not None and currentDateTime < stopRunningAt:
        nextSettings = opt.getNextSettings()
        if nextSettings is not None:
            for setting in nextSettings:
                # Create CRPs and NCDs
                # load weights if this is for a neural net run
                weightMatrix = None
                biases = None
                featureOffset = None
                featureScaling = None
                if NNtype is not None:
                    weightMatrix, biases, featureOffset, featureScaling = get_NN_NCD_params(NNtype, 
                        featureName, learningRate, learningRateBoostFactor, setting['dA Corruption Level'], 
                        numVisible, setting['dA Num Hidden Units'], batchSize,
                        freqStd, numFolders, numFilesPerFolder, timeStacking)

                # create NCD files
                for key in setting.keys():
                    print key, ':', setting[key]
                createNCDfiles(existingNCDs, processPool,
                               featureName, numFeatures, 
                               setting['DownSample Factor'], setting['Time Delay'], 
                               setting['Dimension'], CRPmethod, setting['Neighbourhood Size'], 
                               numFolders, numFilesPerFolder, setting['Sequence Length'], 
                               weightMatrix, biases, featureOffset, featureScaling, timeStacking)
                               
                # create and save a record of the run settings
                runDict = {'featureName': featureName,
                           'method': CRPmethod, 
                           'numFilesPerFolder': numFilesPerFolder}
                if NNtype is not None:
                    runDict['networkType'] = NNtype
                    runDict['timeStacking'] = timeStacking
                    runDict['dA Learning Rate'] = learningRate
                    runDict['dA Batch Size'] = batchSize
                    runDict['dA # Features'] = numFeatures
                for key in setting.keys():
                    runDict[key] = setting[key]
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
                else:
                    print 'No MAP result found!'
                setting['Mean Average Precision'] = MAPresult

            # Increment the number of runs
            numRuns += len(nextSettings)
            
            # Add the results to the optimiser
            opt.addResults(pd.DataFrame(nextSettings))
            print '\nNumber of runs: %i\nIteration: %i\nBest Result: %0.3f\n\n' \
                  %(numRuns, iteration, opt.currentBestResult())
            
            # Save a copy of the results dataframe for the current iteration 
            # after every setting that is run in case of interruption
            df = opt.resultsDataFrame
            df.to_csv(NCDdest + subFolder + '_' + str(iteration) + '.csv')
    
    # Clear optimiser settings for next iteration
    opt.resultsDataFrame = None
    opt.initialiseRandomSettings()
    currentDateTime = datetime.now()
