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

from paths import runHistoryPath, NCDpath, createPath
from optimiser import Optimiser
from calculate_NCDs import calculateNCDs
from ncd_processing import convertNCDs
from getResults2 import getDataFrameMAPresult
from NNs import get_NN_NCD_params

# This script was formally know as run_ncds6.py but has been renamed to reflect
# the type of run that it does i.e. an 'optimisation' over parameters to produce
# the best MAP

################################### Settings ##################################

subFolder = 'run66'

numProcesses = 10
numFolders = 20
numFilesPerFolder = 5 # Set to None to compare all files in each folder

# Feature Settings
featureName = 'CENS_dsf10_wl41'
numFeatures = 12

# CRP settings
CRPmethod = 'fan'
CRPdimensions = [1, 2, 3, 4, 5]
CRPneighbourhoodSizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
CRPtimeDelays = [1, 2, 3, 4, 5, 6, 7]
downSampleFactors = [2, 4, 8, 12, 24]

# NCD settings
sequenceLengths = [300, 500, 700, 900, 1100]

# Neural Network settings
#for weights
NNtype = None
learningRate = 0.05
learningRateBoostFactor = 1.5
batchSize = 40
numVisibles = [12]
numHiddens = [12]
corruptionLevels = [0]
timeStacking = 1
freqStd = False

stopRunningAt = datetime(2015, 9, 23, 23)

###############################################################################

# Create folder for results
createPath(NCDpath + subFolder)

# Create a dictionary of settings to optimise over
settingsDict = {'Dimension': CRPdimensions,
                'DownSample Factor': downSampleFactors,
                'Neighbourhood Size': CRPneighbourhoodSizes,
                'Sequence Length': sequenceLengths,
                'Time Delay': CRPtimeDelays}
                
if NNtype is not None:
    settingsDict['dA Num Hidden Units'] = numHiddens
    settingsDict['dA Num Visible Units'] = numVisibles
    settingsDict['dA Corruption Level'] = corruptionLevels
   
# Initialise
numRuns = 0
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
                # load weights etc. if this is for a neural net run
                if NNtype is not None:
                    weightMatrix, biases, featureOffset, featureScaling = get_NN_NCD_params(NNtype, 
                        featureName, learningRate, learningRateBoostFactor, setting['dA Corruption Level'], 
                        setting['dA Num Visible Units'], setting['dA Num Hidden Units'], batchSize,
                        freqStd, numFolders, numFilesPerFolder, timeStacking)
                else:
                    weightMatrix = biases = featureOffset = featureScaling = None

                # Calculate NCDs
                for key in setting.keys():
                    print key, ':', setting[key]
                NCDlist = calculateNCDs(processPool,
                                        featureName, numFeatures, 
                                        setting['DownSample Factor'], setting['Time Delay'], 
                                        setting['Dimension'], CRPmethod, setting['Neighbourhood Size'], 
                                        numFolders, numFilesPerFolder, setting['Sequence Length'], 
                                        weightMatrix, biases, featureOffset, featureScaling, timeStacking)
                
                # Convert NCD files into a dataframe
                runTime = str(datetime.now()).replace(':', '-')
                MAPresult = None
                if NCDlist is None: # there were errors e.g. in CRP calculation after downsampling
                     MAPresult = 0 # need to use something that is not None for the optimiser to find its best result   
                else:
                    dfNCDs = convertNCDs(NCDlist, dataFrameFileName = runTime)
                     # Get the overall MAP of the run and add to the setting
                    MAPresult = getDataFrameMAPresult(dfNCDs)
                if MAPresult is not None and MAPresult != 0:
                    print 'Mean Average Precision: %0.3f\n' % MAPresult
                else:
                    print 'No MAP result found!'
                setting['Mean Average Precision'] = MAPresult
                
                # Create and save a runDict of the settings and result
                # assign single (non-optimised) settings to the runDict
                runDict = {'featureName': featureName,
                           'method': CRPmethod, 
                           'numFilesPerFolder': numFilesPerFolder}
                if NNtype is not None:
                    runDict['NN Type'] = NNtype
                    runDict['dA Learning Rate'] = learningRate
                    runDict['dA Learning Rate Boost Factor'] = learningRateBoostFactor
                    runDict['dA Batch Size'] = batchSize
                    runDict['dA # Features'] = numFeatures
                    runDict['NN timeStacking'] = timeStacking
                    runDict['NN frequency standardisation'] = freqStd
                # assign settings from optimiser
                for key in setting.keys():
                    runDict[key] = setting[key]
                runDict['Mean Average Precision'] = MAPresult
                
                pickle.dump(runDict, open(runHistoryPath + runTime + '.pkl', 'wb'))

            # Increment the number of runs
            numRuns += len(nextSettings)
            
            # Add the results to the optimiser
            opt.addResults(pd.DataFrame(nextSettings))
            print '\nNumber of runs: %i\nIteration: %i\nBest Result: %0.3f\n\n' \
                  %(numRuns, iteration, opt.currentBestResult())
            
            # Save a copy of the results dataframe for the current iteration 
            # after every setting that is run in case of interruption
            df = opt.resultsDataFrame
            df.to_csv(NCDpath + subFolder + '/' + subFolder + '_' + str(iteration) + '.csv')

    # Clear optimiser settings for next iteration
    opt.resultsDataFrame = None
    opt.initialiseRandomSettings()
    currentDateTime = datetime.now()
