# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""

# This version works with create_ncds5.py i.e. after support for feature transformations
# by neural net weights and works specifically with denoising AutoEncoders

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool

from paths import runHistoryPath, NCDpath, NNweightsPath, createPath, moveFiles
from optimiser import Optimiser
from create_ncds5 import createNCDfiles
from ncd_processing import convertNCDfiles, deleteNCDPickleFiles
from getResults2 import getDataFrameMAPresult



################################### Settings ##################################

numProcesses = 8
subFolder = 'run23'

# Single settings
featureName = 'CENS_dsf2_wl41'
CRPmethod = 'rr'
NNtype = 'dA'
numFilesPerFolder = 5 # Set to None to compare all files in each folder

# Feature Settings
downSampleFactors = [1, 2, 4, 8, 12]
numHiddens = [3, 6, 9, 12]
batchSizes = [10, 20, 30, 40]
learningRates = [0.01, 0.03, 0.1]
corruptionLevels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# CRP settings
CRPdimensions = [1, 2, 3, 4, 5]
CRPneighbourhoodSizes = list(np.arange(0.05, 1, 0.1))
CRPtimeDelays = [1, 2, 3, 4, 5, 6, 7]

# NCD settings
sequenceLengths = [300, 500, 700, 900, 1100]


###############################################################################



weightsPath = NNweightsPath + NNtype + '/' + featureName + '/'

settingsDict = {'Dimension': CRPdimensions,
                'DownSample Factor': downSampleFactors,
                'Neighbourhood Size': CRPneighbourhoodSizes,
                'Sequence Length': sequenceLengths,
                'Time Delay': CRPtimeDelays,
                'dA Learning Rate': learningRates,
                'dA Batch Size': batchSizes,
                'dA Num Hidden Units': numHiddens,
                'dA Corruption Level': corruptionLevels}
                
numRuns = 0
deleteNCDPickleFiles()
existingNCDs = None
processPool = Pool(numProcesses)

iteration = 0
stopRunningAt = datetime(2015, 8, 13, 23)

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
                if NNtype is not None:
                    strLnRt = format(setting['dA Learning Rate'], '.2f').rstrip('0').rstrip('.')
                    weightsFn = 'weights_corruption_%i_epoch_14_nin_12_nhdn_%i_basz_%i_lnrt_%s.csv' \
                                % (setting['dA Corruption Level'], setting['dA Num Hidden Units'], 
                                   setting['dA Batch Size'], strLnRt)
                    weightMatrix = np.matrix(np.loadtxt(weightsPath + weightsFn, delimiter = ',')).transpose()
                # create NCD files
                for key in setting.keys():
                    print key, ':', setting[key]
                createNCDfiles(existingNCDs, processPool,
                               featureName, setting['DownSample Factor'],
                               setting['Time Delay'], setting['Dimension'], CRPmethod, setting['Neighbourhood Size'], 
                               numFilesPerFolder, setting['Sequence Length'], weightMatrix)
                # create and save a record of the run settings
                runDict = {'featureName': featureName,
                           'method': CRPmethod, 
                           'numFilesPerFolder': numFilesPerFolder,
                           'networkType': NNtype}
                for key in setting.keys():
                    runDict[key] = setting[key]
                runTime = str(datetime.now()).replace(':', '-')
                pickle.dump(runDict, open(runHistoryPath + runTime + '.pkl', 'wb'))
                
                # Convert NCD files into a dataframe
                dfNCDs = convertNCDfiles(dataFrameFileName = runTime)

                # Create subfolders and move results files into them
                NCDdest = NCDpath + subFolder + '/'
                runHistDest = runHistoryPath + subFolder + '/'
                createPath(NCDdest)
                createPath(runHistDest)
                moveFiles(NCDpath, NCDdest, '.pkl.res')
                moveFiles(runHistoryPath, runHistDest, '.pkl')
                
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
