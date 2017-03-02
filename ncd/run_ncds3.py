# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""

# This version works with create_ncds4.py i.e. after support for equal size CRPs was introduced

import pickle
import os, shutil
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool

from paths import runHistoryPath, NCDpath, getFileNames
from optimiser import Optimiser
from create_ncds4 import createNCDfiles
from ncd_processing import convertNCDfiles, deleteNCDPickleFiles
from getResults2 import getMAPresult


### Settings ###

numProcesses = 8
featureName = 'CENS_dsf2_wl90'
CRPmethod = 'rr'

CRPdimensions = [1, 2, 3, 4, 5]
CRPneighbourhoodSizes = list(np.arange(0.05, 1, 0.1))
CRPtimeDelays = [1, 2, 3, 4, 5, 6, 7]
downSampleFactors = [1, 2, 4, 8, 12]
sequenceLengths = [300, 500, 700, 900, 1100]
numFilesPerFolder = 5 # Set to None to compare all files in each folder
subFolder = 'run19'

#################


settingsDict = {'Dimension': CRPdimensions,
                'DownSample Factor': downSampleFactors,
                'Neighbourhood Size': CRPneighbourhoodSizes,
                'Sequence Length': sequenceLengths,
                'Time Delay': CRPtimeDelays}
                
numRuns = 0
deleteNCDPickleFiles()
existingNCDs = None
processPool = Pool(numProcesses)


iteration = 0
stopRunningAt = datetime(2015, 8, 9, 12)

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
                createNCDfiles(existingNCDs, processPool,
                               featureName, setting['DownSample Factor'],
                               setting['Time Delay'], setting['Dimension'], CRPmethod, setting['Neighbourhood Size'], 
                               numFilesPerFolder, setting['Sequence Length'])
                runDict = {'featureName': featureName,
                           'downSampleFactor': setting['DownSample Factor'],
                           'timeDelay': setting['Time Delay'], 
                           'dimension': setting['Dimension'],
                           'method': CRPmethod, 
                           'neighbourhoodSize': setting['Neighbourhood Size'],
                           'numFilesPerFolder': numFilesPerFolder,
                           'sequenceLength': setting['Sequence Length']}
                runTime = str(datetime.now()).replace(':', '-')
                pickle.dump(runDict, open(runHistoryPath + runTime + '.pkl', 'wb'))
                # Convert NCD files into a dataframe
                convertNCDfiles(runTime)
                # Create subfolders and move results files into them
                NCDdest = NCDpath + subFolder + '/'
                runHistDest = runHistoryPath + subFolder + '/'
                if not os.path.exists(NCDdest):
                    os.makedirs(NCDdest)
                if not os.path.exists(runHistDest):
                    os.makedirs(runHistDest)
                for fn in getFileNames(NCDpath, '.pkl.res'):
                    shutil.move(NCDpath + fn, NCDdest)
                for fn in getFileNames(runHistoryPath, '.pkl'):
                    shutil.move(runHistoryPath + fn, runHistDest)
                # Get the overall MAP of the run and add to the setting
                MAPresult = getMAPresult(featureName, CRPmethod, setting['Dimension'],
                                         setting['Neighbourhood Size'], setting['Time Delay'], 
                                         setting['DownSample Factor'], numFilesPerFolder,
                                         setting['Sequence Length'], subFolder)
                if MAPresult is not None:
                    print 'Mean Average Precision: %0.3f\n' % MAPresult
                else:
                    print 'No MAP result found!'
                setting['Mean Average Precision'] = MAPresult

         
            # Increment the number of runs
            numRuns += len(nextSettings)
            
            # Add the results to the optimiser
            opt.addResults(pd.DataFrame(nextSettings))
            print '\nNumber of runs: %i\nBest Result: %0.3f\n\n' %(numRuns, opt.currentBestResult())
            
            df = opt.resultsDataFrame
            df.to_csv(NCDdest + subFolder + '_' + str(iteration) + '.csv')
            
    opt.resultsDataFrame = None
    opt.initialiseRandomSettings()
    currentDateTime = datetime.now()
