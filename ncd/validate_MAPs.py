# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""
import os
import pickle
from datetime import datetime
from multiprocessing import Pool

from paths import NCDpath, createPath, loadRunSettingsList
from create_ncds_val_test import createNCDfiles
from ncd_processing import convertNCDfiles
from getResults2 import getDataFrameMAPresult
from NNs import get_NN_NCD_params
from MAPhelpers import cleanCRPfolder, cleanNCDfolder


# Test some settings on the validation set, i.e. all the remaining files in the
# training folder. The results should be slightly better than would get on a test
# set since the pieces are the same and only the performances differ.

numProcesses = 10
settingsFileName = '/u7.swansea/s11/abpg162/project/run_settings/Run Settings Template.csv'
settingsDicts = loadRunSettingsList(settingsFileName)

# Create folder for results
createPath(NCDpath + 'validation runs')

# Initialise
processPool = Pool(numProcesses)

for settingsDict in settingsDicts:
    
    resultsFn = '/u7.swansea/s11/abpg162/project/results_files/validation/' + settingsDict['Run Name'] + '.pkl'
    if not os.path.exists(resultsFn):
        
        cleanCRPfolder()
        cleanNCDfolder()
        startDateTime = datetime.now()
    
        # load weights etc. if this is for a neural net run
        if settingsDict['NN Type'] is not None:
            weightMatrix, biases, featureOffset, featureScaling = get_NN_NCD_params(
                NNtype = settingsDict['NN Type'], 
                featureName = settingsDict['Feature Name'], 
                learningRate = settingsDict['dA Learning Rate'], 
                learningRateBoostFactor = settingsDict['dA Learning Rate Boost Factor'], 
                corruptionLevel = settingsDict['dA Corruption Level'], 
                numVisible = int(settingsDict['dA Num Visible Units']), 
                numHidden = int(settingsDict['dA Num Hidden Units']), 
                batchSize = int(settingsDict['dA Batch Size']),
                freqStd = bool(settingsDict['NN Frequency Standardisation']), 
                NNnumFolders = int(settingsDict['NN Num Folders']), 
                NNnumFilesPerFolder = int(settingsDict['NN Num Files per Folder']), 
                NNtimeStacking = int(settingsDict['NN Time Stacking']))
        else:
            weightMatrix = biases = featureOffset = featureScaling = None
    
        # Create NCD files
        for key in settingsDict.keys():
            print key, ':', settingsDict[key]
        createNCDfiles(existingNCDs = None, 
                       processPool = processPool,
                       featureName = settingsDict['Feature Name'], 
                       numFeatures = int(settingsDict['Number of Features']), 
                       downSampleFactor = int(settingsDict['Feature DownSample Factor']), 
                       timeDelay = int(settingsDict['CRP Time Delay']), 
                       dimension = int(settingsDict['CRP Dimension']), 
                       method = settingsDict['CRP Method'], 
                       neighbourhoodSize = settingsDict['CRP Neighbourhood Size'], 
                       runType = 'validation',
                       sequenceLength = int(settingsDict['NCD Sequence Length']), 
                       weightMatrix = weightMatrix, 
                       biases = biases, 
                       featureOffset = featureOffset, 
                       featureScaling = featureScaling, 
                       timeStacking = int(settingsDict['NN Time Stacking']))
                    
        # Convert NCD files into a dataframe
        dfNCDs = convertNCDfiles(dataFrameFileName = settingsDict['Run Name'])
        
        # Get the overall MAP of the run and add to the setting
        MAPresult = getDataFrameMAPresult(dfNCDs)
        if MAPresult is not None:
            print 'Mean Average Precision: %0.3f\n' % MAPresult
        else:
            print 'No MAP result found!'
        settingsDict['Mean Average Precision'] = MAPresult
        settingsDict['Run Duration'] = datetime.now() - startDateTime
        
        # Save run settings and result
        pickle.dump(settingsDict, open(resultsFn, 'wb'))

