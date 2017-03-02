# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""
import os
import pickle
from datetime import datetime
from multiprocessing import Pool

from paths import loadRunSettingsList, getTestSetFolders, getFolderNames
from calculate_NCDs import calculateNCDs
from getResults2 import getDataFrameMAPresult
from NNs import get_NN_NCD_params
from MAPhelpers import cleanCRPfolder, cleanNCDfolder
from FeatureFileProps import FeatureFileProps as FFP
from ncd_processing import convertNCDs



# Test some settings on the test set, using main memory as much as possible

numProcesses = 10
settingsFileName = '/u7.swansea/s11/abpg162/project/run_settings/Run Settings - Test.csv'
settingsDicts = loadRunSettingsList(settingsFileName)

# Initialise
processPool = Pool(numProcesses)

for settingsDict in settingsDicts:
    
    resultsFn = '/u7.swansea/s11/abpg162/project/results_files/test/' + settingsDict['Run Name'] + '.pkl'
    if not os.path.exists(resultsFn) and settingsDict['Run Name'] is not None:
        
        cleanCRPfolder()
        cleanNCDfolder()
        startDateTime = datetime.now()
        
        # Load base features
        baseFeatureName = settingsDict['Feature Name']
        piecesPath = FFP.getRootPath(baseFeatureName)
        pieceIds = getTestSetFolders(getFolderNames(piecesPath, contains = 'mazurka', orderAlphabetically = True))
        print 'Loading feature file dict...'
        featureFileDict = FFP.loadFeatureFileDictTestSetFolders(piecesPath, settingsDict['Feature Name'])
        print '...done.'

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
                
            if settingsDict['FENS Normalisation Threshold'] is not None:
                
                print 'Generating FENS features...'
                featureFileDict = FFP.generateFENSfeatureFileDict(
                                          featureFileDict, 
                                          int(settingsDict['Number of Features']), 
                                          settingsDict['Feature Name'],
                                          settingsDict['NN Type'], 
                                          int(settingsDict['dA Num Visible Units']), 
                                          int(settingsDict['dA Num Hidden Units']), 
                                          weightMatrix, biases, featureOffset, featureScaling, 
                                          int(settingsDict['NN Time Stacking']),
                                          settingsDict['FENS Downsampling'], 
                                          int(settingsDict['FENS Window Length']),
                                          FENStransformationFunction = settingsDict['FENS Transformation Function'],
                                          FENSnormalisationThreshold = settingsDict['FENS Normalisation Threshold'], 
                                          FENSquantisationSteps = FFP.getFENSQuantisationSteps(settingsDict['FENS Quantisation Step Base'],
                                                                                               settingsDict['FENS Quantisation Step Power']),
                                          FENSquantisationWeights = [settingsDict['FENS Quantisation Weight 1'],
                                                                     settingsDict['FENS Quantisation Weight 2'],
                                                                     settingsDict['FENS Quantisation Weight 3'],
                                                                     settingsDict['FENS Quantisation Weight 4']],
                                          processPool = processPool)
                                          
            else:
                
                print 'Transforming features by weights...'
                featureFileDict = FFP.generateTransformedFeatureFileDict(
                                         featureFileDict,
                                         int(settingsDict['Number of Features']), 
                                         settingsDict['Feature Name'],
                                         settingsDict['NN Type'], 
                                         int(settingsDict['dA Num Visible Units']), 
                                         int(settingsDict['dA Num Hidden Units']), 
                                         weightMatrix, biases, featureOffset, featureScaling, 
                                         int(settingsDict['NN Time Stacking']),
                                         processPool = processPool)
                
        else:
            weightMatrix = biases = featureOffset = featureScaling = None

        # Calculate NCDs
        for key in settingsDict.keys():
            print key, ':', settingsDict[key]
        
        # Name new features
        featureName = baseFeatureName
        if settingsDict['NN Type'] is not None:
            featureName += '-%s-%iv-%ih' % (settingsDict['NN Type'], 
                                            settingsDict['dA Num Visible Units'], 
                                            settingsDict['dA Num Hidden Units'])
        if settingsDict['FENS Normalisation Threshold'] is not None:
            featureName += '-FENS_dsf%i_dswl%i' % (settingsDict['FENS Downsampling'], 
                                                   settingsDict['FENS Window Length'])
        # Assign a value to numFeatures
        if settingsDict['NN Type'] is not None:
            numFeatures = int(settingsDict['dA Num Hidden Units'])
        else:
            numFeatures = int(settingsDict['Number of Features'])
            
        NCDlist = calculateNCDs(processPool,
                                featureName, 
                                numFeatures, 
                                int(settingsDict['Feature DownSample Factor']), 
                                int(settingsDict['CRP Time Delay']), 
                                int(settingsDict['CRP Dimension']), 
                                settingsDict['CRP Method'], 
                                settingsDict['CRP Neighbourhood Size'], 
                                None, None, 
                                int(settingsDict['NCD Sequence Length']), 
                                featureFileDict = featureFileDict, 
                                pieceIds = pieceIds)
        
        # Convert NCD files into a dataframe
        runTime = str(datetime.now()).replace(':', '-')
        MAPresult = None
        if NCDlist is None: # there were errors e.g. in CRP calculation after downsampling
             MAPresult = 0 # need to use something that is not None for the optimiser to find its best result   
        else:
            dfNCDs = convertNCDs(NCDlist, dataFrameFileName = runTime)
             # Get the overall MAP of the run and add to the setting
            print 'Calculating MAP Result...'
            MAPresult = getDataFrameMAPresult(dfNCDs)
        if MAPresult is not None and MAPresult != 0:
            print 'Mean Average Precision: %0.3f\n' % MAPresult
        else:
            print 'No MAP result found!'
        settingsDict['Mean Average Precision'] = MAPresult
        settingsDict['Run Duration'] = datetime.now() - startDateTime
        
        # Save run settings and result
        pickle.dump(settingsDict, open(resultsFn, 'wb'))
