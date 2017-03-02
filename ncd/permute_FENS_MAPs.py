# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""

from __future__ import division

from datetime import datetime
from multiprocessing import Pool
from itertools import product
import pandas as pd
import copy

from ncd_processing import convertNCDs
from getResults2 import getDataFrameMAPresult
from other import divide_timedelta
from MAPhelpers import cleanCRPfolder, cleanNCDfolder, moveResultsFiles
from MAPhelpers import loadFeatureFileDict
from calculate_NCDs import calculateNCDs
from NNs import get_NN_WeightsAndBiases, get_NN_featureOffsetAndScaling
from FeatureFileProps import FeatureFileProps as FFP


# This script runs all the given permutations of settings and finds the MAP for
# each one - useful for comparing the effect of changing one or two settings, 
# but not practical for large numbers of settings due to exponential growth in 
# the number of permutations


################################### Settings ##################################

numProcesses = 10
subFolder = 'report14'

# Single settings
# ===============
# training set size
# -----------------
numFolders = 20
numFilesPerFolder = 5 # Set to None to compare all files in each folder

# Feature Settings
# ================
featureNames = ['logfreqspec']
numsFeatures = [128]
downSampleFactors = [4]

# Neural network (set NNtypes to [None] if not using a learned feature)
# ---------------------------------------------------------------------
NNtypes = ['dA']
batchSizes = [40]
learningRates = [0.05]
learningRateBoostFactors = [1.5]
numsHiddens = [12]
corruptionLevels = [10]
timeStackings = [1]
frequencyStandardisations = [False]

# FENS settings
# =============
FENSnormalisationThresholds = [0.001]
FENStransformationFunctions = ['cube']
FENSquantisationStepBases = [0.05]
FENSquantisationStepPowers = [2.0]
FENSquantisationWeight1s = [0.5, 0.707, 1.0, 1.414, 2.0]
FENSquantisationWeight2s = [0.5, 0.707, 1.0, 1.414, 2.0]
FENSquantisationWeight3s = [0.5, 0.707, 1.0, 1.414, 2.0]
FENSquantisationWeight4s = [0.5, 0.707, 1.0, 1.414, 2.0]
FENSdownsamplings = [2]
FENSwindowLengths = [10]

# CRP settings
# ============
CRPmethods = ['fan']
CRPdimensions = [5]
CRPneighbourhoodSizes = [0.05]
CRPtimeDelays = [3]

# NCD settings
# ============
NCDsequenceLengths = [500]


###############################################################################

cleanCRPfolder()
cleanNCDfolder()
processPool = Pool(numProcesses)
startDateTime = datetime.now()

allSettings = product(featureNames, downSampleFactors, 
                      NNtypes, batchSizes, learningRates, learningRateBoostFactors, 
                      numsHiddens, corruptionLevels, timeStackings,
                      CRPmethods, CRPdimensions, CRPneighbourhoodSizes, CRPtimeDelays,
                      NCDsequenceLengths, numsFeatures, frequencyStandardisations,
                      FENSnormalisationThresholds, FENStransformationFunctions, 
                      FENSquantisationStepBases, FENSquantisationStepPowers,
                      FENSquantisationWeight1s, FENSquantisationWeight2s, 
                      FENSquantisationWeight3s, FENSquantisationWeight4s,
                      FENSdownsamplings, FENSwindowLengths)

totalIterations = len(list(allSettings))
print '%i total iterations to run\n' % totalIterations

# reassign as creating the list messes up the product
allSettings = product(featureNames, downSampleFactors, 
                      NNtypes, batchSizes, learningRates, learningRateBoostFactors, 
                      numsHiddens, corruptionLevels, timeStackings,
                      CRPmethods, CRPdimensions, CRPneighbourhoodSizes, CRPtimeDelays,
                      NCDsequenceLengths, numsFeatures, frequencyStandardisations,
                      FENSnormalisationThresholds, FENStransformationFunctions, 
                      FENSquantisationStepBases, FENSquantisationStepPowers,
                      FENSquantisationWeight1s, FENSquantisationWeight2s, 
                      FENSquantisationWeight3s, FENSquantisationWeight4s,
                      FENSdownsamplings, FENSwindowLengths)

featureFileDict = None


# Load base features
if len(featureNames) == 1:
    featureFileDict, pieceIds = loadFeatureFileDict(featureNames[0], numFolders, numFilesPerFolder)

cleanCRPfolder()
cleanNCDfolder()

iterationsComplete = 0
allResults = []
currentDateTime = datetime.now()

for settings in allSettings:
    
    settingsDict = {'Feature Name': settings[0], 
                    'Feature DownSample Factor': settings[1],
                    'NN Type': settings[2], 
                    'NN Batch Size': settings[3], 
                    'NN Learning Rate': settings[4], 
                    'NN Learning Rate Boost Factor': settings[5], 
                    'NN # Hidden Units': settings[6],
                    'NN Corruption Level': settings[7], 
                    'NN Time Stacking': settings[8],
                    'CRP Method': settings[9], 
                    'CRP Dimension': settings[10], 
                    'CRP Neighbourhood Size': settings[11],
                    'CRP Time Delay': settings[12],
                    'NCD Sequence Length': settings[13],
                    'Number of Features': settings[14],
                    'Frequency Standardisation': settings[15],
                    'FENS Normalisation Threshold': settings[16],
                    'FENS Transformation Function': settings[17],
                    'FENS Quantisation Step Base': settings[18],
                    'FENS Quantisation Step Power': settings[19],
                    'FENS Quantisation Weight 1': settings[20],
                    'FENS Quantisation Weight 2': settings[21],
                    'FENS Quantisation Weight 3': settings[22],
                    'FENS Quantisation Weight 4': settings[23],
                    'FENS Downsampling': settings[24],
                    'FENS Window Length': settings[25]}

    # Load feature file dict if required
    if len(featureNames) != 1:
        featureFileDict, pieceIds = loadFeatureFileDict(featureNames[0], numFolders, numFilesPerFolder)

    # load weights and biases if this is for a neural net run
    weightMatrix = None
    biases = None
    featureOffset = None
    featureScaling = None
    if settingsDict['NN Type'] is not None:
        weightMatrix, biases = get_NN_WeightsAndBiases(settingsDict['NN Type'], settingsDict['Feature Name'], 
                                                       numFolders, numFilesPerFolder,
                                                       settingsDict['NN Batch Size'],
                                                       settingsDict['NN Learning Rate'], settingsDict['NN Learning Rate Boost Factor'], 
                                                       settingsDict['NN Corruption Level'], settingsDict['NN Time Stacking'], 
                                                       settingsDict['Frequency Standardisation'],
                                                       settingsDict['Number of Features'], settingsDict['NN # Hidden Units'])
        featureOffset, featureScaling = get_NN_featureOffsetAndScaling(settingsDict['Feature Name'], numFolders, numFilesPerFolder, 
                                                                       settingsDict['NN Time Stacking'],
                                                                       settingsDict['Number of Features'] * settingsDict['NN Time Stacking'], 
                                                                       settingsDict['Frequency Standardisation'])
    print 'Generating FENS features...'
    FENSfeatureFileDict = FFP.generateFENSfeatureFileDict(
                                    featureFileDict, 
                                    settingsDict['Number of Features'], 
                                    settingsDict['Feature Name'],
                                    settingsDict['NN Type'], 
                                    settingsDict['Number of Features'], 
                                    settingsDict['NN # Hidden Units'], 
                                    weightMatrix, biases, featureOffset, featureScaling, 
                                    settingsDict['NN Time Stacking'],
                                    settingsDict['FENS Downsampling'], 
                                    settingsDict['FENS Window Length'],
                                    FENStransformationFunction = settingsDict['FENS Transformation Function'],
                                    FENSnormalisationThreshold = settingsDict['FENS Normalisation Threshold'], 
                                    FENSquantisationSteps = FFP.getFENSQuantisationSteps(settingsDict['FENS Quantisation Step Base'], 
                                                                                         settingsDict['FENS Quantisation Step Power']),
                                    FENSquantisationWeights = [settingsDict['FENS Quantisation Weight 1'],
                                                               settingsDict['FENS Quantisation Weight 2'],
                                                               settingsDict['FENS Quantisation Weight 3'],
                                                               settingsDict['FENS Quantisation Weight 4']],
                                    processPool = processPool)

    # Calculate NCDs
    for key in settingsDict.keys():
        print key, ':', settingsDict[key]
    featureName = '%s-%s-%iv-%i' % (settingsDict['Feature Name'], 
                                    settingsDict['NN Type'],
                                    settingsDict['Number of Features'], 
                                    settingsDict['NN # Hidden Units'])
                                    
    NCDlist = calculateNCDs(processPool = processPool,
                            featureName = settingsDict['Feature Name'], 
                            numFeatures = settingsDict['NN # Hidden Units'], 
                            downSampleFactor = settingsDict['Feature DownSample Factor'], 
                            timeDelay = settingsDict['CRP Time Delay'], 
                            dimension = settingsDict['CRP Dimension'], 
                            method = settingsDict['CRP Method'], 
                            neighbourhoodSize = settingsDict['CRP Neighbourhood Size'], 
                            numFolders = numFolders, 
                            numFilesPerFolder = numFilesPerFolder, 
                            sequenceLength = settingsDict['NCD Sequence Length'], 
                            featureFileDict = FENSfeatureFileDict, 
                            pieceIds = pieceIds)

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
    settingsDict['Mean Average Precision'] = MAPresult
    
    # Create subfolders and move results files into them
    NCDdest = moveResultsFiles(subFolder)

    # save settings to run history path
    allResults.append(copy.deepcopy(settingsDict))
    dfAllResults = pd.DataFrame(allResults)
    dfAllResults.to_csv(NCDdest + subFolder + '.csv')
    
    cleanCRPfolder()
    cleanNCDfolder()
        
    # Increment the number of runs
    iterationsComplete += 1
    fractionComplete = iterationsComplete / totalIterations
    dtNow = datetime.now()
    estimatedFinishTime = startDateTime + divide_timedelta(dtNow - startDateTime, fractionComplete)
    print '\n%.1f percent complete, estimated finish time = %s\n' % (100 * fractionComplete, str(estimatedFinishTime))

    