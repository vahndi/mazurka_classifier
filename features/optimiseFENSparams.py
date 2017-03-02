# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 19:30:12 2015

@author: vahndi
"""

from datetime import datetime
import pandas as pd
from scipy.stats import norm
from numpy import linspace
import numpy as np
import pickle
import multiprocessing as mp

from adjusters import ValueAdjuster
from paths import getFolderNames, getFileNames, FENSparamsPath
from FeatureFileProps import FeatureFileProps as FFP
from NNs import get_NN_NCD_params
from featureAnalysis import loadFeatureFile, loadAndTransformFeatureFile
from featureConverter import featuresToFENS



################################### Settings ##################################

numProcesses = 8

featureName = 'logfreqspec'
numFeatures = 12

# network and weights
NNtype = 'dA'
numOriginalFeatures = 128
numNewFeatures = 12
batchSize = 40
corruptionLevel = 0
learningRate = 0.05
learningRateBoostFactor = 1.5
timeStacking = 1
frequencyStandardisation = False

numFolders = 20 # set to None to use all folders
numFilesPerFolder = 5 # set to None to use all files

# loops
#windowLengths = [10, 20, 30, 41, 50, 60, 70]
#FENSdownsamplings = [1, 2, 4, 8, 12]

targetDistribution = 'normal'

# Quantisation Parameters
maxQadjustment = 0.01
q1Args = (0.2, 0.01, 0.95, maxQadjustment)
q2Args  = (0.4, 0.01, 0.95, maxQadjustment)
q3Args  = (0.6, 0.01, 0.95, maxQadjustment)
q4Args  = (0.8, 0.01, 0.95, maxQadjustment)

# Quantisation Weights
maxWadjustment = 0.01
w1Args = (1.0, 0.5, 2, maxWadjustment)
w2Args = (1.0, 0.5, 2, maxWadjustment)
w3Args = (1.0, 0.5, 2, maxWadjustment)
w4Args = (1.0, 0.5, 2, maxWadjustment)

# Normalisation Threshold
maxNadjustment = 0.001
nArgs = (0.001, 0.0, 0.01, maxNadjustment)


# Stop Criteria
stopRunningAt = datetime(2015, 9, 16, 23)
maxUnsuccessfulChanges = 50

###############################################################################


def getTargetHistogram(targetDistribution):
    
    assert targetDistribution in ['normal', 'uniform']
    
    if targetDistribution == 'normal':
        x = linspace(0, 1,101)
        pdf_fitted = norm.pdf(x, loc = 0.5, scale = 0.2)
        normalHistogram = 0.1 + 0.9 * (pdf_fitted - pdf_fitted.min()) / (pdf_fitted.max() - pdf_fitted.min())
        return normalHistogram
    
    elif targetDistribution == 'uniform':
    
        x = np.ones(101)
        return x


def multiFeaturesToFENS(args):
    
    features = args[0]
    numFeatures = args[1]
    quantisationSteps = args[2]
    quantisationWeights = args[3]
    normalisationThreshold = args[4]
    
    return featuresToFENS(features, numFeatures, 
                          quantisationSteps = quantisationSteps, 
                          quantisationWeights = quantisationWeights,
                          normalisationThreshold = normalisationThreshold)


pool = mp.Pool(numProcesses)

# TODO: Create function adjuster for common transformations of the feature before
# converting to FENS

# Get the folders (performances)
piecesPath = FFP.getRootPath(featureName)
piecesFolders = getFolderNames(piecesPath, 
                               contains = 'mazurka',
                               orderAlphabetically = True) # added the contains parameter to avoid the new powerspectrum folder

if numFolders is not None:
    piecesFolders = piecesFolders[: numFolders]

# Load weights and biases       
if NNtype is not None:
    weightMatrix, biases, featureOffset, featureScaling = get_NN_NCD_params(
                                                            NNtype, featureName, learningRate, learningRateBoostFactor,
                                                            corruptionLevel, numOriginalFeatures, numNewFeatures, batchSize, 
                                                            freqStd = frequencyStandardisation, NNnumFolders = numFolders, 
                                                            NNnumFilesPerFolder = numFilesPerFolder,
                                                            NNtimeStacking = timeStacking)
# Load (and optionally transform) the feature files
p = 0
featuresDataFrames = []
for piecesFolder in piecesFolders:
    performancesPath = FFP.getFeatureFolderPath(piecesPath + piecesFolder + '/', featureName)
    performances = getFileNames(performancesPath, 
                                orderAlphabetically = True, 
                                endsWith = '.csv')
    if numFilesPerFolder is not None:
        performances = performances[: numFilesPerFolder]
    for performance in performances:
        p+= 1
        print '\rloading feature file %i...' % p,

        performanceFilePath = performancesPath + performance
        if NNtype is None:
            featuresDataFrames.append(loadFeatureFile(performanceFilePath))
        else:
            featuresDataFrames.append(loadAndTransformFeatureFile(performanceFilePath, 
                                                                  featureOffset, featureScaling,
                                                                  timeStacking, weightMatrix, biases))
    print

# Create Target histogram
targetHistogram = getTargetHistogram(targetDistribution)

# Initialise
numRuns = 0
iteration = 0
currentDateTime = datetime.now()

settingsList = []

while currentDateTime < stopRunningAt:
    
    # Initialise Iteration
    iteration += 1
    bestMatchResult = 1e9
    # initialise adjuster values
    q1Adjuster = ValueAdjuster(*q1Args)
    q2Adjuster = ValueAdjuster(*q2Args)
    q3Adjuster = ValueAdjuster(*q3Args)
    q4Adjuster = ValueAdjuster(*q4Args)
    w1Adjuster = ValueAdjuster(*w1Args)
    w2Adjuster = ValueAdjuster(*w2Args)
    w3Adjuster = ValueAdjuster(*w3Args)
    w4Adjuster = ValueAdjuster(*w4Args)
    nAdjuster = ValueAdjuster(*nArgs)
    # get current values from adjusters
    q1Value = q1Adjuster.getCurrentValue()
    q2Value = q2Adjuster.getCurrentValue()
    q3Value = q3Adjuster.getCurrentValue()
    q4Value = q4Adjuster.getCurrentValue()
    w1Value = w1Adjuster.getCurrentValue()
    w2Value = w2Adjuster.getCurrentValue()
    w3Value = w3Adjuster.getCurrentValue()
    w4Value = w4Adjuster.getCurrentValue()
    nValue = nAdjuster.getCurrentValue()
    
    # Loop while maxUnsuccessfulChanges is not exceeded
    while (q1Adjuster.getNumUnsuccessulAdjustments() < maxUnsuccessfulChanges 
           and currentDateTime < stopRunningAt):

        # Create FENS features (in memory - about 38MB for 100 files)
        FENSdataFrames = []
        f = 0
        quantisationSteps = sorted([q1Value, q2Value, q3Value, q4Value], reverse = True)
        print 'Quantisation Steps: [%0.3f, %0.3f, %0.3f, %0.3f]' % (quantisationSteps[0],
                                                                    quantisationSteps[1], 
                                                                    quantisationSteps[2], 
                                                                    quantisationSteps[3])
        quantisationWeights = [w1Value, w2Value, w3Value, w4Value]
        print 'Quantisation Weights: [%0.3f, %0.3f, %0.3f, %0.3f]' % (w1Value, w2Value, w3Value, w4Value)                                           
        print 'Normalisation Threshold: %0.4f' % nValue
        FENSdataFrames = pool.map(multiFeaturesToFENS, [(feature, numFeatures, quantisationSteps, quantisationWeights, nValue)
                                                                                            for feature in featuresDataFrames])

        # Join dataframes        
        print 'Joining features...'
        dfAllFENSfeatures = pd.concat(FENSdataFrames, ignore_index = True)
        # Rescale from 0 to 1
        dfAllFENSfeatures = (dfAllFENSfeatures - dfAllFENSfeatures.min()) / (dfAllFENSfeatures.max() - dfAllFENSfeatures.min())
        
        matchResult = 0
        for col in dfAllFENSfeatures.columns:
            # Take histogram of FENS features
            colHistogram = np.histogram(dfAllFENSfeatures[col], bins = 101)[0]
            # Mutltiply FENS histogram by Target histogram
            # Get result as a fraction of the squared Target distribution
            matchResult += np.square(colHistogram - targetHistogram).mean() / (101 * numFeatures)
        print 'result = %0.3f' % matchResult

        # If result is best so far, keep the current adjuster settings
        if matchResult < bestMatchResult:
            print 'Match improved by %0.3f' % (matchResult - bestMatchResult)
            q1Adjuster.keepAdjustedValue()
            q2Adjuster.keepAdjustedValue()
            q3Adjuster.keepAdjustedValue()
            q4Adjuster.keepAdjustedValue()
            w1Adjuster.keepAdjustedValue()
            w2Adjuster.keepAdjustedValue()
            w3Adjuster.keepAdjustedValue()
            w4Adjuster.keepAdjustedValue()
            nAdjuster.keepAdjustedValue()
            bestMatchResult = matchResult
            bestQuantisationSteps = quantisationSteps
            bestQuantisationWeights = quantisationWeights
            bestNormalisationThreshold = nValue
            FENSparamFileName = FENSparamsPath + featureName + '_nf_%i' % numFeatures
            if NNtype is not None:
                FENSparamFileName += '_NNtype_%s_nvis_%i_nhid_%i_basz_%i_crpt_%i_lr_%0.2f_lrbf_%0.2f_ts_%i_fstd_%s' \
            % (NNtype, numOriginalFeatures, numNewFeatures, batchSize, corruptionLevel, 
               learningRate, learningRateBoostFactor, timeStacking, frequencyStandardisation)
            FENSparamFileName += '_dist_%s_iter_%i' % (targetDistribution, iteration)
            resultsDict = {}
            resultsDict['Best Quantisation Steps'] = bestQuantisationSteps
            resultsDict['Best Quantisation Weights'] = bestQuantisationWeights
            resultsDict['Best Normalisation Threshold'] = bestNormalisationThreshold
            resultsDict['Histogram'] = np.histogram(dfAllFENSfeatures.as_matrix(), bins = 101)
            resultsDict['Mean Squared Error'] = bestMatchResult
            pickle.dump(resultsDict, open(FENSparamFileName, 'wb')) # TODO: fill in filename, including iteration
        else:
            print 'Match did not improve'
            q1Adjuster.keepCurrentValue()
            q2Adjuster.keepCurrentValue()
            q3Adjuster.keepCurrentValue()
            q4Adjuster.keepCurrentValue()
            w1Adjuster.keepCurrentValue()
            w2Adjuster.keepCurrentValue()
            w3Adjuster.keepCurrentValue()
            w4Adjuster.keepCurrentValue()
            nAdjuster.keepCurrentValue()
            
        # Update the weights and biases matrices
        q1Value = q1Adjuster.getAdjustedValue()
        q2Value = q2Adjuster.getAdjustedValue()
        q3Value = q3Adjuster.getAdjustedValue()
        q4Value = q4Adjuster.getAdjustedValue()
        w1Value = w1Adjuster.getAdjustedValue()
        w2Value = w2Adjuster.getAdjustedValue()
        w3Value = w3Adjuster.getAdjustedValue()
        w4Value = w4Adjuster.getAdjustedValue()
        nValue = nAdjuster.getAdjustedValue()
        
        # Increment the number of runs
        numRuns += 1
        print '\nNumber of runs: %i\nIteration: %i\nBest Result: %0.3f\n\n' \
                                          %(numRuns, iteration, bestMatchResult)
        
        # Prepare for next iteration
        currentDateTime = datetime.now()