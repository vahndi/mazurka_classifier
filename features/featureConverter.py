from __future__ import division


import pandas as pd
import numpy as np
import copy
import cPickle, gzip
import pickle
from scipy import hanning
from math import ceil, floor
from multirate import upfirdn
from itertools import islice
from numpy.linalg import norm

from paths import getFileNames
from other import rcut



def smoothDownSampleFeature(dataframe, windowLength, downSampleFactor):
    '''
    Temporal smoothing and downsampling of a feature sequence.
    Adapted from the smoothDownsampleFeature.m file of the Matlab Chroma Toolbox
    at http://resources.mpi-inf.mpg.de/MIR/chromatoolbox/
    '''
    def downsample_to_proportion(rows, proportion = 1):
        return list(islice(rows, 0, len(rows), int(1 / proportion)))    
    
    if windowLength == 1 and downSampleFactor == 1:
        return dataframe
    
    statWindow = hanning(windowLength)
    statWindow = statWindow / statWindow.sum()
    statWindow = np.tile(statWindow, [1, 1])
   
    f_feature = dataframe.as_matrix()
    seg_num = f_feature.shape[0]
    stat_num = int(ceil(seg_num / downSampleFactor))
    f_feature_stat = upfirdn(f_feature, statWindow.transpose(), 1, downSampleFactor)
    cut = floor((windowLength - 1) / (2 * downSampleFactor))
    f_feature_stat = f_feature_stat[cut: stat_num + cut, :]
    
    timeIndex = downsample_to_proportion(dataframe.index, 1 / downSampleFactor)
    dfSmoothed = pd.DataFrame(f_feature_stat, index = timeIndex)
    
    return dfSmoothed


def normaliseFeature(dataframe, normalisationPower, threshold):
    '''
    Normalizes a feature sequence according to the l^p norm
    - If the norm falls below threshold for a feature vector, then the
      normalized feature vector is set to be the unit vector.
    Code adapted from the normalizeFeature.m file of the Matlab Chroma Toolbox
    at http://resources.mpi-inf.mpg.de/MIR/chromatoolbox/
    '''
    unit_vec = np.ones([1, dataframe.shape[1]]) # changed from the Matlab version of using 12 in case this is used for other feature types
    unit_vec = unit_vec / norm(unit_vec, normalisationPower)
    dfValues = dataframe.as_matrix()
    norms = norm(dfValues, ord = normalisationPower, axis = 1)
    smallIndices = np.where(norms < threshold)[0]
    bigIndices = np.where(norms >= threshold)[0]
    
    dfValues[smallIndices, :] = unit_vec
    dfValues[bigIndices, :] = dfValues[bigIndices, :] / np.expand_dims(norms[bigIndices], 1)

    df = pd.DataFrame(dfValues, index = dataframe.index)
    
    return df


def chromaToCENS(dfChroma, 
                 normalisationThreshold = 0.001, 
                 quantisationSteps = [0.4, 0.2, 0.1, 0.05], quantisationWeights = [1, 1, 1, 1],
                 downSampleWindowLength = 41, downSampleFactor = 10.0):
    '''
    Converts a chroma feature .csv file created by Sonic Annotator's Queen Mary Chroma
    plugin into a Chroma Energy Normalised Statistics File
    Code converted from the pitch_to_CENS.m file of the Matlab Chroma Toolbox 
    at http://resources.mpi-inf.mpg.de/MIR/chromatoolbox/
    '''
    
    # Normalise the chroma vectors (Matlab line 123)
    # ----------------------------------------------
    dfChroma['Chroma Sum'] = dfChroma.apply(lambda row: row.sum(), axis = 1)
    dfChroma['Exceeds Threshold'] = dfChroma.apply(lambda row: int(row['Chroma Sum'] > normalisationThreshold), axis = 1)
    dfChromaStandardised = dfChroma[[i for i in range(1, 13)]].multiply(dfChroma['Exceeds Threshold'], axis = 'index').divide(dfChroma['Chroma Sum'], axis = 'index')
    
    # Calculate CENS Feature
    # ----------------------
    
    # Component-wise quantisation of the normalised chroma vectors (Matlab line 134)
    ChromaArray = dfChromaStandardised.as_matrix()
    CENSarray = np.zeros(dfChromaStandardised.shape)
    for i in range(len(quantisationSteps)):
        CENSarray += quantisationWeights[i] * (ChromaArray > quantisationSteps[i])
    dfCENSraw = pd.DataFrame(CENSarray, index = dfChromaStandardised.index)
    
    # Temporal smoothing and downsampling (Matlab line 140)
    dfCENSsmoothed = smoothDownSampleFeature(dfCENSraw, downSampleWindowLength, downSampleFactor)

    # Normalise each vector with its l^2 norm (Matlab line 143)
    dfCENS = normaliseFeature(dfCENSsmoothed, 2, normalisationThreshold)
    
    # Return CENS dataframe
    # ---------------------
    return dfCENS


def featuresToFENS(dfFeatures, numFeatures,
                   transformationFunction = None,
                   normalisationThreshold = 0.001, 
                   quantisationSteps = [0.4, 0.2, 0.1, 0.05], 
                   quantisationWeights = [1, 1, 1, 1],
                   downSampleWindowLength = 41, downSampleFactor = 10.0):
    '''                     
    Mimics the creation of CENS features but using any dataframe of features of any dimension
    Inputs:
        :dfFeatures: a pandas dataframe of the features
        :other settings: see documentation for chromaToCENS
    '''
    # Sort quantisation steps descending in case they have been passed in a different order
    quantisationSteps = sorted(quantisationSteps, reverse = True)
    
    # Transform features if the function name is given
    if transformationFunction is not None:
        dfFeatures = applyFENSTransformationFunction(dfFeatures, transformationFunction)
        
    # Rescale the features to a range from 0 to 0.2
    # N.B. this is an additional step to the chromaToCENS script to reduce the
    # scale of the feature to be the same range as typical chroma features
    dfFeatures = 0.2 * (dfFeatures - dfFeatures.min()) / (dfFeatures.max() - dfFeatures.min())
    
    
    # Standardise the learned feature vectors (Matlab line 123)
    # -------------------------------------------------------
    dfFeatures['Features Sum'] = dfFeatures.apply(lambda row: row.sum(), axis = 1)
    dfFeatures['Exceeds Threshold'] = dfFeatures.apply(lambda row: int(row['Features Sum'] > 
                                                                       normalisationThreshold), 
                                                       axis = 1)
    dfFeaturesStandardised = dfFeatures[[i for i in range(1, numFeatures + 1)]].multiply(dfFeatures['Exceeds Threshold'], 
                                                                                         axis = 'index').divide(dfFeatures['Features Sum'], 
                                                                                                                axis = 'index')
    
    # Calculate FENS Feature
    # ----------------------
    # Component-wise quantisation of the normalised chroma vectors
    FeaturesArray = dfFeaturesStandardised.as_matrix()
    FENSarray = np.zeros(dfFeaturesStandardised.shape)
    for i in range(len(quantisationSteps)):
        FENSarray += quantisationWeights[i] * (FeaturesArray > quantisationSteps[i])
    dfFENSraw = pd.DataFrame(FENSarray, index = dfFeaturesStandardised.index)
    # Temporal smoothing and downsampling
    dfFENSsmoothed = smoothDownSampleFeature(dfFENSraw, downSampleWindowLength, downSampleFactor)
    # Normalise each vector with its l^2 norm
    dfFENS = normaliseFeature(dfFENSsmoothed, 2, normalisationThreshold)
    
    # Return FENS dataframe
    # ---------------------
    return dfFENS


def transformFeatures(featureArray, weightMatrix, biasMatrix, 
                      featureOffset, featureScaling, 
                      timeStacking = None):
    '''
    Transform some features by multiplyling them by a weightMatrix and adding a 
    biasMatrix, learned from a neural network.
    '''
    if type(featureArray) is pd.DataFrame:
        x_t = np.expand_dims(np.array(featureArray.index), 1)
        x_data = featureArray.as_matrix()
    else:
        # split the time and data columns
        x_t = np.expand_dims(featureArray[:,0], 1)
        x_data = featureArray[:, 1:]
    
    # horizontally stack the data array
    if timeStacking is not None:
        numExamples = x_data.shape[0]
        numFeatures= x_data.shape[1]
        x_dataNew = np.zeros([numExamples - timeStacking + 1, 
                              timeStacking * numFeatures])
        for ts in range(timeStacking):
            x_dataNew[:, ts * numFeatures : (ts + 1) * numFeatures] = x_data[ts: numExamples + 1 + ts - timeStacking, :]
        x_data = x_dataNew
        x_t = x_t[0: 1 + numExamples - timeStacking, :]

    # check that the dimensions of the matrix are right for the weights
    if x_data.shape[1] != weightMatrix.shape[0]:
        print 'Warning: feature and weight dimensions misaligned: cropping features from %i columns to %i columns' \
               % (x_data.shape[1], weightMatrix.shape[0])
        x_data = x_data[:, :weightMatrix.shape[0]]

    # standardise feature scaling
    x_data = (x_data + featureOffset) * featureScaling
    
    # multiply by weights and add biases
    xmat_transformed = np.matrix(x_data) * weightMatrix
    xarr_transformed = np.array(xmat_transformed) + np.tile(biasMatrix.transpose(), [x_data.shape[0], 1])
    
    # add back on the time component
    x_transformed = np.append(x_t, xarr_transformed, axis = 1)
    
    if type(featureArray) is pd.DataFrame:
        dfTransformed = pd.DataFrame(x_transformed)
        dfTransformed.set_index(0, inplace = True)
        return dfTransformed
    
    return x_transformed

    
def csvsToTheanoDataSet(inputPaths, outputFn, numFilesPerFolder,
                        trainPercentage = 70.0, validationPercentage = 15.0, testPercentage = 15.0):
    '''
    Convert a batch of .csv files created by SonicAnnotator to a .pkl.gz training and
    testing set for input into Theano
    
    Inputs:
        :inputPaths:  The input folders to get examples from
        :outputFn:  Path to the output file
        :numFilesPerFolder:     The number of files to use from each folder (set to None to use all files)
        :trainPercentage:   The percentage of examples to use for training
        :validationPercentage:  The percentage of examples to use for validation
        :testPercentage:    The percentage of examples to use for testing
    '''

    allFeatures = None
    pieceIndex = 0

    # For each folder (piece)
    for inputPath in inputPaths:

        print 'Converting features in folder %s' % inputPath

        # Get list of numFilesPerFolder feature files
        inputFiles = getFileNames(inputPath, endsWith = '.csv', orderAlphabetically = True)
        if numFilesPerFolder is not None:
            inputFiles = inputFiles[:numFilesPerFolder]

        # For each file (performance)
        for inputFn in inputFiles:
            print '\t%s' % inputFn
            # Read file
            fileFeatures = np.genfromtxt(inputPath + inputFn, delimiter = ',')
            # Drop first column (time)
            fileFeatures = fileFeatures[:, 1:]
            # Drop rows where all columns are zero
            fileFeatures = fileFeatures[~np.all(fileFeatures == 0, axis = 1)]
            # Add a label column to the end
            numFeatures = fileFeatures.shape[1]
            labelledFileFeatures = np.ones([fileFeatures.shape[0], numFeatures + 1]) * pieceIndex
            labelledFileFeatures[:, :-1] = fileFeatures
            # Add to allFeatures array
            if allFeatures is None:
                allFeatures = copy.deepcopy(labelledFileFeatures)
            else:
                allFeatures = np.vstack((allFeatures, copy.deepcopy(labelledFileFeatures)))

        pieceIndex += 1

    # Standardise feature range from 0 to 1
    print 'Standardising range...'
    minFeatureValue = np.min(allFeatures[:, 0: numFeatures])
    maxFeatureValue = np.max(allFeatures[:, 0: numFeatures])
    print 'minFeatureValue = %f, maxFeatureValue = %f' %(minFeatureValue, maxFeatureValue)
    allFeatures[:, 0: numFeatures] = (allFeatures[:, 0: numFeatures] - minFeatureValue) / (maxFeatureValue - minFeatureValue)

    # Shuffle Features 10 times
    print 'Shuffling...'
    for _ in np.arange(10):
        np.random.shuffle(allFeatures)

    # Extract training, validation and test sets
    numExamples = allFeatures.shape[0]
    numFeatures = allFeatures.shape[1] - 1

    trainingExamples = allFeatures[0: int(trainPercentage * numExamples / 100)]
    validationExamples = allFeatures[int(trainPercentage * numExamples / 100): 
                                     int((trainPercentage + validationPercentage) * numExamples / 100)]
    testExamples = allFeatures[int((trainPercentage + validationPercentage) * numExamples / 100): ]

    train_set = (trainingExamples[:, :numFeatures], trainingExamples[:, numFeatures])
    valid_set = (validationExamples[:, :numFeatures], validationExamples[:, numFeatures])
    test_set = (testExamples[:, :numFeatures], testExamples[:, numFeatures])

    # Write file for Theano
    print 'Writing Theano file...'
    outputFn = rcut(outputFn, '.pkl.gz') + '.pkl.gz'
    f = gzip.open(outputFn, 'wb')
    cPickle.dump((train_set, valid_set, test_set), f)
    f.close()


def csvsToTheanoDataSet2(inputPaths, outputFn, numFilesPerFolder, timeStepsPerFeature, 
                         cropFeaturesToSize = None, frequencyStandardisation = False,
                         trainPercentage = 70.0, validationPercentage = 15.0, testPercentage = 15.0):
    '''
    Convert a batch of .csv files created by SonicAnnotator to a .pkl.gz training and
    testing set for input into Theano, with sequential stacking of features to incorporate
    temporal effects
    
    Inputs:
        :inputPaths:  The input folders to get examples from
        :outputFn:  Path to the output file
        :numFilesPerFolder:     The number of files to use from each folder (set to None to use all files)
        :timeStepsPerFeature:   The number of time steps of original features to include in each new feature
        :cropFeaturesToSize:    Set to an integer if only some features should be used - features from the lower end of 
                                 the range will be used i.e. lower frequencies
        :frequencyStandardisation: whether to standardise the range of each frequency band individually
        :trainPercentage:   The percentage of examples to use for training
        :validationPercentage:  The percentage of examples to use for validation
        :testPercentage:    The percentage of examples to use for testing
        TODO: implement a standardisation function argument
    '''

    allFeatures = None
    pieceIndex = 0

    # For each folder (piece)
    for inputPath in inputPaths:

        print 'Converting features in folder %s' % inputPath
        # Get list of numFilesPerFolder feature files
        inputFiles = getFileNames(inputPath, endsWith = '.csv', orderAlphabetically = True)
        if numFilesPerFolder is not None:
            inputFiles = inputFiles[:numFilesPerFolder]

        # For each file (performance)
        for inputFn in inputFiles:
            print '\t%s' % inputFn
            # Read file
            fileFeatures = np.genfromtxt(inputPath + inputFn, delimiter = ',')
            # Drop first column (time)
            fileFeatures = fileFeatures[:, 1:]
            # Drop upper columns if specified
            if cropFeaturesToSize is not None:
                fileFeatures = fileFeatures[:, : cropFeaturesToSize]
            # Drop rows where all columns are zero
            fileFeatures = fileFeatures[~np.all(fileFeatures == 0, axis = 1)]
            # Stack features accoring to the time argument
            numExamples = fileFeatures.shape[0]
            numFeatures = fileFeatures.shape[1]
            ffNew = np.zeros([numExamples - timeStepsPerFeature + 1, 
                              timeStepsPerFeature * numFeatures])
            for ts in range(timeStepsPerFeature):
                ffNew [:, ts * numFeatures : (ts + 1) * numFeatures] = fileFeatures[ts: numExamples + 1 + ts - timeStepsPerFeature, :]
            fileFeatures = ffNew
            # Add a label column to the end
            numFeatures = fileFeatures.shape[1]
            labelledFileFeatures = np.ones([fileFeatures.shape[0], numFeatures + 1]) * pieceIndex
            labelledFileFeatures[:, :-1] = fileFeatures
            # Add to allFeatures array
            if allFeatures is None:
                allFeatures = copy.deepcopy(labelledFileFeatures)
            else:
                allFeatures = np.vstack((allFeatures, copy.deepcopy(labelledFileFeatures)))

        pieceIndex += 1

    # Standardise feature range from 0 to 1
    print 'Standardising range...'
    standardisationFn = rcut(outputFn, '.pkl.gz') + '_standardisationValues.pkl.gz'
    if frequencyStandardisation:
        minFeatureValue = np.min(allFeatures[:, 0: numFeatures], axis = 0)
        maxFeatureValue = np.max(allFeatures[:, 0: numFeatures], axis = 0)
    else:
        minFeatureValue = np.min(np.min(allFeatures[:, 0: numFeatures]))
        maxFeatureValue = np.max(np.max(allFeatures[:, 0: numFeatures]))
    standardisationDict = {'Min Value': minFeatureValue,
                           'Max Value': maxFeatureValue}
    pickle.dump(standardisationDict, open(standardisationFn, 'wb'))
    allFeatures[:, 0: numFeatures] = (allFeatures[:, 0: numFeatures] - 
                                                       minFeatureValue) / (maxFeatureValue - 
                                                                             minFeatureValue)
    print 'minFeatureValue = %s\nmaxFeatureValue = %s' %(minFeatureValue, maxFeatureValue)

    # Shuffle Features 10 times
    print 'Shuffling...'
    for _ in np.arange(10):
        np.random.shuffle(allFeatures)

    # Extract training, validation and test sets
    numExamples = allFeatures.shape[0]
    numFeatures = allFeatures.shape[1] - 1

    trainingExamples = allFeatures[0: int(trainPercentage * numExamples / 100)]
    validationExamples = allFeatures[int(trainPercentage * numExamples / 100): 
                                     int((trainPercentage + validationPercentage) * numExamples / 100)]
    testExamples = allFeatures[int((trainPercentage + validationPercentage) * numExamples / 100): ]

    train_set = (trainingExamples[:, :numFeatures], trainingExamples[:, numFeatures])
    valid_set = (validationExamples[:, :numFeatures], validationExamples[:, numFeatures])
    test_set = (testExamples[:, :numFeatures], testExamples[:, numFeatures])

    # Write file for Theano
    print 'Writing Theano file...'
    outputFn = rcut(outputFn, '.pkl.gz') + '.pkl.gz'
    f = gzip.open(outputFn, 'wb')
    cPickle.dump((train_set, valid_set, test_set), f)
    f.close()
    

def applyFENSTransformationFunction(dataframe, transformationFunction):
        
    assert transformationFunction in ['log', 'exponent', 'square', 'square root', 
                                      'cube', 'cube root', 'inverse', 'fourth', 
                                      'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                                      'tenth', None], 'Error: %s is not a transformation function' % transformationFunction
                                      
    # Normalise the feature range from 0 to 1
    dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
    
    # Transform dataframe
    if transformationFunction == 'log':
        return np.log(dataframe + 1e-9)
    elif transformationFunction == 'exponent':
        return np.exp(dataframe)
    elif transformationFunction == 'square':
        return np.power(dataframe, 2)
    elif transformationFunction == 'square root':
        return np.power(dataframe, 0.5)
    elif transformationFunction == 'cube':
        return np.power(dataframe, 3)
    elif transformationFunction == 'fourth':
        return np.power(dataframe, 4)
    elif transformationFunction == 'fifth':
        return np.power(dataframe, 5)
    elif transformationFunction == 'sixth':
        return np.power(dataframe, 6)
    elif transformationFunction == 'seventh':
        return np.power(dataframe, 7)
    elif transformationFunction == 'eighth':
        return np.power(dataframe, 8)
    elif transformationFunction == 'ninth':
        return np.power(dataframe, 9)
    elif transformationFunction == 'tenth':
        return np.power(dataframe, 10)
    elif transformationFunction == 'cube root':
        return np.power(dataframe, 1.0 / 3.0)
    elif transformationFunction == 'inverse':
        return np.power(dataframe + 1e-9, -1)
        