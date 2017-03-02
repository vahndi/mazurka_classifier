import copy
import numpy as np
import pandas as pd
from math import ceil
from scipy import signal

from paths import mazurkasPath, getFolderNames, getFileNames, newFeaturesPath
from paths import getFENSrootPath, getTransformedFeatureRootPath
from paths import getTestSetFolders, getTestSetPerformances, getValidationSetFolders, getValidationSetPerformances
from featureConverter import featuresToFENS, transformFeatures


def generateFENSfeatureFileDataFrames(dfFeatures, pieceId, performanceId,
                                      weightMatrix, biasMatrix, featureOffset, featureScaling, timeStacking,
                                      numFeatures, FENSrootPath, 
                                      FENSnormalisationThreshold, FENStransformationFunction,
                                      FENSquantisationSteps, FENSquantisationWeights,
                                      FENSdownsampling, FENSwindowLength,
                                      featureName, NNtype, numOriginalFeatures, numNewFeatures,
                                      featureFileId):
    
    dfTransformedFeatures = transformFeatures(dfFeatures, weightMatrix, 
                                              biasMatrix, featureOffset, featureScaling, timeStacking)
                                              
    dfFENs = featuresToFENS(dfTransformedFeatures, numFeatures, 
                            transformationFunction = FENStransformationFunction,
                            normalisationThreshold = FENSnormalisationThreshold,
                            quantisationSteps = FENSquantisationSteps,
                            quantisationWeights = FENSquantisationWeights,
                            downSampleFactor = FENSdownsampling,
                            downSampleWindowLength = FENSwindowLength)

    FENSpath = FENSrootPath + pieceId + '/'
    FENSfn = FENSpath + performanceId + '_%s-%s-%iv-%ih-FENS_dsf%i_dswl%i.csv' \
                                      % (featureName, NNtype, numOriginalFeatures, 
                                         numNewFeatures, FENSdownsampling, FENSwindowLength)
    
    return FENSfn, FENSpath, pieceId, performanceId, dfFENs.reset_index().values, featureFileId


def generateTransformedFeatureFileDataFrames(dfFeatures, pieceId, performanceId,
                                             weightMatrix, biasMatrix, featureOffset, featureScaling, timeStacking,
                                             numFeatures, featuresRootPath, 
                                             featureName, NNtype, numOriginalFeatures, numNewFeatures,
                                             featureFileId):
                                                 
    dfTransformedFeatures = transformFeatures(dfFeatures, weightMatrix, 
                                              biasMatrix, featureOffset, featureScaling, timeStacking)
                                              
    featuresPath = featuresRootPath + pieceId + '/'
    featuresFn = featuresPath + performanceId + '_%s-%s-%iv-%ih.csv' \
                                      % (featureName, NNtype, 
                                         numOriginalFeatures, numNewFeatures)
    
    return featuresFn, featuresPath, pieceId, performanceId, dfTransformedFeatures.reset_index().values, featureFileId


def multi_generateFENSfeatureFileDataFrames(args):
    
    return generateFENSfeatureFileDataFrames(*args)


def multi_generateTransformedFeatureFileDataFrames(args):
    
    return generateTransformedFeatureFileDataFrames(*args)


class FeatureFileProps(object):
    '''
    Holds basic properties about a feature file
    '''
    
    @classmethod
    def getRootPath(cls, featureName):
        
        if featureName in ['chroma', 'mfcc', 'spectrogram', 'constantQspec', 'logfreqspec']:
            return mazurkasPath
        else:
            return newFeaturesPath + featureName + '/'
            
    
    rootPath = {'chroma': mazurkasPath,
                'mfcc': mazurkasPath,
                'spectrogram': mazurkasPath,
                'constantQspec': mazurkasPath,
                'logfreqspec': mazurkasPath,
                'CENS_dsf2_wl10': newFeaturesPath + 'CENS_dsf2_wl10/',
                'CENS_dsf2_wl20': newFeaturesPath + 'CENS_dsf2_wl20/',
                'CENS_dsf2_wl30': newFeaturesPath + 'CENS_dsf2_wl30/',
                'CENS_dsf2_wl41': newFeaturesPath + 'CENS_dsf2_wl41/',
                'CENS_dsf2_wl50': newFeaturesPath + 'CENS_dsf2_wl50/',
                'CENS_dsf2_wl60': newFeaturesPath + 'CENS_dsf2_wl60/',
                'CENS_dsf2_wl70': newFeaturesPath + 'CENS_dsf2_wl70/',
                'CENS_dsf2_wl80': newFeaturesPath + 'CENS_dsf2_wl80/',
                'CENS_dsf2_wl90': newFeaturesPath + 'CENS_dsf2_wl90/',
                'logfreqspec-dA-256v-12h-FENS_dsf2_dswl41': newFeaturesPath + 'logfreqspec-dA-256v-12h-FENS_dsf2_dswl41/'}

    folderPrefix = {'chroma': 'qm-chromagram_standard',
                    'mfcc': 'qm-mfcc-standard',
                    'spectrogram': 'powerspectrum_2048_1024',
                    'constantQspec': 'qm-constantq_standard',
                    'logfreqspec': 'nnls-logfreqspec_standard'}

    fileSuffix = {'chroma': '_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv',
                  'mfcc': '_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv',
                  'CENS_dsf2_wl10': '_CENS_dsf_2.0_dswl_10.csv',
                  'CENS_dsf2_wl20': '_CENS_dsf_2.0_dswl_20.csv',
                  'CENS_dsf2_wl30': '_CENS_dsf_2.0_dswl_30.csv',
                  'CENS_dsf2_wl41': '_CENS_dsf_2.0_dswl_41.csv',
                  'CENS_dsf2_wl50': '_CENS_dsf_2.0_dswl_50.csv',
                  'CENS_dsf2_wl60': '_CENS_dsf_2.0_dswl_60.csv',
                  'CENS_dsf2_wl70': '_CENS_dsf_2.0_dswl_70.csv',
                  'CENS_dsf2_wl80': '_CENS_dsf_2.0_dswl_80.csv',
                  'CENS_dsf2_wl90': '_CENS_dsf_2.0_dswl_90.csv',
                  'CENS_dsf10_wl41': '_CENS_dsf_10_dswl_41.csv',
                  'constantQspec': '_vamp_qm-vamp-plugins_qm-constantq_constantq.csv',
                  'logfreqspec': '_vamp_nnls-chroma_nnls-chroma_logfreqspec.csv',
                  'logfreqspec-dA-256v-12h-FENS_dsf2_dswl41': '_logfreqspec-dA-256v-12h-FENS_dsf2_dswl41.csv'}

    @classmethod
    def getFileSuffix(cls, featureName):
        
        if featureName in FeatureFileProps.fileSuffix.keys():
            return FeatureFileProps.fileSuffix[featureName]
        else:
            return '_%s.csv' % featureName
            

    featureRate = {'chroma': 22.0,
                   'mfcc': 10.0,
                   'CENS_dsf2_wl41': 11.0,
                   'CENS_dsf2_wl90': 11.0,
                   'constantQspec': 22.0,
                   'logfreqspec': 22.0}
    
    featureOffset = {'constantQspec': 0.0,
                     'chroma': 0.0,
                     'CENS_dsf2_wl41': 0.0,
                     'logfreqspec': 0.001883}
    
    featureScaling = {'constantQspec': 1.0 / 0.152199,
                      'chroma': 1.0,
                      'logfreqspec': 1.0 / 23373.600000,
                      'CENS_dsf2_wl41': 1.0}

    
    
    def __init__(self, fileName, filePath, pieceId, performanceId):
        
        self.fileName = fileName
        self.filePath = filePath
        self.pieceId = pieceId
        self.performanceId = performanceId
        self.featureFileData = None


    def getFeatureFileDataFrame(self, numFeatures = None):
        
        if self.featureFileData is None:
            return None
        dfFeatures = pd.DataFrame(self.featureFileData)
        dfFeatures.set_index(0, inplace = True)
        if numFeatures is not None:
            numColumns = len(dfFeatures.columns)
            dfFeatures = dfFeatures.drop([i for i in range(numFeatures + 1, numColumns + 1)], axis = 1)
        
        return dfFeatures
        
        
    @classmethod
    def getFeatureFolderPath(cls, piecePath, featureName):
        '''
        Returns the path of the folder containing specific features within the
        folder for a specific piece
        Inputs:
            :piecePath:     the path to the folder containing the piece feature
                            folders (for features in the original mazurka-dataset folder)
                            or the features themselves (for newly created features such as CENS)
            :featureName:   the name of the feature
        '''
        piecePath = piecePath.rstrip('/') + '/'
        
        if 'CENS' in featureName or 'FENS' in featureName:
            return piecePath
        else:
            featureFolderPrefix = FeatureFileProps.folderPrefix[featureName]
            featureFolders = [fldr for fldr in getFolderNames(piecePath, 
                                                              orderAlphabetically = True) if featureFolderPrefix in fldr]
            assert len(featureFolders) == 1, 'number of feature folders must be 1 (it is %i)' % len(featureFolders)
            featuresFolder = featureFolders[0]
            featuresPath = piecePath + featuresFolder + '/'
            
            return featuresPath


    @classmethod
    def getFeatureFileDict(cls, piecesPath, pieceFolder, featuresPath, featureName, numFiles = None):
        '''
        Returns a dictionary of all the performances of a given piece
        The dictionary keys are the filenames with the standard suffix for that feature
        removed.
        The dictionary values are new dictionaries with keys and associated values for
        FileName, FilePath and PieceId
        If numFiles is None then all files will be returned, 
            otherwise return the first numFiles files alphabetically
        '''
        featureFileDict = {}
        featureFileSuffix = FeatureFileProps.getFileSuffix(featureName)
        featureFileNames = getFileNames(featuresPath, endsWith = featureFileSuffix, orderAlphabetically = True)
        if numFiles is not None:
            featureFileNames = featureFileNames[: numFiles]
        featureFilePaths = [featuresPath + fName for fName in featureFileNames]
        featureFileIds = [fName[:-len(featureFileSuffix)] for fName in featureFileNames]
        for i in range(len(featureFileIds)):
            fileName = featureFileNames[i]
            filePath = featureFilePaths[i]
            pieceId = pieceFolder
            featureFileDict[featureFileIds[i]] = FeatureFileProps(fileName, filePath, pieceId, featureFileIds[i])
            
        return featureFileDict    
        
    
    @classmethod
    def getFeatureFileDictTestSet(cls, pieceFolder, featuresPath, featureName):
        
        featureFileDict = {}
        featureFileSuffix = FeatureFileProps.getFileSuffix(featureName)
        featureFileNames = getFileNames(featuresPath, endsWith = featureFileSuffix, orderAlphabetically = True)
        featureFileNames = getTestSetPerformances(featureFileNames)
        featureFilePaths = [featuresPath + fName for fName in featureFileNames]
        featureFileIds = [fName[:-len(featureFileSuffix)] for fName in featureFileNames]
        for i in range(len(featureFileIds)):
            fileName = featureFileNames[i]
            filePath = featureFilePaths[i]
            pieceId = pieceFolder
            featureFileDict[featureFileIds[i]] = FeatureFileProps(fileName, filePath, pieceId, featureFileIds[i])
            
        return featureFileDict   


    @classmethod
    def getFeatureFileDictValidationSet(cls, pieceFolder, featuresPath, featureName):
        
        featureFileDict = {}
        featureFileSuffix = FeatureFileProps.getFileSuffix(featureName)
        featureFileNames = getFileNames(featuresPath, endsWith = featureFileSuffix, orderAlphabetically = True)
        featureFileNames = getValidationSetPerformances(featureFileNames)
        featureFilePaths = [featuresPath + fName for fName in featureFileNames]
        featureFileIds = [fName[:-len(featureFileSuffix)] for fName in featureFileNames]
        for i in range(len(featureFileIds)):
            fileName = featureFileNames[i]
            filePath = featureFilePaths[i]
            pieceId = pieceFolder
            featureFileDict[featureFileIds[i]] = FeatureFileProps(fileName, filePath, pieceId, featureFileIds[i])
            
        return featureFileDict   
        
        
    @classmethod
    def getFeatureFileDictAllFolders(cls, rootPath, folderNames, featureName, numFilesPerFolder = None):
        '''
        Returns featureFileDicts for the given feature for multiple performances of pieces
        in the given folder names. See getFeatureFileDict for details of the featureFileDict
        If numFilesPerFolder is None then all files will be returned from each folder,
            otherwise numFilesPerFolder files will be returned from each folder
        '''
        featureFileDict = {}
        for pieceFolder in folderNames:
            featuresPath = FeatureFileProps.getFeatureFolderPath(rootPath + pieceFolder + '/', featureName)
            ffDict = copy.deepcopy(FeatureFileProps.getFeatureFileDict(rootPath, pieceFolder, featuresPath, featureName, numFilesPerFolder))
            featureFileDict.update(ffDict)
            
        return featureFileDict
        
    
    @classmethod
    def getFeatureFileDictTestSetFolders(cls, rootPath, featureName):
        '''
        Returns featureFileDicts for the given feature for multiple performances of pieces
        in the given folder names. See getFeatureFileDict for details of the featureFileDict
        If numFilesPerFolder is None then all files will be returned from each folder,
            otherwise numFilesPerFolder files will be returned from each folder
        '''
        featureFileDict = {}
        folderNames = getFolderNames(rootPath, orderAlphabetically = True, contains = 'mazurka')
        testFolderNames = getTestSetFolders(folderNames)
        for pieceFolder in testFolderNames:
            featuresPath = FeatureFileProps.getFeatureFolderPath(rootPath + pieceFolder + '/', featureName)
            ffDict = copy.deepcopy(FeatureFileProps.getFeatureFileDictTestSet(pieceFolder, featuresPath, featureName))
            featureFileDict.update(ffDict)
            
        return featureFileDict


    @classmethod
    def getFeatureFileDictValidationSetFolders(cls, rootPath, featureName):
        '''
        Returns featureFileDicts for the given feature for multiple performances of pieces
        in the given folder names. See getFeatureFileDict for details of the featureFileDict
        If numFilesPerFolder is None then all files will be returned from each folder,
            otherwise numFilesPerFolder files will be returned from each folder
        '''
        featureFileDict = {}
        folderNames = getFolderNames(rootPath, orderAlphabetically = True)
        validationFolderNames = getValidationSetFolders(folderNames)
        for pieceFolder in validationFolderNames:
            featuresPath = FeatureFileProps.getFeatureFolderPath(rootPath + pieceFolder + '/', featureName)
            ffDict = copy.deepcopy(FeatureFileProps.getFeatureFileDictValidationSet(pieceFolder, featuresPath, featureName))
            featureFileDict.update(ffDict)
            
        return featureFileDict
        

    @classmethod
    def loadFeatureFileDictAllFolders(cls, rootPath, folderNames, featureName, numFilesPerFolder = None):
        '''
        Same as getFeatureFileDictAllFolders but loads the features into the .featureFileData property
        '''
        featureFileDict = FeatureFileProps.getFeatureFileDictAllFolders(rootPath, folderNames, featureName, numFilesPerFolder)
        for featureFileId in featureFileDict.keys():
            featureFileDict[featureFileId].featureFileData = np.genfromtxt(featureFileDict[featureFileId].filePath, 
                                                                           delimiter = ',')
        return featureFileDict        


    @classmethod
    def loadFeatureFileDictTestSetFolders(cls, rootPath, featureName):
        '''
        loads a feature file dict for the specified feature using the test set files
        '''
        featureFileDict = FeatureFileProps.getFeatureFileDictTestSetFolders(rootPath, featureName)
        for featureFileId in featureFileDict.keys():
            featureFileDict[featureFileId].featureFileData = np.genfromtxt(featureFileDict[featureFileId].filePath, 
                                                                           delimiter = ',')
        return featureFileDict     


    @classmethod
    def loadFeatureFileDictValidationSetFolders(cls, rootPath, featureName):
        '''
        loads a feature file dict for the specified feature using the test set files
        '''
        featureFileDict = FeatureFileProps.getFeatureFileDictValidationSetFolders(rootPath, featureName)
        for featureFileId in featureFileDict.keys():
            featureFileDict[featureFileId].featureFileData = np.genfromtxt(featureFileDict[featureFileId].filePath, 
                                                                           delimiter = ',')
        return featureFileDict   

    @classmethod
    def generateTransformedFeatureFileDict(cls, originalFeatureFileDict, numFeatures, 
                                                featureName, NNtype, numOriginalFeatures, numNewFeatures,
                                                weightMatrix, biasMatrix, featureOffset, featureScaling, timeStacking,
                                                processPool = None):
        '''
        Generates a FeatureFileDict for in-memory features transformed by weights and biases
        '''
        featuresRootPath = getTransformedFeatureRootPath(featureName, NNtype, 
                                                         numOriginalFeatures, 
                                                         numNewFeatures)
        transformedFeatureFileDictAllFolders = {}
        featureFileList = []
        if processPool is None:
            
            for featureFileId in originalFeatureFileDict.keys():
                ffProps = originalFeatureFileDict[featureFileId]
                dfFeatures = ffProps.getFeatureFileDataFrame(numFeatures = numOriginalFeatures)
                dfTransformedFeatures = transformFeatures(dfFeatures, 
                                                          weightMatrix, biasMatrix, 
                                                          featureOffset, featureScaling, 
                                                          timeStacking)
                # create a FFP object with fileName, filePath, pieceId
                pieceId = ffProps.pieceId
                performanceId = ffProps.performanceId
                featuresPath = featuresRootPath + pieceId + '/'
                featuresFn = featuresPath + performanceId + '_%s-%s-%iv-%ih.csv' \
                                                  % (featureName, NNtype, 
                                                     numOriginalFeatures, numNewFeatures)
                ffpFENS = FeatureFileProps(featuresFn, featuresFn, pieceId, performanceId)
                
                # attach FENS features to FFP as featureFileData
                ffpFENS.featureFileData = dfTransformedFeatures.reset_index().values
                featureFileList.append(featureFileId, copy.deepcopy(ffpFENS))
        else:
            argList = []
            for featureFileId in originalFeatureFileDict.keys():
                ffProps = originalFeatureFileDict[featureFileId]
                dfFeatures = ffProps.getFeatureFileDataFrame(numFeatures = numOriginalFeatures)
                argList.append((dfFeatures, ffProps.pieceId, ffProps.performanceId,
                                weightMatrix, biasMatrix, featureOffset, featureScaling, timeStacking, numFeatures, 
                                featuresRootPath, featureName, NNtype, numOriginalFeatures, numNewFeatures, featureFileId))
            featuresTuples = processPool.map(multi_generateTransformedFeatureFileDataFrames, argList)

            for featuresTuple in featuresTuples:
                ffpFENS = FeatureFileProps(featuresTuple[0], featuresTuple[1], featuresTuple[2], featuresTuple[3])                
                # attach FENS features to FFP as featureFileData
                ffpFENS.featureFileData = featuresTuple[4]
                featureFileList.append((featuresTuple[5], copy.deepcopy(ffpFENS)))
            
        # add to dict
        for featureFileTuple in featureFileList:
            transformedFeatureFileDictAllFolders[featureFileTuple[0]] = featureFileTuple[1]

                
        return transformedFeatureFileDictAllFolders
        
    
    @classmethod
    def generateFENSfeatureFileDict(cls, originalFeatureFileDict, numFeatures, 
                                         featureName, NNtype, numOriginalFeatures, numNewFeatures,
                                         weightMatrix, biasMatrix, featureOffset, featureScaling, timeStacking,
                                         FENSdownsampling, FENSwindowLength,
                                         FENStransformationFunction = None,
                                         FENSnormalisationThreshold = 0.001, 
                                         FENSquantisationSteps = [0.4, 0.2, 0.1, 0.05],
                                         FENSquantisationWeights = [1, 1, 1, 1], 
                                         processPool = None):
        '''
        generates a FeatureFileDict for temporary in-memory FENS features
        '''
        FENSrootPath = getFENSrootPath(featureName, NNtype, numOriginalFeatures, 
                                       numNewFeatures, FENSdownsampling, FENSwindowLength)
        FENSfeatureFileDictAllFolders = {}
        featureFileList = []
        if processPool is None:
            
            for featureFileId in originalFeatureFileDict.keys():
                ffProps = originalFeatureFileDict[featureFileId]
                dfFeatures = ffProps.getFeatureFileDataFrame(numFeatures = numOriginalFeatures)
                dfTransformedFeatures = transformFeatures(dfFeatures, weightMatrix, 
                                                              biasMatrix, featureOffset, featureScaling, timeStacking)
                dfFENs = featuresToFENS(dfTransformedFeatures, numFeatures, 
                                        transformationFunction = FENStransformationFunction,
                                        normalisationThreshold = FENSnormalisationThreshold,
                                        quantisationSteps = FENSquantisationSteps,
                                        quantisationWeights = FENSquantisationWeights,
                                        downSampleFactor = FENSdownsampling,
                                        downSampleWindowLength = FENSwindowLength)
                # create a FFP object with fileName, filePath, pieceId
                pieceId = ffProps.pieceId
                performanceId = ffProps.performanceId
                FENSpath = FENSrootPath + pieceId + '/'
                FENSfn = FENSpath + performanceId + '_%s-%s-%iv-%ih-FENS_dsf%i_dswl%i.csv' \
                                                  % (featureName, NNtype, numOriginalFeatures, 
                                                     numNewFeatures, FENSdownsampling, FENSwindowLength)
                ffpFENS = FeatureFileProps(FENSfn, FENSpath, pieceId, performanceId)
                
                # attach FENS features to FFP as featureFileData
                ffpFENS.featureFileData = dfFENs.reset_index().values
                featureFileList.append(featureFileId, copy.deepcopy(ffpFENS))
        else:
            argList = []
            for featureFileId in originalFeatureFileDict.keys():
                ffProps = originalFeatureFileDict[featureFileId]
                dfFeatures = ffProps.getFeatureFileDataFrame(numFeatures = numOriginalFeatures)
                argList.append((dfFeatures, ffProps.pieceId, ffProps.performanceId,
                                weightMatrix, biasMatrix, featureOffset, featureScaling, timeStacking, numFeatures, 
                                FENSrootPath, FENSnormalisationThreshold, FENStransformationFunction,
                                FENSquantisationSteps, FENSquantisationWeights,
                                FENSdownsampling, FENSwindowLength,
                                featureName, NNtype, numOriginalFeatures, numNewFeatures, featureFileId))
            FENStuples = processPool.map(multi_generateFENSfeatureFileDataFrames, argList)

            for FENStuple in FENStuples:
                ffpFENS = FeatureFileProps(FENStuple[0], FENStuple[1], FENStuple[2], FENStuple[3])                
                # attach FENS features to FFP as featureFileData
                ffpFENS.featureFileData = FENStuple[4]
                featureFileList.append((FENStuple[5], copy.deepcopy(ffpFENS)))
            
        # add to dict
        for featureFileTuple in featureFileList:
            FENSfeatureFileDictAllFolders[featureFileTuple[0]] = featureFileTuple[1]

                
        return FENSfeatureFileDictAllFolders
        
        
    @classmethod
    def downSample(cls, array2d, factor):
        '''
        Used to downsample a feature vector before calculating its CRP
        '''
        numRows, numCols = array2d.shape
        numNewRows = ceil(float(numRows) / factor)
        newArray = np.zeros([numNewRows, numCols])
        newRow = 0
        for rowStart in range(0, numRows, factor):
            newArray[newRow] = array2d[rowStart:min(rowStart + factor, numRows), :].mean(axis = 0)
            newRow += 1
            
        return newArray
        
    
    @classmethod
    def resize(cls, array2d, numFrames):
        if numFrames == 'var':
            return array2d
        else:
            # Resample along each axis
            print array2d.shape
            print 'resizing to %i frames' % numFrames
            return signal.resample(array2d, int(numFrames), axis = 0)
            print 'resized'
    
    @classmethod
    def getFENSQuantisationSteps(cls, base, power):
        '''
        Returns four ordered FENS quantisation steps calculated based on raising
        the base to the power
        '''
        return [base * power**3, base * power**2, base * power, base]
        
        