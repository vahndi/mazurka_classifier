import os, shutil
import pandas as pd



mazurkasPath = '/data/student/abpg162/mazurka-dataset/'
outputBasePath = '/data/student/abpg162/outputs/'
CRPpath = '/data/student/abpg162/outputs/crps/'
NCDpath = '/data/student/abpg162/outputs/ncds/'
runHistoryPath = '/data/student/abpg162/outputs/runhist/'
NNpath = '/data/student/abpg162/outputs/nns/'
NNtrainingPath = NNpath + 'training_files/'
NNweightsPath = NNpath + 'training_results/'
transferPath = '/u7.swansea/s11/abpg162/project/results_files/'
newFeaturesPath = '/data/student/abpg162/outputs/new_features/'
FENSparamsPath = '/u7.swansea/s11/abpg162/project/results_files/FENSparams/'


def getFolderNames(inputPath, startsWith = None, endsWith = None, contains = None, orderAlphabetically = False):
    '''
    Returns the list of folder names in the given folder
    Inputs:
        :inputPath: the path to find folders in
        :startsWith: an optional string that the returned folder names must start with
        :endsWith: an optional string that the returned folder names must end with
        :contains: an optional string that the returned folder names must contain
        :orderAlphabetically: whether to order the returned list of folder names alphabetically
    '''
    folders = [folder for folder in os.listdir(inputPath) if os.path.isdir(inputPath + folder)]
    
    if orderAlphabetically:
        folders = sorted(folders)
    
    if startsWith is not None:
        folders = [f for f in folders if f.startswith(startsWith)]
    if endsWith is not None:
        folders = [f for f in folders if f.endswith(endsWith)]
    if contains is not None:
        folders = [f for f in folders if contains in f]
        
    return folders
    

def getFileNames(inputPath, startsWith = None, endsWith = None, contains = None, orderAlphabetically = False):
    '''
    returns filenames in a folder optionally matching a specific ending string e.g. '.txt'
    '''
    files = [fName for fName in os.listdir(inputPath) if os.path.isfile(inputPath + fName)]
    
    
    if orderAlphabetically:
        files = sorted(files)
    
    if startsWith is not None:
        files = [fName for fName in files if fName.startswith(startsWith)]
    if endsWith is not None:
        files = [fName for fName in files if fName.endswith(endsWith)]
    if contains is not None:
        files = [fName for fName in files if contains in fName]    
    
    return files
   
   
def createPath(pathToCreate):
    '''
    Creates a path, unless it already exists
    '''
    if not os.path.exists(pathToCreate):
        os.makedirs(pathToCreate)


def moveFiles(srcPath, destPath, filesThatEndWith = None):
    '''
    Moves all files in srcPath to destPath
    Optional filter on the end of filenames e.g. '.csv'
    '''
    for fn in getFileNames(srcPath, endsWith = filesThatEndWith):
        shutil.move(srcPath + fn, destPath + fn)


def copyFiles(srcPath, destPath, filesThatEndWith = None):
    '''
    Moves all files in srcPath to destPath
    Optional filter on the end of filenames e.g. '.csv'
    '''
    for fn in getFileNames(srcPath, endsWith = filesThatEndWith):
        shutil.copyfile(srcPath + fn, destPath + fn)
        

def getWeightsPath(NNtype, featureName, numFolders, numFiles,
                   batchSize = None, learningRate = None, 
                   learningRateBoostFactor = None, timeStacking = None, numFeatures = None,
                   frequencyStandardisation = False):
    
    weightsPath = '%s%s/numFolders_%i_numFiles_%i/%s' % (NNweightsPath, NNtype, numFolders, numFiles, featureName)
#    weightsPath =  NNweightsPath + NNtype + '/' + featureName
    
    if batchSize is not None:
        weightsPath += '_bs_%i' % batchSize
    if learningRate is not None:
        weightsPath += '_lr_%0.2f' % learningRate
    if learningRateBoostFactor is not None:
        weightsPath += '_lrbf_%0.2f' % learningRateBoostFactor
    if timeStacking is not None:
        weightsPath += '_ts_%i' % timeStacking
    if numFeatures is not None:
        weightsPath += '_nf_%i' % numFeatures
    if frequencyStandardisation:
        weightsPath += '_freqStd'
    weightsPath += '/'
    
    return weightsPath


def getFENSrootPath(featureName, NNtype, 
                    numOriginalFeatures, numNewFeatures, 
                    FENSdownsampling, FENSwindowLength):

    FENSrootPath = newFeaturesPath + '%s-%s-%iv-%ih-FENS_dsf%i_dswl%i/' \
                                   % (featureName, NNtype, 
                                      numOriginalFeatures, numNewFeatures, 
                                      FENSdownsampling, FENSwindowLength)
    
    return FENSrootPath
    
    
def getTransformedFeatureRootPath(featureName, NNtype, 
                                  numOriginalFeatures, numNewFeatures):
                                      
    transformedFeatureRootPath = newFeaturesPath + '%s-%s-%iv-%ih/' \
                                   % (featureName, NNtype, 
                                      numOriginalFeatures, numNewFeatures)
    
    return transformedFeatureRootPath


def loadRunSettingsList(csvFilePath):
    '''
    Loads a list of predetermined settings from a csv file using pandas and 
    return as a list for use in validation or testing
    '''
    df = pd.read_csv(csvFilePath)
    df = df.where((pd.notnull(df)), None)
    lstDicts = df.to_dict('records')
    return lstDicts


def getTrainingSetFolders(allFolders):
    '''
    Returns the first 20 folders passed, as this was the training set size
    '''
    return allFolders[:20]
    

def getValidationSetFolders(allFolders):
    '''
    The validation set used was the remaining files in the training folder that
    were not in the first 5 used for training.
    '''
    return getTrainingSetFolders(allFolders)
        

def getTestSetFolders(allFolders):
    '''
    The test set is all of the files in all of the remaining folders above the 
    twenty that were used in training and validation.
    '''
    return allFolders[20:]
    
    
def getTrainingSetPerformances(piecePerformances):
    '''
    Returns the first 5 performances of one piece in a list
    N.B. use in conjunction with getTrainingSetFolders
    '''
    return piecePerformances[:5]
    
    
def getValidationSetPerformances(piecePerformances):
    '''
    The validation set used was the remaining files in the training folder that
    were not in the first 5 used for training. 
    '''
    return piecePerformances[5:]
    
    
def getTestSetPerformances(piecePerformances):
    '''
    The test set is all of the files in all of the remaining folders above the 
    twenty that were used in training and validation.
    '''
    return piecePerformances
