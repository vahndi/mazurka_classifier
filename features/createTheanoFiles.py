# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:39:28 2015

@author: Vahndi
"""
from FeatureFileProps import FeatureFileProps
from paths import getFolderNames, NNtrainingPath
from featureConverter import csvsToTheanoDataSet2



######################## Settings ########################

numPieces = 20
numPerformancesPerPiece = 5
#featureNames = ['chroma', 'CENS_dsf2_wl41', 'constantQspec', 'logfreqspec', 'CENS_dsf10_wl41']
featureNames = ['chroma']
timeStacking = 1
numFeaturesToUse = 12
frequencyStandardisation = False

##########################################################


for featureName in featureNames:
    
    print 'Feature Name: %s' % featureName
    piecesRootPath = FeatureFileProps.getRootPath(featureName)
    piecesFolders = getFolderNames(piecesRootPath, orderAlphabetically = True)[: numPieces]
    piecesFeatureFolderPaths = [FeatureFileProps.getFeatureFolderPath(piecesRootPath + pieceFolder, featureName) 
                                for pieceFolder in piecesFolders]
    outputFn = 'feature_%s_numPieces_%i_numPerformances_%s' % (featureName, numPieces, str(numPerformancesPerPiece))
    if numFeaturesToUse is not None:
        outputFn += '_numFeatures_%i' % numFeaturesToUse
    outputFn += '_timeStacking_%i' % timeStacking
    if frequencyStandardisation:
        outputFn +='_freqStd_True'
    csvsToTheanoDataSet2(piecesFeatureFolderPaths,  NNtrainingPath + outputFn, 
                         numPerformancesPerPiece, timeStacking, numFeaturesToUse,
                         frequencyStandardisation = frequencyStandardisation, 
                         trainPercentage = 70.0, validationPercentage = 15.0, testPercentage = 15.0)
