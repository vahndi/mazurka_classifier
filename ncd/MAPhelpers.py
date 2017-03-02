# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 01:38:44 2015

@author: vahndi
"""

import os
import numpy as np

from paths import NCDpath, runHistoryPath, createPath, moveFiles, getFolderNames
from paths import getFileNames, CRPpath
from NCDprops import NCDprops as NCDprops
from CRPprops import CRPprops as CRPprops
from FeatureFileProps import FeatureFileProps as FFP



def loadFeatureFileDict(featureName, numFolders, numFilesPerFolder):
    
    piecesPath = FFP.getRootPath(featureName)
    pieceIds = getFolderNames(piecesPath, contains = 'mazurka', orderAlphabetically = True)[:numFolders]
    print 'Loading feature file dict...'
    featureFileDict = FFP.loadFeatureFileDictAllFolders(piecesPath, pieceIds, featureName, numFilesPerFolder)
    print '...done.'
    return featureFileDict, pieceIds
    

def moveResultsFiles(subFolder):
    
    NCDdest = NCDpath + subFolder + '/'
    runHistDest = runHistoryPath + subFolder + '/'
    createPath(NCDdest)
    createPath(runHistDest)
    moveFiles(NCDpath, NCDdest, '.pkl.res')
    moveFiles(runHistoryPath, runHistDest, '.pkl')
                
    return NCDdest


def cleanCRPfolder():
    
    # Remove CRP files
    crpFiles = getFileNames(CRPpath, endsWith = '.npy')
    for crpFile in  crpFiles:
        os.remove(CRPpath + crpFile)
        

def cleanNCDfolder():
    
    # Remove NCD files
    ncdFiles = getFileNames(NCDpath, endsWith = '.pkl')
    for ncdFile in  ncdFiles:
        os.remove(NCDpath + ncdFile)


def calculateRequiredNCDs(featureFileDict, method, dimension, timeDelay,
                                neighbourhoodSize, downSampleFactor, sequenceLength,
                                featureName):
    
    requiredNCDs = []
    featureFileIds = featureFileDict.keys()
    numFeatureFiles = len(featureFileIds)
    print 'Creating list of required NCDs...'
    for f1 in np.arange(numFeatureFiles - 1):
        featureFilePath1 = featureFileDict[featureFileIds[f1]].filePath
        pc1Id = featureFileDict[featureFileIds[f1]].pieceId
        pc1pfId = featureFileIds[f1]
        for f2 in np.arange(f1, numFeatureFiles):
            featureFilePath2 = featureFileDict[featureFileIds[f2]].filePath
            pc2Id = featureFileDict[featureFileIds[f2]].pieceId
            pc2pfId = featureFileIds[f2]
            ncdProps = NCDprops(pc1Id, pc1pfId, pc2Id, pc2pfId, 
                                method, dimension, timeDelay,
                                neighbourhoodSize, downSampleFactor, sequenceLength,
                                featureName, featureFilePath1, featureFilePath2)
            requiredNCDs.append(ncdProps)
    print 'Number of NCD files required for combination: %i' % len(requiredNCDs)

    return requiredNCDs
    

def calculateRequiredCRPs(requiredNCDs):
    
    print 'Calculating # of required CRP files'
    requiredCRPs = []
    for requiredNCD in requiredNCDs:
        crp1 = requiredNCD.getCRP1()
        crp2 = requiredNCD.getCRP2()
        requiredCRPs.append(crp1)
        requiredCRPs.append(crp2)

    requiredCRPs = CRPprops.uniqueCRPprops(requiredCRPs)
    
    return requiredCRPs

    