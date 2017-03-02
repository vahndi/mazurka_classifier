# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 13:40:21 2015

@author: Vahndi
"""

import os

from paths import  getFileNames, CRPpath, NCDpath, runHistoryPath



def cleanRunFolder(runName = None, cleanCRPfolder = True, cleanNCDfolder = True):
    
    # Remove CRP files
    crpFiles = getFileNames(CRPpath, endsWith = '.npy')
    for crpFile in  crpFiles:
        os.remove(CRPpath + crpFile)
    
    # Remove NCD files
    ncdFiles = getFileNames(NCDpath, endsWith = '.pkl')
    for ncdFile in  ncdFiles:
        os.remove(NCDpath + ncdFile)
    
    # Empty Run Folder and Run History Folder
    if runName is not None:
        resultsPath = NCDpath + runName + '/'
        resultsFiles = getFileNames(resultsPath)
        for resultsFile in resultsFiles:
            os.remove(resultsPath + resultsFile)
        historyPath = runHistoryPath + runName + '/'
        historyFiles = getFileNames(historyPath)
        for historyFile in historyFiles:
            os.remove(historyPath + historyFile)


## Rename FENS features
#FENSpath = '/data/student/abpg162/outputs/new_features/logfreqspec_dA_FENS_dsf2_wl41/'
#folders = getFolderNames(FENSpath)
#for folder in folders:
#    fileNames = getFileNames(FENSpath +  folder + '/')
#    for fileName in fileNames:
#        print fileName
#        newFileName = fileName.replace('FENS', 'logfreqspec-dA-256v-12h-FENS')
#        print newFileName
#        os.rename(os.path.join(FENSpath, folder, fileName), os.path.join(FENSpath, folder, newFileName))
