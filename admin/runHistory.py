# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:54:43 2015

@author: vahndi
"""

from paths import NCDpath, getFolderNames, getFileNames, transferPath
import pandas as pd
import os
import copy
import numpy as np

runDicts = []
optRunDicts = []
outputFolder = transferPath

# For each results folder
for folder in getFolderNames(NCDpath, orderAlphabetically = True):
    
    print '\n', folder
    dfRun = None
    runFolder = NCDpath + folder + '/'
    runResultsFiles = getFileNames(runFolder, endsWith = '.csv', orderAlphabetically = True)
    if len(runResultsFiles) > 0:
        runFileDFs = []
        for resultsFile in runResultsFiles:
            print '\rAppending run dataframe %s...' % resultsFile,
            runFileDFs.append(pd.read_csv(os.path.join(runFolder, resultsFile)))
        dfRun = pd.concat(runFileDFs, axis = 0, ignore_index = True)
        bestRowDict = dict(dfRun.sort('Mean Average Precision', ascending = False).iloc[0])
        bestRowDict['Run Name'] = folder
        optRunDicts.append(copy.deepcopy(bestRowDict))
        runDict = {'Run Name': folder}
        for col in dfRun.columns:
            if col != 'Mean Average Precision':
                uniqueVals = dfRun[col].unique()
                if uniqueVals.dtype == np.dtype('float64'):
                    uniqueVals = np.unique(np.array(["%.2f" % number for number in uniqueVals]))
                runDict[col] = sorted(list(uniqueVals))
        runDicts.append(copy.deepcopy(runDict))


dfRunHistory = pd.DataFrame(runDicts)
dfRunHistory.to_csv(outputFolder + 'Run History.csv')
dfOptRunHistory = pd.DataFrame(optRunDicts)
dfOptRunHistory.to_csv(outputFolder + 'Optimal Run History.csv')
