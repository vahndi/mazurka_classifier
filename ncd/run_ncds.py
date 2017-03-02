# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""

import pickle
from shared import *
from datetime import datetime
from create_ncds3 import createNCDfiles
from ncd_processing import convertNCDfiles, deleteNCDPickleFiles
from multiprocessing import Pool

### Settings ###

numProcesses = 8
featureName = 'chroma'
CRPmethods = ['maxnorm', 'euclidean', 'minnorm', 'rr', 'fan', 'nrmnorm']
CRPdimensions = [1, 2, 3, 4, 5, 6, 7]
CRPneighbourhoodSizes = np.arange(0.05, 1, 0.05)
CRPtimeDelays = [1, 2, 3, 4, 5]
downSampleFactors = [2, 4, 8, 16]
numFilesPerFolder = 5 # Set to None to compare all files in each folder

#################

deleteNCDPickleFiles()

existingNCDs = None
resultsFn = NCDpath + 'NCDs.pkl.res'
processPool = Pool(numProcesses)

#if os.path.exists(resultsFn):
#    dfExistingNCDs = pd.read_pickle(resultsFn)
#    existingNCDs = list(dfExistingNCDs['FileName'])

#method = chooseRandomItem(CRPmethods)
#dimension = chooseRandomItem(CRPdimensions)
#neighbourhoodSize = chooseRandomItem(CRPneighbourhoodSizes)
#timeDelay = chooseRandomItem(CRPtimeDelays)
downSampleFactor = chooseRandomItem(downSampleFactors)

for downSampleFactor in downSampleFactors[::-1]:
    for neighbourhoodSize in CRPneighbourhoodSizes:
        for method in CRPmethods:
            for dimension in CRPdimensions:
                for timeDelay in CRPtimeDelays:

                    createNCDfiles(existingNCDs, processPool,
                                   featureName, downSampleFactor,
                                   timeDelay, dimension, method, neighbourhoodSize, 
                                   numFilesPerFolder)

                    runDict = {'featureName': featureName,
                               'downSampleFactor': downSampleFactor,
                               'timeDelay': timeDelay, 
                               'dimension': dimension,
                               'method': method, 
                               'neighbourhoodSize': neighbourhoodSize,
                               'numFilesPerFolder': numFilesPerFolder}

                    runTime = str(datetime.now()).replace(':', '-')
                    pickle.dump(runDict, open(runHistoryPath + runTime + '.pkl', 'wb'))
                    convertNCDfiles(runTime)
