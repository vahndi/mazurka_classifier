# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:04:56 2015

@author: Vahndi
"""

# This version works with create_ncds4.py i.e. after support for equal size CRPs was introduced

import pickle
from shared import *
from datetime import datetime
from create_ncds4b import createNCDfiles
from ncd_processing import convertNCDfiles, deleteNCDPickleFiles
from multiprocessing import Pool

'''
This is a version to test each of the stages of the process
'''


### Settings ###

numProcesses = 8
featureName = 'chroma'
CRPmethods = ['fan']
CRPdimensions = [3]
CRPneighbourhoodSizes = [0.05]
CRPtimeDelays = [3]
downSampleFactors = [4]
numFilesPerFolder = 1 # Set to None to compare all files in each folder
sequenceLengths = [300, 500, 700, 900, 1100]

#################

deleteNCDPickleFiles()

existingNCDs = None
processPool = Pool(numProcesses)

for downSampleFactor in downSampleFactors[::-1]:
    for neighbourhoodSize in CRPneighbourhoodSizes:
        for method in CRPmethods:
            for dimension in CRPdimensions:
                for timeDelay in CRPtimeDelays:
                    for sequenceLength in sequenceLengths:

                        createNCDfiles(existingNCDs, processPool,
                                       featureName, downSampleFactor,
                                       timeDelay, dimension, method, neighbourhoodSize, 
                                       numFilesPerFolder, sequenceLength)
    
#                        runDict = {'featureName': featureName,
#                                   'downSampleFactor': downSampleFactor,
#                                   'timeDelay': timeDelay, 
#                                   'dimension': dimension,
#                                   'method': method, 
#                                   'neighbourhoodSize': neighbourhoodSize,
#                                   'numFilesPerFolder': numFilesPerFolder,
#                                   'sequenceLength': sequenceLength}
#    
#                        runTime = str(datetime.now()).replace(':', '-')
#                        pickle.dump(runDict, open(runHistoryPath + runTime + '.pkl', 'wb'))
#                        convertNCDfiles(runTime)
