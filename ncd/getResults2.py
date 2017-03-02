# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:29:19 2015

@author: Vahndi
"""

# Get all results for a given downSampleFactor

from ncd_processing import getNCDresults
from evaluation import getMeanAveragePrecision

def getMAPresult(featureName, method, dimension, neighbourhoodSize, timeDelay, 
                 downSampleFactor, numFilesPerFolder, sequenceLength, subFolder):
    '''
    Calculates and returns the MAP for the results matching the arguments
    '''
    try:
        # Get relevant results
        df = getNCDresults(subFolder = subFolder,
                           featureNames = [featureName],
                           downSampleFactors = [downSampleFactor],
                           methods = [method],
                           dimensions = [dimension],
                           timeDelays = [timeDelay],
                           neighbourhoodSizes = [neighbourhoodSize],
                           numFilesPerFolder = [numFilesPerFolder],
                           sequenceLengths = [sequenceLength])
        # Calculate MAP
        meanAveragePrecision = getMeanAveragePrecision(df)
        
        # Return MAP
        return meanAveragePrecision
    
    except:
        return None


def getDataFrameMAPresult(fromDataFrame):
    '''
    Calculates and returns the MAP result for a single dataframe
    '''
    try:
        # Calculate MAP
        meanAveragePrecision = getMeanAveragePrecision(fromDataFrame)
        
        # Return MAP
        return meanAveragePrecision
    
    except:
        return None
