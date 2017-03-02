# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:29:19 2015

@author: Vahndi
"""

# Get all results for a given downSampleFactor

from ncd_processing import getNCDresults
from evaluation import getMeanAveragePrecision
import pandas as pd

featureName = 'CENS_dsf2_wl41'
#CRPmethods = ['maxnorm', 'euclidean', 'minnorm', 'rr', 'fan', 'nrmnorm']
CRPmethods = ['fan']
CRPdimensions = [3]
CRPneighbourhoodSizes = [0.1]
CRPtimeDelays = [5]
downSampleFactors = [1, 2, 4, 8, 12]
numFilesPerFolder = 5 # Set to None to compare all files in each folder
sequenceLengths = [300, 500, 700, 900, 1100]
subFolder = 'run13'
outputFn = subFolder + ' MAPs.pkl' 

meanAveragePrecisions = []
i = 0
for method in CRPmethods:
    for dimension in CRPdimensions:
        for nhSize in CRPneighbourhoodSizes:
            for timeDelay in CRPtimeDelays:
                for downSampleFactor in downSampleFactors:
                    for sequenceLength in sequenceLengths:
                        
                        print i
                        # Get results for run configuration
                        try:
                            df = getNCDresults(subFolder = subFolder,
                                               featureNames = [featureName],
                                               downSampleFactors = [downSampleFactor],
                                               methods = [method],
                                               dimensions = [dimension],
                                               timeDelays = [timeDelay],
                                               neighbourhoodSizes = [nhSize],
                                               numFilesPerFolder = [numFilesPerFolder],
                                               sequenceLengths = [sequenceLength])
                            # Calculate MAP
                            meanAveragePrecision = getMeanAveragePrecision(df)
                            
                            # Add to list of MAPs
                            meanAveragePrecisions.append({'Feature Name': featureName,
                                                          'DownSample Factor': downSampleFactor,
                                                          'CRP Method': method,
                                                          'Dimension': dimension,
                                                          'Time Delay': timeDelay,
                                                          'Neighbourhood Size': nhSize,
                                                          'Files per Folder': numFilesPerFolder,
                                                          'Mean Average Precision': meanAveragePrecision,
                                                          'Sequence Length': sequenceLength})
                        except:
                            pass
                        i += 1

# Create dataframe of MAPs and save
dfMeanAveragePrecisions = pd.DataFrame(meanAveragePrecisions)
dfMeanAveragePrecisions.to_pickle('./' + outputFn)
print 'Maximum MAP: %s' % str(dfMeanAveragePrecisions['Mean Average Precision'].max())
            