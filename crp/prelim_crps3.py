# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:20:29 2015

@author: Vahndi
"""

from crp import crp
import numpy as np


## mfccs
#inputPath = '/data/student/abpg162/mazurka-dataset/mazurka06-1/qm-mfcc-standard.n3_a31f2/'
#inputFn1 = 'pid9088-01_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv'
#inputFn2 = 'pid9093-01_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv'

#inputPath = '/data/student/abpg162/mazurka-dataset/mazurka06-2/qm-mfcc-standard.n3_a31f2/'
#inputFn = 'pid1263-02_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv'


# chroma
#inputPath = '/data/student/abpg162/mazurka-dataset/mazurka06-1/qm-chromagram_standard.n3_9a9b8/'
#inputFn1 = 'pid9088-01_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv'
#inputFn2 = 'pid9093-01_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv'

inputPath = '/data/student/abpg162/mazurka-dataset/mazurka06-2/qm-chromagram_standard.n3_9a9b8/'
inputFn = 'pid1263-02_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv'
 
 
#inputFiles = [inputFn1, inputFn2]
inputFiles = [inputFn]


outputPath = '/data/student/abpg162/outputs/prelim/crps/'


#methods = ['maxnorm', 'euclidean', 'minnorm', 'nrmnorm', 'rr', 'fan', 'inter']
methods = ['euclidean', 'rr', 'fan']
timeDelays = [1, 2, 3, 4 ,5]
dimensions = [1, 2, 3, 4, 5, 6, 7]


for inputFn in inputFiles:
    
    x = np.genfromtxt(inputPath + inputFn, delimiter = ',')
    
    for method in methods:
        i = 0
        print(method)
        for timeDelay in timeDelays:
            for dimension in dimensions:
                i += 1
                print(i)
                y = crp(x, dimension = dimension, timeDelay = timeDelay, method = method)
                fName = outputPath + inputFn[:-4] + '_crp_' + method + '_td_' + str(timeDelay) + '_dim_' + str(dimension) + '.bin'
                np.save(fName, y)
