# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:41:14 2015

@author: vahndi
"""

import numpy as np

songPath = '/data/student/abpg162/mazurka-dataset/mazurka06-1/'

logfreqspecPath = 'nnls-logfreqspec_standard.n3_c509a/'
chromagramPath = 'qm-chromagram_standard.n3_9a9b8/'
constantqPath = 'qm-constantq_standard.n3_2225e/'
mfccPath = 'qm-mfcc-standard.n3_a31f2/'

logfreqspecFn = 'pid1263-01_vamp_nnls-chroma_nnls-chroma_logfreqspec.csv' # 256
chromagramFn = 'pid1263-01_vamp_qm-vamp-plugins_qm-chromagram_chromagram.csv' # 12
constantqFn = 'pid1263-01_vamp_qm-vamp-plugins_qm-constantq_constantq.csv' # 48
mfccFn = 'pid1263-01_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv' # 20

logfreqspec = np.genfromtxt(songPath + logfreqspecPath + logfreqspecFn, delimiter = ',')
print logfreqspec.shape
print logfreqspec[:, 0]

chromagram = np.genfromtxt(songPath + chromagramPath + chromagramFn, delimiter = ',')
print chromagram.shape
print chromagram[:, 0]

constantq = np.genfromtxt(songPath + constantqPath + constantqFn, delimiter = ',')
print constantq.shape
print constantq[:, 0]

mfcc = np.genfromtxt(songPath + mfccPath + mfccFn, delimiter = ',')
print mfcc.shape
print mfcc[:, 0]

dimensions = [1, 2, 3, 4, 5, 6, 7]
timeDelays = [1, 2, 3, 4, 5]

