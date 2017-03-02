# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:18:22 2015

@author: Vahndi
"""

import pandas as pd, os
from featureConverter import chromaToCENS
from paths import getFileNames, getFolderNames

########## Settings ##########

windowLengths = [41]
downsampleFactor = 10

##############################


piecesPath = '/data/student/abpg162/mazurka-dataset/'
piecesFolders = getFolderNames(piecesPath, contains = 'mazurka') # added the contains parameter to avoid the new powerspectrum folder

for windowLength in windowLengths:

    # Create root path for CENS features
    CENSrootPath = '/data/student/abpg162/outputs/new_features/CENS_dsf%i_wl%i/' \
                    % (downsampleFactor, windowLength)
    if not os.path.exists(CENSrootPath):
        print 'Creating Root Path for CENS files...'
        os.makedirs(CENSrootPath)
    
    # For each piece
    for piecesFolder in piecesFolders:
    
        # Create folder for CENS files if it doesn't exist
        CENSpath = CENSrootPath + piecesFolder + '/'
        if os.path.exists(CENSpath):
            print 'CENS folder already exists...'
        else:
            print 'making CENS folder for %s' %piecesFolder
            os.makedirs(CENSpath)
    
        # Get performances of the piece
        chromaFolder = getFolderNames(piecesPath + piecesFolder + '/', contains = 'qm-chromagram_standard')[0]
        chromaPath = piecesPath + piecesFolder + '/' + chromaFolder + '/'
        print 'Looking for performances in %s...' % chromaPath
        performances = getFileNames(chromaPath, endsWith = '.csv')
        print 'found %i performances' %len(performances)
    
        # Create CENS files of each performance
        for performance in performances:
            CENSfn = CENSpath + performance.split('_')[0] + '_CENS_dsf_%i_dswl_%i.csv' % (downsampleFactor, windowLength)
            print 'converting %s' %performance
            dfChroma = pd.read_csv(chromaPath + performance, header = None, index_col = 0)
            dfCENS = chromaToCENS(dfChroma, downSampleFactor = downsampleFactor, downSampleWindowLength = windowLength)
            dfCENS.to_csv(CENSfn, header = False)
