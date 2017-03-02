# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:56:57 2015

@author: Vahndi
"""

import numpy as np


def normalise(X, doNormalisation):
    
    if X.shape[1] >= 2:
        scale = X[:, 0]
        assert (np.diff(scale) >= 0).all(), 'First column of the first vector must be monotonically non-decreasing.'
        idx = np.where(X[:, 1] != np.inf)
        if doNormalisation == True: # Slight concern here that the normalisation only takes place for the first data column - would this work for a multidimensional vector???
            Xnorm = (X[:, 1] - np.mean(X[idx, 1])) / np.std(X[idx, 1])
        else:
            Xnorm = X[:, 1]
    else:
        scale = [np.arange(X.shape[0])]
        idx = np.where(X != np.inf)
        if doNormalisation == True:
            Xnorm = (X - np.mean(X[idx])) / np.std(X[idx])
            
    return scale, Xnorm


def getDistance(X, NX, Y, NY, dimension, timeDelay):

    delays = np.arange(0, dimension * timeDelay, timeDelay)
    s = np.zeros([NX, NY, delays.size])
    for iRow in np.arange(NX):
        ix = iRow + delays
        arrX = X[ix]
        for iCol in np.arange(NY):
            iy = iCol + delays
            arrY = Y[iy]
            s[iRow, iCol, :] = arrX - arrY
    
    return s


def getDistanceNrmNorm(X, NX, Y, NY, dimension, timeDelay):

    delays = np.arange(0, dimension * timeDelay, timeDelay)
    s = np.zeros([NX, NY, delays.size])
    for iRow in np.arange(NX):
        print(iRow)
        ix = iRow + delays
        arrX = X[ix] / np.sqrt(np.power(X[ix], 2).sum())
        for iCol in np.arange(NY):
            iy = iCol + delays
            arrY = Y[iy] / np.sqrt(np.power(Y[iy], 2).sum())
            s[iRow, iCol, :] = arrX - arrY
    
    return s
    

def crp_v2(x, y = None, 
           dimension = 1, timeDelay = 1, neighbourhoodSize = 0.1,
           method = 'euclidean', normalisation = True):
    '''
    Does a recurrence plot of x (and y) and returns a recurrence matrix
    
    If y is not None then does a cross-recurrence plot\n
    If y is None the does an auto-recurrence plot\n
    
    Args:
        :x (2D numpy.ndarray): first time-series (first column is time)
        :y (2D numpy.ndarray): (optional) second time-series (first column is time)
        :dimension (int): dimension of input (not including time column)
        :timeDelay (int): time delay to use for embedding
        :neighbourhoodSize: neighbourhood size
        :method (str): Methods for finding the neighbours of the plot
            (maxnorm: maximum norm, 
            euclidean: Euclidean norm, 
            minnorm: minimum norm, 
            nrmnorm: Euclidean norm of normalized distance, 
            rr: maximum norm fixed RR, 
            fan: fixed amount of nearest neighbours, 
            inter: interdependent neighbours [not implemented], 
            omatrix: order matrix [not implemented], 
            opattern: order pattern [not implemented], 
            distance: distance plot [not implemented])
        :normalisation (boolean): whether to standardise values to mean = 0, stdev = 1
            
    Returns:
        :2D numpy array: Recurrence Matrix
        
    Changes:
        :v2:     Dropped 'inter' method.
        :v2:     Unvectorised the implementation to consume less memory in preparation for use with
                multi-dimensional feature vectors. This was accomplished by
                
                - replacing the previous embedVector() and get_s1() functions for methods ['maxnorm', 
                  'euclidean', 'minnorm', 'rr', 'fan'] with a combined getDistance() function
                - replacing the previous implementation of the 'nrmnorm' method with getDistanceNrmNorm()
    '''
    
    # Methods for finding the neighbours of plot
    # ------------------------------------------
    methods = ['maxnorm', 'euclidean', 'minnorm', 
               'nrmnorm', 'rr', 'fan']

    # Error Checks
    # ------------
    assert dimension >= 1, 'dimension must be at least 1'
    assert timeDelay >= 1, 'timeDelay must be at least 1'
    assert neighbourhoodSize >= 0, 'neighbourhoodSize must not be negative'
    assert method in methods, 'method is invalid'


    # Cross-recurrence or auto-recurrence plot (Matlab line 253)
    # ----------------------------------------------------------
    if y == None:
        y = x

    # Check that matrices are right way up (Matlab line 288)
    # ------------------------------------------------------
    if x.shape[1] > x.shape[0]:
        x = x.transpose()
    if y.shape[1] > y.shape[0]:
        y = y.transpose()

    # Check for and delete rows with nans (Matlab line 279)
    # -----------------------------------------------------
    if True in np.isnan(x).any(axis = 1):
        print('Warning: NaN detected in x - time slice will be removed.')
        x = x[~np.isnan(x).any(axis = 1)]
    if True in np.isnan(y).any(axis = 1):
        print('Warning: NaN detected in x - time slice will be removed.')
        y = y[~np.isnan(y).any(axis = 1)]

    # Calculate embedding vectors lengths (Matlab line 291)
    # -----------------------------------------------------
    NX = x.shape[0] - timeDelay * (dimension - 1)
    assert NX >= 1, 'The embedding vectors cannot be created: dimension and / or timeDelay are too big. Please use smaller values.'
    NY = y.shape[0] - timeDelay * (dimension - 1)
    assert NY >= 1, 'The embedding vectors cannot be created: dimension and / or timeDelay are too big. Please use smaller values.'

    # Normalise the data (Matlab line 297)
    # ------------------------------------
    xScale, x = normalise(x, normalisation)
    yScale, y = normalise(y, normalisation)

    # Computation
    # -----------

    X = np.zeros([NY, NX], np.uint8)

    # local CRP, fixed distance (Matlab line 634)
    if method in ['maxnorm', 'euclidean', 'minnorm', 'rr']:
        
        s1 = getDistance(x, NX, y, NY, dimension, timeDelay)
        
        if method == 'maxnorm':
            # maximum norm (Matlab line 649)
            s = np.abs(s1).max(axis = 2) # take the maximum of the absolute distances across the dimension axis
            
        elif method == 'rr':
            # maximum norm, fixed RR (Matlab line 653)
            s = np.abs(s1).max(axis = 2) # take the maximum of the distances across the dimension axis
            ss = np.sort(s.reshape([s.shape[0] * s.shape[1], 1]), axis = 0) # ss = convert s to vector and sort
            idx = np.ceil(neighbourhoodSize * ss.size) 
            neighbourhoodSize = ss[idx - 1] # set neighbourhood size to be the value at neighbourhoodSize * length of ss along ss
            
        elif method == 'euclidean':
            # euclidean norm (Matlab line 662)
            s = np.sqrt(np.power(s1, 2).sum(axis = 2)) # take the square root of the sum of the squared distances across the dimension axis
            
        elif method == 'minnorm':
            # minimum norm (Matlab line 666)
            s = np.abs(s1).sum(axis = 2) # take the sum of the absolute distances across the dimension axis (should this be min not sum?!?!)
            
        
        X2 = s < neighbourhoodSize
        X = np.uint8(X2)
        
    # local CRP, normalized distance euclidean norm (Matlab line 679)
    elif method == 'nrmnorm':
        
        s1 = getDistanceNrmNorm(x, NX, y, NY, dimension, timeDelay)
        s = np.sqrt(np.power(s1, 2).sum(axis = 2))
        X = np.uint8(s / s.max() < neighbourhoodSize / s.max())
        del s, s1
        
    # local CRP, fixed amount of neigbours (Matlab line 710)
    elif method == 'fan':
        
        assert neighbourhoodSize < 1, 'The value for fixed neigbours amount has to be smaller than one.'
        
        s1 = getDistance(x, NX, y, NY, dimension, timeDelay)
        s = np.power(s1, 2).sum(axis = 2)
        minNS = round(NY * neighbourhoodSize)
        JJ = np.argsort(s, axis = 1)
        X1 = np.zeros(NX * NY, dtype = np.uint8)
        X1[JJ[:, 0: minNS] + np.tile(np.matrix(np.arange(0, NX * NY, NY)).transpose(), [1, minNS])] = np.uint8(1)
        X = X1.reshape(NY, NX).transpose()
        del s, s1, JJ, X1
        

    return X
    
    

        