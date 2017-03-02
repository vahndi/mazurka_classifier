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
            X = (X[:, 1] - np.mean(X[idx, 1])) / np.std(X[idx, 1])
        else:
            X = X[:, 1]
    else:
        scale = [np.arange(X.shape[0])]
        idx = np.where(X != np.inf)
        if doNormalisation == True:
            X = (X - np.mean(X[idx])) / np.std(X[idx])
            
    return scale, X


def embedVector(X, NX, dimension, timeDelay):
    
    i = np.arange(NX)
    j = np.arange(0, dimension * timeDelay, timeDelay)
    iTiled = np.tile(i, [dimension, 1]).transpose()
    jTiled = np.tile(j, [NX, 1])
    ijTiled = iTiled + jTiled
    ij = ijTiled.reshape([dimension * NX, 1])
    X_ij = X[ij]
    XEmbedded = X_ij.reshape([NX, dimension])
    return XEmbedded


def get_s1(x2, y2, NX, NY):
    
    px = np.expand_dims(x2, axis = 2).transpose(0, 2, 1)
    py = np.expand_dims(y2, axis = 2).transpose(2, 0, 1)
    pxTiled = np.tile(px, [1, NY, 1])
    pyTiled = np.tile(py, [NX, 1, 1])
    s1 = pxTiled - pyTiled
    return s1
    

def crp(x, y = None, 
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
            inter: interdependent neighbours, 
            omatrix: order matrix [not implemented], 
            opattern: order pattern [not implemented], 
            distance: distance plot [not implemented])
        :normalisation (boolean): whether to standardise values to mean = 0, stdev = 1
            
    Returns:
        :2D numpy array: Recurrence Matrix
    '''
    
    # Methods for finding the neighbours of plot
    # ------------------------------------------
    methods = ['maxnorm', 'euclidean', 'minnorm', 
               'nrmnorm', 'rr', 'fan', 'inter']

    # Error Checks
    # ------------
    assert dimension >= 1, 'dimension must be at least 1'
    assert timeDelay >= 1, 'timeDelay must be at least 1'
    assert neighbourhoodSize >= 0, 'neighbourhoodSize must not be negative'
    assert method in methods, 'method is invalid'
    if method == 'omatrix':
        assert dimension == 1, 'for order matrix method, a dimension of 1 must be used'
    if method == 'opattern':
        assert dimension > 1, 'for order pattern method, a dimension larger than 1 must be used'


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

    ds = np.eye(dimension)

    # Computation
    # -----------

    # Embedding Vectors (Matlab line 590)
    x2 = embedVector(x, NX, dimension, timeDelay)
    x2 = np.matrix(x2)
    y2 = embedVector(y, NY, dimension, timeDelay)
    y2 = np.matrix(y2) * np.matrix(ds)
    
   
    NX, mx = x2.shape
    NY, my = y2.shape
    X = np.zeros([NY, NX], np.uint8)
    
    # local CRP, fixed distance (Matlab line 634)
    if method in ['maxnorm', 'euclidean', 'minnorm', 'rr']:
        
        s1 = get_s1(x2, y2, NX, NY)
        
        if method == 'maxnorm':
            # maximum norm (Matlab line 649)
            s = np.max(np.abs(s1), axis = 2)
        elif method == 'rr':
            # maximum norm, fixed RR (Matlab line 653)
            s = np.max(np.abs(s1), axis = 2)
            ss = np.sort(s.reshape([s.shape[0] * s.shape[1], 1]), axis = 0)
            idx = np.ceil(neighbourhoodSize * ss.size)
            neighbourhoodSize = ss[idx - 1]
        elif method == 'euclidean':
            # euclidean norm (Matlab line 662)
            s = np.sqrt(np.power(s1, 2).sum(axis = 2))
        elif method == 'minnorm':
            # minimum norm (Matlab line 666)
            s = np.abs(s1).sum(axis = 2)
        
        X2 = s < neighbourhoodSize
        X = np.uint8(X2)
        
    # local CRP, normalized distance euclidean norm (Matlab line 679)
    elif method == 'nrmnorm':
        
        Dx = np.sqrt(np.power(x2, 2).sum(axis = 1))
        Dy = np.sqrt(np.power(y2, 2).sum(axis = 1))
        x1 = x2 / np.tile(Dx, [1, dimension])
        x2 = x1
        y1 = y2 / np.tile(Dy, [1, dimension])
        y2 = y1
        del Dx, Dy, y1, x1
        
        s1 = get_s1(x2, y2, NX, NY)
        s = np.sqrt(np.power(s1, 2).sum(axis = 2))
        X = np.uint8(s / s.max() < neighbourhoodSize / s.max())
        del s, s1
        
    # local CRP, fixed amount of neigbours (Matlab line 710)
    elif method == 'fan':
        
        assert neighbourhoodSize < 1, 'The value for fixed neigbours amount has to be smaller than one.'
        
        s1 = get_s1(x2, y2, NX, NY)
        s = np.power(s1, 2).sum(axis = 2)
        minNS = round(NY * neighbourhoodSize)
        JJ = np.argsort(s, axis = 1)
        X1 = np.zeros(NX * NY, dtype = np.uint8)
        X1[JJ[:, 0: minNS] + np.tile(np.matrix(np.arange(0, NX * NY, NY)).transpose(), [1, minNS])] = np.uint8(1)
        X = X1.reshape(NY, NX).transpose()
        
    # local CRP, interdependent neigbours (Matlab line 756)
    elif method == 'inter':
        
        assert neighbourhoodSize < 1, 'The value for fixed neigbours amount has to be smaller than one.'
        px = np.expand_dims(x2, axis = 2).transpose(0, 2, 1)
        py = np.expand_dims(y2, axis = 2).transpose(0, 2, 1)
        px2 = np.expand_dims(x2, axis = 2).transpose(2, 0, 1)
        py2 = np.expand_dims(y2, axis = 2).transpose(2, 0, 1)
        
        pxTiled = np.tile(px, [1, NX, 1])
        px2Tiled = np.tile(px2, [NX, 1, 1])
        s1 = pxTiled - px2Tiled
        sx = np.sqrt(np.power(s1, 2).sum(axis = 2))
        
        pyTiled = np.tile(py, [1, NY, 1])
        py2Tiled = np.tile(py2, [NY, 1, 1])
        s1 = pyTiled - py2Tiled
        sy = np.sqrt(np.power(s1, 2).sum(axis = 2))

        minNS = round(min(NX, NY) * neighbourhoodSize)
        
        SSx = np.sort(sx, axis = 0)
        JJx = np.argsort(sx, axis = 0)
        SSy = np.sort(sy, axis = 0)
        JJy = np.argsort(sy, axis = 0)
        ey = np.mean(SSy[minNS - 1:minNS + 1, :], axis = 0)
        
        for i in np.arange(min(NX, NY)):
            JJx[(JJx[0: minNS, i] > min(NX, NY)), i]= i + 1
            JJy[(JJy[0: minNS, i] > min(NX, NY)), i]= i + 1
            X[i, JJx[0: minNS, i]] = np.transpose(sy[i, JJx[0: minNS, i]] <= ey[i])
        
        X = X.transpose()

    return X
    
    

        