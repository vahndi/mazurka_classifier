# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 14:35:02 2015

@author: Vahndi
"""

import numpy as np



class ListStepper(object):
    '''
    A class to randomly step along the elements of a list
    '''
    def __init__(self, listValues, initialValue = None):
        
        self._listValues = listValues
        self._numValues = len(listValues)
        if initialValue is not None and initialValue in listValues:
            self._initialValue = initialValue
            self._currentValue = initialValue
            self._adjustedValue = initialValue
        else:
            initialIndex = np.rand.randint(self._numValues)
            self._initialValue = self._listValues[initialIndex]
            self._currentValue = self._initialValue
            self._adjustedValue = self._initialValue
        
        self._numUnsuccessfulAdjustments = 0
        self._lastAdjustmentSuccess = False
        self._currentAdjustment = None

    
    def getCurrentValue(self):
        
        return self._currentValue
        
    
    def getAdjustedValue(self):
        
        currentIndex = self._listValues.index(self._currentValue)
        if currentIndex == 0:
            self._currentAdjustment = np.random.randint(2)
        elif currentIndex == self._numValues - 1:
            self._currentAdjustment = - np.random.randint(2)
        else:
            self._currentAdjustment = np.random.randint(3) - 1

        self._adjustedValue = self._listValues[currentIndex + self._currentAdjustment]
        return self._adjustedValue


    def keepCurrentValue(self):
        
        self._numUnsuccessfulAdjustments += 1
        self._lastAdjustmentSuccess = False
        
    
    def keepAdjustedValue(self):
        
        self._numUnsuccessfulAdjustments = 0
        self._currentValue = self._adjustedValue
        self._lastAdjustmentSuccess = True
        
    
    def getNumUnsuccessulAdjustments(self):
        
        return self._numUnsuccessfulAdjustments



class ValueAdjuster(object):
    
    
    def __init__(self, initialValue, minValue = None, maxValue = None, maxAdjustment = 0.001):
        
        self._initialValue = initialValue
        self._currentValue = initialValue
        self._adjustedValue = initialValue
        self._numUnsuccessfulAdjustments = 0
        self._lastAdjustmentSuccess = False
        self._currentAdjustment = None
        self._minValue = minValue
        self._maxValue = maxValue
        self._maxAdjustment = maxAdjustment
        
        
    def getCurrentValue(self):
        
        return self._currentValue
        
        
    def getAdjustedValue(self):
        
        if not self._lastAdjustmentSuccess or self._currentAdjustment is None:
            self._currentAdjustment = (np.random.rand() - 0.5) * (2 * self._maxAdjustment)
            if self._minValue is not None:
                while self._currentValue + self._currentAdjustment < self._minValue:
                    self._currentAdjustment = (np.random.rand() - 0.5) * (2 * self._maxAdjustment)
            if self._maxValue is not None:
                while self._currentValue + self._currentAdjustment > self._maxValue:
                    self._currentAdjustment = (np.random.rand() - 0.5) * (2 * self._maxAdjustment)
        
        self._adjustedValue = self._currentValue + self._currentAdjustment
        if self._minValue is not None:
            if self._adjustedValue < self._minValue:
                self._adjustedValue = self._minValue
        if self._maxValue is not None:
            if self._adjustedValue > self._maxValue:
                self._adjustedValue = self._maxValue
                
        return self._adjustedValue


    def keepCurrentValue(self):
        
        self._numUnsuccessfulAdjustments += 1
        self._lastAdjustmentSuccess = False
        
    
    def keepAdjustedValue(self):
        
        self._numUnsuccessfulAdjustments = 0
        self._currentValue = self._adjustedValue
        self._lastAdjustmentSuccess = True
        
    
    def getNumUnsuccessulAdjustments(self):
        
        return self._numUnsuccessfulAdjustments
        
        

class ArrayAdjuster(object):
    
    
    def __init__(self, initialArray, maxAdjustment = 0.001):
        '''
        Arguments:
            :initialArray: a numpy array or matrix
        '''
        self._initialArray = initialArray
        self._arrayShape = initialArray.shape
        self._currentArray = initialArray
        self._adjustedArray = initialArray # needed for initial call to keepAdjustedArray in finetune_weights.py
        self._numUnsuccessfulAdjustments = 0
        self._lastAdjustmentSuccess = False
        self._currentAdjustments = None
        self._maxAdjustment = maxAdjustment


    def getCurrentArray(self):
            
        return self._currentArray
        
    
    def getAdjustedArray(self):
        
        if not self._lastAdjustmentSuccess or self._currentAdjustments is None:
            self._currentAdjustments = (np.random.random_sample(self._arrayShape) - 0.5) * (2 * self._maxAdjustment)
            
        self._adjustedArray = self._currentArray + self._currentAdjustments
        return self._adjustedArray
    
    
    def getAdjustmentMean(self):
        
        if self._currentAdjustments is not None:
            return self._currentAdjustments.mean()
        
        return 0.0
        
        
    def getAdjustmentStandardDeviation(self):

        if self._currentAdjustments is not None:        
            return self._currentAdjustments.std()
        return 0.0
    
    
    def keepCurrentArray(self):
        
        self._numUnsuccessfulAdjustments += 1
        self._lastAdjustmentSuccess = False
        
    
    def keepAdjustedArray(self):
        
        self._numUnsuccessfulAdjustments = 0
        self._currentArray = self._adjustedArray
        self._lastAdjustmentSuccess = True
        
    
    def getNumUnsuccessulAdjustments(self):
        
        return self._numUnsuccessfulAdjustments
        
