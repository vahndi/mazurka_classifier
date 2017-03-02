# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 11:52:00 2015

@author: Vahndi
"""

import pandas as pd
from numpy import float64, random, inf



class Optimiser(object):
    
    
    def __init__(self, categoricalSettings, 
                 oldResultsDataFrame = None, resultsColumn = 'Results', 
                 noImprovementStoppingRounds = None, floatRounding = 3):
        '''
        Arguments:
            :categoricalSettings: a {dict of settingName, [setting1, setting2, ...]}
            :oldResultsDataFrame: a pandas DataFrame of existing results to use in the selection of new settings
            :resultsColumn:     the name of the results column in the results DataFrame
            :floatRounding:     the number of decimal places to round float values to
        '''        
        self.categoricalSettings = categoricalSettings
        self.resultsColumn = resultsColumn
        self.resultsDataFrame = oldResultsDataFrame
        self.floatRounding = floatRounding
        self.categories = sorted(list(self.categoricalSettings.keys()))
        self.numCategories = len(self.categories)
        self.currentCategoryIndex = 0
        self.noImprovementStoppingRounds = noImprovementStoppingRounds
        
        # Initialise current settings to random values
        self.initialiseRandomSettings()
    

    def initialiseRandomSettings(self):
        '''
        Randomly set the settings to different values
        '''
        
        self.roundsNoImprovement = 0
        self.currentSettings = {}
        for category in self.categories:
            self.currentSettings[category] = Optimiser._getRandomValue(self.categoricalSettings[category])


    @classmethod
    def _getRandomValue(cls, fromList):
        
        return fromList[random.randint(len(fromList))]
        
    
    def isFinished(self):
        
        return False


    def hasResultFor(self, settings):
        
        if self.resultsDataFrame is None:
            return False
        else:
            dfSub = self.resultsDataFrame
            for category in settings.keys():
                categoryValue = settings[category]
                dfSub = dfSub[dfSub[category] == categoryValue]
            return dfSub.shape[0] > 0
            

    def getNextSettings(self):
        '''
        Returns a list of settings to try next
        '''
        if self.noImprovementStoppingRounds is not None:
            if self.roundsNoImprovement == self.noImprovementStoppingRounds:
                return None

        # Get a list of settings across the dimension of the current category
        nextSettings = []
        numCategoriesTried = 0
        # Loop until some new settings have been acquired or all categories have been tried
        while not nextSettings and numCategoriesTried < self.numCategories:
            
            loopCategory = self.categories[self.currentCategoryIndex]
            for val in self.categoricalSettings[loopCategory]:
                setting = {}
                for category in self.categories:
                    if category == loopCategory:
                        setting[category] = val
                    else:
                        setting[category] = self.currentSettings[category]
                nextSettings.append(setting)
            
            # Remove any settings which already have results for
            nonDuplicates = []
            for setting in nextSettings:
                if not self.hasResultFor(setting):
                    nonDuplicates.append(setting)
            nextSettings = nonDuplicates
            
            # Update loop and category parameters
            numCategoriesTried += 1
            self.currentCategoryIndex += 1
            self.currentCategoryIndex = self.currentCategoryIndex % self.numCategories
            
        # Return the list of settings or None if the run is finished
        if not nextSettings:
            return None
        else:
            self._currentSettingsParameter = self.categories[(self.currentCategoryIndex - 1) % self.numCategories]
            return nextSettings


    def getCurrentSettingsParameter(self):
        
        return self._currentSettingsParameter
        
        
    def currentBestResult(self):
        
        if self.resultsDataFrame is None:
            return -inf
        else:
            return self.resultsDataFrame[self.resultsColumn].max()

    
    def addResults(self, resultsDataFrame):
        '''
        Adds a list of results to the existing results and changes the current
        settings to the best result so far
        '''
        # Add results to any existing results
        if self.resultsDataFrame is None:
            self.resultsDataFrame = resultsDataFrame
        else:
            # Check for improvement
            if self.currentBestResult() >= resultsDataFrame[self.resultsColumn].max():
                self.roundsNoImprovement += 1
            else:
                self.roundsNoImprovement = 0
                
            # Merge results with existing results
            self.resultsDataFrame = pd.concat([self.resultsDataFrame, resultsDataFrame], ignore_index=True)

        df = self.resultsDataFrame
        
        # Get the best result so far and change the current settings to the settings that produced it
        bestResult = df[self.resultsColumn].max()
        bestSettingsRow = df[df[self.resultsColumn] == bestResult].iloc[0]
        for category in self.categories:
            categoryValue = bestSettingsRow[category]
            if type(categoryValue) in (float64, float):
                categoryValue = round(categoryValue, self.floatRounding)
                if categoryValue == round(categoryValue, 0):
                    categoryValue = int(round(categoryValue, 0))

            self.currentSettings[category] = categoryValue


    def saveResults(self, filePath):
        
        self.resultsDataFrame.to_csv(filePath)