# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 00:34:57 2015

@author: vahndi
"""

class MAPcalcProps(object):
    
    
    def __init__(self, initDict):
        
        self._dict = initDict
        self._keys = initDict.keys()
        
    
    def _checkKeyExists(self, key):
        assert key in self._keys, 'Key %s does not exist in the MAPcalcProps dictionary'
        
        
    def getValue(self, key):
        
        self._checkKeyExists(key)
        return self._dict[key]

    def addKeyValue(self, key, value):
        
        self._keyIndex[0] = key
        

        
    def setValue(self, key, value):
        
        self._checkKeyExists(key)
        self._dict[key] = value        
        
