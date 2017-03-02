# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:34:07 2015

@author: Vahndi
"""

#runfile('C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code/dA_jazz_piano_12.py', wdir='C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code')
#runfile('C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code/dA_jazz_piano_13.py', wdir='C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code')
#runfile('C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code/dA_jazz_piano_14.py', wdir='C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code')
#runfile('C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code/dA_jazz_piano_15.py', wdir='C:/Users/Vahndi/Google Drive/MSc Data Science/Project/git code')

from rbm_jazz_piano_1 import test_rbm

featuresPath = 'C:\\Users\\Vahndi\\Google Drive\\MSc Data Science\\Project\\features\\'

# Chromagrams
test_rbm(dataset = featuresPath + 'jazz_piano_chromagrams.pkl.gz',
         n_visible = 12, n_hidden = 12, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_chromagrams.pkl.gz',
         n_visible = 12, n_hidden = 9, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_chromagrams.pkl.gz',
         n_visible = 12, n_hidden = 6, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_chromagrams.pkl.gz',
         n_visible = 12, n_hidden = 3, training_epochs = 10)

# MFCCs
test_rbm(dataset = featuresPath + 'jazz_piano_mfccs.pkl.gz',
         n_visible = 20, n_hidden = 20, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_mfccs.pkl.gz',
         n_visible = 20, n_hidden = 15, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_mfccs.pkl.gz',
         n_visible = 20, n_hidden = 10, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_mfccs.pkl.gz',
         n_visible = 20, n_hidden = 5, training_epochs = 10)
         
# Spectrograms
test_rbm(dataset = featuresPath + 'jazz_piano_spectrograms.pkl.gz',
         n_visible = 48, n_hidden = 48, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_spectrograms.pkl.gz',
         n_visible = 48, n_hidden = 36, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_spectrograms.pkl.gz',
         n_visible = 48, n_hidden = 24, training_epochs = 10)
test_rbm(dataset = featuresPath + 'jazz_piano_spectrograms.pkl.gz',
         n_visible = 48, n_hidden = 12, training_epochs = 10)
