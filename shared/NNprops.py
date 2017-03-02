from NNs import get_NN_WeightsAndBiases, get_NN_featureOffsetAndScaling



def NNprops(object):
    
    def __init__(self, NNtype, featureName, numFolders, numFilesPerFolder,
                 numVisible, numHidden, batchSize, 
                 learningRate, learningRateBoostFactor, corruptionLevel,
                 timeStacking, frequencyStandardisation):
                 
        self._NNtype = NNtype
        self._featureName = featureName
        self._numFolders = numFolders
        self._numFilesPerFolder = numFilesPerFolder
        self._numVisible = numVisible
        self._numHidden = numHidden
        self._batchSize = batchSize
        self._learningRate = learningRate
        self._learningRateBoostFactor = learningRateBoostFactor
        self._corruptionLevel = corruptionLevel
        self._timeStacking = timeStacking
        self._frequencyStandardisation = frequencyStandardisation

    # Property Getters    
    def getNNtype(self):        
        return self._NNtype
    def getFeatureName(self):
        return self._featureName
    def getNumFolders(self):
        return self._numFolders
    def getNumFilesPerFolder(self):
        return self._numFilesPerFolder
    def getNumVisible(self):
        return self._numVisible
    def getBatchSize(self):
        return self._batchSize
    def getLearningRate(self):
        return self._learningRate
    def getLearningRateBoostFactor(self):
        return self._learningRateBoostFactor
    def getCorruptionLevel(self):
        return self._corruptionLevel
    def getTimeStacking(self):
        return self._timeStacking
    def getFrequencyStandardisation(self):
        return self._frequencyStandardisation
        
    
    def getWeightsAndBiases(self):
        
        if self.weights is None:
            self.weights, self.biases =  get_NN_WeightsAndBiases(
                                            self._NNtype, self._featureName, 
                                             self._batchSize, self._learningRate, self._learningRateBoostFactor, 
                                             self._corruptionLevel, self._timeStacking, self._frequencyStandardisation,
                                             self._numVisible, self._numHidden)
        
        return self.weights, self.biases
        
    
    def getFeatureOffsetAndScaling(self):
        
        if self._featureOffset is None:
            self._featureOffset, self._featureScaling = get_NN_featureOffsetAndScaling(
                                                            self._featureName,
                                                            self._numFolders, self._numFilesPerFolder, self._timeStacking,
                                                            self._numVisible, self._frequencyStandardisation)
