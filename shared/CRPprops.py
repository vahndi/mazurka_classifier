import os
from paths import CRPpath
from scipy import signal
import numpy as np

from FeatureFileProps import FeatureFileProps as FFP
from crp_v4 import crp_v4


class CRPprops(object):
    '''
    Holds basic properties about a Cross Recurrence Plot
    '''

    @classmethod
    def toTuples(cls, lstCRPprops):

        lstTuples = []
        for crpProps in lstCRPprops:
            lstTuples.append((crpProps.pieceId, crpProps.performanceId, 
                              crpProps.method, crpProps.dimension, crpProps.timeDelay,
                              crpProps.neighbourhoodSize, crpProps.downSampleFactor,
                              crpProps.sequenceLength,
                              crpProps.featureFilePath))

        return lstTuples


    @classmethod
    def toCRPprops(cls, lstCRPtuples):

        lstCRPprops = []
        for crpTuple in lstCRPtuples:
            lstCRPprops.append(CRPprops(crpTuple[0], crpTuple[1], 
                                        crpTuple[2], crpTuple[3], crpTuple[4], 
                                        crpTuple[5], crpTuple[6], 
                                        crpTuple[7], 
                                        crpTuple[8]))

        return lstCRPprops


    @classmethod
    def uniqueCRPprops(cls, lstCRPprops):

        tuples = CRPprops.toTuples(lstCRPprops)
        tuples = list(set(tuples))
        props = CRPprops.toCRPprops(tuples)
        return props


    def __init__(self, pieceId, performanceId, 
                       method, dimension, timeDelay, neighbourhoodSize, downSampleFactor, 
                       sequenceLength,
                       featureFilePath = None):

        self.pieceId = pieceId
        self.performanceId = performanceId
        self.method = method
        self.dimension = dimension
        self.timeDelay = timeDelay
        self.neighbourhoodSize = neighbourhoodSize
        self.downSampleFactor = downSampleFactor
        self.sequenceLength = sequenceLength
        self.featureFilePath = featureFilePath

        # Neural network parameters
        self.weightMatrix = None
        self.biases = None
        self.featureOffset = 0.0
        self.featureScaling = 1.0
        self.timeStacking = None
        self.numFeatures = None
        
        self.featureFileData = None
        self.featureFileDataFrame = None
        self.CRPdata = None


    def toTuple(self, includeFeatureFilePath = True):
        
        if includeFeatureFilePath:
            return (self.pieceId, self.performanceId, 
                    self.method, self.dimension, self.timeDelay, 
                    self.neighbourhoodSize, self.downSampleFactor,
                    self.sequenceLength,
                    self.featureFilePath)
        else:
            return (self.pieceId, self.performanceId, 
                    self.method, self.dimension, self.timeDelay, 
                    self.neighbourhoodSize, self.downSampleFactor,
                    self.sequenceLength)


    def getFileName(self):
        
        crpFileName = 'CRP_pcId_%s_pfId_%s_mthd_%s_dimn_%s_tdly_%s_nsz_%s_dsf_%s_sqln_%s' \
                      % (self.pieceId, self.performanceId, 
                         self.method, str(self.dimension), str(self.timeDelay), 
                         str(self.neighbourhoodSize), str(self.downSampleFactor),
                         str(self.sequenceLength))

        return crpFileName


    def hasExistingFile(self):

        if os.path.exists(CRPpath + self.getFileName() + '.npy'):
            return True
        else:
            return False
            
    
    @classmethod
    def downSampleCRP(cls, CRP, numFrames):
        '''
        Used to downsample a CRP before calculating the NCD between two CRPs for fixed
        # frames implementation. Use 'var' for numFrames to return the original CRP
        N.B this implementation has been found to sometimes return 2 instead of 1 (0.1% of the time)!!
        '''
        if numFrames == 'var':
            return CRP
        else:
            # Resample along each axis
            CRP_ds1 = signal.resample(CRP, numFrames, axis = 0)
            CRP_ds2 = signal.resample(CRP_ds1, numFrames, axis = 1)
            
            # Convert back to zero or one
            CRP_ds3 = np.array(CRP_ds2 + 0.5, dtype = np.uint8)
            return CRP_ds3
            
            
    @classmethod
    def downSampleCRPmax(cls, CRP, numFrames):
        '''
        Used to downsample a CRP before calculating the NCD between two CRPs for fixed
        # frames implementation. Use 'var' for numFrames to return the original CRP.
        Uses a max filter to favour ones over zeros
        '''
        if numFrames == 'var':
            return CRP
        else:
            newCRP = np.zeros([numFrames, numFrames], dtype = np.uint8)
            filterSize = (int(np.ceil(float(CRP.shape[0]) / numFrames)), int(np.ceil(float(CRP.shape[1]) / numFrames)))
            rowArr = np.linspace(0, CRP.shape[0] - filterSize[0], numFrames, dtype= np.int)
            colArr = np.linspace(0, CRP.shape[1] - filterSize[1], numFrames, dtype= np.int)
            for iRow in range(numFrames):
                for iCol in range(numFrames):
                    newCRP[iRow, iCol] = np.max(CRP[rowArr[iRow]: rowArr[iRow] + filterSize[0], 
                                                    colArr[iCol]: colArr[iCol] + filterSize[1]])
            
            return newCRP
    
    
    def calculateCRP(self):
        '''
        Calculates a CRP for the given file and CRP settings
        Returns True if successful or False if there was an error
        Assumptions:
            The feature file has no header row
            The feature file has a single time column before the feature value columns
        '''
        if self.featureFileData is None:
            # Load file
            try:
                print 'Loading feature data from file...'
                x = np.genfromtxt(self.featureFilePath, delimiter = ',')
            except:
                print 'Error opening feature file for CRP calculation'
        else:
            x = self.featureFileData
            
        try:
            # Crop number of features if necessary
            if self.numFeatures is not None:
                x = x[:, :self.numFeatures + 1]
            # Transform by weight matrix and add bias
            if self.weightMatrix is not None:
                # split the time and data columns
                x_t = np.expand_dims(x[:,0], 1)
                x_data = x[:, 1:]
                # horizontally stack the data array
                if self.timeStacking is not None and self.timeStacking != 1:
                    timeStacking = self.timeStacking
                    numExamples = x_data.shape[0]
                    numFeatures= x_data.shape[1]
                    x_dataNew = np.zeros([numExamples - timeStacking + 1, 
                                          timeStacking * numFeatures])
                    for ts in range(timeStacking):
                        x_dataNew[:, ts * numFeatures : (ts + 1) * numFeatures] = x_data[ts: numExamples + 1 + ts - timeStacking, :]
                    x_data = x_dataNew
                    x_t = x_t[0: 1 + numExamples - timeStacking, :]
                # standardise feature scaling
                x_data = (x_data + self.featureOffset) * self.featureScaling
                # multiply by weights and add biases
                xmat_transformed = np.matrix(x_data) * self.weightMatrix
                xarr_transformed = np.array(xmat_transformed) + np.tile(self.biases.transpose(), [x_data.shape[0], 1])
                # add back on the time component !!! check that time vector is the right length now !!!
                x_transformed = np.append(x_t, xarr_transformed, axis = 1)
            else:
                x_transformed = x
                
            # Downsample
            x_ds = FFP.downSample(x_transformed, self.downSampleFactor)
            
            # Resize
            x_ds = FFP.resize(x_ds, self.sequenceLength)
            
            # Calculate CRP
            y = crp_v4(x_ds, dimension = self.dimension, timeDelay = self.timeDelay, 
                       method = self.method, neighbourhoodSize = self.neighbourhoodSize)
            
#            # Normalise Length
#            y = CRPprops.downSampleCRP(y, self.sequenceLength)
            
            self.CRPdata = y
            
            return y
        
        except:
            print 'Error in CRP calculation'
            return None

            
    def createCRPfile(self):
        '''
        Calculates a CRP and saves a file for the given file and CRP settings
        Returns True if successful or False if there was an error
        Assumptions:
            The feature file has no header row
            The feature file has a single time column before the feature value columns
        '''
        crp = self.calculateCRP()
        if crp is not None:
            outputFilePath = CRPpath + self.getFileName()
            np.save(outputFilePath, crp)
            return True
        return False
    
    
    @classmethod
    def loadCRPfiles(lstCRPprops):
        '''
        Loads all CRP files into memory for the given list of CRP properties
        '''
        CRPfiles = {}
        for crpProps in lstCRPprops:
            try:
                fCRP = open(CRPpath + crpProps.getFileName() + '.npy', 'rb')
                CRPdata = fCRP.read()
                fCRP.close()
                crpProps.featureFilePath = None
                CRPfiles[crpProps.toTuple(False)] = CRPdata
            except:
                pass
            
        return CRPfiles
