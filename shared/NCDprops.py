import pickle, os

from CRPprops import CRPprops
from ncd import memoryNCD
from paths import NCDpath



class NCDprops(object):
    '''
    Holds basic properties about a Normalised Compression Distance between two NCDs
    '''
    
    
    def __init__(self, piece1Id, piece1PerformanceId, piece2Id, piece2PerformanceId, 
                 method, dimension, timeDelay, neighbourhoodSize, downSampleFactor,
                 sequenceLength,
                 featureName, featureFilePath1, featureFilePath2):
        
        self.piece1Id = piece1Id
        self.piece1PerformanceId = piece1PerformanceId
        self.piece2Id = piece2Id
        self.piece2PerformanceId = piece2PerformanceId
        self.method = method
        self.dimension = dimension
        self.timeDelay = timeDelay
        self.neighbourhoodSize = neighbourhoodSize
        self.downSampleFactor = downSampleFactor
        self.sequenceLength = sequenceLength
        self.featureName = featureName
        self.featureFilePath1 = featureFilePath1
        self.featureFilePath2 = featureFilePath2
        
    
    def getCRP1(self):
        
        return CRPprops(self.piece1Id, self.piece1PerformanceId, 
                        self.method, self.dimension, self.timeDelay,
                        self.neighbourhoodSize, self.downSampleFactor,
                        self.sequenceLength,
                        self.featureFilePath1)


    def getCRP2(self):
        
        return CRPprops(self.piece2Id, self.piece2PerformanceId, 
                        self.method, self.dimension, self.timeDelay,
                        self.neighbourhoodSize, self.downSampleFactor,
                        self.sequenceLength,
                        self.featureFilePath2)
        
    def getFileName(self):

        ncdFileName = 'NCD_pc1Id_%s_pc1pfId_%s_pc2Id_%s_pc2pfId_%s_mthd_%s_dimn_%s_tdly_%s_nsz_%s_dsf_%s_feat_%s_sqln_%s' \
                      % (self.piece1Id, self.piece1PerformanceId, self.piece2Id, self.piece2PerformanceId, 
                         self.method, str(self.dimension), str(self.timeDelay), 
                         str(self.neighbourhoodSize), str(self.downSampleFactor),
                         str(self.featureName), str(self.sequenceLength))
    
        return ncdFileName


    @classmethod
    def createNCDfile(cls, NCDfilename, CRP1, CRP2):
        '''
        Calculates the NCD between two CRPs and saves a results file
        Returns True if successful or False if there was an error
        '''
        ncdFilePath = NCDpath + NCDfilename + '.pkl'
        try:
            ncd = memoryNCD(CRP1, CRP2)
            pickle.dump(ncd, open(ncdFilePath, 'wb'))
            return True
        except:
            return False


    @classmethod
    def NCDfileExists(cls, ncdFilename, existingNCDs = None):
        '''
        Returns true if the NCD exists either as a file or a record in the set of 
        existing NCDs, if given
        '''
        if existingNCDs is not None:
            if ncdFilename in existingNCDs:
                return True
        if os.path.exists(NCDpath + ncdFilename + '.pkl'):
            return True
    
        return False