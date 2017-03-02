

class QLearner(object):


    class QLearningStep(object):
        
        def __init__(self, stepSettings):
            
            self._stepSettings = stepSettings
            
        def addNextMove(self, moveName, moveSettings):
            
            pass


    def __init__(self, initValue, targetValue, discountRate, 
                 allPossibleMoves, moveEvaluationCallback, getMoveChildrenCallback):
        '''
        :initValue: the initial value that the 
        :allPossibleMoves: a list of lists of settings that are possible values for next moves
        '''
        self._currentValue = initValue
        self._targetValue = targetValue
        self._allPossibleMoves = allPossibleMoves
    
    
    def addMove(self, moveName, moveSettings, moveReward):
        
        pass
    
    
    def addNextMove(self, currentMoveName, 
                    nextMoveName, nextMoveSettings):

        # adds a next possible move to a current move
        pass


    def _AisBetterThanB(self, A, B):

        return abs(self._targetValue - A) < abs(self._targetValue - B)


    def addNextValue(self, valueToAdd):

        self._valueHistory.append(valueToAdd)


    def getNextMoveSettings(self):
        
        pass
    

    def takeCurrentStep(self):
        
        return True

#pseudo-code
###########

# add all current move possibilities
# for each move m:
#     evaluate Rm
#     add all child moves C
#     for each child move c:
#         evaluate Rc
# for each m:
#     for each c:
#         evaluate reward Rm + gamma * Rc
#
# choose m with the highest Rm + gamma * max(Rc)
