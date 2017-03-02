import pandas as pd
import numpy as np


def _getAveragePrecision(dataFrame, queryPiece):
    '''
    Returns the average precision for a particular query (i.e. performance of a
    piece) from the dataframe
    '''
    def getOmega(row, pieceId):
        
        return int((row['Piece 1 Id'] == pieceId) & 
                   (row['Piece 2 Id'] == pieceId))

    # Filter dataframe to only include NCDs featuring the query piece
    dfPiece = dataFrame[((dataFrame['Piece 1 Id'] == queryPiece[0]) & 
                         (dataFrame['Piece 1 Performance Id'] == queryPiece[1])) |
                        ((dataFrame['Piece 2 Id'] == queryPiece[0]) & 
                         (dataFrame['Piece 2 Performance Id'] == queryPiece[1]))]
           
    # Add Rank and Omega
    dfPieceRanked = dfPiece.sort('NCD')
    dfPieceRanked['Rank'] = 1 + np.argsort(dfPieceRanked['NCD'])
    dfPieceRanked['Omega'] = dfPieceRanked.apply(lambda row: getOmega(row, queryPiece[0]), axis = 1)
    dfPieceRanked = dfPieceRanked[dfPieceRanked['Omega'] == 1]
    
    # Calculate Precision at each row
    sumOmega = 0
    precision = []
    for idx in dfPieceRanked.index:
        sumOmega += dfPieceRanked.ix[idx]['Omega']
        precision.append(float(sumOmega) / dfPieceRanked.ix[idx]['Rank'])
    dfPieceRanked.loc[:, 'Precision'] = pd.Series(precision, index = dfPieceRanked.index)
    
    # Calculate and return Average Precision for the Query
    averagePrecision = dfPieceRanked['Precision'].mean()
    return averagePrecision


def getMeanAveragePrecision(dataFrame):
    '''
    Returns the mean average precision for all queries of results in a dataframe
    The dataframe should therefore only contain results from one configuration of 
    CRP settings
    '''
    # Get unique pieces and performance tuples
    piece1Ids = list(dataFrame['Piece 1 Id'])
    piece1PerformanceIds = list(dataFrame['Piece 1 Performance Id'])
    piecesPerformances1 = zip(piece1Ids, piece1PerformanceIds)
    piece2Ids = list(dataFrame['Piece 2 Id'])
    piece2PerformanceIds = list(dataFrame['Piece 2 Performance Id'])
    piecesPerformances2 = zip(piece2Ids, piece2PerformanceIds)
    pieces = sorted(list(set(piecesPerformances1 + piecesPerformances2)))
    
    # Get Precision for each query
    averagePrecisions = []
    for queryPiece in pieces:
        averagePrecision = _getAveragePrecision(dataFrame, queryPiece)
        averagePrecisions.append(averagePrecision)
    
    # Return Mean Average Precision
    averagePrecisions = np.array(averagePrecisions)
    return averagePrecisions.mean()

