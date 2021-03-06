import numpy as np
from datetime import datetime, timedelta


def chooseRandomItem(fromList):
    
    numItems = len(fromList)
    return fromList[np.random.randint(numItems)]
    

def divide_timedelta(td, divisor):
    """Python 2.x timedelta doesn't support division by float, this function does.

    >>> td = datetime.timedelta(10, 100, 1000)
    >>> divide_timedelta(td, 2) == td / 2
    True
    >>> divide_timedelta(td, 100) == td / 100
    True
    >>> divide_timedelta(td, 0.5)
    datetime.timedelta(20, 200, 2000)
    >>> divide_timedelta(td, 0.3)
    datetime.timedelta(33, 29133, 336667)
    >>> divide_timedelta(td, 2.5)
    datetime.timedelta(4, 40, 400)
    >>> td / 0.5
    Traceback (most recent call last):
      ...
    TypeError: unsupported operand type(s) for /: 'datetime.timedelta' and 'float'

    """
    # timedelta.total_seconds() is new in Python version 2.7, so don't use it
    total_seconds = (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6
    divided_seconds = total_seconds / float(divisor)
    return timedelta(seconds=divided_seconds)


def lcut(stringToCutFrom, stringToCut):
    '''
    cuts a string from the left of another if it starts with it
    '''
    if stringToCutFrom.startswith(stringToCut):
        return stringToCutFrom[len(stringToCut): ]
    return stringToCutFrom


def rcut(stringToCutFrom, stringToCut):
    '''
    cuts a string from the left of another if it starts with it
    '''
    if stringToCutFrom.endswith(stringToCut):
        return stringToCutFrom[: -len(stringToCut)]
    return stringToCutFrom