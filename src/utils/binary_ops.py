# Author: Calebe Elias Ribeiro Brim
# Last Update: 03/06/2018


import numpy as np

def bitsToBytes(values):
    '''
        Generate relative bits integers
        Usage: 

            bitsToBytes(array.shape(1,10)) => array.shape(1,1)
            bitsToBytes(array.shape(2,10)) => array.shape(2,1)
    '''
    ln = values.shape[1] if values.ndim > 1 else values.shape[0]
    processed = np.array(values.dot(2**np.arange(ln)[::-1]))
    return processed


def bitsNeededToNumber(number,nbits=1):
    ar = np.array(np.ones((1, nbits)).dot(2**np.arange(nbits)[::-1]))
    if ar >=number:
        return nbits
    else: 
        return bitsNeededToNumber(number,nbits+1)
        





# tests
def test_bitsNeeded():
    assert bitsNeededToNumber(3)==2


def test_bitsToBytes():
    assert bitsToBytes(np.array([[1, 1]])) == 3


def test_bitsToBytesOneDim():
    assert bitsToBytes(np.array([1, 1])) == 3


def test_bitsToBytesMultiple():
    values =  bitsToBytes(np.array([[1, 1],[1,1]]))
    assert values[0] == 3
    assert values[1] == 3
