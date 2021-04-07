# THIS FILE NO LONGER WORKS

import numpy as np
from lsr import calcSegmentError

def test_calcSegmentError():
    assert calcSegmentError(
        np.array([[0], [1], [2], [3], [4], [5]]), 
        np.array([[1], [3], [5], [7], [9], [11]]),
        np.array([[2], [1]])
        ) == 0, "Should be 0"
    assert calcSegmentError(
        np.array([[0], [1], [2], [3], [4], [5]]), 
        np.array([[1], [3], [5], [7], [9], [13]]),
        np.array([[2], [1]])
    ) == 4, "Should be 4"
    assert calcSegmentError(
        np.array([[0], [1], [2], [3], [4], [5]]), 
        np.array([[0], [2], [4], [6], [8], [10]]),
        np.array([[2], [1]])
    ) == 6, "Should be 6"
    assert calcSegmentError(
        np.array([[0], [1], [2], [3], [4], [5]]), 
        np.array([[0.5], [3.5], [4.5], [7.5], [8.5], [11.5]]),
        np.array([[2], [1]])
    ) == 1.5, "Should be 1.5"

if __name__ == "__main__":
    test_calcSegmentError()
    print("Everything passed")