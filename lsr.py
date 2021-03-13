import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Loads points in from a given filename as a numpy array
def loadPoints(filename):
    points = pd.read_csv(filename, header=None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

# Splits the x and y points into 20 point segments and returns them
def splitPoints(xs, ys):
    numSegments = len(xs) // 20
    xsSplit = np.vsplit(xs, numSegments)
    ysSplit = np.vsplit(ys, numSegments)
    return xsSplit, ysSplit

# Returns the weights from regression
def regression(X, y):
    ones = np.ones((len(X), 1))
    X = np.hstack([X, ones])
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Calculates the estimated points based on the lines
def calcEstimated(xs, ws):
    ones = np.ones((len(xs), 1))
    xs = np.hstack([xs, ones])
    estimates = xs @ ws
    return estimates

# Calculates the error of a 20 point segment
def calcSegmentError(xs, ys, ws):
    esimates = calcEstimated(xs, ws)
    diff = ys - esimates
    diffSquared = diff ** 2
    diffSquaredTotal = np.sum(diffSquared)
    return diffSquaredTotal

# Calculates the total error for every point
def calcTotalError(xsSplit, ysSplit, wsList):
    total = 0
    for i in range(len(xsSplit)):
        total += calcSegmentError(xsSplit[i], ysSplit[i], wsList[i])
    return total

# Plots a series of points on a scatter plot
def plot(xs, ys, wsList):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    for i in range(len(wsList)):
        ws = wsList[i]
        x1 = xs[i * 20]
        x2 = xs[i * 20 + 19]
        y1 = x1 * ws[0][0] + ws[1][0]
        y2 = x2 * ws[0][0] + ws[1][0]
        plt.plot([x1, x2], [y1, y2])
    plt.show()

# Main function
def main():
    xs, ys = loadPoints(sys.argv[1])
    xsSplit, ysSplit = splitPoints(xs, ys)
    wsList = []
    
    for i in range(len(xsSplit)):
        ws = regression(xsSplit[i], ysSplit[i])
        wsList.append(ws)
    
    error = calcTotalError(xsSplit, ysSplit, wsList)
    print(error)

    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":    
            plot(xs, ys, wsList)

if __name__ == "__main__":
    numOfArgs = len(sys.argv)
    if ((numOfArgs == 2) | (numOfArgs == 3)):
        main()
    else:
        print("Invalid argument(s)")