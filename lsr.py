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

# Splits the set of xs or ys into training and validation
def splitTrainingValidation(xsSplit, ysSplit):
    xsSplitTraining = []
    xsSplitValidation = []
    ysSplitTraining = []
    ysSplitValidation = []
    for i in range(len(xsSplit)):
        pos = np.random.permutation(len(xsSplit[i]))
        xsSplit[i] = xsSplit[i][pos]
        ysSplit[i] = ysSplit[i][pos]
        xsSplitTraining.append(xsSplit[i][:12])
        xsSplitValidation.append(xsSplit[i][12:])
        ysSplitTraining.append(ysSplit[i][:12])
        ysSplitValidation.append(ysSplit[i][12:])
    return xsSplitTraining, xsSplitValidation, ysSplitTraining, ysSplitValidation

# Returns the weights from regression
def regressionNormalEquation(X, y):
    # return np.linalg.inv(X.T @ X + reg * np.identity(X.shape[1])) @ X.T @ y 
    return np.linalg.solve(X.T @ X, X.T @ y) # Not sure if the shape of the regulariser is correct

# Linear regression
def linearRegression(xs, y):
    ones = np.ones((len(xs), 1))
    X = np.hstack([xs, ones])
    ws = regressionNormalEquation(X, y)
    return ws

# Calculates the x to the power of values for the equation
def calcXPowers(xs, order):
    XPowersList = []
    for i in range(order + 1):
        values = xs ** i
        XPowersList.insert(0, values)
    XPowers = np.hstack(XPowersList)
    return XPowers

# Returns the weights of polynomialRegression regression
def polynomialRegression(xs, y):
    X = calcXPowers(xs, 3)
    ws = regressionNormalEquation(X, y)
    return ws

# Returns the weight of exponential regression
def exponentialRegression(xs, y):
    ones = np.ones((len(xs), 1))
    exps = np.exp(xs)
    X = np.hstack([exps, ones])
    ws = regressionNormalEquation(X, y)
    return ws

# Performs a specific type of regression based on the function type givenn
def regression(xs, y, func):
    ws = np.array([])
    if func == "linear":
        ws = linearRegression(xs, y)
    elif func == "polynomial":
        ws = polynomialRegression(xs, y)
    elif func == "exponential":
        ws = exponentialRegression(xs, y)
    return ws

# Calculates the estimated points based on the lines
def calcEstimated(xs, ws, func):
    estimates = np.array([])
    if (func == "polynomial") | (func == "linear"):
        line = np.poly1d(ws.flatten()) #Polynomial
        estimates = line(xs) # Polynomial
    if func == "exponential":
        estimates = ws[0] * np.exp(xs) + ws[1]
    return estimates

# Calculates the error of a 20 point segment
def calcSegmentError(xs, ys, ws, func):
    esimates = calcEstimated(xs, ws, func)
    diff = ys - esimates
    diffSquaredTotal = np.sum(diff ** 2)
    return diffSquaredTotal

# Calculates the total error for every point
def calcTotalError(xsSplit, ysSplit, wsList, funcsList):
    total = 0
    for i in range(len(xsSplit)):
        print(funcsList[i], wsList[i])
        total += calcSegmentError(xsSplit[i], ysSplit[i], wsList[i], funcsList[i])
    return total

# Plots a series of points on a scatter plot
def plot(xs, ys, wsList, funcsList):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c = colour)
    for i in range(len(wsList)): # Plot a line for each line segment
        ws = wsList[i]
        xsLine = np.linspace(xs[i * 20], xs[i * 20 + 19], 1000)
        ysLine = np.array([])
        if (funcsList[i] == "linear") | (funcsList[i] == "polynomial"):
            line = np.poly1d(ws.flatten()) # Polynomial
            ysLine = line(xsLine) # Polynomial
        elif funcsList[i] == "exponential":
            ysLine = ws[0] * np.exp(xsLine) + ws[1] # Exponential
        plt.plot(xsLine, ysLine)
    plt.show()

# Main function
def main():
    xs, ys = loadPoints(sys.argv[1])
    xsSplit, ysSplit = splitPoints(xs, ys)

    xsSplitTraining, xsSplitValidation, ysSplitTraining, ysSplitValidation = splitTrainingValidation(xsSplit, ysSplit)

    wsList = []
    funcsList = []

    funcOptions = ["linear", "polynomial", "exponential"]

    # func = "polynomial"

    for i in range(len(xsSplit)):
        ws = np.array([])
        funcUsed = ""

        smallestError = np.Inf
        for func in funcOptions:
            potentialWs = regression(xsSplitTraining[i], ysSplitTraining[i], func)
            error = calcSegmentError(xsSplitValidation[i], ysSplitValidation[i], potentialWs, func)
            if error < smallestError:
                smallestError = error
                ws = potentialWs
                funcUsed = func

        funcsList.append(funcUsed)
        wsList.append(ws)
    
    error = calcTotalError(xsSplit, ysSplit, wsList, funcsList)
    print(error)

    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":    
            plot(xs, ys, wsList, funcsList)

if __name__ == "__main__":
    numOfArgs = len(sys.argv)
    if ((numOfArgs == 2) | (numOfArgs == 3)):
        main()
        print("")
    else:
        print("Invalid argument(s)")