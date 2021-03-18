import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class UnknownSignal:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        
        self.numPoints = len(self.xs)
        self.numSegments = int(self.numPoints / 20)

    def splitIntoSegments(self):
        xsSplit = np.vsplit(self.xs, self.numSegments)
        ysSplit = np.vsplit(self.ys, self.numSegments)

        self.segments = []
        for i in range(self.numSegments):
            lineSegment = LineSegment(xsSplit[i], ysSplit[i])
            self.segments.append(lineSegment)

    def getSegment(self, i):
        return self.segments[i]
    
class LineSegment:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        
        self.numOfPoints = len(self.xs)

    def splitTrainingValidation(self, numTraining):
        pos = np.random.permutation(self.numOfPoints)
        tempXs = self.xs[pos]
        tempYs = self.ys[pos]

        self.xsTraining = tempXs[:numTraining]
        self.ysTraining = tempYs[:numTraining]
        self.xsValidation = tempXs[numTraining:]
        self.ysValidation = tempYs[numTraining:]

    def createLinearX(self):
        ones = np.ones(self.numOfPoints, 1)
        self.XLinear = np.hstack([self.xs, ones])
    
    def createPolynomialX(self):
        order = 3
        XPowersList = []
        for i in range(order + 1):
            values = self.xs ** i
            XPowersList.insert(0, values)
        self.XPolynomial = np.hstack(XPowersList)

    def createSinusoidalX(self):
        ones = np.ones(self.numOfPoints, 1)
        sinxs = np.sin(self.xs)
        self.XSinusoidal = np.hstack([self.sinxs, ones])


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
        xsSplitTraining.append(xsSplit[i][:10])
        xsSplitValidation.append(xsSplit[i][10:])
        ysSplitTraining.append(ysSplit[i][:10])
        ysSplitValidation.append(ysSplit[i][10:])
    return xsSplitTraining, xsSplitValidation, ysSplitTraining, ysSplitValidation

# Returns the weights from regression
def regressionNormalEquation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y 
    # return np.linalg.solve(X.T @ X, X.T @ y) # Not sure if the shape of the regulariser is correct

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

# Returns the weight of sinusoidal regression
def sinusoidalRegression(xs, y):
    ones = np.ones((len(xs), 1))
    sinxs = np.sin(xs)
    X = np.hstack([sinxs, ones])
    ws = regressionNormalEquation(X, y)
    return ws

# Performs a specific type of regression based on the function type givenn
def regression(xs, y, func):
    ws = np.array([])
    if func == "linear":
        ws = linearRegression(xs, y)
    elif func == "polynomial":
        ws = polynomialRegression(xs, y)
    elif func == "sine":
        ws = sinusoidalRegression(xs, y)
    return ws

# Calculates the estimated points based on the lines
def calcEstimated(xs, ws, func):
    estimates = np.array([])
    if func == "linear":
        estimates = ws[0] * xs + ws[1]
    elif func == "polynomial":
        line = np.poly1d(ws.flatten())
        estimates = line(xs)
    elif func == "sine":
        estimates = ws[0] * np.sin(xs) + ws[1]
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
        # print(funcsList[i], wsList[i])
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
        elif funcsList[i] == "sine":
            ysLine = ws[0] * np.sin(xsLine) + ws[1] # Sine
        plt.plot(xsLine, ysLine)
    plt.show()

# Main function
def main():
    xs, ys = loadPoints(sys.argv[1])
    unknownSignal = UnknownSignal(xs, ys)
    unknownSignal.splitIntoSegments()

    wsList = []
    funcsList = []

    funcOptions = ["linear", "polynomial", "sine"]

    for segment in unknownSignal.segments:
        segment.splitTrainingValidation(14)

        ws = np.array([])
        funcUsed = ""

        smallestError = np.Inf
        for func in funcOptions:
            potentialWs = regression(segment.xsTraining, segment.ysTraining, func)
            error = calcSegmentError(segment.xsValidation, segment.ysValidation, potentialWs, func)
            # print(f"{func}: {error}")
            if error < smallestError:
                smallestError = error
                ws = potentialWs
                funcUsed = func

        funcsList.append(funcUsed)
        wsList.append(ws)
    
    print(funcsList)

    error = calcTotalError([segment.xs for segment in unknownSignal.segments], [segment.ys for segment in unknownSignal.segments], wsList, funcsList)
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