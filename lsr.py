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

        self.segments = self.splitIntoSegments()

        self.totalError = self.calcTotalError()
        print(self.totalError)

    def splitIntoSegments(self):
        xsSplit = np.vsplit(self.xs, self.numSegments)
        ysSplit = np.vsplit(self.ys, self.numSegments)
        return [LineSegment(xsSplit[i], ysSplit[i]) for i in range(self.numSegments)]

    def calcTotalError(self):
        return np.sum([segment.totalError for segment in self.segments])


    def getSegment(self, i):
        return self.segments[i]
    
class LineSegment:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.numOfPoints = len(self.xs)
        self.numTrainingPoints = 10

        # Split the data into training data and validation data
        [
            self.xsTraining,
            self.ysTraining,
            self.xsValidation,
            self.ysValidation
        ] = self.splitTrainingValidation()

        # Get X values of each model from training data
        self.XLinear = self.createXLinear()
        self.XPolynomial = self.createXPolynomial()
        self.XSinusoidal = self.createXSinusoidal()

        # Get the weights of each model
        self.wsLinear = self.regressionNormalEquation(self.XLinear)
        self.wsPolynomial = self.regressionNormalEquation(self.XPolynomial)
        self.wsSinusoidal = self.regressionNormalEquation(self.XSinusoidal)

        # Calculate the cross-validation error of each model
        self.errorLinear = self.calcErrorLinear()
        self.errorPolynomial = self.calcErrorPolynomial()
        self.errorSinusoidal = self.calcErrorSinusoidal()

        # Calculate the best model
        self.bestModel = self.calcBestModel()

        # Calculate the total error of the best model
        self.totalError = self.calcTotalError()

    # Splits the data into training and validation
    def splitTrainingValidation(self):
        pos = np.random.permutation(self.numOfPoints)
        tempXs = self.xs[pos]
        tempYs = self.ys[pos]

        xsTraining = tempXs[:self.numTrainingPoints]
        ysTraining = tempYs[:self.numTrainingPoints]
        xsValidation = tempXs[self.numTrainingPoints:]
        ysValidation = tempYs[self.numTrainingPoints:]

        return xsTraining, ysTraining, xsValidation, ysValidation

    # Returns the X values for linear regression
    def createXLinear(self):
        ones = np.ones((self.numTrainingPoints, 1))
        return np.hstack([self.xsTraining, ones])
    
    # Returns the X values for polynomial regression
    def createXPolynomial(self):
        order = 3
        XPowersList = []
        for i in range(order + 1):
            values = self.xsTraining ** i
            XPowersList.insert(0, values)
        return np.hstack(XPowersList)

    # Returns the X values for sinusoidal regression
    def createXSinusoidal(self):
        ones = np.ones((self.numTrainingPoints, 1))
        sinxs = np.sin(self.xsTraining)
        return np.hstack([sinxs, ones])

    # The normal equation for linear regression
    def regressionNormalEquation(self, X):
        ws = np.linalg.inv(X.T @ X) @ X.T @ self.ysTraining
        return ws

    # Returns the cross-validation error for a given set of estimates
    def calcCrossValidationError(self, estimates):
        diff = self.ysValidation - estimates
        return np.sum(diff ** 2)

    # Returns the cross-validation error for linear model
    def calcErrorLinear(self):
        estimates = self.wsLinear[0] * self.xsValidation + self.wsLinear[1]
        return self.calcCrossValidationError(estimates)

    # Returns the cross-validation error for polynomial model
    def calcErrorPolynomial(self):
        estimates = np.poly1d(self.wsPolynomial.flatten())(self.xsValidation)
        return self.calcCrossValidationError(estimates)

    # Returns the cross-validation error for sinusoidal model
    def calcErrorSinusoidal(self):
        estimates = self.wsSinusoidal[0] * np.sin(self.xsValidation) + self.wsSinusoidal[1]
        return self.calcCrossValidationError(estimates)

    # Returns the best model (i.e. the one with the lowest cross-validation error)
    def calcBestModel(self):
        if (self.errorLinear <= self.errorPolynomial) & (self.errorLinear <= self.errorSinusoidal):
            return "linear"
        elif self.errorPolynomial <= self.errorSinusoidal:
            return "polynomial"
        else:
            return "sinusoidal"

    # Calculates the total error of the best model
    def calcTotalError(self):
        estimates = np.array([])
        if self.bestModel == "linear":
            estimates = self.wsLinear[0] * self.xs + self.wsLinear[1]
        elif self.bestModel == "polynomial":
            estimates = np.poly1d(self.wsPolynomial.flatten())(self.xs)
        elif self.bestModel == "sinusoidal":
            estimates = self.wsSinusoidal[0] * np.sin(self.xs) + self.wsSinusoidal[1]
        diff = self.ys - estimates
        return np.sum(diff ** 2)

    


# Loads points in from a given filename as a numpy array
def loadPoints(filename):
    points = pd.read_csv(filename, header=None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

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
        error = calcSegmentError(xsSplit[i], ysSplit[i], wsList[i], funcsList[i])
        total += error 
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

    wsList = []
    funcsList = []

    funcOptions = ["linear", "polynomial", "sine"]

    for segment in unknownSignal.segments:

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