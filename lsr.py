import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class UnknownSignal:
    def __init__(self, xs, ys, plot):
        self.xs = xs
        self.ys = ys
        
        self.numPoints = len(self.xs)
        self.numSegments = self.numPoints // 20

        self.segments = self.splitIntoSegments()

        self.totalError = self.calcTotalError()
        print(self.totalError)

        if plot:
            self.plot()

    def splitIntoSegments(self):
        xsSplit = np.vsplit(self.xs, self.numSegments)
        ysSplit = np.vsplit(self.ys, self.numSegments)
        return [LineSegment(xsSplit[i], ysSplit[i]) for i in range(self.numSegments)]

    def calcTotalError(self):
        return np.sum([segment.totalError for segment in self.segments])

    def getSegment(self, i):
        return self.segments[i]

    def plot(self):
        colour = np.concatenate([[i] * 20 for i in range(self.numSegments)])
        plt.scatter(self.xs, self.ys, c = colour)
        [segment.plot() for segment in self.segments]
        plt.show()
    
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

    def plot(self):
        xsLine = np.linspace(self.xs[0], self.xs[-1], 1000)
        ysLine = np.array([])
        if self.bestModel == "linear":
            ysLine = self.wsLinear[0] * xsLine + self.wsLinear[1]
        elif self.bestModel == "polynomial":
            ysLine = np.poly1d(self.wsPolynomial.flatten())(xsLine)
        elif self.bestModel == "sinusoidal":
            ysLine = self.wsSinusoidal[0] * np.sin(xsLine) + self.wsSinusoidal[1]
        plt.plot(xsLine, ysLine)

# Loads points in from a given filename as a numpy array
def loadPoints(filename):
    points = pd.read_csv(filename, header=None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

# Main function
def main():
    xs, ys = loadPoints(sys.argv[1])
    plot = False
    if len(sys.argv) >= 3:
        plot = (sys.argv[2] == "--plot")

    unknownSignal = UnknownSignal(xs, ys, plot)

if __name__ == "__main__":
    numOfArgs = len(sys.argv)
    if ((numOfArgs == 2) | (numOfArgs == 3)):
        main()
        print("")
    else:
        print("Invalid argument(s)")