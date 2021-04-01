import sys
import os.path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
ORDER = 3
K = 20

class UnknownSignal:
    def __init__(self, xs, ys, plot):
        # The x and y values of an unknown signal
        self.xs = xs
        self.ys = ys
        
        # The number of points and segments of an unknown signal
        self.numPoints = len(self.xs)
        self.numSegments = self.numPoints // 20

        # The segments
        self.segments = self.splitIntoSegments()

        # Output the total error
        self.outputTotalError()

        # Plot the graph
        if plot:
            self.plot()

    # Returns a list of 20-point segments for an unknown signal
    def splitIntoSegments(self):
        xsSplit = np.vsplit(self.xs, self.numSegments)
        ysSplit = np.vsplit(self.ys, self.numSegments)
        return [FullLineSegment(xsSplit[i], ysSplit[i]) for i in range(self.numSegments)]

    # Outputs the total error of the unknown signal
    def outputTotalError(self):
        print(np.sum([segment.totalError for segment in self.segments]), "\n")

    # Plot every point of the unknown signal and the lines for each segment
    def plot(self):
        colour = np.concatenate([[i] * 20 for i in range(self.numSegments)])
        plt.scatter(self.xs, self.ys, c = colour)
        [segment.plot() for segment in self.segments]
        plt.show()
    
class LineSegment:
    # The regression normal equation
    def regressionNormalEquation(self, X, y):
        # return np.linalg.solve(X.T @ X, X.T @ y)
        return np.linalg.inv(X.T @ X) @ X.T @ y

    # Returns the X values for linear regression
    def createXLinear(self, xs):
        ones = np.ones((len(xs), 1))
        return np.hstack([xs, ones])
    
    # Returns the X values for polynomial regression
    def createXPolynomial(self, xs):
        XPowersList = []
        for i in range(ORDER + 1):
            values = xs ** i
            XPowersList.insert(0, values)
        return np.hstack(XPowersList)

    # Returns the X values for sinusoidal regression
    def createXSinusoidal(self, xs):
        ones = np.ones((len(xs), 1))
        sinxs = np.sin(xs)
        return np.hstack([sinxs, ones])

    # Returns the sum squared error for a given set of estimates
    def calcSumSquaredError(self, actual, estimates):
        diff = actual - estimates
        return np.sum(diff ** 2)

    # Returns the sum squared error error for a linear model
    def calcErrorLinear(self, xs, ys, ws):
        estimates = ws[0] * xs + ws[1]
        return self.calcSumSquaredError(estimates, ys)

    # Returns the sum squared error error for a polynomial model
    def calcErrorPolynomial(self, xs, ys, ws):
        estimates = np.poly1d(ws.flatten())(xs)
        return self.calcSumSquaredError(estimates, ys)

    # Returns the sum squared error for a sinusoidal model
    def calcErrorSinusoidal(self, xs, ys, ws):
        estimates = ws[0] * np.sin(xs) + ws[1]
        return self.calcSumSquaredError(estimates, ys)

class FullLineSegment(LineSegment):
    def __init__(self, xs, ys):
        # The x and y values of a line segment
        self.xs = xs
        self.ys = ys

        # The number of points in the line segment
        self.numOfPoints = len(self.xs)

        # K-Fold cross validation
        self.kfolds = self.getKFold()

        # Calculate the cross-validation error of each model
        self.errorLinear = np.sum([kfold.errorLinear for kfold in self.kfolds]) / K
        self.errorPolynomial = np.sum([kfold.errorPolynomial for kfold in self.kfolds]) / K
        self.errorSinusoidal = np.sum([kfold.errorSinusoidal for kfold in self.kfolds]) / K

        # The best model, the weights and the total error
        self.bestModel = self.calcBestModel()
        print(self.bestModel) # For testing purposes only
        self.ws = self.calcWeights()
        self.totalError = self.calcTotalError()

    # Returns the k-fold partitions
    def getKFold(self):
        pos = np.random.permutation(self.numOfPoints)
        xsShuffled = self.xs[pos]
        ysShuffled = self.ys[pos]

        xsPartition = np.split(xsShuffled, K)
        ysPartition = np.split(ysShuffled, K)

        kfolds = []
        for i in range(K):
            xsTraining = np.concatenate(xsPartition[:i] + xsPartition[i + 1:], axis = 0)
            xsValidation = xsPartition[i]
            ysTraining = np.concatenate(ysPartition[:i] + ysPartition[i + 1:], axis = 0)
            ysValidation = ysPartition[i]
            kfolds.append(Partition(xsTraining, xsValidation, ysTraining, ysValidation))

        return kfolds

    # Returns the best model (i.e. the one with the lowest cross-validation error)
    def calcBestModel(self):
        if (self.errorLinear <= self.errorPolynomial) & (self.errorLinear <= self.errorSinusoidal):
            return "linear"
        elif self.errorPolynomial <= self.errorSinusoidal:
            return "polynomial"
        else:
            return "sinusoidal"        

    # Returns the weights of the best model
    def calcWeights(self):
        if self.bestModel == "linear":
            X = self.createXLinear(self.xs)
            return self.regressionNormalEquation(X, self.ys)
        elif self.bestModel == "polynomial":
            X = self.createXPolynomial(self.xs)
            return self.regressionNormalEquation(X, self.ys)
        elif self.bestModel == "sinusoidal":
            X = self.createXSinusoidal(self.xs)
            return self.regressionNormalEquation(X, self.ys)

    # Returns the total error of the best model
    def calcTotalError(self):
        if self.bestModel == "linear":
            return self.calcErrorLinear(self.xs, self.ys, self.ws)
        elif self.bestModel == "polynomial":
            return self.calcErrorPolynomial(self.xs, self.ys, self.ws)
        elif self.bestModel == "sinusoidal":
            return self.calcErrorSinusoidal(self.xs, self.ys, self.ws)

    # Plot the line of the line segment
    def plot(self):
        xsLine = np.linspace(self.xs[0], self.xs[-1], 1000)
        if self.bestModel == "linear":
            ysLine = self.ws[0] * xsLine + self.ws[1]
        elif self.bestModel == "polynomial":
            ysLine = np.poly1d(self.ws.flatten())(xsLine)
        elif self.bestModel == "sinusoidal":
            ysLine = self.ws[0] * np.sin(xsLine) + self.ws[1]
        plt.plot(xsLine, ysLine)

class Partition(LineSegment):
    def __init__(self, xsTraining, xsValidation, ysTraining, ysValidation):
        # The training and validation data for a given partition
        self.xsTraining = xsTraining
        self.xsValidation = xsValidation
        self.ysTraining = ysTraining
        self.ysValidation = ysValidation

        # The number of training points
        self.numTrainingPoints = len(self.xsTraining)

        # X values of each model from training data
        self.XLinear = self.createXLinear(self.xsTraining)
        self.XPolynomial = self.createXPolynomial(self.xsTraining)
        self.XSinusoidal = self.createXSinusoidal(self.xsTraining)

        # The weights of each model
        self.wsLinear = self.regressionNormalEquation(self.XLinear, self.ysTraining)
        self.wsPolynomial = self.regressionNormalEquation(self.XPolynomial, self.ysTraining)
        self.wsSinusoidal = self.regressionNormalEquation(self.XSinusoidal, self.ysTraining)

        # The sum squared error error of each model
        self.errorLinear = self.calcErrorLinear(self.xsValidation, self.ysValidation, self.wsLinear)
        self.errorPolynomial = self.calcErrorPolynomial(self.xsValidation, self.ysValidation, self.wsPolynomial)
        self.errorSinusoidal = self.calcErrorSinusoidal(self.xsValidation, self.ysValidation, self.wsSinusoidal)

# Loads points in from a given filename as a numpy array
def loadPoints(filename):
    points = pd.read_csv(filename, header=None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "An Unknown Signal")
    parser.add_argument("file", metavar = "file", type = str, help = "file name")
    parser.add_argument("--plot", action = "store_true", help = "plot the signal")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File '{args.file}' does not exist.")
        sys.exit()

    xs, ys = loadPoints(args.file)
    UnknownSignal(xs, ys, args.plot)
