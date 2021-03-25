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

    # Returns a list of 20-point segments for an unknown signal
    def splitIntoSegments(self):
        xsSplit = np.vsplit(self.xs, self.numSegments)
        ysSplit = np.vsplit(self.ys, self.numSegments)
        return [LineSegment(xsSplit[i], ysSplit[i]) for i in range(self.numSegments)]

    # Returns the total error of the unknown signal
    def calcTotalError(self):
        return np.sum([segment.totalError for segment in self.segments])

    # Plot every point of the unknown signal and the lines for each segment
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

        # Split the data into training data and validation data
        self.numFolds = 5
        self.kfolds = self.getKFold()

        # Calculate the cross-validation error of each model
        self.errorLinear = np.sum([kfold.errorLinear for kfold in self.kfolds]) / self.numFolds
        self.errorPolynomial = np.sum([kfold.errorPolynomial for kfold in self.kfolds]) / self.numFolds
        self.errorSinusoidal = np.sum([kfold.errorSinusoidal for kfold in self.kfolds]) / self.numFolds

        # Calculate the best model
        self.bestModel = self.calcBestModel()
        print(self.bestModel)

        # Calculates the weights for the best model
        self.ws = self.calcWeights()

        # Calculate the total error of the best model
        self.totalError = self.calcTotalError()

    # Splits the data into training and validation
    def getKFold(self):
        pos = np.random.permutation(self.numOfPoints)
        xsShuffled = self.xs[pos]
        ysShuffled = self.ys[pos]

        xsPartition = np.split(xsShuffled, self.numFolds)
        ysPartition = np.split(ysShuffled, self.numFolds)

        kfolds = []
        for i in range(self.numFolds):
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

    def calcWeights(self):
        temp = Partition(self.xs, np.array([]), self.ys, np.array([]))
        ws = np.array([])
        if self.bestModel == "linear":
            ws = temp.wsLinear
        elif self.bestModel == "polynomial":
            ws = temp.wsPolynomial
        elif self.bestModel == "sinusoidal":
            ws = temp.wsSinusoidal
        return ws

    # Calculates the total error of the best model
    def calcTotalError(self):
        estimates = np.array([])
        if self.bestModel == "linear":
            estimates = self.ws[0] * self.xs + self.ws[1]
        elif self.bestModel == "polynomial":
            estimates = np.poly1d(self.ws.flatten())(self.xs)
        elif self.bestModel == "sinusoidal":
            estimates = self.ws[0] * np.sin(self.xs) + self.ws[1]
        diff = self.ys - estimates
        return np.sum(diff ** 2)

    # Plot the line for the line segment
    def plot(self):
        xsLine = np.linspace(self.xs[0], self.xs[-1], 1000)
        ysLine = np.array([])
        if self.bestModel == "linear":
            ysLine = self.ws[0] * xsLine + self.ws[1]
        elif self.bestModel == "polynomial":
            ysLine = np.poly1d(self.ws.flatten())(xsLine)
        elif self.bestModel == "sinusoidal":
            ysLine = self.ws[0] * np.sin(xsLine) + self.ws[1]
        plt.plot(xsLine, ysLine)

class Partition:
    def __init__(self, xsTraining, xsValidation, ysTraining, ysValidation):
        # The training and validation data for a given partition
        self.xsTraining = xsTraining
        self.xsValidation = xsValidation
        self.ysTraining = ysTraining
        self.ysValidation = ysValidation

        self.numTrainingPoints = len(self.xsTraining)

        # Get X values of each model from training data
        self.XLinear = self.createXLinear()
        self.XPolynomial = self.createXPolynomial()
        self.XSinusoidal = self.createXSinusoidal()

        # Get the weights of each model
        self.wsLinear = self.regressionNormalEquation(self.XLinear)
        self.wsPolynomial = self.regressionNormalEquation(self.XPolynomial)
        self.wsSinusoidal = self.regressionNormalEquation(self.XSinusoidal)

        # Calculate the sum squared error error of each model
        self.errorLinear = self.calcErrorLinear()
        self.errorPolynomial = self.calcErrorPolynomial()
        self.errorSinusoidal = self.calcErrorSinusoidal()

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

    # Returns the sum squared error for a given set of estimates
    def calcSumSquaredError(self, estimates):
        diff = self.ysValidation - estimates
        return np.sum(diff ** 2)

    # Returns the sum squared error error for linear model
    def calcErrorLinear(self):
        estimates = self.wsLinear[0] * self.xsValidation + self.wsLinear[1]
        # print("WS")
        # print(self.wsLinear)
        # print("YS VAL")
        # print(self.ysValidation)
        # print("XS VAL")
        # print(self.xsValidation)



        return self.calcSumSquaredError(estimates)

    # Returns the sum squared error error for polynomial model
    def calcErrorPolynomial(self):
        estimates = np.poly1d(self.wsPolynomial.flatten())(self.xsValidation)
        return self.calcSumSquaredError(estimates)

    # Returns the sum squared error for sinusoidal model
    def calcErrorSinusoidal(self):
        estimates = self.wsSinusoidal[0] * np.sin(self.xsValidation) + self.wsSinusoidal[1]
        return self.calcSumSquaredError(estimates)

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