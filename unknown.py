import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SEGMENTS = [
    ["basic_3.csv", [0]],
    ["basic_4.csv", [1]],
    ["basic_5.csv", [0]],
    ["noise_2.csv", [1]],
    ["noise_3.csv", [1, 2]],
    ["adv_1.csv", [0, 2]],
    ["adv_2.csv", [0, 2]],
    ["adv_3.csv", [0, 1, 2, 4, 5]],
]

def loadPoints(filename):
    points = pd.read_csv(f"training/{filename}", header = None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

def extractSegments(xs, ys, segments):
    numSegments = len(xs) // 20
    xs = [x for i, x in enumerate(np.vsplit(xs, numSegments)) if i in segments]
    ys = [y for i, y in enumerate(np.vsplit(ys, numSegments)) if i in segments]
    return xs, ys

def splitTrainingValidation(xs, ys):
    xsTraining, xsValidation, ysTraining, ysValidation = [], [], [], []
    for x, y in zip(xs, ys):
        pos = np.random.permutation(20)
        x = x[pos]
        y = y[pos]
        xsTraining.append(x[:15])
        xsValidation.append(x[15:])
        ysTraining.append(y[:15])
        ysValidation.append(y[15:])
    return xsTraining, xsValidation, ysTraining, ysValidation

def createXFunc(xs, func):
    ones = np.ones((len(xs), 1))
    funcxs = func(xs)
    return np.hstack([funcxs, ones])

def regressionNormalEquation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)
    # return np.linalg.inv(X.T @ X) @ X.T @ y

if __name__ == "__main__":
    for filenameSegments in SEGMENTS:
        xs, ys = loadPoints(filenameSegments[0])
        xs, ys = extractSegments(xs, ys, filenameSegments[1])
        xsTraining, xsValidation, ysTraining, ysValidation = splitTrainingValidation(xs, ys)
        for i in range(len(xsTraining)):
            xTraining = xsTraining[i]
            xValidation = xsValidation[i]
            yTraining = ysTraining[i]
            yValidation = ysValidation[i]
            print("-" * 50)
            for func in [np.sin, np.cos, np.tan, np.exp]:
                X = createXFunc(xTraining, func)
                ws = regressionNormalEquation(X, yTraining)
                estimates = ws[0] * func(xValidation) + ws[1]
                diff = yValidation - estimates
                error = np.sum(diff ** 2)
                print(f"{filenameSegments[0]:>15} {filenameSegments[1][i]:>5} {str(func):>5} {error:<15}")
            

