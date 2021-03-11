import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def linearRegression(X, y):
    ones = np.ones((len(X), 1))
    X = np.hstack([X, ones])
    ws = np.linalg.inv(X.T @ X) @ X.T @ y
    grad = ws[0][0]
    intercept = ws[1][0]
    return grad, intercept

def loadPoints(filename):
    points = pd.read_csv(filename, header=None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

def plot(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

# Main function
def main():
    xs, ys = loadPoints(sys.argv[1])
    grad, intercept = linearRegression(xs, ys)
    print(f"y = {grad}x + {intercept}")
    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":    
            plot(xs, ys)

if __name__ == "__main__":
    numOfArgs = len(sys.argv)
    if ((numOfArgs == 2) | (numOfArgs == 3)):
        main()
    else:
        print("Invalid argument(s)")