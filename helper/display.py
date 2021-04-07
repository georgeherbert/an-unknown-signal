import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadPoints(filename):
    points = pd.read_csv(filename, header = None)
    xs = np.array([[x] for x in points[0].values])
    ys = np.array([[y] for y in points[1].values])
    return xs, ys

def display(xs, ys):
    colour = np.concatenate([[i] * 20 for i in range(len(xs) // 20)])
    plt.scatter(xs, ys, c = colour)
    plt.show()

if __name__ == "__main__":
    while (filename := input("\t> ")) != "quit":
        if not os.path.exists(filename):
            print(f"File '{filename}' does not exist.")
        else:
            xs, ys = loadPoints(filename)
            display(xs, ys)