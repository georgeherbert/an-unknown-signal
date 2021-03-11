import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Plots the data
def plot(data):
    fix, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1])
    plt.show()

# Main function
def main():
    data = np.genfromtxt(sys.argv[1], delimiter = ',')
    
    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":    
            plot(data)

if __name__ == "__main__":
    numOfArgs = len(sys.argv)
    if ((numOfArgs == 2) | (numOfArgs == 3)):
        main()
    else:
        print("Invalid argument(s)")