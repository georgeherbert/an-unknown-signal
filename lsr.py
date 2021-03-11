import sys
import csv
import numpy as np

def main():
    data = np.genfromtxt(sys.argv[1], delimiter = ',')
    print(data)

def checkArgs():
    valid = False
    if (len(sys.argv) == 2) | (len(sys.argv) == 3):
        valid = True
    return valid

if __name__ == "__main__":
    if checkArgs() == True:
        main()
    else:
        print("Invalid argument(s)")