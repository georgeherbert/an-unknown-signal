import sys
import numpy as np

def checkArgs():
    valid = False
    if (len(sys.argv) == 2) | (len(sys.argv) == 3):
        valid = True
    return valid

def main():
    print(sys.argv[1])

if __name__ == "__main__":
    if checkArgs() == True:
        main()
    else:
        print("Invalid argument(s)")