# An Unknown Signal

'An Unknown Signal' is a piece of coursework I produced for the Data-Driven Computer Science (COMS20011) module at the University of Bristol.

## Description

The coursework instructions are in the file [instructions.pdf](instructions.pdf).

The program takes in a set of 2D points of length a multiple of 20 (e.g. 20, 40, 60) which are supposed to be points of a signal that needs to be reconstructed. To reconstruct the signal, the program then calculates the best model for each 20-point segment, out of three possible forms:
- <i>ax + b</i>
- <i>ax<sup>3</sup> + bx<sup>2</sup> + cx + d</i>
- <i>asin(x) + b</i>

Leave-one-out cross-validation is used to detect overfitting.

The coursework report is in the file [report.pdf](report/report.pdf).

## Getting Started

To begin with, clone the repository:

```bash
git clone https://github.com/georgeherbert/an-unknown-signal.git
```

Install the dependencies:
```bash
pip3 install -r requirements.txt
```

To run the program on the file [training/adv_3.csv](training/adv_3.csv):
```bash
python3 lsr.py training/adv_3.csv
```

To run the program on the file [training/adv_3.csv](training/adv_3.csv) and view the plot add the --plot flag:
```bash
python3 lsr.py training/adv_3.csv --plot
```