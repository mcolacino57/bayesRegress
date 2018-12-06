#181205 created bayesReg1.py

import statistics
import numpy as np
import matplotlib.pyplot as plt


class dataVec():
    def __init__(self, dataL):
        self.vector = dataL # if we need to convert list to tuples or dictionary, can do here


def main():
    trueA = 5
    trueB = 0
    trueSd = 10
    sampleSize = 31

    x = np.array(range(-int((sampleSize-1)/2),int((sampleSize+1)/2)))
    y =  trueA * x + trueB * x +(trueSd * np.random.randn(sampleSize))
    plt.scatter(x, y)
    plt.show()

if __name__ == '__main__':
    main()