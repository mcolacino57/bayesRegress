#181205 created bayesReg1.py

import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pylab as pl


def likelihoodF(a,b,sd,x,y):
    pred = a*x + b
    singlelikelihoods = norm.logsf(y, pred, sd)
    sumll = sum(singlelikelihoods)
    return(sumll)   

def consSlopeLikelihoodF(trueA,trueB,trueSd,x,y):
    seqL = pl.frange(3,7,0.05).tolist()
    slopeLikelihoodL=[]
    for a in seqL:
        slopeLikelihoodL.append(likelihoodF(a,trueB,trueSd,x,y))
    # valA = np.asarray(slopeLikelihoodL)
    return seqL,slopeLikelihoodL



def main():
    trueA = 5
    trueB = 0
    trueSd = 10
    sampleSize = 31

    x = np.array(range(-int((sampleSize-1)/2),int((sampleSize+1)/2)))
    r1=trueA * x
    r2 =trueB * x
    r3=(trueSd * np.random.randn(sampleSize))
    y =  trueA * x + trueB * x +(trueSd * np.random.randn(sampleSize))
    plt.scatter(x, y)
    plt.show()
    rv = likelihoodF(trueA,trueB,trueSd,x,y)
    seqL,slopeLikelihoodL = consSlopeLikelihoodF(trueA,trueB,trueSd,x,y)
    plt.plot(seqL,slopeLikelihoodL)
    plt.show()

if __name__ == '__main__':
    main()