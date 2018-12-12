#181205 created bayesReg1.py
# working through of this 
# https://theoreticalecology.wordpress.com/2010/09/17/metropolis-hastings-mcmc-in-r/

import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform
import pylab as pl
import math as m
import pandas as pd


def likelihoodF(a,b,sd,x,y):
    pred = a*x + b
    singlelikelihoods = norm.logpdf(y, pred, sd)   # This evaluates the probability of the y vector at
                                                # the mean value of ax + b and standard dev of sd
    return(sum(singlelikelihoods))

def consSlopeLikelihoodF(trueA,trueB,trueSd,x,y):
    seqL = pl.frange(3,7,0.05).tolist()
    slopeLikelihoodL=[]
    for a in seqL:
        slopeLikelihoodL.append(likelihoodF(a,trueB,trueSd,x,y))
    return seqL,slopeLikelihoodL

def prior(a,b,sd):
    aprior = uniform.logpdf(a,0,10)
    bprior = norm.logpdf(b,5)
    sdprior = uniform.logpdf(sd,0,30)
    return(aprior+bprior+sdprior)

def posterior(a,b,sd,x,y):
    return(likelihoodF(a,b,sd,x,y)+prior(a,b,sd))

def proposalfunction(meanA):
    return(np.random.normal(meanA,[0.1,0.5,0.3],3))

def run_metropolis_MCMC(startvalueT,x,y,iterationsI):
    chain = np.zeros((iterationsI+1,3)) # create zero array with iterations + 1 cols
    chain[0,]=startvalueT # conversion from tuple to array handled by system
    for  i in range(len(chain)-1):
        proposal = proposalfunction(chain[i,])
        a,b,sd=proposal
        a2,b2,sd2= chain[i,]
        probab = np.exp(posterior(a,b,sd,x,y)-posterior(a2,b2,sd2,x,y))
        if 1 < probab:
            chain[i+1]=proposal
        else:
            chain[i+1]=chain[i,]

        # if uniform.rvs(0,0,1)[0] < probab:
        #     chain[i+1]=proposal
        # else:
        #     chain[i+1]=chain[i,]

    return(chain)

def main():
    trueA = 5
    trueB = 0
    trueSd = 10
    sampleSize = 31

    x = np.array(range(-int((sampleSize-1)/2),int((sampleSize+1)/2)))
    # r1=trueA * x
    # r2 =trueB * x
    # r3=(trueSd * np.random.randn(sampleSize))
    y =  (trueA * x) + trueB +(np.random.normal(0,trueSd,sampleSize))
    # plt.scatter(x, y)
    # plt.show()
    rv = likelihoodF(trueA,trueB,trueSd,x,y)
    seqL,slopeLikelihoodL = consSlopeLikelihoodF(trueA,trueB,trueSd,x,y)
    # plt.plot(seqL,slopeLikelihoodL)
    # plt.show()

    startvalue = (4,0,10)
    chain = run_metropolis_MCMC(startvalue,x,y,10000)
    burnIn = 5000
    df = pd.DataFrame(chain[-burnIn:,])
    print(df.duplicated(None,keep='last').head(20))
    acceptance = 1.0-statistics.mean(df.duplicated(None,keep='last'))
    print(f'a: should be 5: {statistics.mean(df.loc[:,0])}')
    print(f'b: should be 0: {statistics.mean(df.loc[:,1])}')
    print(f'sd: should be 10: {statistics.mean(df.loc[:,2])}')





if __name__ == '__main__':
    main()