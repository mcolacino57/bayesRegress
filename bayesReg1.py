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
    # print(a)
    # print(b)
    # print(list(x))
    # print(list(pred))
    # print(list(y))
    singlelikelihoods = norm.logpdf(y, pred, sd)   # This evaluates the probability of the y vector at
                                                # the mean value of ax + b and standard dev of sd
    return(sum(singlelikelihoods))

def consSlopeLikelihoodF(trueA,trueB,trueSd,x,y):
    seqL = pl.frange(-3.,3.0,0.05).tolist()
    slopeLikelihoodL=[]
    for b in seqL:
        slopeLikelihoodL.append(likelihoodF(trueA,b,trueSd,x,y))
    return seqL,slopeLikelihoodL

def prior(a,b,sd):
    aprior = uniform.logpdf(a,0.0,10.0)
    bprior = norm.logpdf(b,0.0,5.0)
    sdprior = uniform.logpdf(sd,0.0,30.0)
    return(aprior+bprior+sdprior)

def posterior(a,b,sd,x,y):
    return(likelihoodF(a,b,sd,x,y)+prior(a,b,sd))

def proposalfunction(meanA):
    return(norm.rvs(meanA,[0.1,0.5,0.3],3))

def run_metropolis_MCMC(startvalueT,x,y,iterationsI):
    chain = np.zeros((iterationsI+1,3)) # create zero array with iterations + 1 rows
    chain[0,]=startvalueT # conversion from tuple to array handled by system
    for  i in range(len(chain)-1):
        # proposal = proposalfunction(chain[i,])
        proposal = chain[i,]+np.random.normal([0.,0.,0.],[0.1,0.5,0.3],3)
        a,b,sd=proposal
        a2,b2,sd2= chain[i,]
        probab = np.exp(posterior(a,b,sd,x,y)-posterior(a2,b2,sd2,x,y))
        # if 1 < probab:
        #     chain[i+1]=proposal
        # else:
        #     chain[i+1]=chain[i,]
        if uniform.rvs(0.0,1.,1)[0] < probab:
            chain[i+1]=proposal
        else:
            chain[i+1]=chain[i,]
    return(chain)

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(str(var_name)+" Distribution")
    # fig.tight_layout()  # Improves appearance a bit.
    plt.show()

# test = pd.DataFrame(np.random.randn(30, 9), columns=map(str, range(9)))
# draw_histograms(test, test.columns, 3, 3)

def main():
    trueA = 5.
    trueB = 0.
    trueSd = 10.
    sampleSize = 31

    x = np.array(range(-int((sampleSize-1)/2),int((sampleSize+1)/2)))
    y =  (trueA * x) + (trueB) +(np.random.normal(0,trueSd,sampleSize))
    # plt.scatter(x, y)
    # plt.show()
    likelihoodF(trueA,trueB,trueSd,x,y)
    seqL,slopeLikelihoodL = consSlopeLikelihoodF(trueA,trueB,trueSd,x,y)
    plt.plot(seqL,slopeLikelihoodL)
    plt.show()

    startvalue = (4.0,0.0,10.0)
    chain = run_metropolis_MCMC(startvalue,x,y,10000)
    burnIn = 5000
    df = pd.DataFrame(chain[-burnIn:,])
    # print(df.duplicated(None,keep='last').head(20))
    acceptance = 1.0-statistics.mean(df.duplicated(None,keep='last'))
    print(acceptance)
    print(f'a: should be 5: {statistics.mean(df.loc[:,0])}')
    print(f'b mean: should be 0: {statistics.mean(df.loc[:,1])}')
    print(f'b sd: should be ?: {statistics.stdev(df.loc[:,1])}')
    print(f'sd: should be 10: {statistics.mean(df.loc[:,2])}')
    # first_col=df.take([0], axis=1)
    # print(first_col)
    print(df.iloc[:,0])
    draw_histograms(df, df.columns, 1, 3)


if __name__ == '__main__':
    main()