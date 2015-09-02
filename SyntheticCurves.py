#Programmer: Chris Tralie
#Purpose: To create a collection of functions for making random curves and applying
#random rotations/translations/deformations/reparameterizations to existing curves
#to test out the Morse matching algorithm
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from MorseComplex import *
import sklearn.metrics.pairwise as pdist

def makeRandomWalkCurve(res, NPoints, dim):
    #Enumerate all neighbors in hypercube via base 3 counting between [-1, 0, 1]
    Neighbs = np.zeros((3**dim, dim))
    Neighbs[0, :] = -np.ones((1, dim))
    idx = 1
    for ii in range(1, 3**dim):
        N = np.copy(Neighbs[idx-1, :])
        N[0] += 1
        for kk in range(dim):
            if N[kk] > 1:
                N[kk] = -1
                N[kk+1] += 1
        Neighbs[idx, :] = N
        idx += 1
    #Exclude the neighbor that's in the same place
    Neighbs = Neighbs[np.sum(np.abs(Neighbs), 1) > 0, :]

    #Pick a random starting point
    X = np.zeros((NPoints, dim))
    X[0, :] = np.random.choice(res, dim)
    
    #Trace out a random path
    for ii in range(1, NPoints):
        prev = np.copy(X[ii-1, :])
        N = np.tile(prev, (Neighbs.shape[0], 1)) + Neighbs
        #Pick a random next point that is in bounds
        idx = np.sum(N > 0, 1) + np.sum(N < res, 1)
        N = N[idx == 2*dim, :]
        X[ii, :] = N[np.random.choice(N.shape[0], 1), :]
    return X

def applyRandomRigidTransformation(X):
    dim = X.shape[1]
    CM = np.mean(X, 0)
    X = X - CM
    #Make a random rotation matrix
    R = np.random.randn(dim, dim)
    R, S, V = np.linalg.svd(R)
    T = np.std(X)*np.random.randn(1, dim)
    return CM + np.dot(X, R) + np.tile(T, (X.shape[0], 1))

def smoothCurve(X, Fac):
    NPoints = X.shape[0]
    dim = X.shape[1]
    idx = range(NPoints)
    idxx = np.linspace(0, NPoints, NPoints*Fac)
    Y = np.zeros((NPoints*Fac, dim))
    NPointsOut = 0
    for ii in range(dim):
        Y[:, ii] = interp.spline(idx, X[:, ii], idxx)
        #Smooth with box filter
        y = (0.5/Fac)*np.convolve(Y[:, ii], np.ones(Fac*2), mode='same')
        Y[0:len(y), ii] = y
        NPointsOut = len(y)
    Y = Y[0:NPointsOut-1, :]
    Y = Y[2*Fac:-2*Fac, :]
    return Y

if __name__ == "__main__":
    np.random.seed(100)
    X = makeRandomWalkCurve(4, 20, 2)
    Y = smoothCurve(X, 10)
    plt.subplot(1, 2, 1)
    plt.scatter(Y[:, 0], Y[:, 1], 10, 'b')
    plt.plot(Y[:, 0], Y[:, 1], 'r')
    
    #Create morse flow lines
    D = pdist.pairwise_distances(Y, Y)
    c = MorseComplex(D)
    c.makeMesh()
    c.makeFlowLines()
    plt.subplot(1, 2, 2)
    c.plotFlowLines()
    plt.show()
