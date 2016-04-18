import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from FastMarching import *
from DGMTools import *
from Skeleton import *
from SSMTopological import *

def getSSM(t):
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(t*2*np.pi)
    X[:, 1] = np.sin(2*t*2*np.pi)
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    D[D < 0] = 0
    D = np.sqrt(D)
    return D

if __name__ == '__main__':
    N = 200
    NGrid = 10
    GridSkip = N/NGrid
    t1 = np.linspace(0, 1, N)
    t2 = t1**2
    NPeriods = 3
    t3 = 1.1 + np.sin(NPeriods*2*np.pi*t1)
    t3 = np.cumsum(t3)
    t3 = t3/np.max(t3)
    
    #Make warped grids
    sgrid2 = np.zeros(N)
    idx = 0
    for ii in range(1, N):
        while t2[idx] < t1[ii]:
            idx = idx+1
        #Linear interpolation
        a = t2[idx-1]
        b = t2[idx]
        sgrid2[ii] = ((b - t1[ii])*(idx-1) + (t1[ii] - a)*idx)/(b-a)

    sgrid3 = np.zeros(N)
    idx = 0
    for ii in range(1, N):
        while t2[idx] < t1[ii]:
            idx = idx+1
        #Linear interpolation
        a = t2[idx-1]
        b = t2[idx]
        sgrid3[ii] = ((b - t1[ii])*(idx-1) + (t1[ii] - a)*idx)/(b-a)
    
    #SSM1 Plot
    plt.figure(figsize=(10,10))
    D1 = getSSM(t1)
    c1 = SSMComplex(D1)
    c1.makeMesh()
    c1.plotMesh(False)
    plt.hold(True)
    c1.plotCriticalPoints()
    t1 = t1*N
    for ii in range(0, N, GridSkip):
        plt.plot([t1[ii], t1[ii]], [0, N], 'k')
        plt.plot([0, N], [t1[ii], t1[ii]], 'k')

    plt.axis('off')
    plt.title('SSM Parameterized by t')
    plt.savefig('SSMt.svg', bbox_inches='tight')
    
    plt.show()
    #Persistence diagram plots
    print c1.IJoin
    plotDGM(c1.IJoin)
    plt.title('Join Tree Persistence Diagram')
    plt.show()
    print c1.ISplit
    plotDGM(c1.ISplit[:, [1, 0]])
    plt.title('Split Tree Persistence Diagram')
    plt.show()
    
    #SSM2 Plot
    plt.figure(figsize=(10,10))
    D2 = getSSM(t2)
    c2 = SSMComplex(D2)
    c2.makeMesh()
    c2.plotMesh(False)
    plt.hold(True)
    c2.plotCriticalPoints()
    for ii in range(0, N, GridSkip):
        plt.plot([sgrid2[ii], sgrid2[ii]], [0, N], 'k')
        plt.plot([0, N], [sgrid2[ii], sgrid2[ii]], 'k')
    
    plt.axis('off')
    plt.title('SSM Parameterized by u(t)')
    plt.savefig('SSMut.svg', bbox_inches='tight')
    
    #SSM3 Plot
    plt.figure(figsize=(10,10))
    D3 = getSSM(t3)
    c3 = SSMComplex(D3)
    c3.makeMesh()
    c3.plotMesh(False)
    plt.hold(True)
    c3.plotCriticalPoints()
    for ii in range(0, N, GridSkip):
        plt.plot([sgrid3[ii], sgrid3[ii]], [0, N], 'k')
        plt.plot([0, N], [sgrid3[ii], sgrid3[ii]], 'k')
    
    plt.axis('off')
    plt.title('SSM Parameterized by v(t)')
    plt.savefig('SSMvt.svg', bbox_inches='tight')
