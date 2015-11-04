from oct2py import octave
import sys
import os
import scipy.io as sio
import matplotlib.pyplot as plt
sys.path.append('../')
from SSMTopological import *
from DGMTools import *
from multiprocessing import Pool
octave.addpath('../') #Needed to use SSM and CSM

PERSTHRESH = 0.025

def getRowDists(args):
    (dgm1, dgm2, dgm3, dgm4) = args
    (matchidx, dist, D) = getWassersteinDist(dgm1, dgm2)
    #Reflect dgm3 and dgm4 across the diagonal
    if dgm3.size > 0:
        dgm3 = dgm3[:, [1, 0]]
    if dgm4.size > 0:
        dgm4 = dgm4[:, [1, 0]]
    dist += getWassersteinDist(dgm3, dgm4)[1]
    #sys.stdout.write(".")
    #sys.stdout.flush()
    return dist

if __name__ == '__main__':
    Ds1 = sio.loadmat('Ds1.mat')
    Ds1 = Ds1['D']
    Ds2 = sio.loadmat('Ds2.mat')
    Ds2 = Ds2['D']
    CSM1 = octave.mypdist2(Ds1, Ds2)
    
    #Now compute Morse filtrations on all self-similarity images
    IsJoin1 = []
    IsSplit1 = []
    dim = int(np.sqrt(Ds1.shape[1]))
    if not os.path.isfile("Is1.mat"):
        for i in range(Ds1.shape[0]):
            print "Computing %i of %i for SSMs1"%(i, Ds1.shape[0]),
            c = SSMComplex(np.reshape(Ds1[i, :], [dim, dim]))
            c.makeMesh()
            print ", euler = %i (%i mins, %i saddles, %i maxes)"%c.getEuler()
            IsJoin1.append(c.IJoin)
            IsSplit1.append(c.ISplit)
        sio.savemat("Is1.mat", {"IsJoin1":IsJoin1, "IsSplit1":IsSplit1})
    else:
        Is1 = sio.loadmat("Is1.mat")
        IsJoin1 = Is1["IsJoin1"]
        IsJoin1 = [IsJoin1[0][i] for i in range(len(IsJoin1[0]))]
        IsSplit1 = Is1["IsSplit1"]
        IsSplit1 = [IsSplit1[0][i] for i in range(len(IsSplit1[0]))]
    IsJoin2 = []
    IsSplit2 = []
    dim = int(np.sqrt(Ds2.shape[1]))
    if not os.path.isfile("Is2.mat"):
        for i in range(Ds2.shape[0]):
            print "Computing %i of %i for SSMs2"%(i, Ds2.shape[0]),
            c = SSMComplex(np.reshape(Ds2[i, :], [dim, dim]))
            c.makeMesh()
            print ", euler = %i (%i mins, %i saddles, %i maxes)"%c.getEuler()
            IsJoin2.append(c.IJoin)
            IsSplit2.append(c.ISplit)
        sio.savemat("Is2.mat", {"IsJoin2":IsJoin2, "IsSplit2":IsSplit2})
    else:
        Is2 = sio.loadmat("Is2.mat")
        IsJoin2 = Is2["IsJoin2"]
        IsJoin2 = [IsJoin2[0][i] for i in range(len(IsJoin2[0]))]
        IsSplit2 = Is2["IsSplit2"]
        IsSplit2 = [IsSplit2[0][i] for i in range(len(IsSplit2[0]))]
    
    CSM2 = np.zeros((len(IsJoin1), len(IsJoin2)))
    parpool = Pool(processes = 8)
    for i in range(CSM2.shape[0]):
        print "Comparing diagram set %i of %i..."%(i, CSM2.shape[0])
        Z = zip([IsJoin1[i]]*CSM2.shape[1], IsJoin2, [IsSplit1[i]]*CSM2.shape[1], IsSplit2)
        CSM2[i, :] = np.array(parpool.map(getRowDists, Z))
        sio.savemat("CSM2.mat", {"CSM2":CSM2})
