import numpy as np
import time
from heapq import *

(FM_BLACK, FM_RED, FM_GREEN) = (0, 1, 2)

#For examining how many angles are obtuse
def getAllAngles(c):
    scale = np.max(c.D)/c.D.shape[0]
    angles = []
    for f in c.faces:
        V = f.getVerts(scale)
        for i in range(3):
            VRel = V - V[:, i]
            idx = np.arange(3)
            idx = idx[abs(idx - i) > 0]
            VRel = VRel[:, idx]
            norms = np.sqrt(np.sum(VRel**2, 0))
            angles.append(np.arccos(VRel[:, 0].transpose().dot(VRel[:, 1])/np.prod(norms))*180/np.pi)
    return angles

def updateTriangle(n1, n2, n3, FMMDist):
    V = np.zeros((3, 3))
    v1 = n1.get3DCoords()
    v2 = n2.get3DCoords()
    v3 = n3.get3DCoords()
    v1 = v1 - v3
    v2 = v2 - v3
    Q = np.linalg.inv(V.transpose().dot(V))
    col1 = np.ones((2, 1))
    row1 = np.ones((1, 2))
    d = np.ones((2, 1))
    d[0] = FMMDist[n1.i, n1.j]
    d[1] = FMMDist[n2.i, n2.j]
    #Quadratic formula to solve for p (d3)
    a = row1.dot(Q.dot(col1))
    b = -2*row1.dot(Q.dot(d))
    c = (d.transpose()).dot(Q.dot(d)) - 1
    if b**2 < 4*a*c:
        return None
    p = b + np.sqrt(b**2 - 4*a*c)/(2*a)
    

#c: MeshComplex, n0: Initial node
def doFastMarching(c, n0):
    #Attempt to scale xy coordinates so they have a similar range as
    #the height coordinate
    scale = np.max(c.D)/c.D.shape[0]
    
    #Front distances
    FMDists = 0*c.D
    FMTypes = FM_GREEN*np.ones(c.D.shape)
    
    #Initialize first red nodes around black starting node
    #and initialize the front heap
    NBlack = 1
    for n in c.nodes:
        FMDists[n.i, n.j] = np.inf
        FMTypes[n.i. n.j] = FM_GREEN
    FMDists[n0.i, n0.j] = 0.0
    FMTypes[n0.i, n0.j] = FM_BLACK
    front = []
    for n in n0.neighbs:
        FMTypes[n.i, n.j] = FM_RED
        d = n.get3DCoords(scale) - n0.get3DCoords(scale)
        d = np.sqrt(np.sum(d**2))
        FMDists[n.i, n.j] = d
        heappush(front, (d, n))
    
    while NBlack < len(c.nodes):
        #Get the point with the smallest value of d(x) that is on
        #the (red) front
        n1 = heappop(front)
        while FMTypes[n1.i, n1.j] == FM_BLACK and len(front) > 0:
            (d, n1) = heappop(front)
        if FMTypes[n1.i, n1.j] == FM_BLACK:
            break
        #Update all triangles that border n1 whose x3s are not black
        for tri in n1.faces:
            ns = [n for n in tri.nodes]
            while not ns[0] == n1:
                ns = ns[1:] + [ns[0]]
            if not FMTypes[ns[2].i, ns[2].j] == FM_BLACK:
                updateTriangle(ns[0], ns[1], ns[2], FMDists)
                FMTypes[ns[2].i, ns[2].j] = FM_RED
                heappush(front, (FMDists[ns[2].i, ns[2].j], ns[2]))
        #Remove n1 from the unprocessed set (make it "black")
        FMTypes[n1.i, n1.j] = FM_BLACK
        NBlack += 1
