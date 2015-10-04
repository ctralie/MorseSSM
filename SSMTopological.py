#Programmer: Chris Tralie
#Purpose: To build various topological descriptors on top of self-similarity matrices
#http://www2.iap.fr/users/sousbie/web/html/indexd3dd.html?post/Persistence-and-simplification
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from UnionFind import *


#Get index into an upper triangular matrix minus the diagonal
#which is stored as a row major list
def getListIndex(i, j, N):
    return N*(N-1)/2 - (N-i)*(N-1-i)/2 + (j-i-1)

#Comparator for lexographic order
def lexographicLess(n1, n2):
    if n1.i < n2.i:
        return True
    if n1.j < n2.j:
        return True
    return False

#Used to figure out sign changes
def distComparator(n1, n2):
    if n1.d < n2.d:
        return -1
    if n1.d > n2.d:
        return 1
    #Break ties with lexographic order
    if lexographicLess(n1, n2):
        return -1
    return 1

class MorseNode(object):
    (REGULAR, MIN, SADDLE, MAX, UNKNOWN) = (-1, 0, 1, 2, 3)
    def __init__(self, i, j, d):
        self.i = i
        self.j = j
        self.d = d
        self.neighbs = []
        self.flowin = []
        self.flowout = []
        self.index = MorseNode.REGULAR
        self.signChanges = 0
    
    def calcIndex(self):
        signChanges = 0
        nsigns = [distComparator(n, self) for n in self.neighbs]
        nsigns.append(nsigns[0])
        for i in range(len(nsigns)-1):
            if not (nsigns[i] == nsigns[i+1]):
                signChanges += 1
        if signChanges == 0:
            if nsigns[0] < 0:
                self.index = MorseNode.MAX
            else:
                self.index = MorseNode.MIN
        elif signChanges == 2:
            self.index = MorseNode.REGULAR
        elif signChanges >= 4:
            self.index = MorseNode.SADDLE
        else:
            self.index = MorseNode.UNKNOWN
            print "Warning: %i sign changes detected"%signChanges
            print ("%g: " + "%g "*len(self.neighbs))%tuple([self.d] + [n.d for n in self.neighbs])
        self.signChanges = signChanges

class MergeTreeNode(object):
    def __init__(self, morseNode):
        self.morseNode = morseNode
        self.index = morseNode.index
        self.d = morseNode.d
        self.i = morseNode.i
        self.j = morseNode.j
        self.neighbs = []

class SSMComplex(object):
    def __init__(self, D):
        if not len(D.shape) == 2:
            print "Error: Need to pass in 2D self-simlarity matrix"
            return
        if not D.shape[0] == D.shape[1]:
            print "Error: Expecting square self-similarity matrix"
            return
        self.D = D
        self.nodes = []
        self.borderNode = None #Store diagonal/boundary node separately
        self.mins = []
        self.maxes = []
        self.saddles = []
    
    ###################################################################
    ##              TOPOLOGICAL STRUCTURE ALGORITHMS                 ##
    ###################################################################
    #Create all of the vertices and edges between them on the upper triangular
    #part of the mesh minus the diagonal.  Once this is done, classify each
    #point as regular/min/max/saddle
    def makeMesh(self):
        #Num finite edges
        N = self.D.shape[0]
        #Create a node that connects to all boundary points
        idx = np.round(N/2)
        self.borderNode = MorseNode(idx, idx, 0)
        self.borderNode.index = MorseNode.MIN
        d = np.zeros(N*(N-1)/2)
        #Now add all other nodes (only include upper triangular part
        #minus the diagonal)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                self.nodes.append(MorseNode(i, j, self.D[i, j]))
                d[idx] = self.D[i, j]
                idx += 1
        #Add neighbors in graph
        ns = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]
        for thisNode in self.nodes:
            [i, j] = [thisNode.i, thisNode.j]
            for n in ns:
                [ni, nj] = [i + n[0], j + n[1]]
                if nj < ni:
                    continue
                nidx = getListIndex(ni, nj, N)
                #Deal with diagonal cases first
                if abs(n[0]) == abs(n[1]):
                    if ni == nj or ni < 0 or nj < 0 or ni >= N or nj >= N:
                        #Only connect border along straight lines to save space
                        continue
                    elif j == i+1 and nj == ni+1:
                        #Add diagonals along the border
                        thisNode.neighbs.append(self.nodes[nidx])
                    else:
                        #In the general case, add only if this is the steepest
                        #diagonal in a quad
                        a = thisNode
                        b = self.nodes[nidx]
                        c = self.nodes[getListIndex(i, j+n[1], N)]
                        d = self.nodes[getListIndex(i+n[0], j, N)]
                        if abs(a.d - b.d) > abs(c.d - d.d):
                            thisNode.neighbs.append(self.nodes[nidx])
                #Handle the straight line cases
                else:
                    if ni == nj or ni < 0 or nj < 0 or ni >= N or nj >= N:
                        thisNode.neighbs.append(self.borderNode)
                        self.borderNode.neighbs.append(thisNode)
                    else:
                        thisNode.neighbs.append(self.nodes[nidx])
        #Figure out the index of all points (regular/min/max/saddle)
        del self.mins[:]
        del self.maxes[:]
        del self.saddles[:]
        for n in self.nodes:
            n.calcIndex()
            if n.index == MorseNode.MIN:
                self.mins.append(n)
            elif n.index == MorseNode.MAX:
                self.maxes.append(n)
            elif n.index == MorseNode.SADDLE:
                self.saddles.append(n)
        self.mins.append(self.borderNode)

    def getEuler(self):
        nMaxes = len(self.maxes)
        nMins = len(self.mins)
        nSaddles = len(self.saddles)
        return (nMins - nSaddles + nMaxes, nMins, nSaddles, nMaxes)
              
    ###################################################################
    ##                    PLOTTING FUNCTIONS                         ##
    ###################################################################
    def plotCriticalPoints(self):
        for node in self.mins:
            plt.scatter(node.j, node.i, 100, 'b')
        for node in self.maxes:
            plt.scatter(node.j, node.i, 100, 'r')
        for node in self.saddles:
            plt.scatter(node.j, node.i, 100, 'g')
         
    def plotMesh(self, drawEdges = True):
        N = self.D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)
        if drawEdges:
            for node in self.nodes:
                [i1, j1] = [node.i, node.j]
                for n in node.neighbs:
                    [i2, j2] = [n.i, n.j]
                    if i2 == j2:
                        #Handle boundary edge
                        if i1 == 0:
                            [i2, j2] = [i1-1, j1]
                        elif j1 == N-1:
                            [i2, j2] = [i1, j1+1]
                    plt.plot([j1, j2], [i1, i2], 'r')
        self.plotCriticalPoints()
    
    def plotMergeTree(self, minNodes, saddleNodes):
        N = self.D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)
        for node in minNodes:
            plt.scatter(node.j, node.i, 100, 'b')
        for node in saddleNodes:
            plt.scatter(node.j, node.i, 100, 'g')
            [i1, j1] = [node.i, node.j]
            for neighb in node.neighbors:
                [i2, j2] = [neighb.i, neighb.j]
                plt.plot([j1, j2], [i1, i2], 'r')
    
    #For each critical point in the SSM associated with a curve in X,
    #plot the neighborhoods in X that gave rise to that SSM region
    #NOTE: This function is really only useful for 2D curves
    def plotCriticalCurveRegions(self, X, fileprefix, neighbsize = 5):
        [isaddle, imin, imax] = [0, 0, 0]
        for n in self.nodes:
            if n.index == MorseNode.REGULAR:
                continue
            filename = ""
            color = 'k'
            if n.index == MorseNode.MIN:
                filename = "%smin%i.png"%(fileprefix, imin)
                imin += 1
                color = 'b'
            elif n.index == MorseNode.SADDLE:
                filename = "%ssaddle%i.png"%(fileprefix, isaddle)
                isaddle += 1
                color = 'g'
            elif n.index == MorseNode.MAX:
                filename = "%smax%i.png"%(fileprefix, imax)
                imax += 1
                color = 'r'
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(self.D)
            plt.hold(True)
            plt.scatter(n.j, n.i, 100, color)
            plt.subplot(1, 2, 2)
            plt.plot(X[:, 0], X[:, 1], 'b.')
            plt.hold(True)
            n1 = [max(0, n.i - neighbsize), min(self.D.shape[0], n.i+neighbsize)]
            n2 = [max(0, n.j - neighbsize), min(self.D.shape[0], n.j+neighbsize)]
            plt.plot(X[n1[0]:n1[1], 0], X[n1[0]:n1[1], 1], 'r', linewidth = 5.0)
            plt.scatter(X[n.i, 0], X[n.i, 1], 100, 'r')
            plt.plot(X[n2[0]:n2[1], 0], X[n2[0]:n2[1], 1], 'r', linewidth = 5.0)
            plt.scatter(X[n.j, 0], X[n.j, 1], 100, 'r')
            plt.savefig(filename)
        fout = open("%s.html"%fileprefix, 'w')        
        fout.write("<h1>Mins</h1>\n<BR>\n<table>\n<tr>")
        for i in range(imin):
            fout.write("<td><img src = '%smin%i.png'></td>"%(fileprefix, i))
        fout.write("</tr></table><BR>")
        fout.write("<h1>Maxes</h1>\n<BR>\n<table>\n<tr>")
        for i in range(imax):
            fout.write("<td><img src = '%smax%i.png'></td>"%(fileprefix, i))
        fout.write("</tr></table><BR>")
        fout.write("<h1>Saddles</h1>\n<BR>\n<table>\n<tr>")
        for i in range(isaddle):
            fout.write("<td><img src = '%ssaddle%i.png'></td>"%(fileprefix, i))
        fout.write("</tr></table><BR>")
        fout.close()
        
    ###################################################################
    ##                       I/O FUNCTIONS                           ##
    ###################################################################
    def saveOFFMesh(self, fileprefix):
        #First save mesh file
        faces = {}
        N = self.D.shape[0]
        for n in self.nodes:
            if n == self.borderNode:
                continue
            neighbs = [ne for ne in n.neighbs if not (ne == self.borderNode)]
            for i in range(len(neighbs)):
                a = neighbs[i]
                b = neighbs[(i+1)%len(neighbs)]
                idxs = [getListIndex(n.i, n.j, N), getListIndex(b.i, b.j, N), getListIndex(a.i, a.j, N)]
                while not (idxs[0] < idxs[1] and idxs[0] < idxs[2]):
                    idxs = [idxs[2], idxs[0], idxs[1]]
                idxs = [idxs[2], idxs[1], idxs[0]]
                faces["%i:%i:%i"%tuple(idxs)] = idxs
        fout = open("%s.off"%fileprefix, 'w')
        fout.write("OFF\n%i %i 0\n"%(len(self.nodes), len(faces)))
        scale = np.max(self.D)/self.D.shape[0]
        for i in range(len(self.nodes)):
            n = self.nodes[i]
            fout.write("%g %g %g\n"%(n.j*scale, n.i*scale, n.d))
        for fstring in faces:
            f = faces[fstring]
            fout.write("3 %i %i %i\n"%tuple(f))
        fout.close()
        #Now save indices into critical points in the mesh
        mins = []
        saddles = []
        maxes = []
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node.index == MorseNode.MIN:
                mins.append(i)
            elif node.index == MorseNode.SADDLE:
                saddles.append(i)
            elif node.index == MorseNode.MAX:
                maxes.append(i)
        sio.savemat("%s.mat"%fileprefix, {'mins':np.array(mins), 'saddles':np.array(saddles), 'maxes':np.array(maxes)})
        

if __name__ == '__main__':
    N = 1000
    p = 1.68
    thist = 2*np.pi*(np.linspace(0, 1, N)**p)
    ps = np.linspace(0.1, 2, 20)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(thist)
    X[:, 1] = np.sin(2*thist)
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    c = SSMComplex(D)
    c.makeMesh()
    print "euler = %i  (nMins = %i, nSaddles = %i, nMaxes = %i)"%c.getEuler()
#    (minNodes, saddleNodes, I) = c.makeMergeTree()
#    print I
#    plt.plot(I[:, 0], I[:, 1], 'r.')
#    plt.show()
#    
#    c.plotMergeTree(minNodes, saddleNodes)
#    plt.title("p = %g"%p)
#    plt.show()
    
    c.plotMesh(False)
    plt.show()
