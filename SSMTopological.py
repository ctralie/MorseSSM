#Programmer: Chris Tralie
#Purpose: To build various topological descriptors on top of self-similarity matrices
#http://www2.iap.fr/users/sousbie/web/html/indexd3dd.html?post/Persistence-and-simplification
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from FastMarching import *
from DGMTools import *
from Skeleton import *

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

#Given a bunch of persistence points, 
def getXYOrder(IGens):
    print "TODO"

class MorseNode(object):
    (REGULAR, MIN, SADDLE, MAX, UNKNOWN) = (-1, 0, 1, 2, 3)
    TypeStrings = {REGULAR:'REGULAR', MIN:'MIN', SADDLE:'SADDLE', MAX:'MAX', UNKNOWN:'UNKNOWN'}
    def __init__(self, i, j, d, listIdx):
        self.i = i
        self.j = j
        self.d = d
        self.N = N
        self.listIdx = listIdx
        self.neighbs = []
        self.faces = []
        self.index = MorseNode.REGULAR
        self.borderNode = False
        self.addtime = -1
        self.P = self #Pointer for union find
        #The two lists below are for storing merge tree pairings
        self.joinNeighbs = [] #Merge tree up
        self.splitNeighbs = [] #Merge tree down
    
    def isNeighborOf(self, other):
        for n in self.neighbs:
            if n == other:
                return True
        return False
    
    def get3DCoords(self, scale):
        return np.array([self.i*scale, self.j*scale, self.d])

#Check all pairwise edges
def isTriangle(n1, n2, n3):
    if n1.isNeighborOf(n2) and n2.isNeighborOf(n3) and n3.isNeighborOf(n1):
        return True
    return False

class MeshFace(object):
    def __init__(self, nodes):
        self.nodes = nodes
        #Add pointers from nodes to this face
        for n in nodes:
            n.faces.append(self)
    
    #Return the 3D coordinates of each vertex in this face
    #in each column of the return value
    def getVerts(self, scale):
        N = len(self.nodes)
        ret = np.zeros((3, N))
        for i in range(N):
            n = self.nodes[i]
            ret[:, i] = n.get3DCoords(scale)
        return ret

#Union find "find" with path compression
def UFFind(u):
    if not u.P == u:
        u.P = UFFind(u.P) #Path compression
        return u.P
    return u

#Union find "union" with merging to component with earlier birth time
def UFUnion(u, v, SweepUp):
    uP = UFFind(u)
    vP = UFFind(v)
    if uP == vP:
        return #Already in union
    #Merge to the root of the one with the earlier component time
    [ufirst, usecond] = [uP, vP]
    if usecond.addtime < ufirst.addtime:
        [usecond, ufirst] = [ufirst, usecond]
    if not SweepUp:
        [usecond, ufirst] = [ufirst, usecond]
    usecond.P = ufirst

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
        self.mins = []
        self.maxes = []
        self.saddles = []
    
    ###################################################################
    ##              TOPOLOGICAL STRUCTURE ALGORITHMS                 ##
    ###################################################################
    
    #Explicitly make the faces of the mesh.  Constructing face objects
    #will implicitly add pointers from vertices to faces
    def makeFaces(self):
        self.faces = []
        N = self.D.shape[0]
        for i in range(0, N-2):
            for j in range(i+1, N-1):
                n1 = self.nodes[getListIndex(i, j, N)]
                n2 = self.nodes[getListIndex(i, j+1, N)]
                n3 = self.nodes[getListIndex(i+1, j+1, N)]
                if j == i+1:
                    #Border node
                    if isTriangle(n1, n2, n3):
                        self.faces.append(MeshFace([n1, n3, n2]))
                else:
                    n4 = self.nodes[getListIndex(i+1, j, N)]
                    #Check both possible diagonal directions in quad
                    if n1.isNeighborOf(n3):
                        if n1.isNeighborOf(n2) and n2.isNeighborOf(n3):
                            self.faces.append(MeshFace([n1, n3, n2]))
                        if n1.isNeighborOf(n4) and n4.isNeighborOf(n3):
                            self.faces.append(MeshFace([n1, n4, n3]))
                    else:
                        if n1.isNeighborOf(n4) and n1.isNeighborOf(n2):
                            self.faces.append(MeshFace([n1, n4, n2]))
                        if n4.isNeighborOf(n3) and n3.isNeighborOf(n2):
                            self.faces.append(MeshFace([n4, n3, n2]))

    #A helper function that makes a merge tree in one direction, either sweeping
    #up or sweeping down.  Store the merge tree connections implicitly in the 
    #nodes, and return 
    def makeMergeTreeHelper(self, SweepUp = True):
        #Reset union find structure
        for n in self.nodes:
            n.P = n
        #Persistence diagram and associated generators
        I = []
        IGens = []
        #At first all of the mins/maxes represent themselves, but eventually
        #they can be represented by saddles
        repNodes = {}
        #Sweep through points from low to high (or high to low)
        order = self.order
        if not SweepUp:
            #Sweeping down from top; reverse order
            order = self.order[::-1]
        def compare(n1, n2):
            if SweepUp:
                if n1.addtime < n2.addtime:
                    return True
                return False
            else:
                if n1.addtime > n2.addtime:
                    return True
                return False
        for i in order:
            node = self.nodes[i]
            components = [UFFind(n).listIdx for n in node.neighbs if compare(n, node)]
            components = np.unique(components)
            components = [repNodes[self.nodes[i]] for i in components]
            if len(components) == 0:
                #This is a min point or a max point
                if SweepUp:
                    node.index = MorseNode.MIN
                    self.mins.append(node)
                else:
                    node.index = MorseNode.MAX
                    self.maxes.append(node)
                repNodes[node] = node
            elif len(components) == 1:
                #This is a regular point.  Do nothing except merge
                node.index = MorseNode.REGULAR
                UFUnion(components[0], node, SweepUp)
            else:
                #This is a saddle point
                node.index = MorseNode.SADDLE
                self.saddles.append(node)                
                for c in components:
                    if SweepUp:
                        c.joinNeighbs.append(node)
                        node.joinNeighbs.append(c)
                    else:
                        c.splitNeighbs.append(node)
                        node.splitNeighbs.append(c)
                #Figure out which components die and add entries for them
                #into the persistence diagram
                criticalNodes = [UFFind(c) for c in components]
                times = np.array([c.addtime for c in criticalNodes])
                dists = np.array([c.d for c in criticalNodes])
                if SweepUp:
                    persistClass = np.argmin(times)
                else:
                    persistClass = np.argmax(times)
                for k in range(len(components)):
                    if not k == persistClass:
                        I.append([dists[k], node.d])
                        IGens.append(criticalNodes[k])
                #Now merge the components to the saddle
                for c in components:
                    UFUnion(c, node, SweepUp)
                #Update the representative of this class to be the saddle now
                u = UFFind(node)
                repNodes[u] = node
        return (np.array(I), IGens)
        
    #Compute the merge tree
    def makeMergeTree(self):
        for node in self.nodes:
            node.touched = False
        #Sort the points
        self.order = np.argsort(np.array([n.d for n in self.nodes]))
        for i in range(len(self.order)):
            self.nodes[self.order[i]].addtime = i
        #Make the join tree
        (IJoin, IJoinGens) = self.makeMergeTreeHelper(True)
        #Make the split tree
        (ISplit, ISplitGens) = self.makeMergeTreeHelper(False)
        return (ISplit, ISplitGens, IJoin, IJoinGens)
    
    #Create all of the vertices and edges between them on the upper triangular
    #part of the mesh minus the diagonal.  Once this is done, classify each
    #point as regular/min/max/saddle
    def makeMesh(self):
        #Num finite edges
        N = self.D.shape[0]
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                self.nodes.append(MorseNode(i, j, self.D[i, j], idx))
                idx += 1
        #Add neighbors in graph
        ns = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]
        for thisNode in self.nodes:
            [i, j] = [thisNode.i, thisNode.j]
            if j == i+1 or i == 0 or j == 0 or i == N-1 or j == N-1:
                thisNode.borderNode = True
            for n in ns:
                [ni, nj] = [i + n[0], j + n[1]]
                if nj < ni:
                    #Don't add symmetric things on the other side of the diagonal
                    continue
                if nj < 0 or ni < 0 or nj >= N or ni >= N or ni == nj:
                    #Boundary node
                    continue
                nidx = getListIndex(ni, nj, N)
                #Deal with diagonal cases first
                if abs(n[0]) == abs(n[1]):
                    if j == i+1 and nj == ni+1:
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
                elif not (ni == nj):
                        thisNode.neighbs.append(self.nodes[nidx])
        #Figure out the index of all points (regular/min/max/saddle)
        #by computing a merge tree structure
        del self.mins[:]
        del self.maxes[:]
        del self.saddles[:]
        (self.ISplit, self.ISplitGens, self.IJoin, self.IJoinGens) = self.makeMergeTree()
        self.makeFaces()

    def getEuler(self):
        nMaxes = len(self.maxes)
        nMins = len(self.mins)
        nSaddles = len(self.saddles)
        return (nMins - nSaddles + nMaxes, nMins, nSaddles, nMaxes)
        
    def persistenceCleanup(self, thresh, doJoinTree = True, doSplitTree = True):
        #TODO: Clean up dangling saddles
        if doJoinTree:
            toKeep = np.ones(self.IJoin.shape[0], dtype='bool')
            for i in range(self.IJoin.shape[0]):
                if self.IJoin[i, 1] - self.IJoin[i, 0] < thresh:
                    #This is below the persistence threshold
                    toKeep[i] = 0
                    #Remove it from the tree
                    node = self.IJoinGens[i]
                    saddle = node.joinNeighbs[0]
                    node.joinNeighbs = []
                    saddle.joinNeighbs = [n for n in saddle.joinNeighbs if not (n == node)]
            self.IJoin = self.IJoin[toKeep, :]
        if doSplitTree:
            toKeep = np.ones(self.ISplit.shape[0], dtype='bool')
            for i in range(self.ISplit.shape[0]):
                if self.ISplit[i, 0] - self.ISplit[i, 1] < thresh:
                    #This is below the persistence threshold
                    toKeep[i] = 0
                    #Remove it from the tree
                    node = self.ISplitGens[i]
                    saddle = node.splitNeighbs[0]
                    node.splitNeighbs = []
                    saddle.splitNeighbs = [n for n in saddle.splitNeighbs if not (n == node)]
            self.ISplit = self.ISplit[toKeep, :]
                    
    
    ###################################################################
    ##                    PLOTTING FUNCTIONS                         ##
    ###################################################################
    def plotCriticalPoints(self, doMins = True, doMaxes = True, doSaddles = True):
        if doMins:
            for node in self.mins:
                if len(node.joinNeighbs) > 0:
                    plt.scatter(node.j, node.i, 100, 'b')
        if doMaxes:
            for node in self.maxes:
                if len(node.splitNeighbs) > 0:
                    plt.scatter(node.j, node.i, 100, 'r')
        if doSaddles:
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
    
    def plotJoinTree(self):
        N = self.D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)
        for node in self.mins:
            if len(node.joinNeighbs) > 0:
                plt.scatter(node.j, node.i, 100, 'b')
        for node in self.saddles:
            if len(node.joinNeighbs) > 0:
                plt.scatter(node.j, node.i, 100, 'g')
            [i1, j1] = [node.i, node.j]
            for neighb in node.joinNeighbs:
                [i2, j2] = [neighb.i, neighb.j]
                plt.plot([j1, j2], [i1, i2], 'b')
        for node in self.maxes:
            plt.scatter(node.j, node.i, 100, 'r')

    def plotSplitTree(self):
        N = self.D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)
        for node in self.maxes:
            if len(node.splitNeighbs) > 0:
                plt.scatter(node.j, node.i, 100, 'r')
        for node in self.saddles:
            if len(node.splitNeigbhs) > 0:
                plt.scatter(node.j, node.i, 100, 'g')
            [i1, j1] = [node.i, node.j]
            for neighb in node.splitNeighbs:
                [i2, j2] = [neighb.i, neighb.j]
                plt.plot([j1, j2], [i1, i2], 'r')
        for node in self.mins:
            plt.scatter(node.j, node.i, 100, 'b')
    
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
        fout = open("%s.off"%fileprefix, 'w')
        #Don't write out the last node, which is the border node
        fout.write("OFF\n%i %i 0\n"%(len(self.nodes), len(self.faces)))
        #Attempt to scale xy coordinates so they have a similar range as
        #the height coordinate
        scale = np.max(self.D)/self.D.shape[0]
        for n in self.nodes:
            fout.write("%g %g %g\n"%(tuple(n.get3DCoords(scale))))
        for f in self.faces:
            fidx = [n.listIdx for n in f.nodes]
            fout.write("%i "%len(fidx))
            fmtstr = "%i "*len(fidx) + "\n"
            fout.write(fmtstr%tuple(fidx))
        fout.close()
        #Now save indices into critical points in the mesh to export to Matlab
        mins = np.array([m.listIdx for m in self.mins])
        minscoords = np.array([[m.i, m.j] for m in self.mins])
        saddles = np.array([s.listIdx for s in self.saddles])
        saddlescoords = np.array([[s.i, s.j] for s in self.saddles])
        maxes = np.array([m.listIdx for m in self.maxes])
        maxescoords = np.array([[m.i, m.j] for m in self.maxes])
        sio.savemat("%s.mat"%fileprefix, {'mins':mins, 'minscoords':minscoords, 'saddles':saddles, 'saddlescoords':saddlescoords, 'maxes':maxes, 'maxescoords':maxescoords, 'D':self.D})
        

if __name__ == '__main__2':
    N = 200
    p = 1.8
    thist = 2*np.pi*(np.linspace(0, 1, N)**p)
    ps = np.linspace(0.1, 2, 20)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(thist)
    X[:, 1] = np.sin(2*thist)
    
#    asf = "MOCAP/13.asf"
#    amc = "MOCAP/13_11.amc"
#    skeleton = Skeleton()
#    skeleton.initFromFile(asf)
#    animator = SkeletonAnimator(skeleton)
#    X = animator.initFromFileUsingOctave(asf, amc)
#    print "X.shape = ", X.shape
#    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
#    X = X.T
#    print "X.shape = ", X.shape
    
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    D[D < 0] = 0
    D = np.sqrt(D)

    D = sio.loadmat('MusicFeatures/Ds1.mat')
    D = D['D']
    D = np.reshape(D[100, :], (200, 200))

    c = SSMComplex(D)
    c.makeMesh()
    print "euler = %i  (nMins = %i, nSaddles = %i, nMaxes = %i)"%c.getEuler()
    
    plt.subplot(1, 2, 1)
    c.plotMesh(False)
    plt.hold(True)
    c.plotCriticalPoints(True, True, False)
    plt.subplot(1, 2, 2)
    c.persistenceCleanup(0.025)
    c.plotMesh(False)
    plt.hold(True)
    c.plotCriticalPoints(True, True, False)
    plt.show()
    c.plotJoinTree()
    plt.show()
    
    c.saveOFFMesh("MusicD")

if __name__ == '__main__3':
    T = sio.loadmat('TestDists.mat')
    D1 = T['D1']
    D2 = T['D2']
    c1 = SSMComplex(D1)
    c1.makeMesh()
    c2 = SSMComplex(D2)
    c2.makeMesh()
    I1 = c1.IJoin
    I2 = c2.IJoin
    (matchidx, matchdist) = getWassersteinDist(I1, I2)
    plotWassersteinMatching(I1, I2, matchidx)
    plt.show()
    
if __name__ == '__main__':
    #Make a self-similarity image of a figure 8 to use as a test case
    N = 100
    p = 1.8
    thist = 2*np.pi*(np.linspace(0, 1, N)**p)
    ps = np.linspace(0.1, 2, 20)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(thist)
    X[:, 1] = np.sin(2*thist)
    
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    D[D < 0] = 0
    D = np.sqrt(D)
    c1 = SSMComplex(D)
    c1.makeMesh()
    
    #Output SSM mesh as an OFF files
    c1.saveOFFMesh("Figure8")
    
    #Output critical points
    c1.plotMesh(False)
    plt.hold(True)
    c1.plotCriticalPoints()
    plt.savefig("CriticalPoints.png")
    
    #Output join tree
    I1 = c1.IJoin
    I1Gens = c1.IJoinGens
    for i in range(I1.shape[0]):
        plt.clf()
        plt.subplot(1, 2, 1)
        plotDGM(I1)
        plt.hold(True)
        plt.scatter(I1[i, 0], I1[i, 1], 20, 'r')
        plt.subplot(1, 2, 2)
        c1.plotMesh(False)
        plt.hold(True)
        node = I1Gens[i]
        plt.scatter(node.j, node.i, 200, 'r', 'x')
        plt.savefig("Join%i.png"%i)
    
    #Output split tree
    I1 = c1.ISplit
    I1Gens = c1.ISplitGens
    for i in range(I1.shape[0]):
        plt.clf()
        plt.subplot(1, 2, 1)
        plotDGM(I1)
        plt.hold(True)
        plt.scatter(I1[i, 0], I1[i, 1], 20, 'r')
        plt.subplot(1, 2, 2)
        c1.plotMesh(False)
        plt.hold(True)
        node = I1Gens[i]
        plt.scatter(node.j, node.i, 200, 'r', 'x')
        plt.savefig("Split%i.png"%i)
    
