#Programmer: Chris Tralie
#Purpose: To build various topological descriptors on top of self-similarity matrices
#http://www2.iap.fr/users/sousbie/web/html/indexd3dd.html?post/Persistence-and-simplification
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
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

class MorseNode(object):
    (REGULAR, MIN, SADDLE, MAX, UNKNOWN) = (-1, 0, 1, 2, 3)
    TypeStrings = {REGULAR:'REGULAR', MIN:'MIN', SADDLE:'SADDLE', MAX:'MAX', UNKNOWN:'UNKNOWN'}
    def __init__(self, i, j, d, listIdx):
        self.i = i
        self.j = j
        self.d = d
        self.listIdx = listIdx
        self.neighbs = []
        self.index = MorseNode.REGULAR
        self.borderNode = False
        self.addtime = -1
        self.P = self #Pointer for union find
        #The two lists below are for storing merge tree pairings
        self.joinNeighbs = [] #Merge tree up
        self.splitNeighbs = [] #Merge tree down

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
        self.borderNode = None #Store diagonal/boundary node separately
        self.mins = []
        self.maxes = []
        self.saddles = []
    
    ###################################################################
    ##              TOPOLOGICAL STRUCTURE ALGORITHMS                 ##
    ###################################################################

    #A helper function that makes a merge tree in one direction, either sweeping
    #up or sweeping down.  Store the merge tree connections implicitly in the 
    #nodes, and return 
    def makeMergeTreeHelper(self, SweepUp = True):
        #Reset union find structure
        for n in self.nodes:
            n.P = n
        #Persistence diagram
        I = []
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
                times = np.array([UFFind(c).addtime for c in components])
                dists = np.array([UFFind(c).d for c in components])
                if SweepUp:
                    persistClass = np.argmin(times)
                else:
                    persistClass = np.argmax(times)
                for k in range(len(components)):
                    if not k == persistClass:
                        I.append([dists[k], node.d])
                #Now merge the components to the saddle
                for c in components:
                    UFUnion(c, node, SweepUp)
                #Update the representative of this class to be the saddle now
                u = UFFind(node)
                repNodes[u] = node
        return np.array(I)
        
    #Compute the merge tree
    def makeMergeTree(self):
        for node in self.nodes:
            node.touched = False
        #Sort the points
        self.order = np.argsort(np.array([n.d for n in self.nodes]))
        for i in range(len(self.order)):
            self.nodes[self.order[i]].addtime = i
        #Make the join tree
        IJoin = self.makeMergeTreeHelper(True)
        #Make the split tree
        ISplit = self.makeMergeTreeHelper(False)
        return (ISplit, IJoin)
    
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
            if j == i+1 or i == 0 or j == 0:
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
        (self.ISplit, self.IJoin) = self.makeMergeTree()

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
    
    def plotJoinTree(self):
        N = self.D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)
        for node in self.mins:
            plt.scatter(node.j, node.i, 100, 'b')
        for node in self.saddles:
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
            plt.scatter(node.j, node.i, 100, 'r')
        for node in self.saddles:
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
        faces = {}
        for n in self.nodes:
            if n == self.borderNode:
                continue
            neighbs = [ne for ne in n.neighbs if not (ne == self.borderNode)]
            for i in range(len(neighbs)):
                a = neighbs[i]
                b = neighbs[(i+1)%len(neighbs)]
                idxs = [n.listIdx, b.listIdx, a.listIdx]
                while not (idxs[0] < idxs[1] and idxs[0] < idxs[2]):
                    idxs = [idxs[2], idxs[0], idxs[1]]
                idxs = [idxs[2], idxs[1], idxs[0]]
                faces["%i:%i:%i"%tuple(idxs)] = idxs
        fout = open("%s.off"%fileprefix, 'w')
        #Don't write out the last node, which is the border node
        fout.write("OFF\n%i %i 0\n"%(len(self.nodes)-1, len(faces)))
        scale = np.max(self.D)/self.D.shape[0]
        for i in range(len(self.nodes)-1):
            n = self.nodes[i]
            fout.write("%g %g %g\n"%(n.j*scale, n.i*scale, n.d))
        for fstring in faces:
            f = faces[fstring]
            fout.write("3 %i %i %i\n"%tuple(f))
        fout.close()
        #Now save indices into critical points in the mesh to export to Matlab
        mins = np.array([m.listIdx for m in self.mins])
        saddles = np.array([s.listIdx for s in self.saddles])
        maxes = np.array([m.listIdx for m in self.maxes])
        sio.savemat("%s.mat"%fileprefix, {'mins':mins, 'saddles':saddles, 'maxes':maxes})
        

if __name__ == '__main__':
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

#    D = sio.loadmat('MusicFeatures/beatles.mat')
#    D = D['D']
#    D = np.reshape(D[100, :], (200, 200))

    c = SSMComplex(D)
    c.makeMesh()
    print "euler = %i  (nMins = %i, nSaddles = %i, nMaxes = %i)"%c.getEuler()
    
    plt.subplot(1, 2, 1)
    c.plotJoinTree()
    plt.subplot(1, 2, 2)
    plotDGM(c.IJoin)
    #c.plotMesh(False)
    plt.show()

if __name__ == '__main__2':
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
    
