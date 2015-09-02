#http://www2.iap.fr/users/sousbie/web/html/indexd3dd.html?post/Persistence-and-simplification
import numpy as np
import matplotlib.pyplot as plt
import math

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
        self.addtime = 0
    
    def calcIndex(self):
        signChanges = 0
        nsigns = [np.sign(n.addtime - self.addtime) for n in self.neighbs]
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

#Get index into an upper triangular matrix minus the diagonal
#which is stored as a row major list
def getListIndex(i, j, N):
    return N*(N-1)/2 - (N-i)*(N-1-i)/2 + (j-i-1)

class MorseComplex(object):
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
    
    ###################################################################
    ##              MORSE SEGMENTATION ALGORITHMS                    ##
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
        self.borderNode.addtime = -1 #The border is the absolute min to give sphere topology
        d = np.zeros(N*(N-1)/2)
        #Now add all other nodes (only include upper triangular part
        #minus the diagonal)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                self.nodes.append(MorseNode(i, j, self.D[i, j]))
                d[idx] = self.D[i, j]
                idx += 1
        #Compute the order that the nodes appear in the sublevelset filtration
        self.order = np.argsort(d)
        addtimes = np.zeros(self.order.shape)
        for i in range(len(self.order)):
            addtimes[self.order[i]] = i
        for i in range(len(self.nodes)):
            self.nodes[i].addtime = addtimes[i]
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
        for n in self.nodes:
            n.calcIndex()
    
    def makeFlowLines(self):
        #Add nodes in ascending order
        for node in [self.borderNode] + [self.nodes[o] for o in self.order]:
            neighbds = np.array([n.d - node.d for n in node.neighbs])
            if node.index == MorseNode.REGULAR:
                if len(node.flowin) > 0:
                    flowToNode = node.neighbs[np.argmax(neighbds)]
                    node.flowout.append(flowToNode)
                    flowToNode.flowin.append(node)
            elif node.index == MorseNode.SADDLE:
                #Create a list of all groups of pluses that are adjacent
                groups = []
                for i in range(len(node.neighbs)):
                    n = node.neighbs[i]
                    nprev = node.neighbs[(i-1+len(node.neighbs))%len(node.neighbs)]
                    if n.addtime > node.addtime:
                        if i == 0 or nprev.addtime < node.addtime:
                            #Start a new group
                            groups.append([n])
                        else:
                            groups[-1].append(n)
                if node.neighbs[0].addtime > node.addtime and node.neighbs[-1].addtime > node.addtime:
                    #Merge first and last list in a circular fashion
                    groups[0] = groups[0] + groups[-1]
                    groups.pop()
                #For each group, add a path out to the maximum node in that group
                for g in groups:
                    flowToNode = g[np.argmax(np.array([ n.d - node.d for n in g ]))]
                    node.flowout.append(flowToNode)
                    flowToNode.flowin.append(node)            
    
    def getEuler(self):
        nMaxes = 0
        nMins = 1 #Start with a min for the boundary point
        nSaddles = 0
        for n in self.nodes:
            if n.index == MorseNode.SADDLE:
                nSaddles += 1
            elif n.index == MorseNode.MIN:
                nMins += 1
            elif n.index == MorseNode.MAX:
                nMaxes += 1
        return (nMins - nSaddles + nMaxes, nMins, nSaddles, nMaxes)
              
    ###################################################################
    ##                    PLOTTING FUNCTIONS                         ##
    ###################################################################
    def plotCriticalPoints(self, drawRegularPoints = False):
        for node in self.nodes:
            if node.index == MorseNode.REGULAR:
                if drawRegularPoints:
                    plt.scatter(node.j, node.i, 20, 'k')
            elif node.index == MorseNode.MIN:
                plt.scatter(node.j, node.i, 100, 'b')
            elif node.index == MorseNode.MAX:
                plt.scatter(node.j, node.i, 100, 'r')
            elif node.index == MorseNode.SADDLE:
                plt.scatter(node.j, node.i, 100, 'g')
            else:
                plt.scatter(node.j, node.i, 100, 'c')
         
    def plotMesh(self, drawEdges = True, drawRegularPoints = False):
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
        self.plotCriticalPoints(drawRegularPoints)
    
    def plotFlowLines(self):
        D = self.D
        N = D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)   
        self.plotCriticalPoints()
        for node in self.nodes:
            flowout = node.flowout
            for n in flowout:
                [i1, j1] = [node.i, node.j]
                [i2, j2] = [n.i, n.j]
                if i1 == j1:
                    #Handle boundary edge
                    if i2 == 0:
                        [i1, j1] = [i2-1, j2]
                    elif j2 == N-1:
                        [i1, j1] = [i2, j2+1]
                    elif j2 == i2+1:
                        [i1, j1] = [i2, i2]
                plt.plot([j1, j2], [i1, i2], 'r')
        b = np.round(0.01*D.shape[0])
        plt.xlim((-b, D.shape[1]+b))
        plt.ylim((-b, D.shape[0]+b))
        plt.axis('off')
    
    #For each critical point in the SSM associated with a curve in X,
    #plot the neighborhoods in X that gave rise to that SSM region
    def plotCriticalCurveRegions(self, X):
        print "TODO"

if __name__ == '__main__':
    N = 500
    p = 1.62
    thist = 2*np.pi*(np.linspace(0, 1, N)**p)
    ps = np.linspace(0.1, 2, 20)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(thist)
    X[:, 1] = np.sin(2*thist)
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    c = MorseComplex(D)
    c.makeMesh()
    print "euler = %i  (nMins = %i, nSaddles = %i, nMaxes = %i)"%c.getEuler()
    c.makeFlowLines()
    c.plotFlowLines()
    plt.title("p = %g"%p)
    plt.show()

if __name__ == '__main__2':
    N = 1000
    t = np.linspace(0, 1, N)
    ps = np.linspace(0.1, 3, 200)
    for i in range(len(ps)):
        p = ps[i]
        thist = 2*np.pi*t**p
        X = np.zeros((N, 2))
        X[:, 0] = np.cos(thist)
        X[:, 1] = np.sin(2*thist)
        dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
        D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
        c = MorseComplex(D)
        c.makeMesh()
        eulerStr = "euler = %i  (nMins = %i, nSaddles = %i, nMaxes = %i)"%c.getEuler()
        print eulerStr
        c.makeFlowLines()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(X[:, 0], X[:, 1], '.')
        plt.title("p = %g"%p)
        plt.subplot(1, 2, 2)
        c.plotFlowLines()
        plt.title(eulerStr)
        plt.savefig("%i.png"%i)
        plt.close()
