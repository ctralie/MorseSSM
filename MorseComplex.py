#http://www2.iap.fr/users/sousbie/web/html/indexd3dd.html?post/Persistence-and-simplification
import numpy as np
import matplotlib.pyplot as plt
import math

class MorseNode(object):
    (MIN, SADDLE, MAX) = (0, 1, 2)
    def __init__(self, i, j, d):
        self.i = i
        self.j = j
        self.d = d
        self.neighbs = []
        self.index = -1
        self.order = 0
    
    def calcIndex(self):
        print "TODO"

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
    
    #Create all of the vertices and edges between them on the upper triangular
    #part of the mesh minus the diagonal
    def makeMesh(self):
        #Num finite edges
        N = self.D.shape[0]
        #Create a node that connects to all boundary points
        idx = np.round(N/2)
        self.borderNode = MorseNode(idx, idx, -1)
        self.borderNode.index = MorseNode.MIN
        self.borderNode.order = -1 #The border is the absolute min to give sphere topology
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
        for i in range(len(self.nodes)):
            self.nodes[i].order = self.order[i]
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
                    else:
                        thisNode.neighbs.append(self.nodes[nidx])
                            
    def plotMesh(self):
        N = self.D.shape[0]
        implot = plt.imshow(-self.D, interpolation = "none")
        implot.set_cmap('Greys')
        plt.hold(True)
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
        for node in self.nodes:
            plt.scatter(node.j, node.i, 20, 'b')
        plt.show()


if __name__ == '__main__':
    N = 10
    t = np.linspace(0, 2*np.pi, N)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(2*t)
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    c = MorseComplex(D)
    c.makeMesh()
    c.plotMesh()
