class UnionFind(object):
    def __init__(self, N):
        self.N = N
        self.P = range(N) #Pointers
        self.R = [1]*N #Rank
        self.FullyConnected = False
        
    #Union find "find" with path compression
    def Find(self, u):
        if not self.P[u] == u:
            self.P[u] = self.Find(self.P[u]) #Path compression
            return self.P[u]
        return u

    #Union find "union" with rank-based merging and an object 
    #"repObj" which represents the object that goes with this component
    def Union(self, u, v):
        u = self.Find(u)
        v = self.Find(v)
        if self.P[u] == self.P[v]:
            return #Already in union
        #Merge the one with the smaller rank to the root
        #of the larger one
        [usmall, ularge] = [u, v]
        if self.R[v] < self.R[u]:
            [usmall, ularge] = [v, u]
        self.P[usmall] = self.P[ularge]
        self.R[ularge] += self.R[usmall]
        if self.R[ularge] == self.N:
            self.FullyConnected = True
    
    def __str__(self):
        s = "UNION FIND: \n" + str(self.P) + "\n" + str(self.R)
        return s

if __name__ == '__main__':
    U = UnionFind(10)
    print U
    U.Union(0, 1); print U
    U.Union(0, 4); print U
    U.Union(2, 5); print U
    U.Union(9, 8); print U
    U.Union(5, 9); print U
    U.Union(0, 5); print U
    U.Union(2, 7); print U
    U.Union(3, 9); print U
    U.Union(6, 3); print U
    U.Find(0); print U
