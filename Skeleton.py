#Programmer: Chris Tralie
#Purpose: Load in an ASF file for the geometry/topology of a skeleton
#And load in an AMC file with information for animating that skeleton
#http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
#Motion capture data can be found in the CMU MOCAP database
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getRotationMatrix(rx, ry, rz):
    rotX = np.eye(4)
    rotX[1, 1] = np.cos(rx)
    rotX[2, 1] = np.sin(rx)
    rotX[1, 2] = -np.sin(rx)
    rotX[2, 2] = np.cos(rx)
    rotY = np.eye(4)
    rotY[0, 0] = np.cos(ry)
    rotY[2, 0] = np.sin(ry)
    rotY[0, 2] = -np.sin(ry)
    rotY[2, 2] = np.cos(ry)
    rotZ = np.eye(4)
    rotZ[0, 0] = np.cos(rz)
    rotZ[1, 0] = np.sin(rz)
    rotZ[0, 1] = -np.sin(rz)
    rotZ[1, 1] = np.cos(rz)
    
    return rotZ.dot(rotY.dot(rotX))

class SkeletonRoot(object):
    def __init__(self):
        self.id = -1
        self.name = "root"
        self.axis = "XYZ"
        self.order = {}
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0]
        self.children = []
        self.initialRotMatrix = None
    
    def finishInit(self):
        #Precompute Rotation matrix
        angles = [float(a)*np.pi/180.0 for a in self.orientation]
        self.initialRotMatrix = getRotationMatrix(angles[0], angles[1], angles[2])

class SkeletonBone(object):
    def __init__(self):
        self.name = "NONAME"
        self.id = -1
        self.direction = [0, 0, 0]
        self.axis = [0, 0, 0]
        self.length = 0.0
        self.dof = {}
        self.limits = []
        self.children = []
        self.initialRotMatrix = None

class Skeleton(object):
    (PARSE_DEFAULT, PARSE_UNITS, PARSE_DOCUMENTATION, PARSE_ROOT, PARSE_BONEDATA, PARSE_BONEDATALIMITS, PARSE_HIERARCHY, PARSE_FINISHED) = (0, 1, 2, 3, 4, 5, 6, 7)

    def __init__(self):
        self.version = "1.0"
        self.units = []
        self.documentation = []
        self.root = SkeletonRoot()
        self.bones = {'root':self.root}
    
    def initFromFile(self, filename):
        fin = open(filename, 'r')
        lineCount = 0
        parseState = Skeleton.PARSE_DEFAULT
        thisBone = None
        for line in fin:
            lineCount = lineCount + 1
            fields = ((line.lstrip()).rstrip()).split() #Splits whitespace by default
            if len(fields) == 0: 
                continue #Blank line
            if fields[0][0] in ['#', '\0'] or len(fields[0]) == 0:
                continue #Comments and stuff
            if parseState == Skeleton.PARSE_DEFAULT:
                if fields[0] == ":version":
                    self.version = fields[1]
                elif fields[0] == ":name":
                    self.name = fields[1]
                elif fields[0] == ":units":
                    parseState = Skeleton.PARSE_UNITS
                else:
                    print "Unexpected line while in PARSE_DEFAULT: %s"%line
            elif parseState == Skeleton.PARSE_UNITS:
                if fields[0] == ":documentation":
                    parseState = Skeleton.PARSE_DOCUMENTATION
                elif fields[0] == ":root":
                    parseState = Skeleton.PARSE_ROOT
                elif fields[0] == ":bonedata":
                    parseState = Skaleton.PARSE_BONEDATA
                else:
                    self.units.append(line)
            elif parseState == Skeleton.PARSE_DOCUMENTATION:
                if fields[0] == ":root":
                    parseState = Skeleton.PARSE_ROOT
                elif fields[0] == ":bonedata":
                    parseState = Skeleton.PARSE_BONEDATA
                else:
                    self.documentation.append(line)
            elif parseState == Skeleton.PARSE_ROOT:
                if fields[0] == ":bonedata":
                    self.root.finishInit()
                    parseState = Skeleton.PARSE_BONEDATA
                else:
                    #print "ROOT FIELD: |%s|"%fields[0]
                    if fields[0] == "axis":
                        self.root.axis = fields[1]
                    elif fields[0] == "order":
                        orderstr = line.split("order")[1].lstrip()
                        ordervals = orderstr.split()
                        for i in range(len(ordervals)):
                            self.root.order[ordervals[i].lstrip().rstrip()] = i
                    elif fields[0] == "position":
                        point = [float(x) for x in fields[1:]]
                        self.root.position = point
                    elif fields[0] == "orientation":
                        orientation = [float(x) for x in fields[1:]]
                        self.root.orientation = orientation
                    else:
                        print "Warning: unrecognized field %s in root"%fields[0]
            elif parseState == Skeleton.PARSE_BONEDATA:
                #print "BONE FIELD: |%s|"%fields[0]
                if fields[0] == "begin":
                    thisBone = SkeletonBone()
                elif fields[0] == "end":
                    self.bones[thisBone.name] = thisBone
                elif fields[0] == "name":
                    thisBone.name = fields[1]
                elif fields[0] == "id":
                    thisBone.id = int(fields[1])
                elif fields[0] == "direction":
                    direction = np.array([float(x) for x in fields[1:]])
                    thisBone.direction = direction
                elif fields[0] == "length":
                    thisBone.length = float(fields[1])
                elif fields[0] == "axis":
                    axis = np.array([float(x) for x in fields[1:4]])
                    thisBone.axis = axis
                elif fields[0] == "dof":
                    dof = [(x.lstrip().rstrip()).lower() for x in fields[1:]]
                    for i in range(0, len(dof)):
                        thisBone.dof[dof[i]] = i
                elif fields[0] == "limits":
                    parseState = Skeleton.PARSE_BONEDATALIMITS
                    limits = line.split("(")[1]
                    limits = limits.split(")")[0]
                    limits = [float(x) for x in limits.split()]
                    thisBone.limits.append(limits)
                elif fields[0] == ":hierarchy":
                    parseState = Skeleton.PARSE_HIERARCHY
            elif parseState == Skeleton.PARSE_BONEDATALIMITS:
                if fields[0] == "end":
                    self.bones[thisBone.name] = thisBone
                    parseState = Skeleton.PARSE_BONEDATA
                else:
                    limits = line.split("(")[1]
                    limits = limits.split(")")[0]
                    limits = [float(x) for x in limits.split()]
                    thisBone.limits.append(limits)
            elif parseState == Skeleton.PARSE_HIERARCHY:
                if len(fields) == 1 and fields[0] == "begin":
                    parseState = Skeleton.PARSE_HIERARCHY
                elif len(fields) == 1 and fields[0] == "end":
                    parseState = Skeleton.PARSE_FINISHED
                else:
                    parent = fields[0]
                    children = fields[1:]
                    self.bones[parent].children = [self.bones[s] for s in children]
            elif parseState == Skeleton.PARSE_FINISHED:
                print "Warning: Finished, but got line %s"%line
        fin.close()

    #Functions for exporting tree to numpy
    def getEdgesRec(self, node, edges, kindex):
        i1 = kindex[node.name]
        for c in node.children:
            i2 = kindex[c.name]
            edges.append([i1, i2])
            self.getEdgesRec(c, edges, kindex)
    
    def getEdges(self):
        keys = self.bones.keys()
        kindex = {}
        for i in range(len(keys)):
            kindex[keys[i]] = i
        edges = []
        self.getEdgesRec(self.bones['root'], edges, kindex)
        return np.array(edges)

class SkeletonAnimator(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.bonesStates = {}
        self.boneMatrices = {}
        self.bonePositions = {}
        self.NStates = 0
    
    def initMatrices(self, bone, index):
        [rx, ry, rz] = [0]*3
        if "rx" in bone.dof:
            rx = self.bonesStates[bone.name][index][bone.dof["rx"]]*np.pi/180
        if "ry" in bone.dof:
            ry = self.bonesStates[bone.name][index][bone.dof["ry"]]*np.pi/180
        if "rz" in bone.dof:
            rz = self.bonesStates[bone.name][index][bone.dof["rz"]]*np.pi/180
        rotMatrix = getRotationMatrix(rx, ry, rz)
        pos = bone.length*bone.direction
        translationMatrix = np.eye(4)
        translationMatrix[0:3, 3] = pos
        self.boneMatrices[bone.name].append((rotMatrix, translationMatrix))
        for child in bone.children:
            self.initMatrices(child, index)
    
    #translate then rotate, translate then rotate, ...
    def calcPositions(self, bone, parent, index, matrix):
        (R, T) = self.boneMatrices[bone.name][index]
        #Bone is specified in its parent's coordinate system
        pos = matrix.dot(T[:, 3])
        self.bonePositions[bone.name][index, :] = pos[0:3].flatten()
        matrix = matrix.dot(T)
        for child in bone.children:
            self.calcPositions(child, bone, index, matrix)
    
    def initFromFile(self, filename):
        print "Initializing..."
        for bone in self.skeleton.bones:
            self.bonesStates[bone] = []
        #Step 1: Load in states from file
        fin = open(filename, 'r')
        lineCount = 0
        for line in fin:
            lineCount = lineCount + 1
            fields = ((line.lstrip()).rstrip()).split() #Splits whitespace by default    
            if len(fields) == 0: 
                continue #Blank line
            if fields[0][0] in ['#', '\0', 'o'] or len(fields[0]) == 0:
                continue #Comments and stuff
            if fields[0] == ":FULLY-SPECIFIED":
                continue
            if fields[0] == ":DEGREES":
                continue
            if len(fields) == 1:
                continue #The number of the frame, but I don't need to explicitly store this
            bone = fields[0]
            values = [float(a) for a in fields[1:]]
            self.bonesStates[bone].append(values)
        self.NStates = max([len(self.bonesStates[bone]) for bone in self.bonesStates])            
        fin.close()
        #Step 2: Initialize matrices
        for bone in self.bonesStates:
            self.boneMatrices[bone] = []
            self.bonePositions[bone] = np.zeros((self.NStates, 3))
        for index in range(self.NStates):
            #First initialize the root matrix
            bone = self.skeleton.bones['root']
            [TX, TY, TZ, RX, RY, RZ] = [0]*6
            rotorder = bone.order.copy()
            if "TX" in bone.order:
                TX = self.bonesStates[bone.name][index][bone.order["TX"]]
            if "TY" in bone.order:
                TY = self.bonesStates[bone.name][index][bone.order["TY"]]
            if "TZ" in bone.order:
                TZ = self.bonesStates[bone.name][index][bone.order["TZ"]]
            if "RX" in bone.order:
                RX = self.bonesStates[bone.name][index][bone.order["RX"]]*np.pi/180
                rotorder["RX"] = rotorder["RX"] - 3
            if "RY" in bone.order:
                RY = self.bonesStates[bone.name][index][bone.order["RY"]]*np.pi/180
                rotorder["RY"] = rotorder["RY"] - 3
            if "RZ" in bone.order:
                RZ = self.bonesStates[bone.name][index][bone.order["RZ"]]*np.pi/180
                rotorder["RZ"] = rotorder["RZ"] - 3
            translationMatrix = np.eye(4)
            translationMatrix[0:3, 3] = np.array([TX, TY, TZ])
            rotMatrix = getRotationMatrix(RX, RY, RZ)
            self.boneMatrices['root'].append((rotMatrix, translationMatrix))
            for child in bone.children:
                self.initMatrices(child, index)
            matrix = rotMatrix.dot(translationMatrix)
            self.bonePositions['root'][index, :] = matrix[0:3, 3].flatten()
            for child in bone.children:
                self.calcPositions(child, bone, index, matrix)
        print "Finished initializing"
    
    def renderNode(self, bone, parent, index):
        if index >= self.NStates:
            return
        #Endpoint are always matrix[0:3, 3]
        P1 = self.bonePositions[parent.name][index, :]
        P2 = self.bonePositions[bone.name][index, :]
        colors = [ [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1] ]
        C = colors[bone.id%len(colors)]
        glColor3f(C[0], C[1], C[2])
        
        glPointSize(20)
        glBegin(GL_POINTS)
        glVertex3f(P1[0], P1[1], P1[2])
        glEnd()
        
        glLineWidth(10)
        glBegin(GL_LINES)
        glVertex3f(P1[0], P1[1], P1[2])
        glVertex3f(P2[0], P2[1], P2[2])
        glEnd()
        
        for child in bone.children:
            self.renderNode(child, bone, index)
    
    def renderState(self, index):
        root = self.skeleton.bones['root']
        for child in root.children:
            self.renderNode(child, root, index)
    
    def getBBox(self):
        bboxmin = np.min(self.bonePositions['root'], 0)
        bboxmax = np.max(self.bonePositions['root'], 0)
        for bonestr in self.bonePositions:
            pos = self.bonePositions[bonestr]
            minpos = np.min(pos, 0)
            maxpos = np.max(pos, 0)
            bboxmin = [min(bboxmin[i], minpos[i]) for i in range(3)]
            bboxmax = [max(bboxmax[i], maxpos[i]) for i in range(3)]
        return np.array([bboxmin, bboxmax])

    #Functions for exporting data to numpy
    def getState(self, index):
        keys = self.bonePositions.keys()
        N = len(keys)
        X = np.zeros((N, 3))
        for i in range(N):
            X[i, :] = self.bonePositions[keys[i]][index, :]
        return X

if __name__ == '__main__':
    skeleton = Skeleton()
    skeleton.initFromFile("13.asf")
    activity = SkeletonAnimator(skeleton)
    activity.initFromFile("13_11.amc")
    index = 10
    edges = skeleton.getEdges()
    X = activity.getState(index)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.plot(X[:, 0], X[:, 1], X[:, 2], 'b.')
    plt.hold(True)
    for i in range(edges.shape[0]):
        e = edges[i, :].flatten()
        plt.plot(X[e, 0], X[e, 1], X[e, 2], 'r')
    plt.show()
