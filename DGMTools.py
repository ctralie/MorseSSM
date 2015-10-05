# Author: Chris Tralie
# Last updated: June 26, 2015
# Description: Contains methods to plot and compare persistence diagrams
#               Comparison algorithms include grabbing/sorting, persistence landscapes,
#               and the "multiscale heat kernel" (CVPR 2015)
#               Also contains wrappers around SSM computation

import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw #Used for fast landscape triangle rastering
import scipy.misc #Used for downsampling rasterized images avoiding aliasing
import time #For timing kernel comparison

##############################################################################
##########                  Plotting Functions                      ##########    
##############################################################################

def plotDGM(dgm):
    # Create Lists
    X = list(zip(*dgm)[0]);
    Y = list(zip(*dgm)[1]);
    # set axis values
    axMin = min(min(X),min(Y));
    axMax = max(max(X),max(Y));
    axRange = axMax-axMin;
    # plot points
    plt.plot(X, Y,'b.')
    # plot line
    plt.plot([axMin-axRange/5,axMax+axRange/5], [axMin-axRange/5, axMax+axRange/5],'k');
    # adjust axis
    plt.axis([axMin-axRange/5,axMax+axRange/5, axMin-axRange/5, axMax+axRange/5])
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')

##############################################################################
##########            Diagram Comparison Functions                  ##########    
##############################################################################

#Do sorting and grabbing with the option to include birth times
#Zeropadding is also taken into consideration
def sortAndGrab(dgm, NBars = 10, BirthTimes = False):
    dgmNP = np.array(dgm)
    if dgmNP.size == 0:
        if BirthTimes:
            ret = np.zeros(NBars*2)
        else:
            ret = np.zeros(NBars)
        return ret
    #Indices for reverse sort
    idx = np.argsort(-(dgmNP[:, 1] - dgmNP[:, 0])).flatten()
    ret = dgmNP[idx, 1] - dgmNP[idx, 0]
    ret = ret[0:min(NBars, len(ret))].flatten()
    if len(ret) < NBars:
        ret = np.append(ret, np.zeros(NBars - len(ret)))
    if BirthTimes:
        bt = dgmNP[idx, 0].flatten()
        bt = bt[0:min(NBars, len(bt))].flatten()
        if len(bt) < NBars:
            bt = np.append(bt, np.zeros(NBars - len(bt)))
        ret = np.append(ret, bt)
    return ret

def getLandscapeRasterized(dgm, xrange, yrange, UpFac = 10):
    I = np.array(dgm)
    if I.size == 0:
        return np.zeros((yrange.size, xrange.size))
    NX = xrange.size
    NY = yrange.size
    #Rasterize on a finer grid and downsample
    NXFine = UpFac*NX
    NYFine = UpFac*NY
    xrangeup = np.linspace(xrange[0], xrange[-1], NXFine)
    yrangeup = np.linspace(yrange[0], yrange[-1], NYFine)
    dx = xrangeup[1] - xrangeup[0]
    dy = yrangeup[1] - yrangeup[0]
    Y = 0.5*(I[:, 1] - I[:, 0]) #Triangle tips
    L = np.zeros((NYFine, NXFine))
    for ii in range(I.shape[0]):
        x = [I[ii, 0], 0.5*np.sum(I[ii, 0:2]), I[ii, 1]]
        y = [0, Y[ii], 0]
        x = np.round((x - xrangeup[0])/dx)
        y = np.round((y - yrangeup[0])/dy)
        yidx, xidx = skdraw.polygon(y, x)
        #Allow for cropping
        yidx = np.minimum(yidx, L.shape[0]-1)
        xidx = np.minimum(xidx, L.shape[1]-1)
        L[yidx, xidx] += 1
    L = scipy.misc.imresize(L, (NY, NX))
    return L
    

#Get a discretized verison of the solution of the heat flow equation
#described in the CVPR 2015 paper
def getHeatRasterized(dgm, sigma, xrange, yrange, UpFac = 10):
    I = np.array(dgm)
    if I.size == 0:
        return np.zeros((yrange.size, xrange.size))
    NX = xrange.size
    NY = yrange.size
    #Rasterize on a finer grid and downsample
    NXFine = UpFac*NX
    NYFine = UpFac*NY
    xrangeup = np.linspace(xrange[0], xrange[-1], NXFine)
    yrangeup = np.linspace(yrange[0], yrange[-1], NYFine) 
    X, Y = np.meshgrid(xrangeup, yrangeup)
    u = np.zeros(X.shape)
    for ii in range(I.shape[0]):
        u = u + np.exp(-( (X - I[ii, 0])**2 + (Y - I[ii, 1])**2 )/(4*sigma))
        #Now subtract mirror diagonal
        u = u - np.exp(-( (X - I[ii, 1])**2 + (Y - I[ii, 0])**2 )/(4*sigma))
    u = (1.0/(4*np.pi*sigma))*u
    u = scipy.misc.imresize(u, (NY, NX))
    return u
    

#Evaluate the continuous heat-based kernel between dgm1 and dgm2 (more correct
#than L2 on the discretized verison above but may be slower because can't exploit
#Octave's fast matrix multiplication when evaluating many, many kernels)
def evalHeatKernel(dgm1, dgm2, sigma):
    kSigma = 0
    I1 = np.array(dgm1)
    I2 = np.array(dgm2)
    for i in range(I1.shape[0]):
        p = I1[i, 0:2]
        for j in range(I2.shape[0]):
            q = I2[j, 0:2]
            qc = I2[j, 1::-1]
            kSigma += np.exp(-(np.sum((p-q)**2))/(8*sigma)) - np.exp(-(np.sum((p-qc)**2))/(8*sigma))
    return kSigma / (8*np.pi*sigma)

#Return the pseudo-metric between two diagrams based on the continuous 
#heat kernel
def evalHeatDistance(dgm1, dgm2, sigma):
    return np.sqrt(evalHeatKernel(dgm1, dgm1, sigma) + evalHeatKernel(dgm2, dgm2, sigma) - 2*evalHeatKernel(dgm1, dgm2, sigma))
