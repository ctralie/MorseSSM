//Programmer: Chris Tralie
//Purpose: To classify the points in an image as regular, local min, local max, saddle
#include <mex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <queue>
#include <list>
#include <vector>
#include <assert.h>

using namespace std;

//Inputs: D (N x N): Distance Matrix
//Output: I (N x N): Classification of points matrix.  0 min, 1 max, 2 regular point, 2*k k-saddle
void mexFunction(int nOutArray, mxArray *OutArray[], int nInArray, const mxArray *InArray[]) {  
	///////////////MEX INPUTS/////////////////
	const mwSize *dims;
	if (nInArray < 1) {
		mexErrMsgTxt("Error: Self-Similarity matrix required\n");
		return;
	}
	dims = mxGetDimensions(InArray[0]);
	int N = (int)dims[0];
	if ((int)dims[1] != N) {
	    mexErrMsgTxt("Error: Self-Similarity matrix should be square\n");
	    return;
	}
	double* D = (double*)mxGetPr(InArray[0]);
	double* I = new double[N*N];
	double Ns[8];//Neighbors
	const int dis[] = {-1, 0, 1, 1, 1, 0, -1, -1};
	const int djs[] = {1, 1, 1, 0, -1, -1, -1, 0};
	int di, dj;
	
	//Don't bother with boundary points
	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1; j++) {
			for (int neighb = 0; neighb < 8; neighb++) {
				di = dis[neighb];
				dj = djs[neighb];
				if (D[i+di + N*(j+dj)] < D[i + N*j]) {
					Ns[neighb] = -1;
				}
				else {
					Ns[neighb] = 1;
				}
			}
			//Now figure out if this is a regular point, min, max, or saddle
			double switches = 0;
			for (int neighb = 0; neighb < 8; neighb++) {
				
				if (Ns[neighb] != Ns[(neighb+1)%8]) {
					switches++;
				}
			}
			if (switches == 0) {
				if (Ns[0] == -1) {
					//If all points are lower, it's a local max
					switches = 1;
				}
			}
			I[i+j*N] = switches;
		}
	}
	for (int i = 0; i < N; i++) {
		I[i] = 2; //Left column
		I[N*(N-1)+i] = 2;//Right column
		I[i*N] = 2;//Top row
		I[i*N+(N-1)] = 2;//Bottom row
	}
	
	
	///////////////MEX OUTPUTS/////////////////
	mwSize outdims[2];
	outdims[0] = N;
	outdims[1] = N;
	OutArray[0] = mxCreateNumericArray(2, outdims, mxDOUBLE_CLASS, mxREAL);
	double* IPr = (double*)mxGetPr(OutArray[0]);
	memcpy(IPr, I, N*N*sizeof(double));
	
	delete[] I;
}
