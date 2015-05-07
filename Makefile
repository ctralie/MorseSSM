MEX = mex #Using matlab
#MEX = mkoctfile --mex#Using octave
#Change the path below to match your matlab path
MEXINCLUDE = -I/usr/local/MATLAB/R2014b/extern/include/


#LIBS = -lcudart -lcublas

all: classifyCriticalPoints

classifyCriticalPoints: classifyCriticalPoints.cpp
	$(MEX) -g classifyCriticalPoints.cpp $(MEXINCLUDE)
clean:
	rm -f *.mexa64
