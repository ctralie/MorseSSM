%Programmer: Chris Tralie
%Purpose: To exploit Octave's fast matrix multiplication to quickly compute
%self-similarity and cross-similarity matrices on Euclidean data
function [D] = mypdist2(X, Y)
	dotX = dot(X, X, 2);
	dotY = dot(Y, Y, 2);
	D = bsxfun(@plus, dotX, dotY') - 2*(X*Y');	
end
