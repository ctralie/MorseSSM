function [X, boneNames] = getMOCAPTrajectories(asfName, amcName)
    addpath(genpath('MOCAP/HDM05-Parser'));
    [skel, mot] = readMocap(asfName, amcName);
    T = mot.jointTrajectories;
    X = zeros(size(T{1}, 1), length(T), size(T{1}, 2));
    for ii = 1:length(T)
        X(:, ii, :) = T{ii};
    end
    boneNames = skel.boneNames;
end
