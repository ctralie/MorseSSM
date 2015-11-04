function [D, X, SampleDelays, bts, beatIdx] = getBeatSyncDs( filenameout, songfilename, tempobias, BeatsPerBlock, dim )
    addpath('rastamat');
    addpath('coverssrc');

    %Compute beat-synchronous MFCC for both songs
    [XAudio, Fs] = audioread(songfilename);
    if size(XAudio, 2) > 1
        XAudio = mean(XAudio, 2);
    end
    bts = beat(XAudio, Fs, tempobias, 6);
    tempoPeriod = mean(bts(2:end) - bts(1:end-1))
    [X, SampleDelays] = getMFCCTempoWindow(XAudio, Fs, tempoPeriod, 200);
    fprintf(1, 'Computing self-similarity matrices for %s...\n', songfilename);
    
    N = length(bts)-BeatsPerBlock;
    D = zeros(N, dim*dim);
    
    beatIdx = zeros(1, length(bts));
    idx = 1;
    for ii = 1:N
        while(SampleDelays(idx) < bts(ii))
            idx = idx + 1;
        end
        beatIdx(ii) = idx;
    end
    
    %Point center and sphere-normalize point clouds
    validIdx = ones(1, N);
    parfor ii = 1:N
        Y = X(beatIdx(ii)+1:beatIdx(ii+BeatsPerBlock), :);
        if (isempty(Y))
            validIdx(ii) = 0;
            continue;
        end
        Y = bsxfun(@minus, mean(Y), Y);
        Norm = 1./(sqrt(sum(Y.*Y, 2)));
        Y = Y.*(repmat(Norm, [1 size(Y, 2)]));
        dotY = dot(Y, Y, 2);
        thisD = bsxfun(@plus, dotY, dotY') - 2*(Y*Y');
        thisD(thisD < 0) = 0;
        thisD = sqrt(thisD);
        thisD = imresize(thisD, [dim dim]);
        D(ii, :) = thisD(:);
    end
    idx = find(validIdx == 0);
    D = D(1:idx-1, :);
    
    save(filenameout, 'D');
end

