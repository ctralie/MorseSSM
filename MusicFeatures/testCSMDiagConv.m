load('CSM2.mat');
D = CSM2.^2;
D = D';

for W = 2:100
    ND = size(D, 1)-W+1;
    MD = size(D, 2)-W+1;
    CSMDelay = zeros(ND, MD);
    %Do upper right part
    for jj = 1:MD
        d = diag(D, jj-1);
        d2 = cumsum(d);
        d2 = d2(W:end) - [0; cumsum(d(1:end-W))];
        CSMDelay( (jj-1)*ND + 1 + (0:length(d2)-1)*(ND+1)) = d2;
    end
    %Do lower left part
    for ii = 1:ND-1
        d = diag(D, -ii);
        d2 = cumsum(d);
        d2 = d2(W:end) - [0; cumsum(d(1:end-W))];
        CSMDelay((ii+1) + (0:length(d2)-1)*(ND+1)) = d2;
    end
    subplot(1, 2, 1);
    imagesc(sqrt(CSMDelay));
    title(sprintf('W = %i', W));
    subplot(1, 2, 2);
    imagesc(groundTruthKNN(CSMDelay, 40));
    print('-dpng', '-r100', sprintf('%i.png', W));
end