NPoints = 4000;
t1 = linspace(pi/2, 3.5*pi, NPoints);
X1 = [t1(:) cos(t1(:))];
t2 = linspace(0, pi, NPoints);

idx = 0;
for dX2 = linspace(0, 2.5*pi, 2.5*90+1)
    X2 = [t2(:)+dX2 2+sin(t2(:))];
    clf;

    D = pdist2(X1, X2);
    I = classifyCriticalPoints(D);
    [x, y] = meshgrid(1:size(D, 1), 1:size(D, 1));


    subplot(1, 2, 2);
    imagesc(D);
    axis off;
    hold on;
    %Plot mins
    xmins = x(I == 0);
    ymins = y(I == 0);
    plot(xmins(:), ymins(:), 'gx');

    %Plot maxs
    xmaxs = x(I == 1);
    ymaxs = y(I == 1);
    plot(xmaxs(:), ymaxs(:), 'rx');

    %Plot saddles
    xsaddles = x(I >= 4);
    ysaddles = y(I >= 4);
    plot(xsaddles(:), ysaddles(:), 'cx');
    
    subplot(1, 2, 1);
    plot(X1(:, 1), X1(:, 2), 'b.');
    hold on;
    plot(X2(:, 1), X2(:, 2), 'r.');
    title(sprintf('%i Degrees', idx));    
    xs = [xmins; xmaxs; xsaddles];
    ys = [ymins; ymaxs; ysaddles];
    scatter(X2(xs, 1), X2(xs, 2), 20, 'k', 'fill');
    scatter(X1(ys, 1), X1(ys, 2), 20, 'k', 'fill');
    axis off;
    
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 5]);
    print('-dpng', '-r100', sprintf('%i.png', idx));
    idx = idx + 1;
end
