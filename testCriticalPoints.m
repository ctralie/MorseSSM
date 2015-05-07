NPoints = 400;
t = linspace(0, 2*pi, NPoints);
X = [cos(t(:)) sin(2*t(:))];
D = squareform(pdist(X));
I = classifyCriticalPoints(D);

[x, y] = meshgrid(1:size(D, 1), 1:size(D, 1));
idxignore = ~((x > y) & (abs(x - y) > 1));
I(idxignore) = 2;

clf;
imagesc(D);
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
print('-dpng', '-r100', 'AllCriticalPoints.png');

%Now plot neighborhoods of found maxs, mins, and saddles
dNeighb = 5;
C = colormap(sprintf('gray(%i)', NPoints));
AllXs = {xmins, xmaxs, xsaddles};
AllYs = {ymins, ymaxs, ysaddles};
AllNames = {'Min', 'Max', 'Saddle'};

for kk = 1:length(AllXs)
    thisx = AllXs{kk};
    thisy = AllYs{kk};
    name = AllNames{kk};
    for ii = 1:length(thisx)
        clf;
        subplot(2, 2, 1:2);
        scatter(X(:, 1), X(:, 2), 10, C, 'fill');
        title(sprintf('%s %i', name, ii));
        hold on;
        idx = max(thisx(ii)-dNeighb, 1):min(thisx(ii)+dNeighb, size(X, 1));
        X1 = X(idx, :);
        idx = max(thisy(ii)-dNeighb, 1):min(thisy(ii)+dNeighb, size(X, 1));
        X2 = X(idx, :);
        plot(X1(:, 1), X1(:, 2), 'r', 'LineWidth', 4);
        plot(X2(:, 1), X2(:, 2), 'b', 'LineWidth', 4);
        subplot(2, 2, 3);
        imagesc(D);
        colormap('default');
        hold on;
        plot(thisx(ii), thisy(ii), 'rx');
        subplot(2, 2, 4);
        imagesc(pdist2(X1, X2));
        print('-dpng', '-r100', sprintf('%s%i.png', name, ii));
    end
end
