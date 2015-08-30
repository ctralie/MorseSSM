N = 10000;
indexes = zeros(N, 2);
eigvals = zeros(N, 2, 2);

for ii = 1:N
    ii
    vals = randn(1, 5);
    a = vals(1); b = vals(2); c = vals(3); j = vals(4); k = vals(5);

    H1 = [a c; c b];
    H2 = [j*j*a j*k*c; j*k*c k*k*b];

    indexes(ii, 1) = sum(sign(eig(H1)) == 1);
    indexes(ii, 2) = sum(sign(eig(H2)) == 1);
    eigvals(ii, 1, :) = eig(H1);
    eigvals(ii, 2, :) = eig(H2);
end

sum(indexes(:, 1) == indexes(:, 2))

vals = randn(1, 5);
a = vals(1); b = vals(2); c = vals(3); j = vals(4); k = vals(5);
clf;
lambda = linspace(-5, 5, 1000);
y1 = (a*b - c^2 - (a+b)*lambda + lambda.^2);
y2 = ((j*k)^2*(a*b - c^2) - (j^2*a + k^2*b)*lambda + lambda.^2);
plot(lambda, y1, 'b'); hold on; plot(lambda, y2, 'r');
plot(lambda, zeros(size(lambda)), 'k');
plot([0 0], [min([y1 y2]) max([y1 y2])], 'k');