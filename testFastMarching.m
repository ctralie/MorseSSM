addpath(genpath('toolbox_fast_marching'));
load('Figure8.mat');
maxes = maxes+1;
saddles = saddles+1;
saddles = saddles(4);

[V, F] = read_mesh('Figure8.off');

%options.nb_iter_max = Inf;
disp('Performing fast marching...');
[D, S, Q] = perform_fast_marching_mesh(V, F, saddles);
disp('Done Fast Marching');
options.v2v = compute_vertex_ring(F);
options.e2f = compute_edge_face_ring(F);
options.method = 'continuous';
options.verb = 0;
disp('Computing paths...');
paths = compute_geodesic_mesh(D, V, F, maxes, options);
disp('Done computing paths');


figure(1);
clf;
options.colorfx = 'equalize';
plot_fast_marching_mesh(V, F, D, paths, options);
shading interp;

figure(2);
clf;
plot_mesh(V, F);
shading interp;
hold on;
scatter3(V(1, saddles), V(2, saddles), V(3, saddles), 20, 'g', 'fill');
scatter3(V(1, maxes), V(2, maxes), V(3, maxes), 20, 'r', 'fill');
lengths = cellfun(@(x) length(x), paths);
[~, idx] = sort(lengths);
for ii = 1:2
    p = paths{idx(ii)};
    plot3(p(1, :), p(2, :), p(3, :), 'r');
end
