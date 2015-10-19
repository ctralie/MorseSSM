function [] = doFastMarching(V, F, startidx, endidx)
    addpath(genpath('toolbox_fast_marching'));

    %options.nb_iter_max = Inf;
    disp('Performing fast marching...');
    [D, S, Q] = perform_fast_marching_mesh(V, F, startidx);
    disp('Done Fast Marching');
    options.v2v = compute_vertex_ring(F);
    options.e2f = compute_edge_face_ring(F);
    options.method = 'continuous';
    options.verb = 1;
    disp('Computing paths...');
    paths = compute_geodesic_mesh(D, V, F, endidx, options);
    if ~iscell(paths)
        paths = {paths};
    end
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
    scatter3(V(1, startidx), V(2, startidx), V(3, startidx), 20, 'g', 'fill');
    scatter3(V(1, endidx), V(2, endidx), V(3, endidx), 20, 'r', 'fill');
    lengths = cellfun(@(x) length(x), paths);
    [~, idx] = sort(lengths);
    for ii = 1:2
        p = paths{idx(ii)};
        plot3(p(1, :), p(2, :), p(3, :), 'r');
    end
end
