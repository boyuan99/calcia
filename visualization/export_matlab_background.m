%% Export Step 7 background/axon data from MATLAB NAOMi for comparison
% Run this from the naomi_sim/code directory (or add it to path)

clear; close all;

%% Add NAOMi paths
naomi_root = 'C:\Users\boyuan\Documents\GitHub\naomi_sim';
addpath(genpath(fullfile(naomi_root, 'code')));

%% Set random seed
rng(42);

%% Parameters (match Python export)
vol_params.vol_sz   = [60, 60, 30];
vol_params.vres     = 2;
vol_params.N_neur   = 5;
vol_params.N_den    = 7;   % explicit integer (MATLAB default formula gives float)
vol_params.N_bg     = 20;
vol_params.verbose  = 2;

neur_params = [];
vasc_params = [];
dend_params = [];
bg_params   = [];
axon_params = [];
psf_params.objNA = 0.8;
psf_params.NA    = 0.6;
psf_params.n     = 1.35;  % refractive index in tissue

%% Run full pipeline (Steps 1-7)
fprintf('Running simulate_neural_volume (Steps 1-7)...\n');
tic;
[vol_out, vol_params, neur_params, vasc_params, dend_params, bg_params, axon_params] = ...
    simulate_neural_volume(vol_params, neur_params, vasc_params, ...
                           dend_params, bg_params, axon_params, psf_params, 'debug_opt', true);
elapsed = toc;
fprintf('Pipeline complete in %.1f seconds.\n', elapsed);

%% Extract data
volsize = vol_params.vol_sz * vol_params.vres;
N_neur  = vol_params.N_neur;
N_den   = vol_params.N_den;
N_den2  = vol_params.N_den2;
Ncomps  = N_neur + N_den;

gp_vals   = vol_out.gp_vals;
gp_bgvals = vol_out.gp_bgvals;
bg_proc   = vol_out.bg_proc;
neur_num  = vol_out.neur_num;
neur_vol  = vol_out.neur_vol;

fprintf('\nData summary:\n');
fprintf('  Volume size: [%d %d %d]\n', volsize);
fprintf('  N_neur=%d, N_den=%d, N_den2=%d, N_bg=%d\n', N_neur, N_den, N_den2, vol_params.N_bg);

%% Build JSON structure

% Grid shape
result.grid_shape = volsize(:)';

% Parameters
result.params.vol_sz     = vol_params.vol_sz;
result.params.vres       = vol_params.vres;
result.params.N_neur     = N_neur;
result.params.N_den      = N_den;
result.params.N_den2     = N_den2;
result.params.N_bg       = vol_params.N_bg;
result.params.N_bg_actual = size(gp_bgvals, 1);

% Stats
neuron_mask = (neur_num >= 1) & (neur_num <= Ncomps);
bg_mask     = (neur_num > Ncomps) & (neur_num <= Ncomps + N_den2);
result.stats.total_neuron_voxels      = sum(neuron_mask(:));
result.stats.total_bg_dendrite_voxels = sum(bg_mask(:));
total_axon_voxels = 0;
for j = 1:size(gp_bgvals, 1)
    total_axon_voxels = total_axon_voxels + numel(gp_bgvals{j, 1});
end
result.stats.total_axon_voxels = total_axon_voxels;
result.stats.total_empty_voxels = sum(neur_num(:) == 0);

%% Neurons (Steps 2-6 components)
MAX_VOXELS = 5000;
neurons = {};
for kk = 1:N_neur
    locs = gp_vals{kk, 1};  % linear indices (1-based)
    fl   = gp_vals{kk, 2};  % fluorescence
    if size(gp_vals, 2) >= 3
        soma_mask = gp_vals{kk, 3};
    else
        soma_mask = false(size(locs));
    end

    % Soma
    s_idx = locs(soma_mask);
    s_fl  = fl(soma_mask);
    if numel(s_idx) > 2000
        sel = sort(randperm(numel(s_idx), 2000));
        s_idx = s_idx(sel);
        s_fl  = s_fl(sel);
    end
    [sx, sy, sz] = ind2sub(volsize, s_idx);

    % Dendrites
    d_idx = locs(~soma_mask);
    d_fl  = fl(~soma_mask);
    if numel(d_idx) > 3000
        sel = sort(randperm(numel(d_idx), 3000));
        d_idx = d_idx(sel);
        d_fl  = d_fl(sel);
    end
    [dx, dy, dz] = ind2sub(volsize, d_idx);

    n = struct();
    n.id = kk;
    % Convert to 0-based coordinates for consistency with Python
    n.soma_positions    = reshape([sx(:)-1, sy(:)-1, sz(:)-1]', 1, []);
    n.soma_fluorescence = round(s_fl(:)', 4);
    n.dend_positions    = reshape([dx(:)-1, dy(:)-1, dz(:)-1]', 1, []);
    n.dend_fluorescence = round(d_fl(:)', 4);
    neurons{end+1} = n;
end

%% Background dendrites (Step 7A)
bg_dendrites = {};
for kk = (Ncomps+1):size(gp_vals, 1)
    locs = gp_vals{kk, 1};
    fl   = gp_vals{kk, 2};
    if isempty(locs)
        continue;
    end

    total_vox = numel(locs);
    if numel(locs) > MAX_VOXELS
        sel = sort(randperm(numel(locs), MAX_VOXELS));
        locs = locs(sel);
        fl   = fl(sel);
    end
    [bx, by, bz] = ind2sub(volsize, locs);

    bg = struct();
    bg.id = kk;
    bg.total_voxels = total_vox;
    bg.positions    = reshape([bx(:)-1, by(:)-1, bz(:)-1]', 1, []);
    bg.fluorescence = round(fl(:)', 4);
    bg_dendrites{end+1} = bg;
end

%% Axons (Step 7B)
axons = {};
for kk = 1:size(gp_bgvals, 1)
    locs = gp_bgvals{kk, 1};
    fl   = gp_bgvals{kk, 2};
    if isempty(locs)
        continue;
    end

    total_vox = numel(locs);
    if numel(locs) > MAX_VOXELS
        sel = sort(randperm(numel(locs), MAX_VOXELS));
        locs = locs(sel);
        fl   = fl(sel);
    end
    [ax, ay, az] = ind2sub(volsize, locs);

    a = struct();
    a.id = kk;
    a.total_voxels = total_vox;
    a.positions    = reshape([ax(:)-1, ay(:)-1, az(:)-1]', 1, []);
    a.fluorescence = round(fl(:)', 4);
    axons{end+1} = a;
end

%% Assemble and write JSON
result.neurons       = neurons;
result.bg_dendrites  = bg_dendrites;
result.axons         = axons;

out_path = fullfile('C:\Users\boyuan\Documents\GitHub\calcia\visualization', ...
                    'background_matlab.json');
json_str = jsonencode(result);
fid = fopen(out_path, 'w');
fwrite(fid, json_str, 'char');
fclose(fid);

fprintf('\nExported to %s\n', out_path);
fprintf('  JSON size: %.1f MB\n', numel(json_str) / 1024 / 1024);
fprintf('  Neurons: %d (%d voxels)\n', N_neur, result.stats.total_neuron_voxels);
fprintf('  Bg dendrites: %d (%d voxels)\n', N_den2, result.stats.total_bg_dendrite_voxels);
fprintf('  Axons: %d (%d voxels)\n', size(gp_bgvals,1), result.stats.total_axon_voxels);
fprintf('  Empty: %d voxels\n', result.stats.total_empty_voxels);
fprintf('Done!\n');
