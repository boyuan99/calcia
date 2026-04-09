% export_matlab_neurons.m
% Generate neuron data from MATLAB for comparison with Python implementation

% Add path to NAOMi code
addpath(genpath('C:\Users\boyuan\Documents\GitHub\naomi_sim\code'));

% Set random seed for reproducibility
rng(42);

% Use MATLAB default params (from check_neur_params.m)
neur_params = check_neur_params(struct());
neur_params.n_samps = 200;

% Generate single neuron
fprintf('Generating single neuron...\n');
[Vcell, Vnuc, Tri, angles] = generateNeuralBody(neur_params);

% Export to JSON format
data = struct();
data.neurons = {};

neuron = struct();
neuron.soma.vertices = Vcell;
neuron.soma.faces = Tri - 1;  % Convert to 0-based indexing
neuron.nucleus.vertices = Vnuc;
neuron.nucleus.faces = Tri - 1;
neuron.position = [0, 0, 0];
neuron.rotation = angles;

data.neurons{1} = neuron;

% Save as JSON
jsonStr = jsonencode(data);
fid = fopen('matlab_neuron_data.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);
fprintf('Saved: matlab_neuron_data.json\n');

% Generate multiple neurons
fprintf('Generating 9 neurons...\n');
rng(42);  % Reset seed

data_multi = struct();
data_multi.neurons = {};

spacing = 25.0;
for i = 1:9
    row = floor((i-1) / 3);
    col = mod(i-1, 3);
    pos = [col * spacing, row * spacing, 0];

    [Vcell, Vnuc, Tri, angles] = generateNeuralBody(neur_params);

    % Translate to position
    Vcell = Vcell + pos;
    Vnuc = Vnuc + pos;

    neuron = struct();
    neuron.soma.vertices = Vcell;
    neuron.soma.faces = Tri - 1;
    neuron.nucleus.vertices = Vnuc;
    neuron.nucleus.faces = Tri - 1;
    neuron.position = pos;
    neuron.rotation = angles;

    data_multi.neurons{i} = neuron;
end

jsonStr = jsonencode(data_multi);
fid = fopen('matlab_neurons_data.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);
fprintf('Saved: matlab_neurons_data.json\n');

% Generate different neuron types
fprintf('Generating neuron types...\n');
types = {'pyr', 'other'};  % Skip 'peanut' as it has issues in MATLAB too
type_names = {'Pyramidal', 'Spherical'};

data_types = struct();
data_types.neurons = {};

for i = 1:length(types)
    rng(42);  % Same seed for fair comparison

    params = check_neur_params(struct());  % Get default params
    params.neur_type = types{i};
    params.max_ang = 0;  % No rotation for comparison
    params.n_samps = 200;

    [Vcell, Vnuc, Tri, angles] = generateNeuralBody(params);

    % Offset for side-by-side display
    offset = [(i-1) * 20, 0, 0];
    Vcell = Vcell + offset;
    Vnuc = Vnuc + offset;

    neuron = struct();
    neuron.name = type_names{i};
    neuron.type = types{i};
    neuron.soma.vertices = Vcell;
    neuron.soma.faces = Tri - 1;
    neuron.nucleus.vertices = Vnuc;
    neuron.nucleus.faces = Tri - 1;
    neuron.position = offset;

    data_types.neurons{i} = neuron;
end

jsonStr = jsonencode(data_types);
fid = fopen('matlab_neuron_types_data.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);
fprintf('Saved: matlab_neuron_types_data.json\n');

fprintf('\nDone! MATLAB data exported.\n');
fprintf('Compare with Python data in the Three.js viewer.\n');
