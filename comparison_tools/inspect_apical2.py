import numpy as np
import sys
sys.path.insert(0, 'C:/Users/boyuan/Documents/GitHub/calcia')
from calcia import import_pipeline_output

out = import_pipeline_output('C:/Users/boyuan/Documents/GitHub/calcia/examples/output/output.npz')
nn_ad = out.neur_num_ad
nn = out.neur_num

py_ids = np.unique(nn_ad[nn_ad > 0])
N_neur = out.params['vol_params'].N_neur
N_den = out.params['vol_params'].N_den

print(f'N_neur={N_neur}, N_den={N_den}')
print(f'Expected Step4 apical IDs: 1..{N_neur} (neuron IDs)')
print(f'Expected Step5 through-vol IDs: {N_neur+1}..{N_neur+N_den}')
print()
print(f'=== neur_num_AD unique IDs: {len(py_ids)}, range [{py_ids.min()}, {py_ids.max()}]')
print(f'IDs in neuron range (1..{N_neur}): {np.sum(py_ids <= N_neur)}')
print(f'IDs in apical range ({N_neur+1}..{N_neur+N_den}): {np.sum((py_ids > N_neur) & (py_ids <= N_neur+N_den))}')
print(f'IDs beyond apical (>{N_neur+N_den}): {np.sum(py_ids > N_neur+N_den)}')
if np.sum(py_ids > N_neur+N_den) > 0:
    beyond = py_ids[py_ids > N_neur+N_den]
    print(f'  Beyond IDs: {beyond[:10].tolist()}...')

# Compare with neur_num to understand the beyond-750 IDs
nn_ids = np.unique(nn[nn > 0])
print(f'\n=== neur_num unique IDs: {len(nn_ids)}, range [{nn_ids.min()}, {nn_ids.max()}]')
print(f'IDs beyond {N_neur+N_den}: {np.sum(nn_ids > N_neur+N_den)}')

# Check what the >750 IDs in neur_num_ad correspond to in neur_num
beyond_mask = (nn_ad > N_neur+N_den) & (nn_ad > 0)
print(f'\nVoxels with neur_num_ad > {N_neur+N_den}: {np.sum(beyond_mask)}')
if np.sum(beyond_mask) > 0:
    corresponding_nn = nn[beyond_mask]
    print(f'  neur_num at those voxels: unique={np.unique(corresponding_nn)[:10].tolist()}')
