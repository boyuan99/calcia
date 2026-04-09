import numpy as np
import scipy.io
import sys
sys.path.insert(0, 'C:/Users/boyuan/Documents/GitHub/calcia')
from calcia import import_pipeline_output

out = import_pipeline_output('C:/Users/boyuan/Documents/GitHub/calcia/examples/output/output.npz')
nn_ad = out.neur_num_ad
py_ids = np.unique(nn_ad[nn_ad > 0])
print('=== Python neur_num_AD ===')
print(f'Unique IDs: {len(py_ids)}, range [{py_ids.min()}, {py_ids.max()}]')
print(f'IDs <= 625: {np.sum(py_ids <= 625)}')
print(f'IDs 626-750: {np.sum((py_ids > 625) & (py_ids <= 750))}')
print(f'IDs > 750: {np.sum(py_ids > 750)}')
print(f'Sample first 10: {py_ids[:10].tolist()}')
print(f'Sample last 10: {py_ids[-10:].tolist()}')

m = scipy.io.loadmat(
    'C:/Users/boyuan/Documents/GitHub/naomi_sim/comparison_tools/matlab_phase1_stats.mat',
    squeeze_me=True, struct_as_record=False)
s = m['stats']
print('=== MATLAB ===')
print(f'n_apical_components: {int(s.n_apical_components)}')
print(f'neur_num_AD_nonzero: {int(s.neur_num_AD_nonzero)}')
