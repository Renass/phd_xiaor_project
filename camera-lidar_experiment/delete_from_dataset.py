import os
import json
import sys
import h5py
import numpy as np

'''
This script work with set of hdf task-state-action dataset to delete some dataset instances from h5 files

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)
'''

DATASET_FOLDER = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/cola'
DATASET = 'tsa-trajs_2024-05-02_18-33-28.h5'
NEW_DATASET = 'tsa-trajs_2024-05-02_18-33-28_e.h5'
DELETE_INSTANCE_WITH_IND = 1



if __name__ == '__main__':
    combined_path = os.path.join(DATASET_FOLDER, DATASET)
    combined_path_new = os.path.join(DATASET_FOLDER, NEW_DATASET)



    
    dataset_counters = {'states': 0, 'maps': 0, 'costmaps': 0, 'pose': 0, 'actions': 0}
    with h5py.File(combined_path_new, 'w') as hf_combined:
        states_group = hf_combined.create_group('states')
        actions_group = hf_combined.create_group('actions')
        map_group = hf_combined.create_group('maps')
        costmap_group = hf_combined.create_group('costmaps')
        pose_group = hf_combined.create_group('pose')

        with h5py.File(combined_path, 'r') as file:
            for group_name in dataset_counters.keys():
                group = hf_combined[group_name]
                for dataset_name, dataset in file[group_name].items():
                    if dataset_name !=  f"data_{DELETE_INSTANCE_WITH_IND}":
                        new_dataset_name = f"data_{dataset_counters[group_name]}"
                        group.create_dataset(new_dataset_name, data=dataset,dtype = np.float32, compression = 'gzip')
                        dataset_counters[group_name] += 1

