import os
import json
import sys
import h5py
import numpy as np

'''
This script work with set of hdf task-state-action dataset files to create one with all of them combined in a single file

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)
'''

DATASET_FOLDER = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/real'
COMBINED_FILE = 'tsa_combined.h5'



if __name__ == '__main__':
    combined_path = os.path.join(DATASET_FOLDER, COMBINED_FILE)

    #Mapinfo load and check all matched
    mapinfo_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('_mapinfo.json') and f != COMBINED_FILE]
    with open(os.path.join(DATASET_FOLDER, mapinfo_files[0]), 'r') as file:
        mapinfo = json.load(file)
    for mapinfo_file in mapinfo_files:
        with open(os.path.join(DATASET_FOLDER, mapinfo_file), 'r') as file:
            mapinfo_i = json.load(file)
        if mapinfo_i != mapinfo:
            print('Mapinfo files are different!')
            sys.exit()    
    print('Mapinfo data is matched')

    
    dataset_counters = {'states': 0, 'maps': 0, 'costmaps': 0, 'pose': 0, 'actions': 0}
    episode_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.h5') and f != COMBINED_FILE]
    with h5py.File(combined_path, 'w') as hf_combined:
        states_group = hf_combined.create_group('states')
        actions_group = hf_combined.create_group('actions')
        map_group = hf_combined.create_group('maps')
        costmap_group = hf_combined.create_group('costmaps')
        pose_group = hf_combined.create_group('pose')




        for episode_file in episode_files:
            episode_path = os.path.join(DATASET_FOLDER, episode_file)
            with h5py.File(episode_path, 'r') as hdf:
                for group_name in dataset_counters.keys():
                    group = hf_combined[group_name]
                    for dataset_name, dataset in hdf[group_name].items():
                        new_dataset_name = f"data_{dataset_counters[group_name]}"
                        group.create_dataset(new_dataset_name, data=dataset,dtype = np.float32, compression = 'gzip')
                        dataset_counters[group_name] += 1
