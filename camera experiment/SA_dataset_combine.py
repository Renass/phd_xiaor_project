import h5py
import os
import numpy as np

'''
This script work with set of hdf state-action dataset files to create one with all of them combined in a single file
'''

DATASET_FOLDER = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/real_pink_gates'
COMBINED_FILE = 'sa-trajs_combined.h5'


if __name__ == '__main__':
    combined_path = os.path.join(DATASET_FOLDER, COMBINED_FILE)

    states_data = []
    actions_data = []

    episode_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.h5') and f != COMBINED_FILE]
    for episode_file in episode_files:
        episode_path = os.path.join(DATASET_FOLDER, episode_file)

        try:
            with h5py.File(episode_path, 'r') as episode_hf:
                episode_states = episode_hf['states']['data'][:]
                episode_actions = episode_hf['actions']['data'][:]
                states_data.append(episode_states)
                actions_data.append(episode_actions)
                print(f'Episode from {episode_file} added to aggregation')
        except KeyError as e:
            print(f'Warning: {episode_file} skipped due to missing dataset: {e}')

    states_data = np.concatenate(states_data, axis=0)
    actions_data = np.concatenate(actions_data, axis=0)

    with h5py.File(combined_path, 'w') as combined_hf:
        states_group = combined_hf.create_group('states')
        actions_group = combined_hf.create_group('actions')

        states_group.create_dataset('data', data=states_data, dtype=np.float32, compression='gzip')
        actions_group.create_dataset('data', data=actions_data, dtype=np.float32, compression='gzip')