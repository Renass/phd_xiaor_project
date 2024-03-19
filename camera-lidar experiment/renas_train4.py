import torch
import h5py
import os 
import json
from torch.utils.data import Dataset

'''
Behavioral cloning Renas  transformer camera-lidar TRAIN LOOP

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for transformer: position (x,y), orinetation (yaw), (final_state)  

State: im, map, costmap, pose, mapinfo, prompt

1. TEXT-Image encoding using ViLT (trainable) (modality encoding)
2. Text-Image cls tokens and action tokens (positional-encoding?) (modality-encoding?) 
3. (Text-Image)-(action) causal Transformer GPT 
'''

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/tsa-trajs_2024-03-19_15-09-09.h5'
TEST_PART = 0.2

class StateActionPromptDataset(Dataset):
    def __init__(self, im, map, costmap, pose, action, mapinfo, prompt):
        self.im = im
        self.map = map
        self.costmap = costmap
        self.pose = pose
        self.action = action
        self.mapinfo = mapinfo
        self.prompt = prompt


    def __len__(self):
        return len(self.im)

    def __getitem__(self, idx):
        im = self.im[idx]
        map = self.map[idx]
        costmap = self.costmap[idx]
        pose = self.pose[idx]
        action = self.action[idx]
        mapinfo = self.mapinfo
        prompt = self.prompt[idx]
        return im, map, costmap, pose, action, mapinfo, prompt


if __name__ == '__main__':

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')

    im = []
    map = []
    costmap = []
    pose = []
    action = []
    prompt = []
    with h5py.File(DATASET, 'r') as hdf:
        im_group = hdf['states']
        map_group =hdf['maps']
        costmap_group =hdf['costmaps']
        pose_group = hdf['pose']
        action_group = hdf['actions']

        for i, im_episode in enumerate(im_group):
            im.append(torch.from_numpy(im_group[im_episode][:]))
        for i, map_episode in enumerate(map_group):
            map.append(torch.from_numpy(map_group[map_episode][:]))
        for i, costmap_episode in enumerate(costmap_group):
            costmap.append(torch.from_numpy(costmap_group[costmap_episode][:]))
        for i, pose_episode in enumerate(pose_group):
            pose.append(torch.from_numpy(pose_group[pose_episode][:]))
        for i, action_episode in enumerate(action_group):
            action.append(torch.from_numpy(action_group[action_episode][:]))
    
    mapinfo_filename = f"{os.path.splitext(DATASET)[0]}_mapinfo.json"
    with open(mapinfo_filename, 'r') as file:
        mapinfo = json.load(file)
    mapinfo = torch.tensor([
        mapinfo['resolution'],
        mapinfo['width'],
        mapinfo['height'],
        mapinfo['origin']['position']['x'],
        mapinfo['origin']['position']['y'],
        mapinfo['origin']['position']['z'],
        mapinfo['origin']['orientation']['x'],
        mapinfo['origin']['orientation']['y'],
        mapinfo['origin']['orientation']['z'],
        mapinfo['origin']['orientation']['w'] 
    ], dtype=torch.float32)
    prompt_filename = f'{os.path.splitext(DATASET)[0]}_tasks.txt'
    with open(prompt_filename, 'r') as file:
        for p in file:
            prompt.append(p.strip())
    dataset =  StateActionPromptDataset(im, map, costmap, pose, action, mapinfo, prompt)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    print("Dataset episodes load: ",len(dataset))
