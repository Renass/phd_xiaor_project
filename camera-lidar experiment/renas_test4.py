from renas_train4 import Renas, StateActionPromptDataset, padding_collate
import torch
import trajectories_gather6
import threading
import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
import h5py
import os
import json
from torch.utils.data import DataLoader

'''
Behavioral cloning Renas  transformer camera-lidar TESTING

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. TEXT-Image encoding using ViLT (trainable) 
2. >Text-Image token + lidar map, costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 

'''

LOAD_WEIGHTS = '/home/renas/pythonprogv2/phd_xiaor_project/weights/early_renas4.pt'
DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/real/test/tsa-trajs_2024-04-09_20-45-40.h5'
BATCH_SIZE = 1

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)


    model = Renas(device=device)
    model = model.to(device)  
    model.eval()

    model.load_state_dict(torch.load(LOAD_WEIGHTS))
    print('weights loaded from file.')

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
            im.append(torch.from_numpy(im_group[im_episode][:]).float()/255.0)
        for i, map_episode in enumerate(map_group):
            map.append(torch.from_numpy(map_group[map_episode][:]).float()/100.0)
        for i, costmap_episode in enumerate(costmap_group):
            costmap.append(torch.from_numpy(costmap_group[costmap_episode][:]).float()/100.0)
        for i, pose_episode in enumerate(pose_group):
            pose.append(torch.from_numpy(pose_group[pose_episode][:]))
        for i, action_episode in enumerate(action_group):
            a = torch.from_numpy(action_group[action_episode][:])
            action.append(torch.cat((a, torch.zeros((1,4))), dim=0))

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
    dataset =  StateActionPromptDataset(im, map, costmap, mapinfo, pose, action, prompt)
    print("Dataset episodes load: ",len(dataset))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=padding_collate)
    for i, batch in enumerate(loader):
        if i ==4:
            testing_state = 1
            j = testing_state
            im, map, costmap, mapinfo, pose, action, prompt = batch
            im = im[0][j].unsqueeze(0).unsqueeze(0)
            map = map[0][j].unsqueeze(0).unsqueeze(0)
            costmap = costmap[0][j].unsqueeze(0).unsqueeze(0)
            pose = pose[0][j].unsqueeze(0).unsqueeze(0)
            action = action[0][j].unsqueeze(0).unsqueeze(0)
            #action = torch.zeros((1,1,4))
            print(im.shape)            
            print(map.shape)
            print(costmap.shape)
            print(mapinfo.shape)
            print(pose.shape)
            print(action.shape) 
            output1 = model((im, map, costmap, mapinfo, pose, action, prompt))
            output2 = model(batch)
            print(output1)
            print(output2)