from renas_train4 import Renas, StateActionPromptDataset, padding_collate
import torch
import h5py
import os
import json
from torch.utils.data import DataLoader

'''
CODE NEEDS TO REVISE WITH H5PY DATASET PROBLEM: data_19 earlier than data_2 (alphabetical sorting)

Behavioral cloning Renas  transformer camera-lidar TESTING

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. TEXT-Image encoding using ViLT (trainable) 
2. >Text-Image token + lidar map, costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 

'''

LOAD_WEIGHTS = '/home/renas/pythonprogv2/phd_xiaor_project/weights/early_renas4.pt'
DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/real/tsa_combined.h5'
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

        n = len(im_group)

        for i in range(n):
            episode = 'data_'+str(i)
            im.append(torch.from_numpy(im_group[episode][:]).float()/255.0)
            map.append(torch.from_numpy(map_group[episode][:]).float()/100.0)
            costmap.append(torch.from_numpy(costmap_group[episode][:]).float()/100.0)
            pose.append(torch.from_numpy(pose_group[episode][:]))

            a = torch.from_numpy(action_group[episode][:])
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
        print(batch[6])
        print('target:', batch[5])
        output2 = model(batch)
        print('output2: ',output2)