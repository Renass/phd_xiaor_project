from renas_train4_1 import Renas, StateActionPromptDataset, padding_collate
import torch
import h5py
import os
import json
from torch.utils.data import DataLoader
from transformers import ViltImageProcessor

'''
Behavioral cloning Renas  transformer camera-lidar TESTING
SINGLE AND STATIC MAP VERSION

State: im, map (single per dataset), NO costmap, pose, mapinfo (single per dataset), prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. TEXT-Image encoding using ViLT (trainable) 
2. >Text-Image token + lidar map, NO costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 

'''

LOAD_WEIGHTS = '/home/renas/pythonprogv2/phd_xiaor_project/weights/early_trash.pt'
DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/real/tsa_combined.h5'
BATCH_SIZE = 10

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
    
    
    map_processor = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    map_processor.do_resize = True
    map_processor.do_rescale = False
    map_processor.do_normalize = False


    im = []
    map = []
    pose = []
    action = []
    prompt = []
    with h5py.File(DATASET, 'r') as hdf:
        im_group = hdf['states']
        map_group =hdf['maps']
        pose_group = hdf['pose']
        action_group = hdf['actions']

        n = len(im_group)

        for i in range(n):
            episode = 'data_'+str(i)
            im.append(torch.from_numpy(im_group[episode][:]).float()/255.0)
            if i==0:
                map_i = torch.from_numpy(map_group[episode][:]).float()/100.0
                map_i = map_i.unsqueeze(1).repeat(1, 3, 1, 1)
                map_i = map_processor(images=map_i, return_tensors="pt", padding=True)['pixel_values']
                map_i = map_i[:, 0, :, :]
                map_patch_size = 64
                map_i = map_i.unfold(1, map_patch_size, map_patch_size).unfold(2, map_patch_size, map_patch_size)
                map_i = map_i.contiguous().view(-1, 6 * 6, map_patch_size * map_patch_size)
                map.append(map_i)
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
    dataset =  StateActionPromptDataset(im, map, mapinfo, pose, action, prompt)
    print("Dataset episodes load: ",len(dataset))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=padding_collate)
    for i, batch in enumerate(loader):
        print(batch[5])
        print('target:', batch[4])
        output2 = model(batch)
        print('output2: ',output2)