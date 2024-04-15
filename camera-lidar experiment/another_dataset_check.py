import torch
import h5py
import os 
import json

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/real/tsa_combined.h5'

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
        n = len(im_group)
        
        for i in range(n):
            episode = 'data_'+str(i)
            im.append(torch.from_numpy(im_group[episode][:]).float()/255.0)
            map.append(torch.from_numpy(map_group[episode][:]).float()/100.0)
            costmap.append(torch.from_numpy(costmap_group[episode][:]).float()/100.0)
            pose.append(torch.from_numpy(pose_group[episode][:]))

            a = torch.from_numpy(action_group[episode][:])
            action.append(torch.cat((a, torch.zeros((1,4))), dim=0))
        
        
        #for i, map_episode in enumerate(map_group):
        #    #print(map_episode)
        #    map.append(torch.from_numpy(map_group[map_episode][:]).float()/100.0)
        #for i, costmap_episode in enumerate(costmap_group):
            #costmap.append(torch.from_numpy(costmap_group[costmap_episode][:]).float()/100.0)
        #for i, pose_episode in enumerate(pose_group):
        #    pose.append(torch.from_numpy(pose_group[pose_episode][:]))
        #for i, action_episode in enumerate(action_group):
            #a = torch.from_numpy(action_group[action_episode][:])
            #print('action number ', i, ': ',a)
            #action.append(torch.cat((a, torch.zeros((1,4))), dim=0))

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
    #print(prompt)