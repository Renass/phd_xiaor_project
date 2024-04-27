import torch
import h5py
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

'''
Behavioral cloning Renas  transformer camera-lidar TRAIN LOOP

State: im-map concatenation (reworked h5), prompt 
states organized as sequences - episodes

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for model are explored (im-prompt description) and set as tokens vocabulary

1. TEXT-Image(camera+map concatenation) encoding using ViLT (trainable) 
2. (im_prompt)-(action) causal Transformer GPT 
'''

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/tsa_combined_reworked.h5'
POSES = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52_action_vocab.h5'

class StateActionPromptDataset(Dataset):
    def __init__(self, im, action, prompt):
        self.im = im
        self.action = action
        self.prompt = prompt
    def __len__(self):
        return len(self.im)
    def __getitem__(self, idx):
        im = self.im[idx]
        action = self.action[idx]
        prompt = self.prompt[idx]
        return im, action, prompt
    
def action2token_vocab(action, action_vocab_token, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    selected_tokens = [action_vocab_token[idx] for idx in max_indices]
    selected_tokens = torch.stack(selected_tokens, dim=0)
    return selected_tokens


if __name__ == '__main__':
    
    im = []
    action = []
    prompt = []
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device('cuda:0')
            device_i = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device_i, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
        device = torch.device('cpu')
    print('Current device: ',device)

    action_vocab_token = []
    action_vocab_action = []
    with h5py.File(POSES, 'r') as hdf2:
        num_poses = len(hdf2['tokens'])
        for i in range(num_poses):
            action_vocab_token.append(torch.from_numpy(hdf2['tokens']['data_'+str(i)][:]))
            action_vocab_action.append(torch.from_numpy(hdf2['actions']['data_'+str(i)][:])[0])
        action_vocab_token = torch.stack(action_vocab_token, dim=0)
        #additional end_token of ones
        action_vocab_token = torch.cat((action_vocab_token, torch.ones((1, 768))))

        action_vocab_action = torch.stack(action_vocab_action, dim=0)
        #additional end_token of ones
        action_vocab_action = torch.cat((action_vocab_action, torch.ones((1, 4))))  

    with h5py.File(DATASET, 'r') as hdf:
        num_episodes = len(hdf['states'])
        for i in range(num_episodes):
            episode_i = 'data_'+str(i)
            im_i = torch.from_numpy(hdf['states'][episode_i][:]).float()
            im.append(im_i)

            a = torch.from_numpy(hdf['actions'][episode_i][:])
            a = torch.cat((a, torch.ones((1,4))), dim=0)
            a = action2token_vocab(a, action_vocab_token, action_vocab_action)
            action.append(a)
    
    prompt_filename = DATASET[:-12]+'_tasks.txt'
    with open(prompt_filename, 'r') as file:
        for p in file:
            prompt.append(p.strip())

    dataset =  StateActionPromptDataset(im, action, prompt)