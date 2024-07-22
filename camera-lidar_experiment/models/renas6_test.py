import torch
import h5py
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
import time
import shutil
from transformers import ViltProcessor, ViltModel, ViltImageProcessor
import math
from transformers import OpenAIGPTConfig, OpenAIGPTModel
from renas6_train import StateActionPromptDataset, Renas, action2label_vocab, action2token_vocab, padding_collate
import matplotlib.pyplot as plt
import numpy as np

'''
Behavioral cloning Renas  transformer camera-lidar TEST

State: im-map concatenation (reworked h5), prompt 
states organized as sequences - episodes

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for model are explored (im-prompt description) and set as tokens vocabulary

1. TEXT-Image(camera+map concatenation) encoding using ViLT (trainable) 
2. (im_prompt)-(action) causal Transformer GPT 
'''

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/test/tsa-trajs_2024-04-30_17-15-25_reworked.h5'
POSES = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52_action_vocab.h5'
BATCH_SIZE = 5

WEIGHTS_DIR = '/home/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'renas6.pt'


if __name__ == '__main__':
    
    im = []
    action = []
    a_label = []
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
            a_label_i = action2label_vocab(a, action_vocab_action)
            a_label.append(a_label_i)
            a = action2token_vocab(a, action_vocab_token, action_vocab_action)
            action.append(a)
    
    prompt_filename = DATASET[:-12]+'_tasks.txt'
    with open(prompt_filename, 'r') as file:
        for p in file:
            prompt.append(p.strip())

    dataset =  StateActionPromptDataset(im, action, a_label, prompt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=padding_collate)
    
    model = Renas(device).to(device)
    model.eval()
    
    if os.path.isfile(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS)):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del model_dict, pretrained_dict
        print('weights loaded from file.')

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch[0] = batch[0][:, :-1, :, :, :]
            
            batch[1] = batch[1][:, 1:, :]
            print('correct labels: ', batch[2])
            batch[2] = None
            print('debug: prompts to model:', batch[3])
            output = model(batch, action_vocab_token)
            print('model output: ', output)

            #plt.imshow(batch[0][0][0].numpy().transpose(1,2,0))
            #plt.show()