import torch
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor, ViltModel
import numpy as np
from transformers import OFATokenizer, OFAModel

'''
Preparing action tokens vocabulary from action-prompt forward pass 

TEXT-Image(camera+map concatenation) (input reworked.h5) encoding using OFA  
'''

#EXPLORING POSES TO CODE AS ACTION OPTIONS 
#(Reworked h5 required) 
DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52_reworked.h5'
BATCH_SIZE = 1

class StatePromptDataset(Dataset):
    def __init__(self, im, prompt):
        self.im = im
        self.prompt = prompt
    def __len__(self):
        return len(self.im)
    def __getitem__(self, idx):
        im = self.im[idx]
        prompt = self.prompt[idx]
        return im, prompt
    

class ActionCoder(torch.nn.Module):
    def __init__(self, device):
        super(ActionCoder, self).__init__()
        self.device = device
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = False
        self.processor.current_processor.do_normalize = False

        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.d_model = self.vilt_model.config.hidden_size
    
    def forward(self, batch):
        im, prompt = batch
        i1, i2, i3, i4, i5 = im.size()
        if i2 != 1:
            print('Warning: Episodes should contain single image')
        im = im.view(i1*i2, i3, i4, i5)   
        prompt = [prompt for prompt in prompt for _ in range(i2)]
        im_prompt = self.processor(images=im, text=prompt, return_tensors="pt", padding=True).to(self.device)
        tokens = self.vilt_model(**im_prompt).pooler_output
        return tokens

if __name__ == '__main__':
    ckpt_dir = '/home/renas/pythonprogv2/phd_xiaor_project/OFA-base'
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device('cuda:0')
            device_i = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device_i, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
        device = torch.device('cpu')
    print('Current device: ',device)

    
    
    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
    model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
    model = model.to(device)
    model.eval()
    
    
    
    im = []
    action = []
    prompt = []
    with h5py.File(DATASET, 'r') as hdf:
        n = len(hdf['states'])
        print('Actions to code to tokens: ', n)
        for i in range(n):
            episode_i = 'data_'+str(i)
            im_i = torch.from_numpy(hdf['states'][episode_i][:]).float()
            #Every episode should contain only 1 image desribing action result
            im_i= im_i[0]
            im.append(im_i)
            actions_i = hdf['actions'][episode_i][:]
            action.append(actions_i)
    
    prompt_filename = f'{os.path.splitext(DATASET)[0][:-9]}_tasks.txt'
    print(prompt_filename)
    with open(prompt_filename, 'r') as file:
        for p in file:
            p = p.strip()
            p = tokenizer([p], return_tensors="pt").input_ids.squeeze(0)
            prompt.append(p)
            
    dataset =  StatePromptDataset(im, prompt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    id = 0
    with h5py.File(DATASET[:-12]+"_action_vocab.h5", 'w') as new_hdf:
        new_hdf_actions_group = new_hdf.create_group('actions')
        new_hdf_tokens_group = new_hdf.create_group('tokens')
        with torch.no_grad():
            for batch in dataloader:
                print(batch[0].shape)
                print(batch[1].shape)
                tokens = model.encoder.forward(input_ids = batch[1].to(device), patch_images=batch[0].to(device))
                tokens = tokens.last_hidden_state
                tokens = torch.mean(tokens, dim=1)
                print('here', tokens.shape)
                for i in range(batch[0].shape[0]):
                    new_hdf_actions_group.create_dataset('data_'+str(id), data=action[id], dtype = np.float32, compression = 'gzip')
                    new_hdf_tokens_group.create_dataset('data_'+str(id), data=tokens[i].cpu(), dtype = np.float32, compression = 'gzip')
                    id += 1
