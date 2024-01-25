import torch
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import h5py
from sklearn.model_selection import train_test_split
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import  destroy_process_group
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
import math 
from transformers import ViltProcessor, ViltModel
from transformers import OpenAIGPTConfig, OpenAIGPTModel


'''
Behavioral cloning Renas  transformer TRAIN LOOP
Actions are resolved as a regression task
hdf Dataset

1. TEXT-Image encoding using ViLT (trainable) (modality encoding?)
2. Text-Image cls tokens and action tokens (positional-encoding?) (modality-encoding?) 
3. (Text-Image)-(action) causal Transformer GPT 
'''

LR = 10e-5
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

# How many data samples to take from every data file
DATA_SAMPLES = None  #For ALL data set: None
BATCH_SIZE = 1
SEQ_LENGTH = 100
TEST_PART = 0.2
CHECKPOINT_INTERVAL = 10
DEVICE_NUM = 2



LOAD_WEIGHTS = 'renas3'
SAVE_WEIGHTS = 'renas.pt'

DATASET1 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_13-37-35.h5'
DATASET2 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_14-31-15.h5'
#PROMPT = ["Go to the cube"]* SEQ_LENGTH*BATCH_SIZE
PROMPT1 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_13-37-35_prompt.txt'
PROMPT2 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_14-31-15_prompt.txt'   


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class EncodingVector(torch.nn.Module):
    def __init__(self, d_model):
        super(EncodingVector, self).__init__()
        self.modality_vector = torch.nn.Parameter(torch.randn(d_model))
    def forward(self, x):
        return x + self.modality_vector.unsqueeze(0).unsqueeze(0)

class Renas(torch.nn.Module):
    def __init__(self, device):
        super(Renas, self).__init__()

        self.device = device

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = True
        self.processor.current_processor.do_normalize = False
        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.d_model = self.vilt_model.config.hidden_size
        for param in self.vilt_model.parameters():
            param.requires_grad = True

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2, 768),
        #    torch.nn.GELU(),
        #    torch.nn.Linear(768, 768)
            )
        self.states_enc_vector = EncodingVector(d_model=self.d_model)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model)
        self.pos_enc = PositionalEncoding(d_model=self.d_model)

        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=4, n_head=12)
        self.gpt_model = OpenAIGPTModel(self.gpt_config)

        self.fc2 = torch.nn.Linear(768, 2)


    def forward(self, states_tensor, actions_tensor, prompt):
        i1, i2, i3, i4, i5 = states_tensor.size()
        prompt = [prompt for prompt in prompt for _ in range(i2)]
        states_tensor = states_tensor.view(i1*i2, i3, i4, i5)
        states_tensor = torch.clamp(states_tensor, 0, 1)
        states_tensor = states_tensor.float()
        states_tensor = self.processor(images=states_tensor, text=prompt, return_tensors="pt", padding=True).to(self.device)
        del  prompt
        states_tensor = self.vilt_model(**states_tensor).pooler_output
        states_tensor = states_tensor.view(i1, i2, self.d_model)
        actions_tensor = self.fc(actions_tensor.to(self.device))

        states_tensor = self.states_enc_vector(states_tensor)
        actions_tensor = self.actions_enc_vector(actions_tensor)

        states_tensor = self.pos_enc(states_tensor)
        actions_tensor = self.pos_enc(actions_tensor)
        #tokens = torch.cat((states_tensor, actions_tensor), dim=1)
        tokens = torch.zeros(i1, i2*2, self.d_model, device=self.device)
        tokens[:, 0::2, :] = states_tensor
        tokens[:, 1::2, :] = actions_tensor
        del states_tensor, actions_tensor
        tokens = self.gpt_model(inputs_embeds = tokens).last_hidden_state
        tokens = self.fc2(tokens[:, 0::2, :])        
        return tokens
    




class StateActionPromptDataset(Dataset):
    def __init__(self, states, actions, prompts):
        self.states = states
        self.actions = actions
        self.prompts = prompts


    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        prompt = self.prompts[idx]
        return state, action, prompt
    
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    


def ddp_train_loop(rank, world_size, train_dataset, test_dataset):
    ddp_setup(rank, world_size)
    model = Renas(rank).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=LR_WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=LR_DECAY_EPOCHS, eta_min= LR/10)
    scheduler3 = ConstantLR(optimizer, factor=LR/10, total_iters= 100000)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[LR_WARMUP_EPOCHS, LR_WARMUP_EPOCHS+LR_DECAY_EPOCHS])
    criterion = torch.nn.MSELoss()

    if rank == 0:
        ten_board_writer = SummaryWriter()
    

    if os.path.isfile(LOAD_WEIGHTS):
        model_dict = model.module.state_dict()
        pretrained_dict = torch.load(LOAD_WEIGHTS)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.module.load_state_dict(model_dict)
        del model_dict, pretrained_dict
        print('cuda:',rank,'  weights loaded from file.')


    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler)
    del train_dataset
    
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=test_sampler)
    del test_dataset


    epoch = 0
    while True:
        epoch += 1
        total_loss = 0
        test_total_loss = 0
        epoch_train_time_start = time.time()
        optimizer.zero_grad()
        for batch in train_dataloader:
            batch_state = batch[0]
            batch_action = batch[1]
            batch_prompt = batch[2]
            output = model(batch_state, batch_action, batch_prompt) 
            del batch_state, batch_prompt
            batch_action = batch_action.to(rank)
            loss = criterion(output, batch_action)
            del output, batch_action
            total_loss += loss
            loss.backward()
            del loss
        optimizer.step()
        scheduler.step()
        average_loss = total_loss/len(train_dataloader)
        if rank==0:
            ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
            print('\nEpoch: ', epoch,"  Training Loss:", average_loss.item())

        with torch.no_grad():
            for batch in test_dataloader:
                batch_state  = batch[0]
                batch_action = batch[1]
                batch_prompt = batch[2]
                output = model(batch_state, batch_action, batch_prompt) 
                del batch_state, batch_prompt
                batch_action = batch_action.to(rank)
                test_loss = criterion(output, batch_action)
                del output, batch_action
                test_total_loss += test_loss  
                del test_loss
            test_average_loss = test_total_loss/len(test_dataloader)   
            if rank==0:     
                ten_board_writer.add_scalar('Test_Loss', test_average_loss.item(), epoch)
        epoch_train_time_end = time.time()
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)
    
    
        if epoch % CHECKPOINT_INTERVAL == 0 and rank==0:
            torch.save(model.module.state_dict(), 'temp_'+ SAVE_WEIGHTS)
            shutil.move('temp_'+ SAVE_WEIGHTS, SAVE_WEIGHTS)
            print('weights saved')
    destroy_process_group()



if __name__ == '__main__':
    world_size = DEVICE_NUM

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
    
    #Taking dataset from 2 files
    data_file1 = h5py.File(DATASET1, 'r')
    states_tensor1 = data_file1['states']['data'][:DATA_SAMPLES]
    states_tensor1 = torch.from_numpy(states_tensor1)
    actions_tensor1 = data_file1['actions']['data'][:DATA_SAMPLES]
    actions_tensor1 = torch.from_numpy(actions_tensor1)
    prompts1 = open(PROMPT1, 'r').read().splitlines()*states_tensor1.shape[0]
    
    data_file2 = h5py.File(DATASET2, 'r')
    states_tensor2 = data_file2['states']['data'][:DATA_SAMPLES]
    states_tensor2 = torch.from_numpy(states_tensor2)
    actions_tensor2 = data_file2['actions']['data'][:DATA_SAMPLES]
    actions_tensor2 = torch.from_numpy(actions_tensor2)
    prompts2 = open(PROMPT2, 'r').read().splitlines()*states_tensor2.shape[0]

    states_tensor = torch.cat((states_tensor1, states_tensor2), dim=0)
    del states_tensor1, states_tensor2
    actions_tensor = torch.cat((actions_tensor1, actions_tensor2), dim=0)
    del actions_tensor1, actions_tensor2
    prompts = prompts1 + prompts2

    train_states, test_states, train_actions, test_actions, train_prompts, test_prompts = train_test_split(
        states_tensor, 
        actions_tensor, prompts, 
        test_size=TEST_PART)
    print('state-action dataset size:')
    print(states_tensor.shape)
    print(actions_tensor.shape)
    del states_tensor, actions_tensor, prompts

    train_dataset = StateActionPromptDataset(train_states, train_actions, train_prompts)
    test_dataset = StateActionPromptDataset(test_states, test_actions, test_prompts)

    mp.spawn(ddp_train_loop,args=(world_size,train_dataset, test_dataset), nprocs=world_size, join=True)


    




    

