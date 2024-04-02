import torch
import h5py
import os 
import json
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.utils.tensorboard import SummaryWriter
import time
from transformers import ViltProcessor, ViltModel, ViltImageProcessor
from torch.nn.utils.rnn import pad_sequence
import math
from transformers import OpenAIGPTConfig, OpenAIGPTModel
import shutil

'''
Behavioral cloning Renas  transformer camera-lidar TRAIN LOOP

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. TEXT-Image encoding using ViLT (trainable) 
2. >Text-Image token + lidar map, costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 
'''

LR = 10e-5
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/tsa-trajs_2024-03-19_15-09-09.h5'
TEST_PART = 0.2
CHECKPOINT_INTERVAL = 10
DEVICE_NUM = 2
BATCH_SIZE = 10

WEIGHTS_DIR = '/home/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'renas4.pt'
SAVE_WEIGHTS = 'renas4.pt'

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

class StateActionPromptDataset(Dataset):
    def __init__(self, im, map, costmap, mapinfo, pose, action, prompt):
        self.im = im
        self.map = map
        self.costmap = costmap
        self.mapinfo = mapinfo
        self.pose = pose
        self.action = action
        self.prompt = prompt


    def __len__(self):
        return len(self.im)

    def __getitem__(self, idx):
        im = self.im[idx]
        map = self.map[idx]
        costmap = self.costmap[idx]
        mapinfo = self.mapinfo
        pose = self.pose[idx]
        action = self.action[idx]
        prompt = self.prompt[idx]
        return im, map, costmap, mapinfo, pose, action, prompt

class Renas(torch.nn.Module):
    def __init__(self, device):
        super(Renas, self).__init__()
        self.device = device
        
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = True
        self.processor.current_processor.do_normalize = False

        self.map_processor = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.map_processor.do_resize = True
        self.map_processor.do_rescale = False
        self.map_processor.do_normalize = False

        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.d_model = self.vilt_model.config.hidden_size
        for param in self.vilt_model.parameters():
            param.requires_grad = True

        self.mid_transformer = torch.nn.TransformerEncoder(
             torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8),
             num_layers=4
        )
        

        self.pos_enc = PositionalEncoding(d_model=self.d_model)
        self.states_enc_vector = EncodingVector(d_model=self.d_model)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model)

        self.map2token = torch.nn.Linear(4096, self.d_model) 
        self.mapinfo2token = torch.nn.Linear(10, self.d_model)
        self.pose2token = torch.nn.Linear(4, self.d_model)
        self.action2token = torch.nn.Linear(4, self.d_model)


        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=4, n_head=12)
        self.gpt_model = OpenAIGPTModel(self.gpt_config)

        self.fc2 = torch.nn.Linear(self.d_model, 4)


    def forward(self, batch):
        im, map, costmap, mapinfo, pose, action, prompt = batch
        i1, i2, i3, i4, i5 = im.size()
        im = im.view(i1*i2, i3, i4, i5)

        prompt = [prompt for prompt in prompt for _ in range(i2)]


        im_prompt = self.processor(images=im, text=prompt, return_tensors="pt", padding=True).to(self.device)
        im_prompt = self.vilt_model(**im_prompt).pooler_output.unsqueeze(1)
        
        map = map.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        m1, m2, m3, m4, m5 = map.size()
        map = map.view(m1*m2, m3, m4, m5)
        map = self.map_processor(images=map, return_tensors="pt", padding=True)['pixel_values'].to(self.device)
        map = map[:, 0, :, :]
        map_patch_size = 64
        map = map.unfold(1, map_patch_size, map_patch_size).unfold(2, map_patch_size, map_patch_size)
        map = map.contiguous().view(m1*m2, 6 * 6, map_patch_size * map_patch_size)
        map = self.map2token(map)
        map = self.pos_enc(map)

        costmap = costmap.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        costmap = costmap.view(m1*m2, m3, m4, m5)
        costmap = self.map_processor(images=costmap, return_tensors="pt", padding=True)['pixel_values'].to(self.device)
        costmap = costmap[:, 0, :, :]
        costmap = costmap.unfold(1, map_patch_size, map_patch_size).unfold(2, map_patch_size, map_patch_size)
        costmap = costmap.contiguous().view(m1*m2, 6 * 6, map_patch_size * map_patch_size)
        costmap = self.map2token(costmap)
        costmap = self.pos_enc(costmap)

        mapinfo = self.mapinfo2token(mapinfo.to(self.device)).unsqueeze(1).repeat(m2, 1, 1)
        
        pose = self.pose2token(pose.to(self.device)).view(m1*m2, -1).unsqueeze(1)

        tokens = torch.cat((im_prompt, map, costmap, mapinfo, pose), dim=1)


        states = self.mid_transformer(tokens)[:,0,:].view(i1,i2, -1)
        
        #print(states.shape)
        #print(action.shape)

        actions = self.action2token(action.to(self.device))
        
        states = self.states_enc_vector(states)
        actions = self.actions_enc_vector(actions)

        states = self.pos_enc(states)
        actions = self.pos_enc(actions)
        #tokens = torch.cat((states_tensor, actions_tensor), dim=1)
        tokens = torch.zeros(i1, i2*2, self.d_model, device=self.device)
        tokens[:, 0::2, :] = states
        tokens[:, 1::2, :] = actions

        tokens = self.gpt_model(inputs_embeds = tokens).last_hidden_state
        #print(tokens.shape)
        tokens = self.fc2(tokens[:, 0::2, :])
        #print(tokens.shape)        
        return tokens





def padding_collate(batch):
    new_batch = []
    for i in range(6): # 7 data types: im, map, costmap, mapinfo, pose, action, prompt
        new_batch.append([item[i] for item in batch])
        new_batch[i] = pad_sequence(new_batch[i], batch_first=True)
    new_batch.append([item[6] for item in batch])
    return new_batch

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_train_loop(rank, world_size, train_dataset, test_dataset):
    ddp_setup(rank, world_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler, collate_fn=padding_collate)
    del train_dataset
    
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=test_sampler, collate_fn=padding_collate)
    del test_dataset


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
    

    if os.path.isfile(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS)):
        model_dict = model.module.state_dict()
        pretrained_dict = torch.load(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.module.load_state_dict(model_dict)
        del model_dict, pretrained_dict
        print('cuda:',rank,'  weights loaded from file.')


    
    epoch = 0
    min_loss = 10000
    while True:
        epoch += 1
        total_loss = 0
        test_total_loss = 0
        epoch_train_time_start = time.time()
        optimizer.zero_grad()
        for batch in train_dataloader:
            output = model(batch)
            loss = criterion(output, batch[5].to(rank))
            total_loss += loss
            loss.backward()
        optimizer.step()
        scheduler.step()
        average_loss = total_loss/len(train_dataloader)
        if rank==0:
            ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
            print('\nEpoch: ', epoch,"  Training Loss:", average_loss.item())

        with torch.no_grad():
            for batch in test_dataloader:
                output = model(batch)
                test_loss = criterion(output, batch[5].to(rank))
                test_total_loss += test_loss
            
            torch.distributed.all_reduce(test_total_loss, op=torch.distributed.ReduceOp.SUM)
            test_average_loss = test_total_loss/len(test_dataloader)/world_size   
            if rank==0:     
                ten_board_writer.add_scalar('Test_Loss', test_average_loss.item(), epoch)
        epoch_train_time_end = time.time()
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)


        if epoch % CHECKPOINT_INTERVAL == 0 and rank==0:
            torch.save(model.module.state_dict(), os.path.join(WEIGHTS_DIR, 'temp_'+ SAVE_WEIGHTS))
            shutil.move(os.path.join(WEIGHTS_DIR, 'temp_'+ SAVE_WEIGHTS), os.path.join(WEIGHTS_DIR, SAVE_WEIGHTS))

            if test_average_loss.item()<min_loss:
                torch.save(model.module.state_dict(), os.path.join(WEIGHTS_DIR, 'temp_'+ 'early_'+ SAVE_WEIGHTS))
                shutil.move(os.path.join(WEIGHTS_DIR, 'temp_'+'early_'+ SAVE_WEIGHTS), os.path.join(WEIGHTS_DIR, 'early_'+ SAVE_WEIGHTS))
                min_loss = test_average_loss.item()
                print('Early stopping with loss', min_loss, 'at the epoch', epoch)
            print('weights saved')
    destroy_process_group()


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
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    print("Dataset episodes load: ",len(dataset))

    mp.spawn(ddp_train_loop,args=(DEVICE_NUM,train_dataset, test_dataset), nprocs=DEVICE_NUM, join=True)