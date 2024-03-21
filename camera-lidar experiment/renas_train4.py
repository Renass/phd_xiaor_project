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

'''
Behavioral cloning Renas  transformer camera-lidar TRAIN LOOP

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for transformer: position (x,y), orinetation (yaw), (final_state)  

State: im, map, costmap, pose, mapinfo, prompt

1. TEXT-Image encoding using ViLT (trainable) (modality encoding)
2. Text-Image cls tokens and action tokens (positional-encoding?) (modality-encoding?) 
3. (Text-Image)-(action) causal Transformer GPT 
'''

LR = 10e-5
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/tsa-trajs_2024-03-19_15-09-09.h5'
TEST_PART = 0.2
DEVICE_NUM = 2
BATCH_SIZE = 2

WEIGHTS_DIR = '/home/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'renas4.pt'
SAVE_WEIGHTS = 'renas4.pt'

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

        self.mapinfo2token = torch.nn.Linear(10, self.d_model)
        self.pose2token = torch.nn.Linear(4, self.d_model)


    def forward(self, batch):
        im, map, costmap, mapinfo, pose, action, prompt = batch
        i1, i2, i3, i4, i5 = im.size()
        im = im.view(i1*i2, i3, i4, i5)
        #im = torch.clamp(im, 0, 1)
        #im = im.float()
        prompt = [prompt for prompt in prompt for _ in range(i2)]


        im_prompt = self.processor(images=im, text=prompt, return_tensors="pt", padding=True).to(self.device)
        im_prompt = self.vilt_model(**im_prompt).pooler_output
        
        map = map.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        print('here')
        print(map.shape)
        # batch_size and seq_length should be clamped as for image
        map = self.map_processor(images=map, return_tensors="pt", padding=True).to(self.device)
        costmap = self.map_processor(images=costmap, return_tensors="pt", padding=True).to(self.device)


        mapinfo = self.mapinfo2token(mapinfo.to(self.device))
        pose = self.pose2token(pose.to(self.device))
        print(map.shape)






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
    criterion = torch.nn.CrossEntropyLoss()

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
            #print(type(batch[0]))
    
            print('success')

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
            action.append(torch.from_numpy(action_group[action_episode][:]))
    print(torch.max(map[0][0]))
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