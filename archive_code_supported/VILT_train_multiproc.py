import torch
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import h5py
from sklearn.model_selection import train_test_split
from transformers import ViltProcessor, ViltModel
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
'''
Behavioral cloning ViLT transformer TRAIN LOOP
Actions are resolved as a regression task
hdf Dataset

Image encoding with ViLT vision transformer patching
'''

LR = 10e-4
BATCH_SIZE = 2
SEQ_LENGTH = 100
TEST_PART = 0.2
CHECKPOINT_INTERVAL = 10
DEVICE_NUM = 2



LOAD_WEIGHTS = 'VILT.pt'
SAVE_WEIGHTS = 'VILT.pt'

DATASET1 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_13-37-35.h5'
DATASET2 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_14-31-15.h5'
#PROMPT = ["Go to the cube"]* SEQ_LENGTH*BATCH_SIZE
PROMPT1 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_13-37-35_prompt.txt'
PROMPT2 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-08_14-31-15_prompt.txt'   


class VILT(torch.nn.Module):
    def __init__(self, device):
        super(VILT, self).__init__()
        self.device = device



        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = False
        self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        #print('here', self.encoder.encoder.layer[11])
        for param in self.model.parameters():
            param.requires_grad = True


        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 2)
            )


    def forward(self, states_tensor, prompt):
        i1, i2, i3, i4, i5 = states_tensor.size()
        states_tensor = states_tensor.view(i1*i2, i3, i4, i5)
        #states_tensor = torch.clamp(states_tensor, 0, 1)
        states_tensor = states_tensor.float()
        prompt = prompt*(i2)
        
        inputs = self.processor(images=states_tensor, text=prompt, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        outputs = outputs.pooler_output
        outputs = self.fc2(outputs)
        return outputs   
    




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
    model = VILT(rank).to(rank)
    model = DDP(model, device_ids=[rank])  
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    if rank == 0:
        ten_board_writer = SummaryWriter()
    

    if os.path.isfile(LOAD_WEIGHTS):
        model.module.load_state_dict(torch.load(LOAD_WEIGHTS))
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
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_state, batch_label = batch[0], batch[1]
            batch_prompt = batch[2]
            j1,j2,j3 = batch_label.size()
            batch_label = batch_label.view(j1*j2, j3).to(rank)
            output = model(batch_state, batch_prompt) 
            del batch_state, batch_prompt
            loss = criterion(output, batch_label)
            del output, batch_label
            total_loss += loss
            loss.backward()
            optimizer.step()
            del loss
        average_loss = total_loss/len(train_dataloader)
        if rank==0:
            ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
            print('Epoch: ', epoch,"  Training Loss:", average_loss.item())

        with torch.no_grad():
            for batch in test_dataloader:
                batch_state, batch_label = batch[0], batch[1]
                batch_prompt = batch[2]
                j1,j2,j3 = batch_label.size()
                batch_label = batch_label.view(j1*j2, j3).to(rank)
                output = model(batch_state, batch_prompt) 
                del batch_state, batch_prompt
                test_loss = criterion(output, batch_label)
                del output, batch_label
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
    

    data_file1 = h5py.File(DATASET1, 'r')
    states_tensor1 = data_file1['states']['data'][:]
    states_tensor1 = torch.from_numpy(states_tensor1)
    actions_tensor1 = data_file1['actions']['data'][:]
    actions_tensor1 = torch.from_numpy(actions_tensor1)
    prompts1 = open(PROMPT1, 'r').read().splitlines()*states_tensor1.shape[0]
    
    data_file2 = h5py.File(DATASET2, 'r')
    states_tensor2 = data_file2['states']['data'][:]
    states_tensor2 = torch.from_numpy(states_tensor2)
    actions_tensor2 = data_file2['actions']['data'][:]
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


    




    

