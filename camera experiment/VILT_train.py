import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import time
import h5py
from transformers import BertTokenizer, VisualBertModel
from sklearn.model_selection import train_test_split
from transformers import ViltProcessor, ViltModel, ViltImageProcessor
from transformers import BertModel, BertTokenizer

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
        prompt = prompt*i2
        
        inputs = self.processor(images=states_tensor, text=prompt, return_tensors="pt", padding=True).to(device)
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
    



 


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    ten_board_writer = SummaryWriter()


    model = VILT(device=device)
    model = model.to(device)  
    model.train()


    weights_file = 'VILT.pt'
    if os.path.isfile(weights_file):
        model.load_state_dict(torch.load(weights_file))
        print('weights loaded from file.')

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
    


    dataset = StateActionPromptDataset(train_states, train_actions, train_prompts)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del dataset
    test_dataset = StateActionPromptDataset(test_states, test_actions, test_prompts)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    del test_dataset


    

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    epoch = 0
    while True:
        epoch += 1
        total_loss = 0
        test_total_loss = 0
        epoch_train_time_start = time.time()
        for batch in dataloader:
            optimizer.zero_grad()
            batch_state = batch[0].to(device)
            batch_label = batch[1].to(device)
            batch_prompt = batch[2]
            j1,j2,j3 = batch_label.size()
            batch_label = batch_label.view(j1*j2, j3)
            output = model.forward(batch_state, batch_prompt) 
            del batch_state, batch_prompt
            loss = criterion(output, batch_label)
            del output, batch_label
            total_loss += loss
            loss.backward()
            optimizer.step()
            del loss
        average_loss = total_loss/len(dataloader)
        ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
        print("Training Loss:", average_loss.item())

        with torch.no_grad():
            for batch in test_dataloader:
                batch_state = batch[0].to(device)
                batch_label = batch[1].to(device)
                batch_prompt = batch[2]
                j1,j2,j3 = batch_label.size()
                batch_label = batch_label.view(j1*j2, j3)
                output = model.forward(batch_state, batch_prompt) 
                del batch_state, batch_prompt
                test_loss = criterion(output, batch_label)
                del output, batch_label
                test_total_loss += test_loss  
                del test_loss
            test_average_loss = test_total_loss/len(test_dataloader)        
            ten_board_writer.add_scalar('Test_Loss', test_average_loss.item(), epoch)
        epoch_train_time_end = time.time()
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)
    
    
        if epoch % CHECKPOINT_INTERVAL == 0: 
            torch.save(model.state_dict(), 'VILT.pt')
            print('weights saved')