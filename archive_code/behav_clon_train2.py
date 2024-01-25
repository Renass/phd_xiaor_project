import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import os
import h5py
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import keyboard as k
import time

'''
Behavioral cloning transformer TRAIN LOOP
Actions are resolved as a regression task
hdf Dataset

im1,    mask,   mask, mask, action_tken
im1, im2,   mask,   mask,     action_token
im1,    im2,    im3,   mask,    action_token
im1, im2,    im3,     im4,    action_token
'''

class CustomEncoder(torch.nn.Module):
    def __init__(self, num_dim):
        super(CustomEncoder, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT').features
        for param in self.mobilenetv2.parameters():
            param.requires_grad = False

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(num_dim, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 768)
            )   
        
        self.transformer_encoder_config = GPT2Config(
            vocab_size= 2,
            n_positions= 101,
            pad_token_id=50257,
            bos_token_id=50256   
            )
        self.transformer_encoder_model = GPT2Model(self.transformer_encoder_config)
        self.transformer_encoder_config = self.transformer_encoder_model.config
        self.transformer_encoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', config = self.transformer_encoder_config)



        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 2)
            )  


    def forward(self, states_tensor):
        cnn_outputs = []
        with torch.no_grad():
            for trajectory_states_tensor in states_tensor:
                cnn_output = self.mobilenetv2(trajectory_states_tensor)
                i1, i2, i3, i4 = cnn_output.size()
                cnn_output = cnn_output.view(i1, i2*i3*i4)
                cnn_outputs.append(cnn_output)
        del cnn_output
        del states_tensor

        #feed_forward_outputs = []
        #for cnn_output in cnn_outputs:
        #    feed_forward_output = self.feed_forward(cnn_output)
        #    feed_forward_outputs.append(feed_forward_output)
        #del feed_forward_output
        #del cnn_outputs
        #feed_forward_outputs = torch.stack(feed_forward_outputs, dim=0)
        
        cnn_outputs = torch.stack(cnn_outputs, dim=0)
        feed_forward_outputs = self.feed_forward(cnn_outputs)
        del cnn_outputs



        
        #for sequence_states in feed_forward_outputs:
            
        #del feed_forward_outputs
        
        #encoder_output = encoder_output['last_hidden_state'][:,-1,:]
        #fc2_output = self.fc2(encoder_output)
        #return fc2_output   




BATCH_SIZE = 2
LR = 10e-4
CNN_LAST_HIDDEN = 1280*7*7
DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-10-24_17-28-03.h5'


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    ten_board_writer = SummaryWriter()

    encoder_model = CustomEncoder(num_dim=CNN_LAST_HIDDEN)
    encoder_model = encoder_model.to(device)  
    encoder_model.train()

    weights_file = 'encoder.pt'
    if os.path.isfile(weights_file):
        encoder_model.load_state_dict(torch.load(weights_file))
        print('weights loaded from file.')



    data_file = h5py.File(DATASET, 'r')
    states_tensor = data_file['states']['data'][:]
    states_tensor = torch.from_numpy(states_tensor)
    actions_tensor = data_file['actions']['data'][:]
    actions_tensor = torch.from_numpy(actions_tensor)
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('state-action dataset size:')
    print(states_tensor.shape)
    print(actions_tensor.shape)


    encoder_optimizer = torch.optim.SGD(encoder_model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    epoch = 0


    while not k.is_pressed('alt') and not k.is_pressed('ctrl'):
        epoch += 1
        total_loss = 0
        epoch_train_time_start = time.time()
        encoder_optimizer.zero_grad()
        for batch in dataloader:
            batch_state = batch[0].to(device)
            batch_action = batch[1].to(device)
            encoder_output = encoder_model.forward(batch_state)

            loss = criterion(encoder_output, batch_action)
            total_loss += loss
            loss.backward()
        average_loss = total_loss/len(dataloader)
        ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
        encoder_optimizer.step()
        epoch_train_time_end = time.time()
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)
        print("Training Loss:", average_loss.item())

    
    
    
    if k.is_pressed('alt'):
        torch.save(encoder_model.state_dict(), 'encoder.pt')
        print('weights saved')
    else:
        print('weights were not saved')
    print('Train finish')