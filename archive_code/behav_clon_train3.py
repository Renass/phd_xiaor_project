import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer, GPT2Model
import pickle
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import keyboard as k
import os
import h5py

'''
Behavioral cloning transformer TRAIN LOOP
Actions are resolved as a regression task
hdf dataset
'''

class CustomEncoder(torch.nn.Module):
    def __init__(self, num_dim, model_name):
        super(CustomEncoder, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT').features
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(num_dim, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 768)
            )   
        self.transformer_encoder = AutoModel.from_pretrained(model_name)


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

        feed_forward_outputs = []
        for cnn_output in cnn_outputs:
            feed_forward_output = self.feed_forward(cnn_output)
            feed_forward_outputs.append(feed_forward_output)

        del cnn_outputs
        del feed_forward_output
        feed_forward_outputs = torch.stack(feed_forward_outputs, dim=0)
        
        
        encoder_output = self.transformer_encoder(inputs_embeds=feed_forward_outputs)
        del feed_forward_outputs
        return encoder_output    





class CustomDecoder(torch.nn.Module):
    def __init__(self, model_name, state_representation_size, action_dim):
        super(CustomDecoder, self).__init__()
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(state_representation_size+action_dim, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 768)
            ) 
        self.transformer_decoder = GPT2Model.from_pretrained(model_name)
        self.action_output = torch.nn.Linear(self.transformer_decoder.config.hidden_size, action_dim)

    def forward(self, state_representation, previous_actions):
        combined_inputs = torch.cat((state_representation, previous_actions), dim=-1)
        del state_representation
        del previous_actions
        combined_inputs = self.feed_forward(combined_inputs)
        decoder_outputs = self.transformer_decoder(inputs_embeds=combined_inputs)
        del combined_inputs
        action_predictions = self.action_output(decoder_outputs.last_hidden_state)
        del decoder_outputs
        return action_predictions       




SEQUENCE_LENGTH = 100
CNN_LAST_HIDDEN = 1280*7*7
LR = 10e-4
BATCH_SIZE = 5

ENCODER_MODEL_NAME = 'bert-base-uncased'
DECODER_MODEL_NAME = 'gpt2'
DECODER_STATE_REPRES = 768



if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    ten_board_writer = SummaryWriter()





    
    encoder_model = CustomEncoder(num_dim=CNN_LAST_HIDDEN, model_name = ENCODER_MODEL_NAME)
    encoder_model = encoder_model.to(device)  
    encoder_model.train()

    decoder_model = CustomDecoder(model_name = DECODER_MODEL_NAME, state_representation_size=DECODER_STATE_REPRES, action_dim = 2)
    decoder_model = decoder_model.to(device)
    decoder_model.train()

    encoder_weights_file = 'encoder3.pt'
    decoder_weights_file = 'decoder3.pt'
    if os.path.isfile(encoder_weights_file) and os.path.isfile(decoder_weights_file):
        encoder_model.load_state_dict(torch.load(encoder_weights_file))
        decoder_model.load_state_dict(torch.load(decoder_weights_file))
        print('weights loaded from file.')

    dataset_filename = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-10-15_18-52-31.h5'
    data_file = h5py.File(dataset_filename, 'r')
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
    decoder_optimizer = torch.optim.SGD(decoder_model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    epoch = 0
    while not k.is_pressed('alt') and not k.is_pressed('ctrl'):
        epoch += 1
        total_loss = 0
        epoch_train_time_start = time.time()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        for batch in dataloader:
            batch_state = batch[0].to(device)
            batch_action = batch[1].to(device)
            encoder_output = encoder_model.forward(batch_state)
            decoder_output = decoder_model.forward(encoder_output.last_hidden_state, batch_action)
            loss = criterion(decoder_output, batch_action)
            #print('Decoder output: ',decoder_output[0][-1].data)
            total_loss += loss
            loss.backward()
        average_loss = total_loss/len(dataloader)
        ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
        encoder_optimizer.step()
        decoder_optimizer.step()
        epoch_train_time_end = time.time()
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)
        print("Training Loss:", average_loss.item())

    
    
    
    if k.is_pressed('alt'):
        torch.save(encoder_model.state_dict(), 'encoder3.pt')
        torch.save(decoder_model.state_dict(), 'decoder3.pt')
        print('weights saved')
    else:
        print('weights were not saved')
    print('Train finish')