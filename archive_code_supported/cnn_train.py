import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, TensorDataset
import keyboard as k
import time
import h5py

'''
CNN for behavioral cloning
regression task
'''

class CubeClassifier(torch.nn.Module):
    def __init__(self):
        super(CubeClassifier, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT').features
        for param in self.mobilenetv2.parameters():
            param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(CNN_LAST_HIDDEN, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 2)
            )   


    def forward(self, states_tensor):
        i1, i2, i3, i4, i5 = states_tensor.size()
        states_tensor = states_tensor.view(i1*i2, i3, i4, i5)
        with torch.no_grad():
            cnn_output = self.mobilenetv2(states_tensor)
            del states_tensor
            i1, i2, i3, i4 = cnn_output.size()
            cnn_output = cnn_output.view(i1, i2*i3*i4)

        feed_forward_output = self.fc(cnn_output)
        del cnn_output
        return feed_forward_output   
    

LR = 10e-4
BATCH_SIZE = 2
#DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-10-24_17-28-03.h5'
DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-10-15_18-52-31.h5'
CNN_LAST_HIDDEN = 1280*7*7



if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    ten_board_writer = SummaryWriter()


    cnn_model = CubeClassifier()
    cnn_model = cnn_model.to(device)  
    cnn_model.train()

    cnn_weights_file = 'cnn_model.pt'
    if os.path.isfile(cnn_weights_file):
        cnn_model.load_state_dict(torch.load(cnn_weights_file))
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
    

    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    epoch = 0
    while not k.is_pressed('alt') and not k.is_pressed('ctrl'):
        epoch += 1
        total_loss = 0
        epoch_train_time_start = time.time()
        optimizer.zero_grad()
        for batch in dataloader:
            batch_state = batch[0].to(device)
            batch_label = batch[1].to(device)
            j1,j2,j3 = batch_label.size()
            batch_label = batch_label.view(j1*j2, j3)
            cnn_output = cnn_model.forward(batch_state)
            loss = criterion(cnn_output, batch_label)
            total_loss += loss
            loss.backward()
        average_loss = total_loss/len(dataloader)
        ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
        optimizer.step()
        epoch_train_time_end = time.time()
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)
        print("Training Loss:", average_loss.item())

    
    
    
    if k.is_pressed('alt'):
        torch.save(cnn_model.state_dict(), 'cnn_model.pt')
        print('weights saved')
    else:
        print('weights were not saved')
    print('Train finish')