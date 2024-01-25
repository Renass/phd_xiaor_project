import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import keyboard as k
import time
import h5py
from transformers import LxmertTokenizer, LxmertModel
from sklearn.model_selection import train_test_split
from torchvision.ops import roi_align
from torchvision.transforms import functional as F



'''
Behavioral cloning LXMERT transformer TRAIN LOOP
Actions are resolved as a regression task
hdf Dataset

'''

class LXMERT(torch.nn.Module):
    def __init__(self, device):
        super(LXMERT, self).__init__()
        self.device = device

        self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.faster_rcnn.eval()
        for param in self.faster_rcnn.parameters():
            param.requires_grad = False

        self.fc_roi = torch.nn.Linear(2048, 2048)
        self.fc_pos = torch.nn.Linear(4, 2048)
        self.layer_norm = torch.nn.LayerNorm(2048)

        self.encoder = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        #print('here', self.encoder.encoder.layer[11])
        for param in self.encoder.parameters():
            param.requires_grad = False
        #for i in [8, 9,10,11]:
        #    for param in self.encoder.encoder.layer[i].parameters():
        #        param.requires_grad = True
        self.tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')


        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 2)
            )


    def forward(self, states_tensor, prompt):
        i1, i2, i3, i4, i5 = states_tensor.size()
        states_tensor = states_tensor.view(i1*i2, i3, i4, i5)

        with torch.no_grad():
            backbone_features = self.faster_rcnn.backbone(states_tensor)
            proposals, _ = self.faster_rcnn.rpn(states_tensor, backbone_features)
            pooled_features = [roi_align(backbone_features['out'], [proposal], output_size=(7, 7), sampling_ratio=2) for proposal in proposals]
        
        final_embeddings = []
        for p_features, proposal in zip(pooled_features, proposals):
            # Position features
            pos_features = F.to_tensor(proposal).view(-1, 4)

            # Apply fully connected layers
            roi_out = self.fc_roi(p_features)
            pos_out = self.fc_pos(pos_features)

            # Layer normalization and averaging
            norm_roi = self.layer_norm(roi_out)
            norm_pos = self.layer_norm(pos_out)
            vj = (norm_roi + norm_pos) / 2

            final_embeddings.append(vj)
        
        
        
        
        
        print('here', len(final_embeddings))
        
        prompt = prompt*i2
        inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.device)
        visual_features = self.encoder.visual_proj(states_tensor).to(self.device)
        visual_attention_mask = torch.ones(visual_features.shape[:-1], dtype=torch.float).to(self.device)

        inputs.update({
            "visual_embeds": visual_features,
            "visual_attention_mask": visual_attention_mask
            }
        )



        outputs = self.encoder(**inputs)
        del inputs
        outputs = outputs.pooled_output
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
    



LR = 10e-4
BATCH_SIZE = 1
SEQ_LENGTH = 100
TEST_PART = 0.2
DATASET1 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-03_18-14-26.h5'
DATASET2 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-03_18-41-17.h5'
#PROMPT = ["Go to the cube"]* SEQ_LENGTH*BATCH_SIZE
PROMPT1 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-03_18-14-26_pormpt.txt'
PROMPT2 = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/sa-trajs2023-11-03_18-41-17_prompt.txt'   


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    ten_board_writer = SummaryWriter()


    model = LXMERT(device=device)
    model = model.to(device)  
    model.train()


    weights_file = 'LXMERT.pt'
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
    while not k.is_pressed('alt') and not k.is_pressed('ctrl'):
        epoch += 1
        total_loss = 0
        test_total_loss = 0
        epoch_train_time_start = time.time()
        optimizer.zero_grad()
        for batch in dataloader:
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
            del loss
        optimizer.step()
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
    
    
    if k.is_pressed('alt'):
        torch.save(model.state_dict(), 'visualBERT.pt')
        print('weights saved')
    else:
        print('weights were not saved')
    print('Train finish')