import torch

'''
Behavioral cloning Renas  transformer camera-lidar INFERENCE

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. TEXT-Image encoding using ViLT (trainable) 
2. >Text-Image token + lidar map, costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 
'''

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)