import torch
import os
import trajectories_gather5
import threading
import rospy
import time
from geometry_msgs.msg import Twist
from transformers import BertModel, BertTokenizer
import torchvision.models as models
import math
from transformers import ViltProcessor, ViltModel
from transformers import OpenAIGPTConfig, OpenAIGPTModel
import numpy as np

'''
Behavioral cloning Renas  transformer INFERENCE
Actions are resolved as a CLASSIFICATION task
hdf Dataset

1. TEXT-Image encoding using ViLT (trainable) (modality encoding?)
2. Text-Image cls tokens and action tokens (positional-encoding?) (modality-encoding?) 
3. (Text-Image)-(action) causal Transformer GPT 

'''

BUFFER_SIZE = 1
IM_RESOLUTION = (640, 480)
SEQUENCE_LENGTH = 20

#CMD_PUBLISH_TOPIC = '/cmd_vel'
CMD_PUBLISH_TOPIC = 'robot_base_velocity_controller/cmd_vel'

#IMAGE_TOPIC = 'camera/rgb/image_raw'
IMAGE_TOPIC = '/image_raw'

LOAD_WEIGHTS = '/home/renas/pythonprogv2/phd_xiaor_project/weights/renas3.1_env.pt'

PROMPT = 'Go to the ball' 

VELOCITY_PAIRS = np.array([
        [0.5, 1],
        [0.5, 0],
        [0.5, -1],
        [0, 1],
        [0, 0],
        [0, -1],
        [-0.5, 1],
        [-0.5, 0],
        [-0.5, -1]
    ])

def actions_to_options(actions, velocity_pairs=None):
    '''Switch [batch_size, seq_length, 2] numpy actions to [batch_size, seq_length, 9] action options'''
    batch_size, seq_length, _ = actions.shape
    if velocity_pairs is None:
        print('Velocity pairs are not defined. Standart teleop_twist_keyboard applied.')
        velocity_pairs = np.array([
            [0.5, 1],
            [0.5, 0],
            [0.5, -1],
            [0, 1],
            [0, 0],
            [0, -1],
            [-0.5, 1],
            [-0.5, 0],
            [-0.5, -1]
        ])
    encoded_actions = np.zeros((batch_size, seq_length, 9), dtype=int)
    distances = np.sum((actions[:, :, np.newaxis, :] - velocity_pairs[np.newaxis, np.newaxis, :, :]) ** 2, axis=3)
    closest_indices = np.argmin(distances, axis=2)
    encoded_actions[np.arange(batch_size)[:, np.newaxis], np.arange(seq_length), closest_indices] = 1
    return encoded_actions

def publish_twist(publisher, a):
    twist_msg = Twist()
    twist_msg.linear.x = a[0]
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = a[1]
    publisher.publish(twist_msg)


def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass


def behav_clon_inference_thread():
    while not rospy.is_shutdown():
        print('inference step starts')
        while traj_buffer.gather == True:
            if len(traj_buffer.states_buffer) > 0:
                if len(traj_buffer.states_buffer[-1]) >  0:
                    start_time = time.time()
                    states = torch.stack(traj_buffer.states_buffer[-1], dim=0).unsqueeze(0)
                    actions = torch.tensor(traj_buffer.actions_buffer[-1]).unsqueeze(0)
                    actions = torch.tensor(actions_to_options(actions.numpy(), velocity_pairs=VELOCITY_PAIRS)).float()
                    if states.shape[1] == actions.shape[1]:
                        output = model.forward(states, actions, prompt=[PROMPT])
                        #print('cnn_output', cnn_output.shape)
                        action = output[-1,-1,:].cpu().detach().numpy()
                        action = VELOCITY_PAIRS[np.argmax(action)]
                        publish_twist(driv_pub, action)
 
                    
                    end_time = time.time()
                    print('one_move time :', end_time - start_time)

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
            torch.nn.Linear(9, 768),
        #    torch.nn.GELU(),
        #    torch.nn.Linear(768, 768)
            )
        self.states_enc_vector = EncodingVector(d_model=self.d_model)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model)
        self.pos_enc = PositionalEncoding(d_model=self.d_model)

        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=4, n_head=12)
        self.gpt_model = OpenAIGPTModel(self.gpt_config)

        self.fc2 = torch.nn.Linear(768, 9)


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
    


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)


    model = Renas(device=device)
    model = model.to(device)  
    model.eval()

    model.load_state_dict(torch.load(LOAD_WEIGHTS))
    print('weights loaded from file.')

    traj_buffer = trajectories_gather5.TrajectoryBuffer(
        buffer_size=BUFFER_SIZE,  
        im_resolution=IM_RESOLUTION,
        im_preproc= False, 
        num_transitions=SEQUENCE_LENGTH, 
        always=True,
        image_topic= IMAGE_TOPIC,
        cmd_vel_topic= CMD_PUBLISH_TOPIC, 
        reset_environment=False)
    
    driv_pub = rospy.Publisher(CMD_PUBLISH_TOPIC, Twist, queue_size=1)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('Cube Classifier inference starts')