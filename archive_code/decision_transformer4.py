import threading
import trajectories_gather
import rospy
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import torch 
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from torch.utils.tensorboard import SummaryWriter
#from typing import List, Dict
import time
import torchvision.models as models
from torchvision import transforms
from PIL import Image

'''
file of decision_tramsformer (with MobileNetV2 images feature encoder)
imports: trajectories_gather
work with: reset_env, reward_publisher
'''

def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass


def inference_thread():
    while not rospy.is_shutdown():
        if len(traj_buffer.traj) > 0:
            if len(traj_buffer.traj[-1]) >  0:
                start_time = time.time()
                current_trajectory = traj_buffer.traj[-1] 
                #print('current_trajectory',current_trajectory)
                states_list = []
                actions_list = []
                returns_list = []
                for transition in current_trajectory:
                    state = transition[0]
                    action = transition[1]
                    returnn = transition[2]
                    
                    state = transforms.ToPILImage()(state)
                    state = preprocess(state)

                    states_list.append(state)
                    actions_list.append(action)
                    returns_list.append(returnn)

                #print('watch here')
                #print(len(actions_list))
                #print(actions_list[0])
            
 
                states_tensor = torch.stack(states_list, dim=0)  # Stack states along a new batch dimension
                #states_tensor = states_tensor.view(-1, IM_CHANNELS_NUMBER, IM_RESOLUTION[0],IM_RESOLUTION[1])

                actions_tensor = torch.tensor(actions_list, dtype=torch.float32)
                actions_tensor = actions_tensor.view(1, -1, 2) 

                returns_np = np.array(returns_list, dtype=np.float32)
                rewards_sequence = torch.tensor(returns_np, dtype=torch.float32).view(1, -1, 1) 
                timesteps_sequence = torch.arange(len(returns_list), dtype=torch.long).view(1, -1)
                returns_to_go = torch.full_like(rewards_sequence, fill_value=100)
                batch_size, sequence_length, _ = actions_tensor.size()
                attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.long)
                attention_mask[:, -1] = 1 
                #print('shapes:')
                #print('states tensor:', states_tensor.shape)
                #print(actions_tensor.shape)
                #print(rewards_sequence.shape)
                #print(returns_to_go.shape)
                #print(timesteps_sequence.shape)
                #print(attention_mask.shape)
                states_tensor = states_tensor.to(device)
                actions_tensor = actions_tensor.to(device)
                rewards_sequence = rewards_sequence.to(device)
                returns_to_go = returns_to_go.to(device)
                timesteps_sequence = timesteps_sequence.to(device)
                attention_mask = attention_mask.to(device)


                with torch.no_grad():
                    features = mobilenetv2(states_tensor)
                    #print('features',features.shape)
                    features = features.view(1, -1, DT_STATE_DIM)
                    #print('features for DT', features.shape)
                    state_preds, action_preds, return_preds = model.original_forward(
                        states=features,
                        actions=actions_tensor,
                        rewards=rewards_sequence,
                        returns_to_go=returns_to_go,
                        timesteps=timesteps_sequence,
                        attention_mask=attention_mask,
                        return_dict=False,
                        )
                publish_twist(driv_pub, action_preds[0][-1])
                end_time = time.time()
                print('one_move time :', end_time - start_time)

def publish_twist(publisher, a):
    twist_msg = Twist()
    twist_msg.linear.x = a[0]
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = a[1]
    publisher.publish(twist_msg)


def training_thread():
    epoch = 1
    while not rospy.is_shutdown():
        if len(traj_buffer.traj) > 1:
                current_trajectory = traj_buffer.traj[-2] 
                states_list = []
                actions_list = []
                returns_list = []
                total_return = 0 


                for transition in current_trajectory:
                    state = transition[0]
                    action = transition[1]
                    returnn = transition[2]

                    state = transforms.ToPILImage()(state)
                    state = preprocess(state)

                    states_list.append(state)
                    actions_list.append(action)
                    returns_list.append(returnn)
                    total_return += float(returnn[0])

                #print('state_len', len(states_list))
                #print('actions_len', len(actions_list))
                #print('returns_len', len(returns_list))

                states_tensor = torch.stack(states_list, dim=0)  # Stack states along a new batch dimension
                #states_tensor = states_tensor.view(1, -1, STATE_DIM)

                actions_tensor = torch.tensor(actions_list, dtype=torch.float32)
                #print(actions_tensor.shape)
                actions_tensor = actions_tensor.view(1, -1, 2) 
                #print(actions_tensor.shape)

                returns_np = np.array(returns_list, dtype=np.float32)
                rewards_sequence = torch.tensor(returns_np, dtype=torch.float32).view(1, -1, 1) 
                timesteps_sequence = torch.arange(len(returns_list), dtype=torch.long).view(1, -1)

                returns_to_go = torch.tensor([total_return] * len(current_trajectory), dtype=torch.float32)
                returns_to_go = returns_to_go.view(1, -1, 1)

                batch_size, sequence_length, _ = actions_tensor.size()
                attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.long)
                attention_mask[:, -1] = 1 


                states_tensor = states_tensor.to(device)
                actions_tensor = actions_tensor.to(device)
                rewards_sequence = rewards_sequence.to(device)
                returns_to_go = returns_to_go.to(device)
                timesteps_sequence = timesteps_sequence.to(device)
                attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    features = mobilenetv2(states_tensor)
                    features = features.view(1, -1, DT_STATE_DIM)

                loss = model.forward(
                        states=features,
                        actions=actions_tensor,
                        rewards=rewards_sequence,
                        returns_to_go=returns_to_go,
                        timesteps=timesteps_sequence,
                        attention_mask=attention_mask,
                        return_dict=False,
                        )
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                ten_board_writer.add_scalar('Loss', loss['loss'].item(), epoch)
                epoch +=1
                print("Training Loss:", loss['loss'].item())



class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)
    


                        
if __name__ == '__main__':
    IM_CHANNELS_NUMBER = 3
    BUFFER_SIZE = 2
    IM_AMOUNT = 1
    IM_RESOLUTION = (224,224)
    STATE_DIM = IM_RESOLUTION[0]*IM_RESOLUTION[1]*IM_CHANNELS_NUMBER*IM_AMOUNT
    DT_STATE_DIM = 1280*7*7

    device = torch.device('cuda:0')

    traj_buffer = trajectories_gather.TrajectoryBuffer(buffer_size=BUFFER_SIZE, im_amount=IM_AMOUNT, im_resolution=IM_RESOLUTION, always = True)
    driv_pub = rospy.Publisher('robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)

    ten_board_writer = SummaryWriter()


    #mobilenetv2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
    #mobilenetv2.eval()
    #mobilenetv2 = mobilenetv2.features
    mobilenetv2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT').features.to(device)
    mobilenetv2.eval()



    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match MobileNetV2 input size
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(            # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
                        ),
    ])


    configuration = DecisionTransformerConfig()
    configuration.state_dim = DT_STATE_DIM
    configuration.act_dim = 2 
    model = TrainableDT(configuration)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=inference_thread)
    t3 = threading.Thread(target=training_thread)
    t1.start()
    print('Traj gather stars')
    t2.start()
    print('Inference starts')
    t3.start()
    #print('Training starts')

    t1.join()
    t2.join()
    t3.join()
    #trainer.save_model("decision_transformer_model")