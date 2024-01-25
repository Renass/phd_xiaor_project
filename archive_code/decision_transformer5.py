import threading
import trajectories_gather3
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
decision_tramsformer (with MobileNetV2 images feature encoder)
Train/Inference not at the same time
imports: (trajectories_gather2 or trajectories_gather3)
work with: reward_publisher
(Used running)
'''

def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass


def train_inference_thread():
    epoch = 1
    while not rospy.is_shutdown():
        print('inference step starts')
        while traj_buffer.gather == True:
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


        
        
        
        
        
        print('train step starts')
        for local_epoch in range(EPOCH_PER_TRAIN):
            states_list = []
            actions_list = []
            returns_list = []
            total_return = []

            trajectories = traj_buffer.traj
            for i,trajectory in enumerate(trajectories):
                states_list.append([])
                actions_list.append([])
                returns_list.append([])
                total_return.append(0)
                for transition in trajectory:
                        state = transition[0]
                        action = transition[1]
                        returnn = transition[2][0]

                        state = transforms.ToPILImage()(state)
                        state = preprocess(state)

                        states_list[i].append(state)
                        actions_list[i].append(action)
                        returns_list[i].append(returnn)
                        total_return[i] = total_return[i] + float(returnn)

            #print('states list',len(states_list))
            #print('actions list', actions_list)
            #print('returns list', returns_list)
            #print('total return', total_return)
            all_states_tensor = []
            all_actions_tensor = []
            all_returns_tensor = []
            all_returns_to_go = []

            for i in range(len(trajectories)):
                states_tensor = torch.stack(states_list[i], dim=0)
                all_states_tensor.append(states_tensor)
        
                actions_tensor = torch.tensor(actions_list[i], dtype=torch.float32)
                actions_tensor = actions_tensor.view(1, -1, 2)
                all_actions_tensor.append(actions_tensor)

                returns_np = np.array(returns_list[i], dtype=np.float32)
                rewards_sequence = torch.tensor(returns_np, dtype=torch.float32).view(1, -1, 1)
                all_returns_tensor.append(rewards_sequence)
        
                returns_to_go = torch.full_like(rewards_sequence, fill_value=total_return[i])
                all_returns_to_go.append(returns_to_go)

            # Stack tensors for all trajectories
            all_states_tensor = torch.cat(all_states_tensor, dim=0)
            #print(all_states_tensor.shape)
            all_actions_tensor = torch.cat(all_actions_tensor, dim=0)
            #print(all_actions_tensor.shape)
            all_returns_tensor = torch.cat(all_returns_tensor, dim=0)
            all_returns_to_go = torch.cat(all_returns_to_go, dim=0)
            batch_size = len(traj_buffer.traj)
            sequence_length = NUM_TRANSITIONS
            timesteps_sequence = torch.arange(sequence_length, dtype=torch.long).view(1, -1)
            attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.long)
            attention_mask[:, -1] = 1
            with torch.no_grad():
                features = mobilenetv2(all_states_tensor)
                features = features.view(batch_size, -1, DT_STATE_DIM)

            loss = model.forward(
                    states=features,
                    actions=all_actions_tensor,
                    rewards=all_returns_tensor,
                    returns_to_go=all_returns_to_go,
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
        traj_buffer.new_data()

def publish_twist(publisher, a):
    twist_msg = Twist()
    twist_msg.linear.x = a[0]
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = a[1]
    publisher.publish(twist_msg)






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
    EPOCH_PER_TRAIN = 10
    IM_CHANNELS_NUMBER = 3
    BUFFER_SIZE = 10
    IM_AMOUNT = 1
    IM_RESOLUTION = (224,224)
    NUM_TRANSITIONS = 100
    #STATE_DIM = IM_RESOLUTION[0]*IM_RESOLUTION[1]*IM_CHANNELS_NUMBER*IM_AMOUNT
    DT_STATE_DIM = 1280*7*7
    traj_buffer = trajectories_gather3.TrajectoryBuffer(buffer_size=BUFFER_SIZE, im_amount=IM_AMOUNT, im_resolution=IM_RESOLUTION, always = False, num_transitions=NUM_TRANSITIONS)
    driv_pub = rospy.Publisher('robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)

    ten_board_writer = SummaryWriter()


    mobilenetv2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
    mobilenetv2.eval()
    mobilenetv2 = mobilenetv2.features

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
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=train_inference_thread)
    t1.start()
    print('Traj gather stars')
    t2.start()
    print('Train/Inference starts')

    t1.join()
    t2.join()
    #trainer.save_model("decision_transformer_model")