import rospy
import tf
from gazebo_msgs.msg import ModelStates
import os
import math
from std_msgs.msg import String

'''
Running node to generate reward for Reinforcement learning
(Used running, work while running)
'''

def quaternion_to_yaw(orientation):
    # Convert the quaternion to a tuple (x, y, z, w)
    quat = (orientation.x, orientation.y, orientation.z, orientation.w)

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    euler = tf.transformations.euler_from_quaternion(quat)

    # Extract the yaw angle (rotation around the Z-axis)
    yaw = euler[2]  # The third element is the yaw angle

    return yaw


def calculate_reward(yaw,xyz,target_xyz):
    distance_to_target = (xyz.x-target_xyz.x)**2+(xyz.y-target_xyz.y)**2+(xyz.z-target_xyz.z)**2
    target_yaw = math.atan2(target_xyz.y - xyz.y, target_xyz.x - xyz.x)
    yaw_difference = abs(yaw - target_yaw)
    distance_reward = 50 - distance_to_target
    alignment_reward = math.cos(yaw_difference) * 50
    reward = distance_reward + alignment_reward
    return str(reward)

def model_states_callback(msg):
    os.system('clear')
    robot_base_index = msg.name.index("robot_base")
    robot_base_pose = msg.pose[robot_base_index]
    box_pose = msg.pose[msg.name.index('box_model')]

    robot_base_yaw = quaternion_to_yaw(robot_base_pose.orientation)
    reward = calculate_reward(robot_base_yaw,xyz=robot_base_pose.position,target_xyz=box_pose.position)
    pub.publish(reward)
    print(reward)

if __name__ == '__main__':
    rospy.init_node('reward_publisher', anonymous=True)
    sub = rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback,queue_size=1) 
    pub = rospy.Publisher('/gazebo/rl_reward', String, queue_size=1)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass