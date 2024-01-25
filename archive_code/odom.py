import rospy
import tf
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import os
import math

'''
Get poses and orientations of robot and objects in gazebo
(Used as a demonstration code to write other scripts)
'''

def quaternion_to_yaw(orientation):
    # Convert the quaternion to a tuple (x, y, z, w)
    quat = (orientation.x, orientation.y, orientation.z, orientation.w)

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    euler = tf.transformations.euler_from_quaternion(quat)

    # Extract the yaw angle (rotation around the Z-axis)
    yaw = euler[2]  # The third element is the yaw angle

    return yaw

def model_states_callback(msg):
    os.system('clear')
    robot_base_index = msg.name.index("robot_base")  # Find the index of "robot_base" in the names list
    robot_base_pose = msg.pose[robot_base_index]  # Get the pose of "robot_base" using the index
    print("Robot Base Pose:")
    print("Position: x={}, y={}, z={}".format(robot_base_pose.position.x,
                                               robot_base_pose.position.y,
                                               robot_base_pose.position.z))
    yaw = quaternion_to_yaw(robot_base_pose.orientation)
    print("Robot Heading (Yaw):", math.degrees(yaw))

if __name__ == '__main__':
    rospy.init_node('ground_truth_subscriber')
    sub = rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback,queue_size=1) 

    try:
        rospy.spin()
    except:
        pass