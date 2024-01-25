import rospy
from std_srvs.srv import Empty
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String

'''
Running node to restart all objects in simulation to inital coordinates
(Used running, work while running)
'''

if __name__ == "__main__":
    rospy.init_node('world_resetter')

    # Wait for the Gazebo reset_world service to become available
    rospy.wait_for_service('/gazebo/reset_world')

    # Create a service proxy for the reset_world service
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    twist_pub = rospy.Publisher('/robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)
    zero_twist = Twist()
    reset_pub = rospy.Publisher('/reset_signal', String, queue_size=1)

    # Call the reset_world service to reset the Gazebo world
    while not rospy.is_shutdown():
        try:
            reset_world()
            reset_pub.publish('reset')
            twist_pub.publish(zero_twist)
            print("Gazebo world reset successful!")
            time.sleep(5)
        except:
            pass

