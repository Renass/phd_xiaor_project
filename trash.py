from geometry_msgs.msg import Twist
import rospy
import threading
import time

CMD_PUBLISH_TOPIC = 'rob/cmd_vel'

def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass


def behav_clon_inference_thread():
    time.sleep(1)
    publish_twist(driv_pub, [0, 0])

def publish_twist(publisher, a):
    twist_msg = Twist()
    twist_msg.linear.x = a[0]
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = a[1]
    publisher.publish(twist_msg)


if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    driv_pub = rospy.Publisher(CMD_PUBLISH_TOPIC, Twist, queue_size=1)


    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('Cube Classifier inference starts')