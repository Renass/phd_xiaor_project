import rospy
import tf
from tf.transformations import euler_from_quaternion

rospy.init_node('pose_extractor')

listener = tf.TransformListener()

rate = rospy.Rate(10.0)
while not rospy.is_shutdown():
    try:
        # Look up the transformation from target frame to source frame
        (trans, rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        print("Translation: ", trans)
        print("Rotation (quaternion): ", rot)
        roll, pitch, yaw = euler_from_quaternion(rot)
        print("Roll: ", roll)
        print("Pitch: ", pitch)
        print("Yaw: ", yaw)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue

    rate.sleep()