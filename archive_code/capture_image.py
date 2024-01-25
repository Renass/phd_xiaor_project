import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import time

'''
Running node of preparing torch tensors from gazebo camera stream
(used to import classes of it to other scripts)
'''

class ImageBuffer:
    def __init__(self, buffer_size=100, im_resolution=(640,480)):
        self.im_resolution = im_resolution
        self.gather = True
        self.buffer_size = buffer_size
        self.buffer = []
        self.tensor_images = None
        self.bridge = CvBridge()

        # Initialize the ROS node and subscribe to the image topic
        rospy.init_node('multimodal_inference_node', anonymous=True)
        rospy.Subscriber('/image_raw', Image, self.callback_image)

    def callback_image(self, msg):
        # Convert ROS Image to OpenCV format
        if self.gather:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.buffer.append(cv_image)

            # Check if we have received the desired number of images
            if len(self.buffer) == self.buffer_size:
                self.gather = False
                self.get_last_images_as_tensor()


 

    def get_last_images_as_tensor(self):
        
        # Get the last images from the buffer as a PyTorch tensor
        valid_images = [img for img in self.buffer if img is not None]
        if len(valid_images) == 0:
            return None
        
        stacked_images = np.stack(valid_images)
        self.tensor_images = torch.tensor(stacked_images.transpose((0, 3, 1, 2)), dtype=torch.float32)
        self.tensor_images = torch.reshape(self.tensor_images,(3,self.im_resolution[1]*self.buffer_size,self.im_resolution[0]) )
        #self.tensor_images = torch.cat((self.tensor_images,), dim=1)
        print('tensor images processed')
        #rospy.signal_shutdown("Received the desired number of images")


    def fresh_buffer(self):
        self.buffer = []
        self.tensor_images = None
        self.gather = True
        #rospy.init_node('multimodal_inference_node', anonymous=True)
        #rospy.Subscriber('/image_raw', Image, self.callback_image)
        #rospy.spin()
    
if __name__ == '__main__':
    # Create the ImageBuffer instance
    image_buffer = ImageBuffer(buffer_size=5, im_resolution=(224,224))
    while len(image_buffer.buffer)<image_buffer.buffer_size:
        time.sleep(0.1)
    print(image_buffer.buffer[0].shape)
    print(image_buffer.tensor_images.shape)

    #image_buffer.fresh_buffer()
    #image_buffer.fresh_buffer()
    # Run the program indefinitely
    try:
        rospy.spin()
    except KeyboardInterrupt:

        pass
