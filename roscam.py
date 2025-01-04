#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class WebcamSubscriber(Node):
    def __init__(self):
        super().__init__('webcam_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            'webcam/image/compressed',
            self.image_callback,
            10)
        
        # 창 생성
        cv2.namedWindow('Webcam View', cv2.WINDOW_NORMAL)
        
    def image_callback(self, msg):
        # 압축된 이미지 데이터를 numpy 배열로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 이미지 표시
        cv2.imshow('Webcam View', image)
        cv2.waitKey(1)
    
    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    webcam_subscriber = WebcamSubscriber()
    
    try:
        rclpy.spin(webcam_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        webcam_subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
