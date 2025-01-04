#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.publisher = self.create_publisher(CompressedImage, 'webcam/image/compressed', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz로 발행
        
        # V4L2를 사용하여 웹캠 연결
        self.cap = cv2.VideoCapture('/dev/video0')
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error('웹캠을 열 수 없습니다!')
            return

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임을 읽을 수 없습니다!')
            return

        # 이미지 압축
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = np.array(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])[1]).tobytes()
        
        self.publisher.publish(msg)

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    webcam_publisher = WebcamPublisher()
    
    try:
        rclpy.spin(webcam_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        webcam_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
