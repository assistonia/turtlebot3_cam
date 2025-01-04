#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

# Wayland 경고 메시지 제거
os.environ["QT_QPA_PLATFORM"] = "xcb"

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # CvBridge 초기화
        self.bridge = CvBridge()
        
        # ROS2 이미지 토픽 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/webcam/image/compressed',
            self.image_callback,
            10)
        
        # ArUco 딕셔너리 설정 - 4x4만 사용
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # 파라미터 조정
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 0.5
        
        # 이미지 저장 디렉토리 생성
        self.save_dir = 'aruco_images'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 마지막 처리 시간 저장
        self.last_process_time = datetime.now()
        
        # 창 생성
        cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
        
    def image_callback(self, msg):
        try:
            current_time = datetime.now()
            time_diff = (current_time - self.last_process_time).total_seconds()
            
            # 압축된 이미지를 numpy 배열로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 이미지 전처리
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # ArUco 마커 감지
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
            
            # 마커가 감지되면
            if ids is not None:
                # 마커 표시 (바운딩 박스와 ID)
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # 각 마커에 대해 추가 정보 표시
                for i, (corner, id_num) in enumerate(zip(corners, ids)):
                    # 바운딩 박스의 코너 좌표 계산
                    corner = corner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corner
                    
                    # 정수형으로 변환
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    
                    # 바운딩 박스 그리기
                    cv2.line(cv_image, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(cv_image, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(cv_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(cv_image, bottomLeft, topLeft, (0, 255, 0), 2)
                    
                    # 마커 중심점 계산
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    
                    # 마커 ID 표시
                    cv2.putText(cv_image, f"ID: {id_num[0]}", (cX - 25, cY - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 1초마다 이미지 저장
                    if time_diff >= 1.0:
                        self.last_process_time = current_time
                        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                        filename = f"marker_{id_num[0]}_{timestamp}.jpg"
                        filepath = os.path.join(self.save_dir, filename)
                        cv2.imwrite(filepath, cv_image)
                        self.get_logger().info(f'Detected marker {id_num[0]} and saved to {filepath}')
            
            # 이미지 표시 (항상 수행)
            cv2.imshow('Camera View', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    
    try:
        rclpy.spin(aruco_detector)
    except KeyboardInterrupt:
        pass
    finally:
        aruco_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
