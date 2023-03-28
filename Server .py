import cv2
import zmq
import base64
import numpy as np
import paddlehub as hub
import time
import math
import socket
import sys
class HeadPost(object):
    def __init__(self):
        self.module = hub.Module(name="face_landmark_localization")
        # 头部三维关键点坐标
        self.model_points = np.array([
            [6.825897, 6.760612, 4.402142],
            [1.330353, 7.122144, 6.903745],
            [-1.330353, 7.122144, 6.903745],
            [-6.825897, 6.760612, 4.402142],
            [5.311432, 5.485328, 3.987654],
            [1.789930, 5.393625, 4.413414],
            [-1.789930, 5.393625, 4.413414],
            [-5.311432, 5.485328, 3.987654],
            [2.005628, 1.409845, 6.165652],
            [-2.005628, 1.409845, 6.165652],
            [2.774015, -2.080775, 5.048531],
            [-2.774015, -2.080775, 5.048531],
            [0.000000, -3.116408, 6.097667],
            [0.000000, -7.415691, 4.070434]
        ], dtype='float')
        # 头部投影点
        self.reprojectsrc = np.float32([
            [10.0, 10.0, 10.0],
            [10.0, -10.0, 10.0],
            [-10.0, 10.0, 10.0],
            [-10.0, -10.0, 10.0]])
        # 投影点连线
        self.line_pairs = [
            [0, 2], [1, 3], [0, 1], [2, 3]]


    def get_image_points(self, face_landmark):
        image_points = np.array([
            face_landmark[17], face_landmark[21],
            face_landmark[22], face_landmark[26],
            face_landmark[36], face_landmark[39],
            face_landmark[42], face_landmark[45],
            face_landmark[31], face_landmark[35],
            face_landmark[48], face_landmark[54],
            face_landmark[57], face_landmark[8]
        ], dtype='float')
        return image_points


    def get_pose_vector(self, image_points):
        # 设定相机的焦距、图像的中心位置
        center = (self.photo_size[1] / 2, self.photo_size[0] / 2)
        focal_length = self.photo_size[1]
        # 相机内参数矩阵
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]],
            dtype="float")
        # 畸变矩阵（假设不存在畸变）
        dist_coeffs = np.zeros((4, 1))
        # 函数solvepnp接收一组对应的3D坐标和2D坐标，以及相机内参camera_matrix和dist_coeffs进行反推图片的外参rotation_vector,translation_vector
        ret, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)
        # 函数projectPoints根据所给的3D坐标和已知的几何变换来求解投影后的2D坐标
        reprojectdst, ret = cv2.projectPoints(self.reprojectsrc, rotation_vector, translation_vector, camera_matrix,
                                              dist_coeffs)
        return rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst

    # 将旋转向量转换为欧拉角
    def get_euler_angle(self, rotation_vector, translation_vector):
        # 通过罗德里格斯公式将旋转向量和旋转矩阵之间进行转换
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        return euler_angle



    def pose_euler_angle(self, photo):
        self.photo_size = photo.shape
        res = self.module.keypoint_detection(images=[photo], use_gpu=False)
        face_landmark = res[0]['data'][0]
        image_points = self.get_image_points(face_landmark)
        rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst = self.get_pose_vector(image_points)
        euler_angle = self.get_euler_angle(rotation_vector, translation_vector)
        pitch = euler_angle[0]
        yaw = euler_angle[1]
        roll = euler_angle[2]
        if(pitch<-20 or pitch>20 or yaw<-30 or yaw >30):
            print("distracted")
            command = str(0)  # 向树莓派发送信号0
            footage_socket.send_string(command)

            # time.sleep(1)
        else:
            print("concentrate")
            command = str(1)  # 向树莓派发送信号0
            footage_socket.send_string(command)
        # 画出投影框
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(4, 2)))
        for start, end in self.line_pairs:
            cv2.line(photo, reprojectdst[start], reprojectdst[end], (0, 0, 255))
        # 标注14个人脸关键点
        for (x, y) in image_points:
            cv2.circle(photo, (int(x), int(y)), 2, (0, 0, 255), -1)
        # 显示参数
        cv2.putText(photo, "pitch: " + "{:5.2f}".format(float(pitch)), (15, int(self.photo_size[0] / 2 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(photo, "yaw: " + "{:6.2f}".format(float(yaw)), (15, int(self.photo_size[0] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(photo, "roll: " + "{:6.2f}".format(float(roll)), (15, int(self.photo_size[0] / 2 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('headpost', photo)
        cv2.waitKey(50)

time.sleep(5)
"""实例化用来接收帧的zmq对象"""
context = zmq.Context()
"""zmq对象建立TCP链接"""
footage_socket = context.socket(zmq.PAIR)
footage_socket.bind('tcp://*:5555')
#译码
while True:
    frame = footage_socket.recv_string()  # 接收TCP传输过来的图像数据
    print("rec")
    img = base64.b64decode(frame) #把数据进行base64解码后储存到内存img变量中
    npimg = np.frombuffer(img, dtype=np.uint8) #把这段缓存解码成一维数组
    source = cv2.imdecode(npimg, 1) #将一维数组解码为图像source
    HeadPost().pose_euler_angle(photo=source)


