# coding=UTF-8
# This Python file uses the following encoding: utf-8
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import zmq
import base64
import sys
import os

import threading

import pygame
from pygame import mixer
from gpiozero import LED
#from time import sleep
# from gpiozero import Button
import smbus
import LCD1602 as LCD

# 定义震动模块端口
red = LED(17)
red.off()

IP = '192.168.43.233' #图像接受端的IP地址
    
"""实例化用来发送帧的zmq对象"""
contest = zmq.Context()
"""zmq对象使用TCP通讯协议"""
footage_socket = contest.socket(zmq.PAIR)
"""zmq对象和视频接收端建立TCP通讯协议"""
footage_socket.connect('tcp://%s:5555'%IP)
print(IP)

# 定义注意力分散反应函数
def distracted_action():
    red.on()
    # 生成一个pygame的界面
    pygame.display.set_mode([300, 300])
    # 初始化音响
    mixer.init()
    mixer.music.set_volume(1)
    mixer.music.load('test.mp3')
    mixer.music.play()
    # 初始化lcd
    LCD.init_lcd()
    LCD.print_lcd(3, 0, "Distracted")
    # 震动模块震动5次
#    num = 0
#    while num < 2:
#        red.on()
#        time.sleep(1)
#       red.off()
#        time.sleep(1)
#        num = num + 1
    #mixer.music.pause()

    # 点击×可以关闭界面的代码
#    while 1:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT:
#                sys.exit()

def send():
    while True:
        camera = PiCamera()
        camera.resolution = (640, 480)#设置相机分辨率
        camera.framerate = 32#设置刷新率
        camera.hflip = True#是否进行水平翻转
        camera.vflip = True#是否进行垂直翻转
        camera.capture('/home/pi/Desktop/image.jpg')#拍摄图片并保存到该路径
        camera.close()#解除摄像头占用
        
        reload(sys)#
        sys.setdefaultencoding('utf8')#python2.x的默认编码是ascii，而代码中可能由utf-8的字符导致，解决方法是设置utf-8
        img = cv2.imread('/home/pi/Desktop/image.jpg')
        
        
        encoded, buffer = cv2.imencode('.jpg', img) #把图像数据再次转换成流数据，
                                                    # 并且把流数据储存到内吨buffer中
        jpg_as_test = base64.b64encode(buffer) #把内存中的图像流数据进行base64编码
        footage_socket.send(jpg_as_test) #把编码后的流数据发送给图像的接收端
        os.remove('/home/pi/Desktop/image.jpg')
        time.sleep(0.5)

def recv():
    while True:
        jj
        command = footage_socket.recv_string()
        
        if int(command)==0:
            print('Distracted')
            distracted_action()
        elif int(command)==1:
            print('Concentrate')
            red.off()
            LCD.print_lcd(3, 0, "Concentrate")
    
def main():
   
    add_thread = threading.Thread(target=send,name = 'T1')
    #启动线程1来运行job函数
    add_thread2 = threading.Thread(target=recv,name = 'T2')
    #启动线程2来运行T2函数
    add_thread.start()
    #add_thread启动
    add_thread2.start()
    #add_thread2启动
    
main()
