# -*- coding: utf-8 -*-
# !/usr/bin/python3
# SkillFramework 0.2.2 python demo

import os
import cv2
import time
import hilens

from utils import *
from socket import *
from detect_traffic_light import traffic_light

HOST = ''
PORT = 7778
bufsize = 1024
socket_3399 = socket(AF_INET, SOCK_STREAM)
socket_3399.bind((HOST, PORT))
socket_3399.listen()

def run(work_path):
    light = 0
    # 系统初始化，参数要与创建技能时填写的检验值保持一致
    hilens.init("driving")

    # 初始化自带摄像头与HDMI显示器,
    # hilens studio中VideoCapture如果不填写参数，则默认读取test/camera0.mp4文件，
    # 在hilens kit中不填写参数则读取本地摄像头
    camera = hilens.VideoCapture()
    display = hilens.Display(hilens.HDMI)

    # 初始化模型a
    # model_path = os.path.join(work_path, 'model/convert-38ef.om')
    model_path = os.path.join(work_path,'model/convert-5095.om')
    # model_path2 = os.path.join(work_path,'model/driving2.om')

    driving_model = hilens.Model(model_path)
    # driving_model1 = hilens.Model(model_path1)
    # driving_model2 = hilens.Model(model_path2)
    while True:
        # frame_index += 1
        # 1. 设备接入 #####
        input_yuv = camera.read()  # 读取一帧图片(YUV NV21格式)

        # 2. 数据预处理 #####
        # img_bgr = cv2.cvtColor(
        #     input_yuv, cv2.COLOR_YUV2BGR_NV21)  # 转为BGR格式
        # img_preprocess, img_w, img_h = preprocess(img_bgr)  # 缩放为模型输入尺寸
        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_YUV2RGB_NV21)  # 转为RGB格式
        img_preprocess, img_w, img_h = preprocess(img_rgb)  # 缩放为模型输入尺寸

        # 3. 模型推理 #####
        output = driving_model.infer([img_preprocess.flatten()])
        # 4. 获取检测结果 #####
        bboxes = get_result(output, img_w, img_h)
        for box in bboxes:
            if box[-2] == 3:
                triffic_light_bbox = ((int(box[0]),int(box[1])),(int(box[2]), int(box[3])))
                light = traffic_light(img_rgb[triffic_light_bbox[0][1]:triffic_light_bbox[1][1],triffic_light_bbox[0][0]:triffic_light_bbox[1][0]],triffic_light_bbox)
        img_rgb, labelName = draw_boxes(img_rgb, bboxes)
        detection_info = get_boxes(bboxes,light)
        socketSendMsg(socket_3399,detection_info)
        output_yuv = hilens.cvt_color(img_rgb, hilens.RGB2YUV_NV21)
        display.show(output_yuv)  # 显示到屏幕上
    socket_3399.close()
    hilens.terminate()


if __name__ == "__main__":
    run(os.getcwd())