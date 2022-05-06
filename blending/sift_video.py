import cv2
import numpy as np
from sift import *
import screeninfo

if __name__ =="__main__":


    screen = screeninfo.get_monitors()[0]  # 0 1920*1080   1 2560*1440
    width, height = screen.width, screen.height
    print(width, height)

    captureA = cv2.VideoCapture("A.mp4")
    frames = captureA.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frames)
    fps = captureA.get(cv2.CAP_PROP_FPS)               # 返回视频的fps--帧率
    width = captureA.get(cv2.CAP_PROP_FRAME_WIDTH)     # 返回视频的宽
    height = captureA.get(cv2.CAP_PROP_FRAME_HEIGHT)   # 返回视频的高

    captureB = cv2.VideoCapture("B.mp4")
    frames = captureB.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frames)

    windowname = 'stitcher'
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while captureB.isOpened() and captureA.isOpened():
        #while capture.isOpened():
            # 摄像头打开，读取图像
        flag, imageA = captureA.read()
        #imageA = cv2.resize(imageA,(360,640))
        #print(flag, imageA.shape)
        #cv2.imshow("imageA", imageA)


        flag, imageB = captureB.read()
        #imageB = cv2.resize(imageB,(360,640))
        #print(flag, imageB.shape)
        #cv2.imshow("imageB", imageB)


        # 计算SIFT特征点和特征向量
        (kpsA, featuresA) = detectAndCompute(imageA)
        (kpsB, featuresB) = detectAndCompute(imageB)

        # 基于最近邻和随机取样一致性得到一个单应性矩阵
        (M, matches, status) = matchKeyPoints(kpsA, kpsB, featuresA, featuresB)

        # 绘制匹配结果
        drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

        # 拼接
        stich_image = stich(imageA, imageB, M)
        #cv2.imshow("stich_image", stich_image)


        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        (status, pano) = stitcher.stitch((imageA, imageB))
        print(pano.shape)
        #pano = cv2.resize(pano,(1920,1024))


        cv2.imshow(windowname, pano)

        k = cv2.waitKey(100)
        if k == ord("q") or k == ord("Q"):
            break        

    # 释放摄像头
    captureA.release()
    captureB.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

