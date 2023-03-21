# coding = utf-8
import cv2
import os
import numpy as np


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)

def useCamera():
    # 获取摄像头
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,512)
    capture.set(cv2.CAP_PROP_FPS, 30)
    window_name = "VideoCapture"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_EVENT_LBUTTONDOWN)

    #im = np.zeros([240, 320], dtpe = np.uint8)
    #cv2.polylines(im, reg,True, (100,200,255))
    while capture.isOpened():
        # 摄像头打开，读取图像
        flag, frame = capture.read()
        if flag :
            cv2.putText(frame,'OpenCV',org=(100,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,
                    color=(255,100,100),thickness=2,lineType=cv2.LINE_AA) 
                                                                                                                                                          
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(20)
        if k == ord('s') or k == ord("S"):
            cv2.imwrite("shoot.jpg", frame)
        elif k == ord("q") or k == ord("Q"):
            break
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    useCamera()