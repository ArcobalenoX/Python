# coding = utf-8
import cv2
import os
import numpy as np
import shutil

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)

def useCamera():
    # 获取摄像头
    capture = cv2.VideoCapture(1)
    capture.set(3, 512)
    cv2.namedWindow("image")

    reg = np.array([[[119,111], [247,227], [225,263], [405,533]]], dtype = np.int32)
    print(reg.shape)
    #im = np.zeros([240, 320], dtpe = np.uint8)
    #cv2.polylines(im, reg,True, (100,200,255))
    while capture.isOpened():
        # 摄像头打开，读取图像
        flag, image = capture.read()
        cv2.putText(image,'OpenCV',org=(100,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,
                    color=(255,100,100),thickness=2,lineType=cv2.LINE_AA) 
        #cv2.FONT                 
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

        cv2.imshow("image", image)
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite("test.jpg", image)
        elif k == ord("q") or k == ord("Q"):
            break
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    useCamera()