# -*- coding:utf-8 -*-

import cv2
import sys
import os


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(major_ver, minor_ver, subminor_ver)

if __name__ == '__main__':

    # Set up tracker1.
    # Instead of MIL, you can also use

    tracker_types = ['MIL', 'KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'MOSSE', 'BOOSTING']
    tracker_type = tracker_types[2]

    if tracker_type == 'MIL':
        tracker1 = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker1 = cv2.TrackerKCF_create()
    if tracker_type == 'CSRT':
        tracker1 = cv2.TrackerCSRT_create()

    if tracker_type == 'TLD':
        tracker1 = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker1 = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'MOSSE':
        tracker1 = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == 'BOOSTING':
        tracker1 = cv2.legacy.TrackerBoosting_create()

    # Read video
    # video_path = ""
    # video = cv2.VideoCapture(video_path)
    # #video = cv2.VideoCapture(0)
    # # Exit if video not opened.
    # if not video.isOpened():
    #     print("Could not open video")
    #     sys.exit()

    # # Read first frame.
    # ok, frame = video.read()
    # if not ok:
    #     print('Cannot read video file')
    #     sys.exit()


    imgs_path = r"E:\Download\PETS09-S2L1\img1"
    frame = cv2.imread(os.path.join(imgs_path,"000001.jpg"))

    # Define an initial bounding box
    bbox = (287, 23, 86, 320) # Uncomment the line below to select a different bounding box
    # Initialize tracker1 with first frame and bounding box
    bbox1 = cv2.selectROI(frame, False)
    ok1 = tracker1.init(frame, bbox1)

    #tracker2 = cv2.TrackerKCF_create()
    # bbox2 = cv2.selectROI(frame, False)
    # ok2 = tracker2.init(frame, bbox2)

    cv2.destroyWindow("ROI selector")

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    fps = 30
    h,w = frame.shape[:2]
    out = cv2.VideoWriter('track.mp4', fourcc, fps , (w, h))

    #while True:
        # Read a new frame
        #ok, frame = video.read()
        #if not ok1:
        #    print("no frame")
        #    break

    for i in os.listdir(imgs_path):
        frame = cv2.imread(os.path.join(imgs_path,i))
        # cv2.imshow("origin",frame)
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker1
        ok1, bbox1 = tracker1.update(frame)
        print("bbox1", bbox1)
        # ok2, bbox2 = tracker2.update(frame)
        # print("bbox2", bbox2)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok1:
            # Tracking success
            p1 = (int(bbox1[0]), int(bbox1[1]))
            p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # if ok2:
        #     # Tracking success
        #     p1 = (int(bbox2[0]), int(bbox2[1]))
        #     p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
        #     cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
        # else:
        #     # Tracking failure
        #     cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)


        # Display tracker1 type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display result
        cv2.imshow("Tracking", frame)

        out.write(frame)        
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord("q") or k == ord("Q"): break
        if k == ord("s") or k == ord("S"): 
            bbox = cv2.selectROI(frame, False)
            tracker1.init(frame, bbox)
            ok1, bbox1 = tracker1.update(frame)
        

    out.release()
    cv2.destroyAllWindows()
