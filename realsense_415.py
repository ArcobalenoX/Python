import pyrealsense2 as rs
import numpy as np
import cv2
'''
# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

pipe_profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
try:
    while True:
        # 等待一副连贯的帧：深度和颜色
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # 将图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # 内部和外部
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        # 深度刻度-深度框架内值的单位，即如何将值转换为1米的单位
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        #将深度映射到颜色
        # depth_pixel = [int(n) for n in input().split()]
        depth_pixel = [240, 320]  # Random pixel
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)

        color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
        color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())
        i = 640 * 200 + 200
        print('depth: ', [np.float(vtx[i][2])])
finally:

    # Stop streaming
    pipeline.stop()
'''
'''
import pyrealsense2 as rs
import numpy as np
import cv2

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipe_profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Map depth to color
    depth_pixel = [240, 320]  # Random pixel
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)

    color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
    color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
    print('depth: ', color_point)
    print('depth: ', color_pixel)
'''


import pyrealsense2 as rs
import numpy as np
import cv2
pc = rs.pointcloud()
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
pipe_profile = pipeline.start(config)
# align_to = rs.stream.color
# align = rs.align(align_to)
# print(pipeline.isOpened())
while True:
    frames = pipeline.wait_for_frames()
    # aligned_frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())

    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    # print(vtx)
    i = 640 * 300 + 300
    print('depth: ', [np.float(vtx[i][0]), np.float(vtx[i][1]), np.float(vtx[i][2])])
    cv2.imshow('frame', img_color)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
pipeline.stop()



