import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Create a pipeline for the RealSense camera:
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Load the YOLOv8 model and set its confidence threshold:
model = YOLO(r"C:\Users\Owner\Desktop\robot shit\v17_1536x864.pt")

# Fetch frames from the camera, resize them and run them through the model for inference:
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    # Resize color_frame to the desired size
    color_image = cv2.resize(color_image , (1536, 864))

    # Run model (the return is Results object)
    results = model(color_image)

    for result in results:
        # Get the central point of the bounding box:
        center_x = (result.xyxy[0] + result.xyxy[2]) / 2
        center_y = (result.xyxy[1] + result.xyxy[3]) / 2
        
        # Get the depth from the center of the bounding box:
        depth = depth_frame.get_distance(center_x, center_y)

        print('Location: ({}, {}) Depth: {}'.format(center_x, center_y, depth))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
