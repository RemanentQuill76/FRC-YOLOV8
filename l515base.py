import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO   
#Use if troubleshooting
debug_mode=True
# Define the debugging function
def display_bbox_values(bbox):
    if debug_mode:
        print(f"Raw bounding box values: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")  
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # depth stream at 720p
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)  # color stream at 1080p 
hole_filling = rs.hole_filling_filter()
colorizer = rs.colorizer() 
# Start streaming
profile = pipeline.start(config)
# Get the device
dev = profile.get_device()
# Get the depth sensor
depth_sensor = dev.first_depth_sensor()
# Set the preset
preset = rs.option.visual_preset
depth_sensor.set_option(preset, 3.0)  # Low Ambient Light
time.sleep(2)  # Add a delay to ensure the camera is ready 
#align depth fov to rgb camera
align_to = rs.stream.color
align = rs.align(align_to) 
# Load the model
model = YOLO(r"C:\Users\Owner\Desktop\robot shit\v17_1536x864.pt") 
#set camera fov for usage
hfov = 70
vfov = 43         
# Initialize an empty list to store the results
recent_results = []
# Prepare font settings for OpenCV's putText function
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White
line_type = 2  
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue   
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Create a blank image for text display
        text_image = np.zeros((500, 1000, 3), dtype="uint8")   
        # Run model prediction on the color image
        results = model.predict(source=color_image, show=True, conf=0.6, half=False)  # Set show to False  
        # Get the frame width and height
        width = depth_frame.get_width()
        height = depth_frame.get_height()  
        for result in results:
            # Get bounding boxes, classes, and confidence scores
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()  
        # Iterate over each detection
        for i in range(len(boxes_xyxy)):
            bbox = boxes_xyxy[i] 
            display_bbox_values(bbox)  
            # Calculate the scaling factors
            scale_x = 1920 / 1536
            scale_y = 1080 / 864   
            # Scale the bounding box coordinates
            scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y] 
            # Calculate center of scaled bounding box
            center_x = int((scaled_bbox[0] + scaled_bbox[2]) / 2)
            center_y = int((scaled_bbox[1] + scaled_bbox[3]) / 2)  
            # Clamp the coordinates to the image dimensions
            center_x = max(0, min(center_x, width - 1))
            center_y = max(0, min(center_y, height - 1))   
            # Create a hole filling filter
            hole_filling = rs.hole_filling_filter()
            # Apply the filter to the depth frame
            filled_frame = hole_filling.process(depth_frame)   
            # Convert the filled frame back to a depth frame
            filled_depth_frame = rs.depth_frame(filled_frame)  
            # Calculate depth at the center of the bounding box in inches
            depth = filled_depth_frame.get_distance(center_x, center_y) * 39.37
        
            # Calculate the coordinates of the inner 1/4 area of the bounding box
            inner_x1 = int(scaled_bbox[0] + (scaled_bbox[2] - scaled_bbox[0]) / 4)
            inner_y1 = int(scaled_bbox[1] + (scaled_bbox[3] - scaled_bbox[1]) / 4)
            inner_x2 = int(scaled_bbox[2] - (scaled_bbox[2] - scaled_bbox[0]) / 4)
            inner_y2 = int(scaled_bbox[3] - (scaled_bbox[3] - scaled_bbox[1]) / 4)
        
            # Extract the depth values in the inner 1/4 area
            inner_depth_values = depth_image[inner_y1:inner_y2, inner_x1:inner_x2]
        
            # Calculate the average depth in the inner 1/4 area
            average_depth = np.mean(inner_depth_values) * 39.37  # convert to inches
        
            # Calculate horizontal and vertical offset in degrees
            offset_x = ((center_x / width) - 0.5) * hfov
            offset_y = ((center_y / height) - 0.5) * vfov  
        
            # Create a dictionary to store the result
            res = {
                'bbox': bbox,
                'class': classes[i],
                'conf': confidences[i],
                'depth': depth,
                'average_depth': average_depth,  # Add average depth
                'offset_x': offset_x,
                'offset_y': offset_y
            }  
            # Add the result to the list
            recent_results.append(res) 
            # Add detection info to the text image
            text = f"Class: {res['class']}, Conf: {res['conf']:.2f}, Depth: {res['depth']:.2f}, Offset: ({res['offset_x']:.2f}, {res['offset_y']:.2f})"
            y = (i + 1) * 25  # Position the text for each detection on a new line
            cv2.putText(text_image, text, (10, y), font, font_scale, font_color, line_type)
        # If there are more than 3 elements in the list, remove the oldest one
        while len(recent_results) > 3:
            recent_results.pop(0)  
        # Display the text image in a window
        cv2.namedWindow('Text', cv2.WINDOW_NORMAL)
        cv2.imshow('Text', text_image)
        cv2.waitKey(1) 
        # Clear the text image after each frame
        text_image.fill(0) 
        # Clear the text image after each frame
        text_image.fill(0) 
finally:
    # Stop streaming
    pipeline.stop()
