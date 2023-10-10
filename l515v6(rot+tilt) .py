import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO   
from networktables import NetworkTables
# Create a blank 300x300 black image
image = np.zeros((300,600,3), np.uint8)

#Camera position inputs
camera_x = -10.5
camera_y = 38.75
camera_z = 4.52
camera_tilt_deg = 16
obj_offset_x = 0
obj_offset_y = 0
obj_depth = 0
distance = 0
angle_deg = 0
angle_rad = 0
rotation_deg = 0
d=50
res=0

#Use if troubleshooting
debug_mode=True
# Initialize the network tables
NetworkTables.initialize(server='roborio-7054-frc.local')
# Create a custom tab
sd = NetworkTables.getTable("Shuffleboard").getSubTable("inference")
# Define the debugging function
def display_bbox_values(bbox):
    if debug_mode:
        print(f"Raw bounding box values: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")  


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)  # depth stream at 720p
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
model = YOLO(r"C:\Users\Owner\Desktop\Tiller Shiz\robot shit\v17_1536x864.pt") 
#set camera fov for usage
hfov = 69
vfov = 42            
# Initialize an empty list to store the results
recent_results = []
# Prepare font settings for OpenCV's putText function
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White   
line_type = 2  
# Create a blank image
img = np.zeros((400, 512, 3), np.uint8)

# Define the font type
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the font scale
font_scale = 0.75

# Define the font color
font_color = (255, 255, 255)

# Define the line thickness
line_type = 2

# Calculate the scaling factors
scale_x = 1920/1536
scale_y = 1080/864
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

        # Run model prediction on the color image
        results = model.predict(source=color_image, show=True, conf=0.7, half=True)  # Set show to False  

        # Get the frame width and height
        width = depth_frame.get_width()
        height = depth_frame.get_height()  
        # Create a list to store recent results
        recent_results = []
        for result in results:
            # Get bounding boxes, classes, and confidence scores
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()  
            # Iterate over each detection
            for i in range(len(boxes_xyxy)):
                bbox = boxes_xyxy[i] 
            
                # Scale the bounding box coordinates
                scaled_bbox = bbox * [scale_x, scale_y, scale_x, scale_y]
            
                # Calculate center of scaled bounding box
                box_x = ((scaled_bbox[0] + scaled_bbox[2]) / 2).astype(int)
                box_y = ((scaled_bbox[1] + scaled_bbox[3]) / 2).astype(int) 
            
                # Clamp the coordinates to the image dimensions
                box_x = np.clip(box_x, 0, width - 1)
                box_y = np.clip(box_y, 0, height - 1)
            
                # Apply the filter to the depth frame
                filled_frame = hole_filling.process(depth_frame)   
                # Convert the filled frame back to a depth frame
                filled_depth_frame = rs.depth_frame(filled_frame)  
                # Calculate depth at the center of the bounding box in inches
                depth = filled_depth_frame.get_distance(box_x, box_y) * 39.37
            
                # Calculate the coordinates of the inner 1/25 area of the bounding box
                inner_coords = scaled_bbox + 2 * (scaled_bbox[2:] - scaled_bbox[:2]) / 5
            
                # Extract the depth values in the inner 1/25 area
                inner_depth_values = depth_image[int(inner_coords[1]):int(inner_coords[3]), int(inner_coords[0]):int(inner_coords[2])]
            
                # Calculate the median depth in the inner 1/25 area
                median_depth = np.median(inner_depth_values) * 39.37
            
                # Calculate horizontal and vertical offset in degrees
                offset_x = ((box_x / width) - 0.5) * hfov
                offset_y = ((box_y / height) - 0.5) * vfov
                print("offset_x" )
                print (offset_x)
                print("offset_y" )
                print(offset_y)
                print("point depth")
                print(depth)
            
                # Calculate the l/r, f/b, up/down position of the object relative to the camera
                x = depth * (np.sin(np.radians(offset_x))) # x position (left- and right+)
                y = depth * (np.sin(np.radians(90)) / (np.sin(np.radians(90-offset_x))))
                z = depth * (np.sin(np.radians(offset_y)))
                print("x")
                print(x)
                print("y")
                print(y)
                print("z")
                print(z)  


                # Create a dictionary to store the result
                res = {
                'bbox': bbox,
                'class': classes[i],
                'conf': confidences[i],
                'depth': median_depth,
                'average_depth': depth,  # Add average depth
                'offset_x': offset_x,
                'offset_y': offset_y
                }

                # Loop over the results
                for res in recent_results:
                    # Create a blank image
                    img = np.zeros((400, 512, 3), np.uint8)
                
                    # Put the text on the image
                    cv2.putText(img, f"offset_x: {res['offset_x']}", (10, 30), font, font_scale, font_color, line_type)
                    cv2.putText(img, f"offset_y: {res['offset_y']}", (10, 60), font, font_scale, font_color, line_type)
                    cv2.putText(img, f"point depth: {res['depth']}", (10, 90), font, font_scale, font_color, line_type)
                    cv2.putText(img, f"x: {res['x']}", (10, 120), font, font_scale, font_color, line_type)
                    cv2.putText(img, f"y: {res['y']}", (10, 150), font, font_scale, font_color, line_type)
                    cv2.putText(img, f"z: {res['z']}", (10, 180), font, font_scale, font_color, line_type)
                
                    # Display the image
                    cv2.imshow('Image', img)
                    cv2.waitKey(0)

        # Create a NetworkTables instance
        NetworkTables.initialize("roboRIO-7054-FRC.local")  # Replace "XXXX" with your team number
        sd = NetworkTables.getTable("SmartDashboard")
        for i, res in enumerate(recent_results):
            table = sd.getSubTable(f"Object{i}")
            
            # Send the class, 3D location, and time as entries in the sub-table
            table.putString("class", res.get("class"))
            table.putNumber("x", res.get("x", 0))  # 0 is the default value if 'x' doesn't exist
            table.putNumber("y", res.get("y", 0))  # 0 is the default value if 'y' doesn't exist
            table.putNumber("z", res.get("z", 0))  # 0 is the default value if 'z' doesn't exist
            table.putNumber("time", time.time())

            # Send the values over network tables
            sd.putNumber("distance", distance)
            sd.putNumber("angle", angle_deg)
            sd.putNumber("rotation", rotation_deg)

finally:
    # Stop streaming
    pipeline.stop()


#NOTE use cscore in the future for sending video feed maybe??
#current stopgap may just be for windows inference use !!   