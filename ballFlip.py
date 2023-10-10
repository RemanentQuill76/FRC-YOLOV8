import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from ultralytics import YOLO


pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
profile = pipe.start(config)
model = YOLO(r"C:\Users\TDKin\OneDrive\Desktop\Targeting\v17_1536x864.pt") 
show=True
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
xyxy = 0

try:
    while True:
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        res = color.copy()

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # Predict using YOLOv8 model
        results = model.predict(color)

        depth = np.asanyarray(aligned_depth_frame.get_data())

        # List to store detected objects and their info
        detected_objects = []

        for result in results:
            # get the bounding box coordinates
            boxes = result.boxes
            if len(boxes) > 0:  # check if boxes is not empty
                xyxy = boxes.xyxy[0].tolist()
                x, y, x2, y2 = [int(i) for i in xyxy]
                # the rest of your code
            else:
                continue
            x, y, x2, y2 = [int(i) for i in xyxy]
            # scale the bounding box coordinates back to the original image size
            x, x2 = [int(i * color.shape[1] / 1536) for i in [x, x2]]
            y, y2 = [int(i * color.shape[0] / 864) for i in [y, y2]]
            w = x2 - x
            h = y2 - y

            # get the class ID and class name
            if result.probs is not None:
                class_id = np.argmax(result.probs[0])
                class_name = result.names[str(class_id)]
            else:
                continue

            # Crop depth data:
            depth_crop = depth[y:y+h, x:x+w].astype(float)

            if depth_crop.size == 0:
                continue

            depth_res = depth_crop[depth_crop != 0]

            # Get data scale from the device and convert to meters
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_res = depth_res * depth_scale

            if depth_res.size == 0:
                continue

            dist = min(depth_res)
            print("Minimum depth: ", dist)

            # Store detected object info
            detected_objects.append({
                'bbox': (x, y, w, h),
                'class_name': class_name,
                'dist': dist
            })

        # Sort detected objects by distance
        detected_objects.sort(key=lambda x: x['dist'])

        for i, obj in enumerate(detected_objects):
            x, y, w, h = obj['bbox']
            class_name = obj['class_name']
            dist = obj['dist']

            # Draw bounding box and depth value on both streams
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.rectangle(colorized_depth, (x, y), (x + w, y + h), (0, 255, 0), 3)
            text = f"Depth: {dist:.2f}, Class: {class_name}"
            cv2.putText(res, text, (x, y - 5), font, fontScale, fontColor, lineType)
            cv2.putText(colorized_depth, text, (x, y - 5), font, fontScale, fontColor, lineType)

            # Draw crosshairs for the closest object
            if i == 0:
                cv2.drawMarker(res, (x + w // 2, y), (0, 0, 255), cv2.MARKER_CROSS, markerSize=10, thickness=3)
                cv2.drawMarker(res, (x + w // 2, y + h), (0, 0, 255), cv2.MARKER_CROSS, markerSize=10, thickness=3)
                cv2.drawMarker(res, (x, y + h // 2), (0, 0, 255), cv2.MARKER_CROSS, markerSize=10, thickness=3)
                cv2.drawMarker(res, (x + w, y + h // 2), (0, 0, 255), cv2.MARKER_CROSS, markerSize=10, thickness=3)
        
        cv2.namedWindow('RBG', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RBG', res)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', colorized_depth)

        cv2.waitKey(1)
    
finally:
    pipe.stop()
