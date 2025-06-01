import cv2
import numpy as np
import os

def frame_cropped(frame, square_size, grid_x, grid_y, target_size=(640, 640)):
    video_h, video_w = frame.shape[:2]
    x1, y1 = max(0, grid_x), max(0, grid_y)
    x2, y2 = min(video_w, grid_x + square_size), min(video_h, grid_y + square_size)
    cropped_frame = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    cropped_frame[:y2 - y1, :x2 - x1] = frame[y1:y2, x1:x2]
    resized_frame = cv2.resize(cropped_frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_frame
