import cv2
import numpy as np
import os

def frame_cropped(frame_trigger, square_size, grid_x, grid_y, target_size=(640, 640)):
    video_h, video_w = frame.shape[:2]
    x1, y1 = max(0, grid_x), max(0, grid_y)
    x2, y2 = min(video_w, grid_x + square_size), min(video_h, grid_y + square_size)
    cropped_frame = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    cropped_frame[:y2 - y1, :x2 - x1] = frame[y1:y2, x1:x2]
    resized_frame = cv2.resize(cropped_frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_frame

def crop_boxes_from_frame(frame, perc_top, perc_bottom, box_size, box_distance, box_offset_x):
    # Dimensões do frame
    frame_height, frame_width = frame.shape[:2]

    # Define as posições das linhas
    line_limit_top = int(frame_height * perc_top)
    line_limit_bottom = int(frame_height * perc_bottom)

    # Cálculo do centro vertical da faixa de trigger
    center_y = int((line_limit_top + line_limit_bottom) // 2)

    # Coordenadas dos centros dos quadrados
    center_x = frame_width // 2 + box_offset_x
    left_x = center_x - (box_distance // 2)
    right_x = center_x + (box_distance // 2)

    # Limites do recorte (left box)
    left_top_y = max(center_y - box_size // 2, 0)
    left_bottom_y = min(center_y + box_size // 2, frame_height)
    left_left_x = max(left_x - box_size // 2, 0)
    left_right_x = min(left_x + box_size // 2, frame_width)

    # Limites do recorte (right box)
    right_top_y = max(center_y - box_size // 2, 0)
    right_bottom_y = min(center_y + box_size // 2, frame_height)
    right_left_x = max(right_x - box_size // 2, 0)
    right_right_x = min(right_x + box_size // 2, frame_width)

    # Recortes dos quadrados na imagem
    left_crop = frame[left_top_y:left_bottom_y, left_left_x:left_right_x]
    right_crop = frame[right_top_y:right_bottom_y, right_left_x:right_right_x]

    # Redimensiona para 640x640
    left_crop_resized = cv2.resize(left_crop, (640, 640))
    right_crop_resized = cv2.resize(right_crop, (640, 640))

    return left_crop_resized, right_crop_resized
