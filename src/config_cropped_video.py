import cv2
import numpy as np

# Caminho dos arquivos
video_path = r"data\videos\2025-04-03_12-50-37.mp4"
config_path = r"app\config.txt"

def read_config_cropped(config_path):
    config = {}
    try:
        with open(config_path, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                config[key.strip()] = float(value.strip())
    except FileNotFoundError:
        print("Arquivo de configuração não encontrado. Criando novo.")
    return config

def save_config_cropped(config, config_path):
    with open(config_path, "w") as file:
        for key, value in config.items():
            file.write(f"{key} = {value}\n")

def frame_cropped(frame, square_size, grid_x, grid_y, target_size=(640, 640)):
    video_h, video_w = frame.shape[:2]
    x1, y1 = max(0, grid_x), max(0, grid_y)
    x2, y2 = min(video_w, grid_x + square_size), min(video_h, grid_y + square_size)
    cropped_frame = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    cropped_frame[:y2 - y1, :x2 - x1] = frame[y1:y2, x1:x2]
    resized_frame = cv2.resize(cropped_frame, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_frame

def configure_crop(video_path, config_path):
    config = read_config_cropped(config_path)
    square_size = int(config.get("square_size", 50))
    grid_x = int(config.get("grid_x", 100))
    grid_y = int(config.get("grid_y", 100))
    step_size, resize_step = 10, 10
    scale_percent = 70  # Definição da escala
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    video_h, video_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv2.namedWindow("Configuração", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Configuração", int(video_w * (scale_percent / 100)), int(video_h * (scale_percent / 100)))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Voltar ao início quando o vídeo acabar
            continue

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (grid_x, grid_y), (grid_x + square_size, grid_y + square_size), (0, 255, 0), 2)
        cv2.imshow("Configuração", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 13:  # Enter
            config["square_size"], config["grid_x"], config["grid_y"] = square_size, grid_x, grid_y
            save_config_cropped(config, config_path)
            break
        elif key in (ord('w'), 82):
            grid_y = max(0, grid_y - step_size)
        elif key in (ord('s'), 84):
            grid_y = min(video_h - square_size, grid_y + step_size)
        elif key in (ord('a'), 81):
            grid_x = max(0, grid_x - step_size)
        elif key in (ord('d'), 83):
            grid_x = min(video_w - square_size, grid_x + step_size)
        elif key == ord('i'):
            square_size = min(video_w - grid_x, video_h - grid_y, square_size + resize_step)
        elif key == ord('u') and square_size > resize_step:
            square_size -= resize_step

    cap.release()
    cv2.destroyAllWindows()

def play_cropped_video(video_path, config_path):
    config = read_config_cropped(config_path)
    square_size = int(config.get("square_size", 50))
    grid_x = int(config.get("grid_x", 100))
    grid_y = int(config.get("grid_y", 100))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    cv2.namedWindow("Vídeo Cortado", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame_cropped(frame, square_size, grid_x, grid_y)
        cv2.imshow("Vídeo Cortado", cropped_frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # ESC para sair
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    configure_crop(video_path, config_path)
    play_cropped_video(video_path, config_path)