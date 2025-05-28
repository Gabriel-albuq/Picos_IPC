import cv2
import numpy as np
import os

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
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC para sair
            break
        elif key == 13:  # ENTER para sair também
            print("Enter pressionado. Encerrando reprodução.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def save_cropped_video(video_path, config_path, output_path, fps=30):
    config = read_config_cropped(config_path)
    square_size = int(config.get("square_size", 50))
    grid_x = int(config.get("grid_x", 100))
    grid_y = int(config.get("grid_y", 100))
    target_size = (640, 640)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    print("Salvando vídeo cortado")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame_cropped(frame, square_size, grid_x, grid_y, target_size)
        out.write(cropped_frame)

    cap.release()
    out.release()
    print(f"Vídeo salvo com sucesso em: {output_path}")

def extract_frames_by_interval(video_path, output_folder, interval_seconds=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps * interval_seconds)

    os.makedirs(output_folder, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"{saved_count} imagens salvas em: {output_folder}")

if __name__ == "__main__":
    configure_crop(video_path, config_path)
    play_cropped_video(video_path, config_path)

    folder, filename = os.path.split(video_path)
    filename_no_ext, _ = os.path.splitext(filename)
    cropped_filename = f"cropped_{filename}"
    output_path = os.path.join(folder, cropped_filename)

    save_cropped_video(video_path, config_path, output_path)

    frames_output_folder = os.path.join(f"data", "train_images", filename_no_ext)
    extract_frames_by_interval(video_path, frames_output_folder, interval_seconds=0.2)