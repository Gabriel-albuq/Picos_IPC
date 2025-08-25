import os
import sys
import torch
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
print(root_path)

from src.func import (
    device_config,
    device_start,
    device_start_capture,
    device_start_capture_multiples,
    start_application_interface,
    load_settings, 
    save_settings,
    load_model
)

if __name__ == '__main__':
    # INPUTS
    type_model = 'FRCNN_RN50'
    model = load_model(type_model)
    config_path = r'app\config.txt'
    exposure_value = 0.0
    sec_run_model = 0.4
    wait_key = 16

    # Verificar se a GPU está disponível e configurar o dispositivo
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDispositivo de processamento utilizado: {torch_device}')

    current_directory = os.path.dirname(os.path.abspath(__file__))   # Diretório atual
    parent_directory = os.path.dirname(current_directory)   # Diretório pai (pasta acima)

    (
        linha,
        device_name,
        device_path,
        camera_backend,
        option_visualize,
        perc_top,
        perc_bottom,
        perc_median,
        min_score,
        limit_center,
        save_dir,
        salvar_maiores,
        salvar_menores,
        deslocamento_esquerda,
        deslocamento_direita,
        box_size,
        box_distance,
        box_offset_x,
    ) = load_settings(config_path)

    # Iniciar a aplicação
    linha, device_name, device_path, camera_backend, option_visualize, perc_top, perc_bottom, \
            perc_median, min_score, limit_center, save_dir, salvar_maiores, salvar_menores, deslocamento_esquerda, deslocamento_direita, \
            box_size, box_distance, box_offset_x = start_application_interface(config_path)

    # Caso seja uma câmera, converter em número
    try:
        device_path = int(device_path)  # Tenta converter para inteiro
    except ValueError:
        pass

    # Inicia o Device
    (device,device_fps,device_width,device_height,device_exposure) = device_start(device_name, camera_backend, device_path)

    if device:
        if camera_backend == "OpenCV":
            device = device_config(device_name, device, device_fps, device_width, device_height, device_exposure)

        device_start_capture(device_path, option_visualize, camera_backend, torch_device, device_name, device, device_fps, type_model, model,
                              option_visualize, sec_run_model, perc_top, perc_bottom, perc_median, deslocamento_esquerda, deslocamento_direita,
                              box_size, box_distance, box_offset_x, wait_key, config_path, 
                              exposure_value, min_score, limit_center, save_dir, salvar_maiores, salvar_menores, linha
        )

        # device_start_capture_multiples(camera_backend, torch_device, device_name, device, device_fps, type_model, model,
        #                       option_visualize, sec_run_model, perc_top, perc_bottom, perc_median, deslocamento_esquerda, deslocamento_direita,
        #                       box_size, box_distance, box_offset_x, wait_key, config_path, 
        #                       exposure_value, min_score, limit_center, save_dir, linha
        # )