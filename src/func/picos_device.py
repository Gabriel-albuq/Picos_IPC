import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from .picos_rules_detection import rules_detection, no_rules_detection
from .picos_run_model import run_model
from .picos_trigger import trigger_frame, trigger_test
from .picos_interface import save_settings
from .picos_cropped import frame_cropped
from src.class_camera_gx import GxiCapture

def device_start(device_name, camera_backend, device_path):
    
    print(f"Verificando device '{device_name}'")
    if camera_backend == 'GxCam':
        device = GxiCapture(device_path) if device_path is not None else None
    else:
        device = cv2.VideoCapture(device_path) if device_path is not None else None

    if not device or not device.isOpened():
        print(f"Erro ao abrir a câmera '{device_name}'.")
        return None, None, None, None, None

    device_fps = device.get(cv2.CAP_PROP_FPS)
    device_width = int(device.get(cv2.CAP_PROP_FRAME_WIDTH))
    device_height = int(device.get(cv2.CAP_PROP_FRAME_HEIGHT))
    device_exposure = device.get(cv2.CAP_PROP_EXPOSURE)  # Pode retornar -1 se não for suportado

    print('-----------------------------------')
    print(f'FPS: {device_fps}')
    print(f'Largura: {device_width}, Altura: {device_height}')
    print(f'Exposição: {device_exposure}')
    print('-----------------------------------')

    return device, device_fps, device_width, device_height, device_exposure


def device_config(device_name, device, device_fps, device_width,
                  device_height, device_exposure):
    
    print(f"Configurando device '{device_name}'")
    device.set(cv2.CAP_PROP_FRAME_WIDTH, device_width)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, device_height)
    device.set(cv2.CAP_PROP_EXPOSURE, device_exposure)
    device.set(cv2.CAP_PROP_FPS, device_fps)
    print('-----------------------------------')

    return device


def calculate_fps_video(frame_count, start_time_fps):
    
    frame_count += 1
    elapsed_time = (time.time() - start_time_fps)  # Tempo decorrido desde o início
    fps_display = (frame_count / elapsed_time if elapsed_time > 0 else 0)  # Calcular FPS

    return fps_display


def device_start_capture(camera_backend, torch_device, device_name, device, device_fps, type_model, 
                         model, visualize, sec_run_model, perc_top, perc_bottom, wait_key,
                         config_path, exposure_value, min_score, limit_center, save_dir, linha,
                         cropped_image, square_size, grid_x, grid_y):
    
    frame_delay = int(device_fps * sec_run_model)  # Número de quadros para rodar o modelo

    start_process = 'OFF'
    frame_count = 0
    time_run_model = 0
    count_trigger = 0
    wait_trigger_off = True
    start_time_fps = time.time()
    restart_return_camera = True
    return_camera = None
    device_width = int(device.get(cv2.CAP_PROP_FRAME_WIDTH))
    device_height = int(device.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nAbrindo câmera '{device_name}'...")
    while True:
        frame_count += 1

        # Verifica se a imagem capturada da câmera mudou de estado V > F ou F > V, caso sim printa um aviso e exibe a imagem
        if device is not None:
            if return_camera != True:
                restart_return_camera = True

            # Capturar a imagem
            if camera_backend == "OpenCV":
                return_camera, frame_original = device.read()
            if camera_backend == "GxCam":
                return_camera, frame_original = device.read()  

            if restart_return_camera == True:
                restart_return_camera = False
                print(f"Imagem da câmera '{device_name}' capturada.")
                print('Captura desligada')
                print('-----------------------------------')

        else:
            if return_camera != False:
                restart_return_camera = True

            return_camera, frame_original = False, np.zeros((device_width, device_height, 3), dtype=np.uint8)  # Frame preto para câmera 1

            if restart_return_camera == True:
                restart_return_camera = False
                print(f"Falha ao capturar imagem da câmera '{device_name}'.")

        # Cortar caso necessário
        if cropped_image == 1:
            try:
                frame_original = frame_cropped(frame_original.copy(), square_size, grid_x, grid_y, target_size=(640, 640))
            except:
                return_camera, frame_original = False, np.zeros((device_width, device_height, 3), dtype=np.uint8)

        # Teste do Trigger
        frame_trigger, result_trigger, trigger_top_value, trigger_bottom_value = trigger_test(frame_original.copy(), perc_top, perc_bottom)

        # Tratamento da imagem com o Trigger
        frame_trigger = trigger_frame(frame_trigger, trigger_top_value, trigger_bottom_value,
                                      perc_top, perc_bottom)

        if start_process == 'OFF':
            # Abrir a imagem de configuração
            frame_config = frame_trigger.copy()

            cv2.rectangle(frame_config,  (0, 0),  (frame_config.shape[1], 50), (80, 43, 30),  -1)

            cv2.putText(frame_config, f'{str(start_process)} (O)', (5, frame_config.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame_config, f'Abertura superior: {int(perc_bottom*100)}%   (Q - W)',  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(frame_config,  f'Abertura inferior: {int(perc_top*100)}%   (A - S)',  (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 255),  1,  cv2.LINE_AA)
            
            #cv2.putText(frame_config,  f'Abertura da camera (iluminacao): {exposure_value}   (R - T)',  (300, 20), cv2.FONT_HERSHEY_SIMPLEX,  0.4,  (255, 255, 255),  1,  cv2.LINE_AA)

            # Exibe o quadro ao vivo
            if visualize == 1:
                cv2.imshow(f'Ao vivo: {device_name}', frame_config)

            # Botões para configurar
            key = cv2.waitKey(wait_key) & 0xFF
            if key:
                # Ligar o detector
                if key == ord('o') or key == ord('O'):
                    start_process = 'ON'
                    save_settings(config_path, exposure_value, perc_top, perc_bottom, min_score,limit_center, sec_run_model, wait_key, save_dir,square_size, grid_x, grid_y)
                    print('Captura ligada')
                    print('-----------------------------------')

                # Abertura Superior
                if key == ord('q') or key == ord('Q'):
                    if perc_top >= 0.05:
                        perc_top = ((perc_top * 100) - 5) / 100
                if key == ord('w') or key == ord('W'):
                    if perc_top <= 0.95:
                        perc_top = ((perc_top * 100) + 5) / 100

                # Abertura Inferior
                if key == ord('a') or key == ord('A'):
                    if perc_bottom >= 0.05:
                        perc_bottom = ((perc_bottom * 100) - 5) / 100
                if key == ord('s') or key == ord('S'):
                    if perc_bottom <= 0.95:
                        perc_bottom = ((perc_bottom * 100) + 5) / 100

                # Trigger superior
                if key == ord('r') or key == ord('R'):
                    exposure_value -= 0.5
                    device.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
                elif key == ord('t') or key == ord('T'):
                    exposure_value += 0.5
                    device.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

                if key == 27:
                    break

        if start_process == 'ON':
            # Calcula o FPS do video
            fps_display = calculate_fps_video(frame_count, start_time_fps)

            # Exibe o quadro ao vivo
            if visualize == 1:
                cv2.putText(frame_trigger, f'{str(start_process)} (O)', (5, frame_trigger.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(f'Ao vivo: {device_name}', frame_trigger)

            # Para ativar o trigger, ele tem que ficar off pelo menos uma vez (para não contar o mesmo biscoito 2x)
            if (
                result_trigger == False
            ):   # O trigger tem que ficar off pelo menos uma vez para poder ser ativado novamente
                wait_trigger_off = True

            # Quando a contagem de frame entra no intervalo de número de frames para rodar no modelo, e ele vem de uma não contagem, ativa para rodar o modelo
            if frame_count % frame_delay == 0 and wait_trigger_off:
                time_run_model = 1

            # Rodar o modelo
            if time_run_model:
                if result_trigger:
                    start_time = time.time()   # Início da medição de tempo
                    count_trigger += 1
                    time_run_model = 0
                    wait_trigger_off = False

                    print('-')

                    data_hora_trigger = datetime.now().strftime('%Y%m%d_%H%M%S')
                    data_hora_trigger_text = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                    print(f'Trigger ativado as {data_hora_trigger_text}, rodando o modelo')

                    detections_sorted = run_model(torch_device, type_model, model, frame_original)
                    frame_detect, total_detections = rules_detection(frame_original.copy(), detections_sorted,perc_top, perc_bottom,
                                                                     min_score, limit_center)
                    # frame_detect, total_detections = no_rules_detection(frame_original.copy(), detections_sorted,perc_top, perc_bottom,
                    #                                                  min_score, limit_center)
                    if total_detections > 0:
                        if visualize == 1:
                            cv2.imshow(f'Aplicacao do Modelo: : {device_name}', frame_detect)
                        if save_dir:
                            save_frame(frame_original, frame_detect, linha, data_hora_trigger, total_detections, save_dir)

                    end_time = time.time()  # Fim da medição de tempo
                    processing_time = end_time - start_time
                    print(
                        f'(Tempo de Processamento: {processing_time:.4f}s) - Detecções: {len(detections_sorted)}'
                    )

            # Botões para configurar
            key = cv2.waitKey(wait_key) & 0xFF
            if key:
                # Desliga o detector
                if key == ord('o') or key == ord('O'):
                    start_process = 'OFF'
                    print('Captura desligada')
                    print('-----------------------------------')

                if key == 27:
                    break

        key = cv2.waitKey(wait_key) & 0xFF
        if key:
            # Pressione 'ESC' para sair
            if key == 27:
                break

    device.release()
    cv2.destroyAllWindows()


def save_frame(frame_original, frame_detection, linha, data_hora_atual,
               total_detections, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    caminho_SM = os.path.join(save_dir, 'SM')
    os.makedirs(caminho_SM, exist_ok=True)
    frame_SM_path = os.path.join(caminho_SM, f'SM_{linha}_{data_hora_atual}.jpg')
    csv_SM_path = os.path.join(caminho_SM, f'SM_{linha}_{data_hora_atual}.csv')
    cv2.imwrite(frame_SM_path, frame_original)

    with open(csv_SM_path, mode='w', newline='') as csv_SM_file:
        writer = csv.writer(csv_SM_file)
        writer.writerow(['DataHora', 'Linha', 'TotalDeteccoes'])
        writer.writerow([data_hora_atual, linha, total_detections])

    caminho_CM = os.path.join(save_dir, 'CM')
    os.makedirs(caminho_CM, exist_ok=True)
    frame_CM_path = os.path.join(caminho_CM, f'CM_{linha}_{data_hora_atual}.jpg')
    csv_CM_path = os.path.join(caminho_CM, f'CM_{linha}_{data_hora_atual}.csv')
    cv2.imwrite(frame_CM_path, frame_detection)

    with open(csv_CM_path, mode='w', newline='') as csv_CM_file:
        writer = csv.writer(csv_CM_file)
        writer.writerow(['DataHora', 'Linha', 'TotalDeteccoes'])
        writer.writerow([data_hora_atual, linha, total_detections])

    print(f"Salvo em {frame_CM_path} e {frame_SM_path}")