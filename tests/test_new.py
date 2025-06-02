import os
import sys
import torch
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox
import time
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
print(root_path)

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


def device_start_capture(camera_backend, torch_device, device_name, device, device_fps, type_model, 
                         model, visualize, sec_run_model, perc_top, perc_bottom, wait_key,
                         min_score, limit_center, save_dir, linha):
    
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

            # Exibe o quadro ao vivo
            if visualize == 1:
                cv2.imshow(f'Ao vivo: {device_name}', frame_config)

            # Botões para configurar
            key = cv2.waitKey(wait_key) & 0xFF
            if key:
                # Ligar o detector
                if key == ord('o') or key == ord('O'):
                    start_process = 'ON'
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


def load_model(type_model, torch_device):
    # if type_model == 'YOLO':
    #     # Carregar o modelo YOLO treinado
    #     model_path = os.path.join(
    #         parent_directory,
    #         'data',
    #         'inputs',
    #         'ia_models',
    #         'YoloV11 Nano 20241030',
    #         'weights',
    #         'best.pt',
    #     )
    #     model = YOLO(model_path).to(torch_device)

    if type_model == 'FRCNN_RN50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')     # Carregar o modelo treinado
        num_classes = 2  # Inclua o número de classes (background + classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes))
        model.to(torch_device)   # Mover o modelo para o dispositivo

        # model_path = os.path.join(parent_directory, 'data', 'ia_models', 'FRCNN Resnet50', 'best_faster_rcnn_model_20241114.pth')
        model_path = os.path.join(
            #parent_directory,
            'data',
            'inputs',
            'ia_models',
            'FRCNN Resnet50',
            #'best_faster_rcnn_model_20241203.pth'
            'best_faster_rcnn_model_20250409_171546.pth'
        )
        model.load_state_dict(torch.load(model_path, map_location=torch_device))   # Carregar o modelo salvo, mapeando para o dispositivo correto
        model.eval()  # Colocar o modelo em modo de avaliação

    if type_model == 'FRCNN_MNV3L':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2))
        model.to(torch_device)

        model_path = os.path.join(
            #parent_directory,
            'data',
            'inputs',
            'ia_models',
            'FRCNN MobilenetV3 Large',
            'best_faster_rcnn_model_large.pth',
        )
        model.load_state_dict(torch.load(model_path, map_location=torch_device))   # Carregar o modelo salvo, mapeando para o dispositivo correto
        model.eval()  # Colocar o modelo em modo de avaliação

    if type_model == 'FRCNN_MNV3S':
        # Definir número de classes
        num_classes = 2  # Background + classe

        # Carregar o backbone MobileNetV3 Small, com saída ajustada para o Faster R-CNN
        backbone = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT').features
        backbone.out_channels = 576  # Saída final do MobileNetV3 Small

        # Definir o AnchorGenerator com tamanhos e razões customizados para o MobileNetV3
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        # Definir a camada de pooling e o box head do Faster R-CNN
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'], 
            output_size=7, 
            sampling_ratio=2,
        )

        # Construir o modelo Faster R-CNN com o backbone MobileNetV3 Small
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )

        model.to(torch_device)
        model_path = os.path.join(
            #parent_directory,
            'data',
            'inputs',
            'ia_models',
            'FRCNN MobileNetV3 Small',
            'best_faster_rcnn_mobilenetv3_small.pth',
        )
        model.load_state_dict(
            torch.load(model_path, map_location=torch_device)
        )
        model.eval()   # Colocar o modelo em modo de avaliação

    print(f'Model: {type(model).__name__}')  # Exibe o nome da classe do modelo
    return model


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDispositivo de processamento utilizado: {torch_device}')

# INPUTS
type_model = 'FRCNN_RN50'
model = load_model(type_model, torch_device)
config_path = r'app\config.txt'
exposure_value = 0.0
sec_run_model = 0.4
wait_key = 16

option_visualize = 1
perc_top = 0
perc_bottom = 1
min_score = 0.5
limit_center = 8
save_dir = None
linha = '14'
cropped_image = False
square_size = 0
grid_x = 0
grid_y = 0
device_name = 'teste'
camera_backend = 'opencv'
device_path = r'data\inputs\test_videos\2025-05-29_11-56-33.mp4'
perc_top: 0.4
perc_bottom: 0.8
min_score: 0.5
limit_center: 8
save_dir: 'data\\outputs\\capturas'
square_size: 640
grid_x: 0
grid_y: 0
crop_image: 1  # 1 = Sim

visualize = 1


(device,device_fps,device_width,device_height,device_exposure) = device_start(device_name, camera_backend, device_path)

if device:
    if camera_backend == "OpenCV":
        device = device_config(device_name, device, device_fps, device_width, device_height, device_exposure)

    device_start_capture(camera_backend, torch_device, device_name, device, device_fps, type_model, 
                            model, visualize, sec_run_model, perc_top, perc_bottom, wait_key,
                            min_score, limit_center, save_dir, linha)