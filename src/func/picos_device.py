import csv
import os
import time
from datetime import datetime
import math

import cv2
import numpy as np

from .picos_rules_detection import rules_detection, no_rules_detection
from .picos_cropped import crop_boxes_from_frame
from .picos_run_model import run_model
from .picos_trigger import trigger_frame, trigger_test
from .picos_interface import save_settings
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


def detectar_slugs(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    limite_inferior_marrom_escuro = np.array([0, 50, 20])     # Marrom mais escuro
    limite_superior_marrom_escuro = np.array([20, 255, 200])  # Marrom médio
    
    limite_inferior_marrom_claro = np.array([10, 30, 40])      # Marrom claro
    limite_superior_marrom_claro = np.array([25, 150, 255])   # Marrom muito claro

    # Cria máscaras para ambas as faixas de marrom
    mascara_marrom_escuro = cv2.inRange(frame_hsv, limite_inferior_marrom_escuro, limite_superior_marrom_escuro)
    mascara_marrom_claro = cv2.inRange(frame_hsv, limite_inferior_marrom_claro, limite_superior_marrom_claro)

    # Combina as máscaras
    mascara_combinada = cv2.bitwise_or(mascara_marrom_escuro, mascara_marrom_claro)

    # Operações morfológicas para melhorar a máscara
    kernel = np.ones((5,5), np.uint8)
    mascara_combinada = cv2.morphologyEx(mascara_combinada, cv2.MORPH_CLOSE, kernel)  # Fecha pequenos buracos
    mascara_combinada = cv2.morphologyEx(mascara_combinada, cv2.MORPH_OPEN, kernel)   # Remove ruídos

    # Encontrar contornos na máscara combinada
    contornos, _ = cv2.findContours(mascara_combinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_filtrados = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)
        if area > 100 and w > 150:
            contornos_filtrados.append(contorno)

    return contornos_filtrados

def device_start_capture(camera_backend, torch_device, device_name, device, device_fps, type_model, 
                         model, visualize, sec_run_model, perc_top, perc_bottom, perc_median, deslocamento_esquerda, deslocamento_direita, 
                         box_size, box_distance, box_offset_x,  wait_key,
                         config_path, exposure_value, min_score, limit_center, save_dir, linha):
    
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

        frame_height, frame_width = frame_original.shape[:2]
        # Teste do Trigger
        frame_trigger, result_trigger, trigger_top_value, trigger_bottom_value = trigger_test(frame_original.copy(), perc_top, perc_bottom, deslocamento_esquerda, deslocamento_direita)

        # Tratamento da imagem com o Trigger
        frame_trigger = trigger_frame(frame_trigger, trigger_top_value, trigger_bottom_value,
                                      perc_top, perc_bottom, deslocamento_esquerda, deslocamento_direita,
                                      box_size, box_distance, box_offset_x)

        if start_process == 'OFF':
            # Abrir a imagem de configuração
            frame_config = frame_trigger.copy()

            cv2.rectangle(frame_config,  (0, 0),  (frame_config.shape[1], 50), (80, 43, 30),  -1)

            cv2.putText(frame_config, f'{str(start_process)} (O)', (5, frame_config.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame_config, f'Abertura superior: {int(perc_bottom*100)}%   (Q - W)',  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_config,  f'Abertura inferior: {int(perc_top*100)}%   (A - S)',  (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 255),  1,  cv2.LINE_AA)
            
            cv2.putText(frame_config, f'Deslocamento Esquerda: {int(perc_bottom*100)}%   (Q - W)',  (310, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_config,  f'Deslocamento Direita: {int(perc_top*100)}%   (A - S)',  (310, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 255),  1,  cv2.LINE_AA)

            cv2.putText(frame_config, f'Tamanho do Quadrado de Corte: {int(perc_bottom*100)}%   (Z - X)',  (610, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_config,  f'Distancia Entre Quadrados de Corte: {int(perc_top*100)}%   (C - V)',  (610, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 255),  1,  cv2.LINE_AA)
            cv2.putText(frame_config, f'Deslocamento Horizontal do Quadrado de Corte: {int(perc_bottom*100)}%   (N - M)',  (1010, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Exibe o quadro ao vivo
            if visualize == 1:
                cv2.imshow(f'Ao vivo: {device_name}', frame_config)

            # Botões para configurar
            key = cv2.waitKey(wait_key) & 0xFF
            if key:
                # Ligar o detector
                if key == ord('o') or key == ord('O'):
                    start_process = 'ON'
                    save_settings(config_path, exposure_value, perc_top, perc_bottom, min_score,limit_center, sec_run_model, wait_key, save_dir, deslocamento_esquerda, deslocamento_direita, box_size, box_distance, box_offset_x)
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

                # Deslocamento Esquerdo
                if key == ord('r') or key == ord('R'):
                    deslocamento_esquerda += 10
                if key == ord('t') or key == ord('T'):
                    deslocamento_esquerda -= 10

                # Deslocamento Direito
                if key == ord('f') or key == ord('F'):
                        deslocamento_direita += 10
                if key == ord('g') or key == ord('G'):
                    if perc_bottom <= 0.95:
                        deslocamento_direita -= 10

                # Tamanho do Quadrado de Corte
                if key == ord('z') or key == ord('Z'):
                    box_size -= 10
                if key == ord('x') or key == ord('X'):
                    box_size += 10

                # Distancia do Quadrado de Corte
                if key == ord('c') or key == ord('C'):
                        box_distance  -= 10
                if key == ord('v') or key == ord('V'):
                    if perc_bottom <= 0.95:
                        box_distance  += 10

                # Deslocamento Horizontal do Quadrado de Corte
                if key == ord('n') or key == ord('N'):
                    box_offset_x  -= 10
                if key == ord('m') or key == ord('M'):
                    box_offset_x  += 10

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
                    frame_left, frame_right = crop_boxes_from_frame(frame_original.copy(), perc_top, perc_bottom, box_size, box_distance, box_offset_x)

                    start_time = time.time()   # Início da medição de tempo
                    count_trigger += 1
                    time_run_model = 0
                    wait_trigger_off = False

                    print('-')

                    data_hora_trigger = datetime.now().strftime('%Y%m%d_%H%M%S')
                    data_hora_trigger_text = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                    print(f'Trigger ativado as {data_hora_trigger_text}, rodando o modelo')

                    # Esquerda
                    detections_sorted_left = run_model(torch_device, type_model, model, frame_left)
                    frame_detect_left, total_detections_left = rules_detection(frame_left.copy(), detections_sorted_left, 0, 1, perc_median,
                                                                    min_score, limit_center) # 0 e 1 porque nao estou fazendo com a imagem já cortada

                    # Direita
                    detections_sorted_right = run_model(torch_device, type_model, model, frame_right)
                    frame_detect_right, total_detections_right = rules_detection(frame_right.copy(), detections_sorted_right, 0, 1, perc_median,
                                                                    min_score, limit_center) # 0 e 1 porque nao estou fazendo com a imagem já cortada

                    total_detections = total_detections_left + total_detections_right

                    id_image = ''
                    if total_detections > 0:
                        if visualize == 1:
                            cv2.imshow(f'Aplicacao do Modelo no Lado Esquerdo: {device_name}', frame_detect_left)
                            cv2.imshow(f'Aplicacao do Modelo no Lado Direito: {device_name}', frame_detect_right)
                        if save_dir:
                            save_frame(frame_left, frame_detect_left, linha, id_image, data_hora_trigger, total_detections_left, save_dir)
                            save_frame(frame_right, frame_detect_right, linha, id_image, data_hora_trigger, total_detections_right, save_dir)

                    end_time = time.time()  # Fim da medição de tempo
                    processing_time = end_time - start_time
                    # print(
                    #     f'(Tempo de Processamento: {processing_time:.4f}s) - Detecções: {len(detections_sorted)}'
                    # )

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


def device_start_capture_multiples(camera_backend, torch_device, device_name, device, device_fps, type_model, 
                         model, visualize, sec_run_model, perc_top, perc_bottom, perc_median, deslocamento_esquerda, deslocamento_direita, 
                         box_size, box_distance, box_offset_x,  wait_key,
                         config_path, exposure_value, min_score, limit_center, save_dir, linha):
    
    frame_delay = int(device_fps * sec_run_model)  # Número de quadros para rodar o modelo

    start_process = 'ON'
    frame_count = 0
    time_run_model = 0
    count_trigger = 0
    wait_trigger_off = True
    start_time_fps = time.time()
    restart_return_camera = True
    return_camera = None
    device_width = int(device.get(cv2.CAP_PROP_FRAME_WIDTH))
    device_height = int(device.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configurações do tracker
    DISTANCIA_MAX_ASSOCIACAO = 200  # pixels
    MAX_FRAMES_SEM_DETECCAO = 10    # número de frames sem detecção antes de remover
    MIN_AREA = 100                  # área mínima para considerar um objeto
    MIN_LARGURA = 150               # largura mínima para considerar um objeto

    # Estrutura para armazenar objetos rastreados
    objetos_rastreados = {}  # Formato: {id: {'bbox': (x,y,w,h), 'historia': [(x,y)], 'frames_sem_detecao': int, 'total_detections': int}}
    proximo_id = 1

    print(f"\nAbrindo câmera '{device_name}'...")
    while True:
        frame_count += 1
        # Para cada objeto existente, marque como não detectado neste frame
        for obj_id in list(objetos_rastreados.keys()):
            objetos_rastreados[obj_id]['detectado_este_frame'] = False

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

        # Adiciona padding de 20 pixels (preto) ao redor da imagem
        padding = 50
        frame_original = cv2.copyMakeBorder(
            frame_original,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Cor preta (BGR)
        )

        altura, largura = frame_original.shape[:2]

        # Detectar slugs
        contornos = detectar_slugs(frame_original.copy())
        
        # Cópia da imagem original
        frame_marcado = frame_original.copy()
        altura, largura = frame_original.shape[:2]

        padding = 20
        margem_borda = 10
        contador = 1

        slugs_atuais = []
        for contorno in contornos:
            contador += 1

            area = cv2.contourArea(contorno)
            x, y, w, h = cv2.boundingRect(contorno)
            lado = max(w, h)

            centro_x = x + w // 2
            centro_y = y + h // 2

            lado_com_padding = lado + 2 * padding

            # Recalcula coordenadas com padding centralizado
            novo_x_pad = max(centro_x - lado_com_padding // 2, 0)
            novo_y_pad = max(centro_y - lado_com_padding // 2, 0)
            fim_x_pad = min(centro_x + lado_com_padding // 2, largura)
            fim_y_pad = min(centro_y + lado_com_padding // 2, altura)

            # Ajusta para manter dentro dos limites da imagem
            lado_pad = min(fim_x_pad - novo_x_pad, fim_y_pad - novo_y_pad)

            # Define as bounding boxes
            bbox_atual = (novo_x_pad, novo_y_pad, lado_pad, lado_pad)
            bbox_sem_padding = (centro_x - lado // 2, centro_y - lado // 2, lado, lado)

            if novo_y_pad < 30 or fim_y_pad > altura - 30:
                continue  # Ignora essa marcação

            # Adiciona à lista de detecções atuais
            slugs_atuais.append({
                'bbox': bbox_atual,
                'bbox_sem_padding': bbox_sem_padding,
                'centro': (centro_x, centro_y)
            })

            for slug in slugs_atuais:
                bbox_atual = slug['bbox']
                centro_atual = slug['centro']
                
                objeto_associado = None
                menor_distancia = DISTANCIA_MAX_ASSOCIACAO
                
                # Procura o objeto rastreado mais próximo
                for obj_id, dados in objetos_rastreados.items():
                    centro_rastreado = (dados['bbox'][0] + dados['bbox'][2]//2, 
                                    dados['bbox'][1] + dados['bbox'][3]//2)
                    distancia = math.hypot(centro_atual[0] - centro_rastreado[0], 
                                        centro_atual[1] - centro_rastreado[1])
                    
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        objeto_associado = obj_id
                
                # Se encontrou um objeto para associar
                if objeto_associado is not None:
                    # Atualiza os dados do objeto rastreado
                    objetos_rastreados[objeto_associado]['bbox'] = bbox_atual
                    objetos_rastreados[objeto_associado]['bbox_sem_padding'] = bbox_sem_padding
                    objetos_rastreados[objeto_associado]['historia'].append(centro_atual)
                    objetos_rastreados[objeto_associado]['detectado_este_frame'] = True
                    objetos_rastreados[objeto_associado]['frames_sem_detecao'] = 0
                else:
                    # Cria um novo objeto rastreado
                    objetos_rastreados[proximo_id] = {
                        'bbox': bbox_atual,
                        'bbox_sem_padding': bbox_sem_padding,
                        'historia': [centro_atual],
                        'frames_sem_detecao': 0,
                        'detectado_este_frame': True,
                        'total_detections': 0
                    }

                    objeto_associado = proximo_id
                    proximo_id += 1

                for obj_id in list(objetos_rastreados.keys()):
                    if not objetos_rastreados[obj_id]['detectado_este_frame']:
                        objetos_rastreados[obj_id]['frames_sem_detecao'] += 1
                    else:
                        objetos_rastreados[obj_id]['frames_sem_detecao'] = 0

                    # Remove objetos que não foram detectados por muitos frames
                    if objetos_rastreados[obj_id]['frames_sem_detecao'] > MAX_FRAMES_SEM_DETECCAO:
                        del objetos_rastreados[obj_id]


                # Processar detecções e salvar imagens para objetos detectados
                data_hora_model = datetime.now().strftime('%Y%m%d_%H%M%S')
                data_hora_model_text = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                for obj_id, dados in objetos_rastreados.items():
                    if dados['detectado_este_frame']:
                        bbox = dados['bbox']
                        bbox_sem_padding = dados['bbox_sem_padding']
                        novo_x, novo_y, lado_pad, _ = bbox
                        x_sem_padding, y_sem_padding, lado_sem_padding, _ = bbox_sem_padding
                        
                        # Recortar a região de interesse
                        recorte = frame_original[novo_y:novo_y + lado_pad, novo_x:novo_x + lado_pad]
                        if recorte.size == 0:  # Pula se o recorte estiver vazio
                            continue
                            
                        imagem_expandida = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_LINEAR)

                        # Roda o modelo
                        total_detections = 0
                        start_time = time.time()
                        print(f'Modelo rodando as {data_hora_model_text} para objeto ID {obj_id}')
                        detections_sorted_left = run_model(torch_device, type_model, model, imagem_expandida)

                        # Aplica as regras de detecção
                        frame_detectado, total_detections = rules_detection(imagem_expandida.copy(), detections_sorted_left, 0, 1, perc_median, min_score, limit_center) # 0 e 1 porque nao estou fazendo com a imagem já cortada

                        end_time = time.time()  # Fim da medição de tempo
                        processing_time = end_time - start_time
                        print(f'(Tempo de Processamento: {processing_time:.4f}s)')

                        # Atualiza o total de detecções para este objeto
                        objetos_rastreados[obj_id]['total_detections'] = total_detections

                        if total_detections > 0 and save_dir:
                            save_frame(imagem_expandida, frame_detectado, linha, obj_id, data_hora_model, total_detections, save_dir)

                        # Marcar Imagem com o Retangulo sem padding
                        cv2.rectangle(frame_marcado, 
                                      (x_sem_padding, y_sem_padding), (x_sem_padding + lado_sem_padding,  y_sem_padding + lado_sem_padding), 
                                      (255, 0, 0), 
                                      2
                        )

                        # Desenha o retângulo e ID
                        cv2.rectangle(frame_marcado, 
                                    (novo_x, novo_y), (novo_x + lado_pad, novo_y + lado_pad), 
                                    (255, 0, 0), 2)

                        texto_id = f"ID {obj_id}"
                        fonte = cv2.FONT_HERSHEY_SIMPLEX
                        escala = 0.6
                        espessura = 2

                        (tamanho_texto, _) = cv2.getTextSize(texto_id, fonte, escala, espessura)
                        largura_texto, altura_texto = tamanho_texto
                        x_texto = novo_x
                        y_texto = novo_y - 10

                        # Fundo do texto de ID
                        cv2.rectangle(frame_marcado,
                                    (x_texto, y_texto - altura_texto),
                                    (x_texto + largura_texto, y_texto + 5),
                                    (255, 0, 0), thickness=-1)

                        # Texto de ID
                        cv2.putText(frame_marcado, texto_id,
                                    (x_texto, y_texto),
                                    fonte, escala, (255, 255, 255), espessura)

                        # Exibir total de detecções no centro
                        texto_total = str(total_detections)
                        (w_texto, h_texto), _ = cv2.getTextSize(texto_total, fonte, 2.5, 4)
                        centro_texto_x = novo_x + lado_pad // 2 - w_texto // 2
                        centro_texto_y = novo_y + lado_pad // 2 + h_texto // 2

                        cv2.putText(frame_marcado, texto_total,
                                    (centro_texto_x, centro_texto_y),
                                    fonte, 2.5, (0, 255, 255), 4)

        # Exibe o quadro ao vivo
        if visualize == 1:
            cv2.imshow(f'Ao vivo: {device_name}', frame_marcado)

        key = cv2.waitKey(wait_key) & 0xFF
        if key:
            # Pressione 'ESC' para sair
            if key == 27:
                break

    device.release()
    cv2.destroyAllWindows()


def save_frame(frame_original, frame_detection, linha, id_image, data_hora_atual,
               total_detections, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    caminho_SM = os.path.join(save_dir, 'SM')
    os.makedirs(caminho_SM, exist_ok=True)
    frame_SM_path = os.path.join(caminho_SM, f'SM_{linha}_{id_image}_{data_hora_atual}.jpg')
    csv_SM_path = os.path.join(caminho_SM, f'SM_{linha}_{id_image}_{data_hora_atual}.csv')
    cv2.imwrite(frame_SM_path, frame_original)

    with open(csv_SM_path, mode='w', newline='') as csv_SM_file:
        writer = csv.writer(csv_SM_file)
        writer.writerow(['DataHora', 'Linha', 'TotalDeteccoes'])
        writer.writerow([data_hora_atual, linha, total_detections])

    caminho_CM = os.path.join(save_dir, 'CM')
    os.makedirs(caminho_CM, exist_ok=True)
    frame_CM_path = os.path.join(caminho_CM, f'CM_{linha}_{id_image}_{data_hora_atual}.jpg')
    csv_CM_path = os.path.join(caminho_CM, f'CM_{linha}_{id_image}_{data_hora_atual}.csv')
    cv2.imwrite(frame_CM_path, frame_detection)

    with open(csv_CM_path, mode='w', newline='') as csv_CM_file:
        writer = csv.writer(csv_CM_file)
        writer.writerow(['DataHora', 'Linha', 'TotalDeteccoes'])
        writer.writerow([data_hora_atual, linha, total_detections])

    print(f"Salvo em {frame_CM_path} e {frame_SM_path}")

