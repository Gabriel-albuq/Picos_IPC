import cv2
import torch
import torchvision.transforms as transforms
import gc


def run_model(torch_device, type_model, model, frame):
    if type_model in ['YOLO', 'RTDETR']:
        with torch.no_grad():
            results = model.predict(
                source=frame, 
                conf=0.1, # Manter 0.1, pois vamos filtrar o score depois             
                imgsz=1280,      
                device=torch_device,
                verbose=False
            )

            detections = results[0].boxes  # Obter as detecções

            # Extrair boxes e scores em um único loop
            detections_info = [(detection.xyxy[0].tolist(), float(detection.conf)) for detection in detections]
            
            boxes, scores = (zip(*detections_info) if detections_info else ([], []))  # Separar os boxes e scores em variáveis distintas

            detections = []  # Cria uma lista para armazenar as detecções

            # Itera sobre as caixas e pontuações para criar pares (box, score)
            for box, score in zip(boxes, scores):
                detections.append([box, score])  # Não é necessário usar box.tolist() aqui

            # Ordena as detecções pelo valor de x_min
            detections_sorted = sorted(detections, key=lambda det: det[0][0])  # Ordena pelo x_min

    if type_model in ['FRCNN_RN50', 'FRCNN_MNV3L', 'FRCNN_MNV3S']:
        # Pré-processar a imagem
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converter BGR para RGB
        transform = (transforms.ToTensor())  # Define a transformação para ToTensor
        image_tensor = (transform(image).unsqueeze(0).to(torch_device))  # Converte a imagem para tensor e adiciona uma dimensão de batch

        # Aplicar o modelo na imagem
        with torch.no_grad():  # Desativar cálculo de gradientes para economizar memória
            predictions = model(image_tensor)  # Fazer previsões

        boxes = (predictions[0]['boxes'].cpu().detach().numpy())   # Converte as previsões para arrays do numpy
        scores = (predictions[0]['scores'].cpu().detach().numpy())   # Converte as previsões para arrays do numpy

        detections = []   # Cria uma lista para armazenar as detecções

        # Itera sobre as caixas e pontuações para criar pares (box, score)
        for box, score in zip(boxes, scores):
            detections.append([box.tolist(), float(score)])

        # Obter as detecções
        detections_sorted = sorted(detections, key=lambda deteccao: deteccao[0][0])   # Ordena pelo x_min

        # for i, deteccao in enumerate(detections_sorted):
        #     score_da_deteccao = deteccao[1]
        #     print(f"Detecção {i + 1}: Score = {score_da_deteccao:.4f}") # Formata o score para 4 casas decimais
        # print("------------------------------------")

        # LIMPEZA CRÍTICA
        del predictions
        del image_tensor

    # --- LIMPEZA DE CACHE FINAL ---
    # Só rodamos se a GPU estiver sendo usada
    if "cuda" in str(torch_device) or str(torch_device) == "0":
        torch.cuda.empty_cache()
        gc.collect()

    return detections_sorted
