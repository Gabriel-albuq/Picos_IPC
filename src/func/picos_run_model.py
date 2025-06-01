import cv2
import torch
import torchvision.transforms as transforms


def run_model(torch_device, type_model, model, frame):
    if type_model == 'YOLO':
        results = model(frame)  # Realizar a detecção com o YOLO, sem marcações antes, imagem pura
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

    return detections_sorted
