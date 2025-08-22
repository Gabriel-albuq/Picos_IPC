import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np
from matplotlib.widgets import Button
import shutil
from tqdm import tqdm

base_dir = os.path.abspath(os.path.join(os.getcwd()))
print(base_dir)

sys.path.append(base_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Definir o dispositivo (GPU ou CPU)

class CustomTransform: # Transformação das imagens
    def __call__(self, img, target):
        if isinstance(img, torch.Tensor):
            return img.to(device), target  # Verifica se a imagem é tensor, caso seja já envia para o dispositivo

        img = transforms.ToTensor()(img)  # Caso não seja Tensor, transforma em tensor
        return img.to(device), target  # Envia a imagem para o dispositivo
    
class CustomDataset(CocoDetection): # Configuração do dataset personalizado
    def __init__(self, root, annotation, transforms=None):
        super().__init__(root, annotation)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if self.transforms:
            img, target = self.transforms(img, target)  # Aplica a transformação na imagem e no target

        return img, target

def load_settings():
    """Função para ler as configurações de um arquivo .txt e atribuir os valores diretamente às variáveis."""
    # Valores padrão
    defaults = {
        'perc_top': 0.4,
        'perc_bottom': 0.8,
        'min_score': 0.5,
        'limit_center': 8,
        'save_dir': 'data\\outputs\\capturas',
        'square_size': 640,
        'grid_x': 0,
        'grid_y': 0,
        'crop_image': 1  # 1 = Sim
    }

    return (
        defaults['perc_top'],
        defaults['perc_bottom'],
        defaults['min_score'],
        defaults['limit_center'],
        defaults['save_dir'],
        defaults['square_size'],
        defaults['grid_x'],
        defaults['grid_y'],
        defaults['crop_image'],
    )

def create_model(weights, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def load_pretrained_model(device, model, pretrained_model_path):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Modelo carregado de: {pretrained_model_path}")

    else:
        print("Treinando modelo do zero...")

    model.to(device)
    return model

def load_model_eval(device, model, pretrained_model_path):  # Colocar o modelo em modo de avaliação
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.to(device)
    model.eval() 
    
    return model

def load_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")  # Carrega a imagem e converte para RGB
    transform = transforms.ToTensor()  # Define a transformação para ToTensor
    image_tensor = transform(image).unsqueeze(0)  # Converte a imagem para tensor e adiciona uma dimensão de batch
    
    return image_tensor

def rules_detection(frame, detections_sorted, perc_top, perc_bottom, min_score, limit_center):
    height, width = frame.shape[:2]

    # Define as posições das linhas
    line_limit_top = int(height * perc_top)    # Só conta quando a mediana entrar nesse range
    line_limit_bottom = int(height * perc_bottom)   # Só conta quando a mediana entrar nesse range
    dif_limit = line_limit_bottom - line_limit_top

    # Linha de limite superior
    cv2.line(
        frame, 
        (0, line_limit_top), 
        (frame.shape[1], line_limit_top), 
        (0, 255, 0),  # Verde
        2,
    )

    # Linha de limite inferior
    cv2.line(
        frame, 
        (0, line_limit_bottom), 
        (frame.shape[1], line_limit_bottom), 
        (0, 255, 0),  # Verde
        2,
    )

    line_top_detect = None  # É calculada a mediana das detecções, e só são contados biscoitos que estão naquela mediana + - um valor de range
    line_bottom_detect = None   # É calculada a mediana das detecções, e só são contados biscoitos que estão naquela mediana + - um valor de range

    # Lista para armazenar centros de detecções
    centers = []
    all_centers_y = ([])  # Lista para armazenar as coordenadas y dos centros detectados

    total_detections = 0  # Contador total de detecções
    detections_info = []  # Lista para armazenar informações das detecções aceitas

    ### CALCULO DA MEDIANA
    for idx, detection in enumerate(detections_sorted):
        score = detection[1]   # Pontuação de confiança
        x_min, y_min, x_max, y_max = detection[0]  # Coordenadas da caixa - y cresce de cima para baixo

        # Verificar se a pontuação é maior que o limite e se está entre as linhas de contagem
        if (score > min_score and y_max > line_limit_top and y_min < line_limit_bottom):
            center_y = (y_min + y_max) // 2   # Calcular o centro da caixa de detecção
            all_centers_y.append(center_y)  # Adiciona y à lista de centros

    if all_centers_y:
        median_y = int(np.median(all_centers_y))  # Obtém a mediana
        line_bottom_detect = int(median_y + int(dif_limit / 2))
        line_top_detect = int(median_y - int(dif_limit / 2))
        
        cv2.line(
            frame, 
            (640, median_y), 
            (640 + 640, median_y), 
            (255, 0, 0), 
            2,
        )  # Desenhar a linha horizontal na moda

    ### MARCACAO
    if ( all_centers_y and line_limit_bottom > median_y > line_limit_top):   # Se tiver pelo menos uma marcação dentro dos limites e a mediana for dentro dos limites
        for idx, detection in enumerate(detections_sorted):
            score = detection[1]   # Pontuação de confiança
            x_min, y_min, x_max, y_max = detection[0]  # Coordenadas da caixa - y cresce de cima para baixo

            center_x = int((x_min + x_max) // 2)   # Calcular o centro da caixa de detecção
            center_y = int((y_min + y_max) // 2)

            test_score = score > min_score   # Verifica score da deteccao
            test_center = not any(np.linalg.norm(np.array([center_x, center_y]) - np.array(center))< limit_center for center in centers)   # Verifica se está próximo de algum centro
            test_center_x = not any(abs(center_x - center[0]) < limit_center for center in centers)   # Verifica se está próximo de algum x dos centros
            test_median = (y_max > line_top_detect and y_min < line_bottom_detect)   # Verifica se está na mediana +- range

            if test_score and test_center and test_median:
                total_detections += 1
                
                # Adiciona informações da detecção
                detections_info.append({
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],  # COCO format: [x,y,width,height]
                    'score': float(score),
                    'center': [center_x, center_y]
                })

                centers.append((center_x, center_y))  # Adiciona o centro à lista

                cv2.circle(
                    frame, 
                    (center_x, center_y), 
                    limit_center,
                    (0, 0, 255), 
                    -1,
                )   # Desenhar uma bolinha (círculo) no centro

                cv2.circle(
                    frame, 
                    (center_x, center_y), 
                    limit_center, 
                    (255, 0, 0), 
                    1,
                )  # Círculo vermelho de limite

    cv2.line(
        frame, 
        (640, line_top_detect), 
        (640 + 640, line_top_detect), 
        (255, 0, 0), 
        2,
    )
    cv2.line(
        frame,
        (640,
         line_bottom_detect),
         (640 + 640, line_bottom_detect),
         (255, 0, 0),
         2,
    )

    return frame, total_detections, detections_info

def process_images_in_folder(folder_path, model, output_base_dir, threshold=0.5, limit_center=6, perc_top=0.35, perc_bottom=0.6):
    # Criar pastas de saída
    approved_dir = os.path.join(output_base_dir, 'aprovados')
    rejected_dir = os.path.join(output_base_dir, 'recusados')
    os.makedirs(approved_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    
    # Lista para armazenar informações COCO
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}]
    }
    
    annotation_id = 1
    
    # Listar todas as imagens na pasta
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processando imagens"):
        image_path = os.path.join(folder_path, image_file)
        
        # Carregar imagem
        image_tensor = load_image_tensor(image_path)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_detected = (image_np * 255).astype(np.uint8).copy()
        frame_bgr = cv2.cvtColor(image_detected, cv2.COLOR_RGB2BGR)
        
        # Obter detecções
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            predictions = model(image_tensor)
        
        prediction = predictions[0]
        boxes_pred = prediction['boxes'].cpu().detach().numpy()
        scores = prediction['scores'].cpu().detach().numpy()
        
        # Preparar detecções
        detections_sorted = [
            (box, score) for box, score in zip(boxes_pred, scores) if score > threshold
        ]
        
        # Aplicar regras
        frame_bgr, total_detections, detections_info = rules_detection(
            frame=frame_bgr,
            detections_sorted=detections_sorted,
            perc_top=perc_top,
            perc_bottom=perc_bottom,
            min_score=threshold,
            limit_center=limit_center,
        )
        
        # Mostrar imagem para revisão
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f"Imagem: {image_file} - Detecções: {total_detections}\nPressione 'A' para aceitar ou 'R' para recusar")
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        
        # Esperar input do usuário
        while True:
            key = input("Pressione 'A' para ACEITAR ou 'R' para RECUSAR: ").upper()
            if key in ['A', 'R']:
                break
            print("Input inválido. Use 'A' para aceitar ou 'R' para recusar")
        
        plt.close()
        
        # Processar decisão
        if key == 'A':  # Aceitar
            dest_dir = approved_dir
            
            # Adicionar informações ao dataset COCO
            image_id = len(coco_data["images"]) + 1
            height, width = frame_bgr.shape[:2]
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })
            
            for detection in detections_info:
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": detection['bbox'],
                    "area": detection['bbox'][2] * detection['bbox'][3],
                    "iscrowd": 0,
                    "score": detection['score']
                })
                annotation_id += 1
                
        else:  # Recusar
            dest_dir = rejected_dir
            
        # Copiar imagem para a pasta correspondente
        shutil.copy(image_path, os.path.join(dest_dir, image_file))
    
    # Salvar anotações COCO para imagens aprovadas
    if coco_data["images"]:
        coco_file = os.path.join(approved_dir, 'annotations_coco.json')
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"Anotações COCO salvas em: {coco_file}")

if __name__ == "__main__":
    # Configurações iniciais
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo
    weights = 'FasterRCNN_ResNet50_FPN_Weights.COCO_V1'
    num_classes = 2
    model = create_model(weights, num_classes)
    pretrained_model_path = os.path.join(base_dir, 'data', 'inputs', 'ia_models', 'FRCNN Resnet50', 'best_faster_rcnn_model_20250326_154439.pth')
    model = load_pretrained_model(device, model, pretrained_model_path)
    model = load_model_eval(device, model, pretrained_model_path)
    
    # Pasta com imagens para processar
    input_folder = r'C:\Projetos Python\PICOS\data\outputs\cropped_generate'
    output_base_dir = r'C:\Projetos Python\PICOS\data\outputs\reviewed_images'
    
    # Processar todas as imagens na pasta
    process_images_in_folder(
        folder_path=input_folder,
        model=model,
        output_base_dir=output_base_dir,
        threshold=0.1,
        limit_center=12,
        perc_top=0.1,
        perc_bottom=0.9
    )