import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import copy
from datetime import datetime

# Configurações iniciais
base_dir = os.path.abspath(os.path.join(os.getcwd()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_tensor(image_path):
    """Carrega uma imagem como tensor"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)

def get_all_images(images_dir):
    """Obtém todas as imagens de um diretório"""
    annotations_path = os.path.join(images_dir, '_annotations.coco.json')
    
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    # Mapear anotações por imagem
    annotations_dict = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append(annotation)
    
    images_info = []
    for image_info in coco_data['images']:
        image_path = os.path.join(base_dir, images_dir, image_info['file_name'])
        image_id = image_info['id']
        annotations = annotations_dict.get(image_id, [])
        images_info.append((image_path, annotations, image_info))
    
    return images_info, coco_data

def group_close_boxes(boxes, distance_threshold):
    """Agrupa caixas próximas sem usar bibliotecas externas"""
    centers = [(x + w/2, y + h/2) for x, y, w, h in boxes]
    groups = []
    visited = set()
    
    for i in range(len(centers)):
        if i not in visited:
            group = [i]
            queue = [i]
            visited.add(i)
            
            while queue:
                current = queue.pop()
                for j in range(len(centers)):
                    if j not in visited:
                        dx = centers[current][0] - centers[j][0]
                        dy = centers[current][1] - centers[j][1]
                        distance = (dx**2 + dy**2)**0.5
                        
                        if distance <= distance_threshold:
                            visited.add(j)
                            group.append(j)
                            queue.append(j)
            
            if len(group) > 1:
                groups.append(group)
    
    return groups

def create_square_roi(min_x, min_y, max_x, max_y, img_width, img_height, padding):
    """Cria uma região quadrada com padding"""
    width = max_x - min_x
    height = max_y - min_y
    
    # Aplicar padding
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(img_width, max_x + padding)
    max_y = min(img_height, max_y + padding)
    
    # Tornar quadrado (usar o maior lado)
    new_width = max_x - min_x
    new_height = max_y - min_y
    
    if new_width > new_height:
        diff = new_width - new_height
        min_y = max(0, min_y - diff // 2)
        max_y = min(img_height, max_y + diff - diff // 2)
    else:
        diff = new_height - new_width
        min_x = max(0, min_x - diff // 2)
        max_x = min(img_width, max_x + diff - diff // 2)
    
    return int(min_x), int(min_y), int(max_x), int(max_y)

def process_group(image_np, boxes, annotations, group_indices, roi, output_size=(640, 640)):
    """Processa um grupo individual e retorna imagem e anotações redimensionadas"""
    # Cortar a região de interesse
    min_x, min_y, max_x, max_y = roi
    cropped = image_np[min_y:max_y, min_x:max_x]
    
    # Obter dimensões originais e novas
    original_height = max_y - min_y
    original_width = max_x - min_x
    new_width, new_height = output_size
    
    # Calcular fatores de escala
    width_scale = new_width / original_width
    height_scale = new_height / original_height
    
    # Redimensionar a imagem
    resized_img = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)
    
    # Processar anotações do grupo
    group_annotations = []
    for idx in group_indices:
        ann = copy.deepcopy(annotations[idx])
        x, y, w, h = ann['bbox']
        
        # Ajustar coordenadas para a ROI cortada
        x -= min_x
        y -= min_y
        
        # Redimensionar as anotações
        x = x * width_scale
        y = y * height_scale
        w = w * width_scale
        h = h * height_scale
        
        # Atualizar a bounding box
        ann['bbox'] = [x, y, w, h]
        group_annotations.append(ann)
    
    return resized_img, group_annotations, (width_scale, height_scale)

def draw_annotations_on_image(image, annotations):
    """Desenha anotações na imagem e retorna a imagem anotada"""
    annotated_image = image.copy()
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Desenhar retângulo
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Adicionar texto com ID
        cv2.putText(annotated_image, f"ID:{ann['id']}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return annotated_image

def create_merged_coco(output_dir, all_images_data):
    """Cria um arquivo COCO consolidado com todas as imagens e anotações"""
    # Estrutura básica do COCO
    merged_coco = {
        "info": {
            "year": "2025",
            "version": "2",
            "description": "Exported from roboflow.com",
            "contributor": "",
            "url": "https://public.roboflow.com/object-detection/undefined",
            "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        },
        "licenses": [{
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }],
        "categories": [
            {"id": 0, "name": "biscoito-Y6ed", "supercategory": "none"},
            {"id": 1, "name": "biscoito", "supercategory": "biscoito-Y6ed"}
        ],
        "images": [],
        "annotations": []
    }
    
    # Contadores para IDs únicos
    image_id_counter = 0
    annotation_id_counter = 0
    
    # Processar cada grupo de cada imagem
    for img_data in all_images_data:
        original_img_name = os.path.splitext(os.path.basename(img_data['original_path']))[0]
        
        for group_id, group_data in enumerate(img_data['groups'], 1):
            # Adicionar informação da imagem
            img_filename = f"{original_img_name}_g{group_id}.jpg"
            
            merged_coco["images"].append({
                "id": image_id_counter,
                "license": 1,
                "file_name": img_filename,
                "height": 640,
                "width": 640,
                "date_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "extra": {"name": f"{original_img_name}_g{group_id}.jpg"}
            })
            
            # Adicionar anotações
            for ann in group_data['annotations']:
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = annotation_id_counter
                new_ann["image_id"] = image_id_counter
                new_ann["category_id"] = 1  # Todos são biscoitos
                new_ann["area"] = new_ann["bbox"][2] * new_ann["bbox"][3]
                new_ann["segmentation"] = []
                new_ann["iscrowd"] = 0
                
                merged_coco["annotations"].append(new_ann)
                annotation_id_counter += 1
            
            image_id_counter += 1
    
    # Salvar arquivo COCO consolidado
    merged_path = os.path.join(output_dir, "_annotations.coco.json")
    with open(merged_path, 'w') as f:
        json.dump(merged_coco, f, indent=2)
    
    return merged_path

def process_all_images(images_dir, output_base_dir, 
                     distance_threshold=50, padding=10, 
                     output_size=(640, 640)):
    """Processa todas as imagens de um diretório"""
    # Obter todas as imagens e anotações
    images_info, original_coco = get_all_images(images_dir)
    
    # Criar diretórios de saída
    output_dir = os.path.join(output_base_dir, 'images')
    annotated_dir = os.path.join(output_base_dir, 'annotated_images')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    
    all_images_data = []
    
    # Processar cada imagem
    for img_idx, (image_path, annotations, image_info) in enumerate(images_info, 1):
        print(f"\nProcessando imagem {img_idx}/{len(images_info)}: {image_info['file_name']}")
        
        try:
            # Carregar imagem
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_tensor = load_image_tensor(image_path)
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            # Extrair bounding boxes
            boxes = [ann['bbox'] for ann in annotations]
            
            # Agrupar caixas próximas
            groups = group_close_boxes(boxes, distance_threshold)
            
            image_data = {
                "original_path": image_path,
                "groups": []
            }
            
            # Processar cada grupo
            for group_id, group_indices in enumerate(groups, 1):
                # Calcular ROI para o grupo
                group_boxes = [boxes[i] for i in group_indices]
                x_coords = [x for x, y, w, h in group_boxes]
                y_coords = [y for x, y, w, h in group_boxes]
                widths = [w for x, y, w, h in group_boxes]
                heights = [h for x, y, w, h in group_boxes]
                
                min_x = min(x_coords)
                min_y = min(y_coords)
                max_x = max(x + w for x, y, w, h in zip(x_coords, y_coords, widths, heights))
                max_y = max(y + h for x, y, w, h in zip(x_coords, y_coords, widths, heights))
                
                # Criar ROI quadrada
                roi = create_square_roi(min_x, min_y, max_x, max_y, 
                                      image_np.shape[1], image_np.shape[0], padding)
                
                # Processar grupo
                group_img, group_ann, _ = process_group(
                    image_np, boxes, annotations, group_indices, roi, output_size)
                
                # Salvar imagens
                img_filename = f"{image_name}_g{group_id}.jpg"
                img_path = os.path.join(output_dir, img_filename)
                cv2.imwrite(img_path, cv2.cvtColor(group_img, cv2.COLOR_RGB2BGR))
                
                # Salvar imagem anotada
                annotated_img = draw_annotations_on_image(group_img, group_ann)
                annotated_path = os.path.join(annotated_dir, f"{image_name}_g{group_id}_annotated.jpg")
                cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                
                # Armazenar dados para o COCO consolidado
                image_data["groups"].append({
                    "image_path": img_path,
                    "annotations": group_ann
                })
                
                print(f"  Grupo {group_id} processado e salvo")
            
            all_images_data.append(image_data)
            
        except Exception as e:
            print(f"Erro ao processar imagem {image_path}: {str(e)}")
            continue
    
    # Criar COCO consolidado
    if all_images_data:
        merged_path = create_merged_coco(output_base_dir, all_images_data)
        print(f"\nProcessamento concluído. Arquivo COCO consolidado salvo em: {merged_path}")
    else:
        print("\nNenhuma imagem foi processada com sucesso.")

# Exemplo de uso
if __name__ == "__main__":
    images_dir = 'data/inputs/train_images/COCO_2025.05/test'
    output_base_dir = os.path.join(base_dir, 'data', 'outputs', 'processed_groups')
    
    process_all_images(
        images_dir=images_dir,
        output_base_dir=output_base_dir,
        distance_threshold=60,
        padding=15,
        output_size=(640, 640)
    )