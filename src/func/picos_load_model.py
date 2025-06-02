import os

import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
#rom ultralytics import YOLO

# Verificar se a GPU está disponível e configurar o dispositivo
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")  # Mantenha em CPU se não houver GPU disponível
print(f'\nDispositivo utilizado: {torch_device}')

current_directory = os.path.dirname(os.path.abspath(__file__))   # Diretório atual
parent_directory = os.path.dirname(os.path.dirname(current_directory))  # Dois diretórios acima

def load_model(type_model):
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
            parent_directory,
            'data',
            'inputs',
            'ia_models',
            'FRCNN Resnet50',
            'best_faster_rcnn_model_20250520_171637.pth'
        )
        model.load_state_dict(torch.load(model_path, map_location=torch_device))   # Carregar o modelo salvo, mapeando para o dispositivo correto
        model.eval()  # Colocar o modelo em modo de avaliação

    if type_model == 'FRCNN_MNV3L':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2))
        model.to(torch_device)

        model_path = os.path.join(
            parent_directory,
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
            parent_directory,
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


if __name__ == '__main__':
    type_model = 'YOLO'
    model = load_model(type_model)
    print(model)