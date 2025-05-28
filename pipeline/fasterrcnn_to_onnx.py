import torch
import torchvision
import os
import io

def load_model(torch_device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')     # Carregar o modelo treinado
    num_classes = 2  # Inclua o número de classes (background + classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes))
    model.to(torch_device)   # Mover o modelo para o dispositivo

    model_path = os.path.join(
        'data', 'inputs', 'ia_models', 'FRCNN Resnet50', 'best_faster_rcnn_model_20241203.pth',
    )
    model.load_state_dict(torch.load(model_path, map_location=torch_device))   # Carregar o modelo salvo
    model.eval()  # Colocar o modelo em modo de avaliação

    return model

def export_to_onnx(model, torch_device, onnx_path):
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(torch_device)  # Tamanho de entrada esperado
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=12)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Carregar o modelo
    model = load_model(device)
    
    # Definir o caminho onde o modelo ONNX será salvo
    onnx_path = "faster_rcnn_model.onnx"
    
    # Exportar para o formato ONNX
    export_to_onnx(model, device, onnx_path)
    print(f"Modelo exportado para ONNX e salvo em: {onnx_path}")
