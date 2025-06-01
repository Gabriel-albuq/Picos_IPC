import onnx
import onnx_tensorrt.backend as backend

# Carregar o modelo ONNX
onnx_model = onnx.load("faster_rcnn_model.onnx")

# Converter para o formato TensorRT
engine = backend.prepare(onnx_model, device='CUDA:0')

# Salvar o modelo TensorRT
with open("faster_rcnn_model.trt", "wb") as f:
    f.write(engine.serialize())
    
print("Modelo convertido para TensorRT e salvo como 'faster_rcnn_model.trt'")
