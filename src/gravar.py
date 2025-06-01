import cv2
from datetime import datetime
import os

from src.class_camera_gx import GxiCapture

# Nome do arquivo de vídeo de saída
output_dir = os.path.join("data", "videos")
output_file = os.path.join(output_dir, "gravacao.mp4")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Definir o codec de vídeo e criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # Inicialmente, sem gravar

# Capturar o vídeo da câmera
cap = cv2.VideoCapture(0) # Para webcam
#cap = GxiCapture(0) # Para camera Gx
gravando = False  # Controle de gravação

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Largura: {width}, Altura: {height}")

while True:
    # Ler o quadro da câmera
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    cv2.imshow('Camera', frame)

    if gravando and out is not None:     # Se estiver gravando, escrever o quadro no arquivo de saída
        out.write(frame)
        cv2.putText(frame, "Gravando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if gravando: # Exibir o status na tela
        cv2.putText(frame, "Gravando... Pressione 'P' para parar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Pressione 'G' para gravar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Camera', frame) # Mostrar o frame com as instruções

   
    key = cv2.waitKey(1) & 0xFF  # Verificar as teclas pressionadas
    
    
    if key == 27: # Parar se a tecla ESC for pressionada
        print("Programa finalizado.")
        break

    
    if key == ord('g') and not gravando: # Iniciar a gravação quando a tecla 'G' for pressionada
        output_file = os.path.join("videos", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.mp4')
        print("Gravação iniciada:", output_file)
        
        gravando = True
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))
    
    if key == ord('p') and gravando: # Parar a gravação quando a tecla 'P' for pressionada
        print("Gravação parada.")
        gravando = False
        out.release()
        out = None

# Liberar os objetos e fechar as janelas
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
