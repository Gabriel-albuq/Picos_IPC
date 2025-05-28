import cv2
from datetime import datetime
import numpy as np
import os

from src.class_camera_gx import GxiCapture

# Nome do diretório de saída dos vídeos
output_dir = os.path.join("data", "videos")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Codec de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # Inicialmente, sem gravar

# Capturar o vídeo da câmera
cap = GxiCapture(1)  # Para câmera Gx
gravando = False  # Controle de gravação
frame_size = None  # Para armazenar o tamanho do vídeo

while True:
    ret, frame = cap.read()

    if ret == False:
        cv2.putText(frame, "Sem imagem", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if frame_size is None:
        frame_size = (frame.shape[1], frame.shape[0])  # Definir o tamanho da gravação baseado no primeiro frame

    if gravando and out is not None:
        out.write(frame)
        cv2.putText(frame, "Gravando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir status na tela
    status_text = "Gravando... Pressione 'P' para parar" if gravando else "Pressione 'G' para gravar"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not gravando else (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir imagem capturada
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressionar ESC para sair
        print("Programa finalizado.")
        break

    if key == ord('g') and not gravando:  # Iniciar gravação com 'G'
        output_file = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.mp4')
        print("Gravação iniciada:", output_file)

        gravando = True
        out = cv2.VideoWriter(output_file, fourcc, 20.0, frame_size)  # Mantém a resolução original

    if key == ord('p') and gravando:  # Parar gravação com 'P'
        print("Gravação parada.")
        gravando = False
        out.release()
        out = None

# Liberar recursos
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
