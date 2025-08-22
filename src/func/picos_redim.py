import cv2
#from screeninfo import get_monitors

def redim_frame(frame, screen_width = 1920, screen_height = 1080):

    # Obter as dimensões do primeiro monitor
    # monitor = get_monitors()[0]
    # screen_width = monitor.width
    # screen_height = monitor.height

    # Ajustar as dimensões para garantir que a janela não ocupe a tela inteira,
    # deixando espaço para a barra de tarefas ou outros elementos da interface.
    # Por exemplo, reduzindo a altura e largura em 10%
    max_width = int(screen_width * 0.9)
    max_height = int(screen_height * 0.9)

    # Obter as dimensões atuais do frame
    frame_height, frame_width = frame.shape[:2]

    # Calcular a proporção de redimensionamento
    # Se a imagem é maior que o espaço máximo, calcule o fator de escala
    if frame_width > max_width or frame_height > max_height:
        scale_width = max_width / frame_width
        scale_height = max_height / frame_height
        scale = min(scale_width, scale_height) # Usa a menor proporção para manter a imagem completa

        # Redimensionar o frame
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    else:
        # Se a imagem já couber, não precisa redimensionar
        frame_resized = frame.copy()

    return frame_resized