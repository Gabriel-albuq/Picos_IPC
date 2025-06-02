import cv2
import numpy as np

def is_brown_or_gray(mean_color):
    color_gray = np.array([128, 128, 128])  # Cinza médio para preto/branco
    color_brown = np.array([42, 42, 165])  # Aproximação de marrom

    dist1 = np.linalg.norm(np.array(mean_color) - color_gray)  # Distância da cor 1
    dist2 = np.linalg.norm(np.array(mean_color) - color_brown)  # Distância da cor 2

    return 0 if dist1 < dist2 else 1

def is_brown(mean_color):
    # Limites para marrom em HSV (ajuste conforme necessário)
    limite_inferior = np.array([10, 100, 20])
    limite_superior = np.array([20, 255, 200])

    # Converte a cor média de BGR para HSV
    bgr_pixel = np.uint8([[mean_color]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]

    # Verifica se cada componente está dentro dos limites
    dentro_dos_limites = np.all(hsv_pixel >= limite_inferior) and np.all(hsv_pixel <= limite_superior)

    return dentro_dos_limites

def calculate_mean_color(frame, frame_width, line_limit, deslocamento_esquerda, deslocamento_direita):
    x1, y1 = (int((frame_width / 2) - deslocamento_esquerda), line_limit)  # Posição do canto superior esquerdo
    x2, y2 = (int((frame_width / 2) - deslocamento_direita), line_limit + 20)   # Posição do canto inferior direito

    cropped_image = frame.copy()[y1:y2, x1:x2]   # Cortando a região de interesse (ROI)
    mean_color = cv2.mean(cropped_image)[:3]  # Apenas os canais BGR

    return mean_color

def frame_trigger_lines(frame, frame_width, trigger_value, line_limit, deslocamento_esquerda, deslocamento_direita):
    # Desenhar as linhas, a cor depende do trigger
    rectangle_width = 40  # Largura do retângulo
    rectangle_height = 20  # Altura do retângulo
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    x1, y1 = (int((frame_width / 2) - deslocamento_esquerda), line_limit)  # Posição do canto superior esquerdo
    x2, y2 = (int((frame_width / 2) - deslocamento_direita), line_limit + 20)   # Posição do canto inferior direito
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    text = str(int(trigger_value))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (rectangle_width - text_size[0]) // 2  # Centraliza o texto no retângulo
    text_y = (line_limit + text_size[1] // 2)     # Alinha o texto ao centro vertical do retângulo
    
    if trigger_value == 1:
        cv2.line(frame, (0, line_limit), (frame_width, line_limit), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, line_limit - rectangle_height // 2), (rectangle_width, line_limit + rectangle_height // 2), (0, 255, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    else:
        cv2.line(frame, (0, line_limit), (frame_width, line_limit), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, line_limit - rectangle_height // 2), (rectangle_width, line_limit + rectangle_height // 2), (0, 0, 255), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return frame

def write_cropped_boxes(frame, frame_width, line_limit_top, line_limit_bottom, box_size, box_distance, box_offset_x):
    # Cálculo do centro vertical da faixa de trigger
    center_y = int((line_limit_top + line_limit_bottom) // 2)

    # Coordenadas dos quadrados
    center_x = frame_width // 2 + box_offset_x
    left_x = center_x - (box_distance // 2)
    right_x = center_x + (box_distance // 2)

    # Desenhar os quadrados no frame_trigger
    cv2.rectangle(frame, (left_x - box_size//2, center_y - box_size//2),
              (left_x + box_size//2, center_y + box_size//2), (255, 0, 0), 2)

    cv2.rectangle(frame, (right_x - box_size//2, center_y - box_size//2),
              (right_x + box_size//2, center_y + box_size//2), (255, 0, 0), 2)

    return frame

def trigger_test(frame, perc_top, perc_bottom, deslocamento_esquerda, deslocamento_direita):
    """
    Analisa regiões específicas da imagem (topo e base) e determina se ambas
    apresentam coloração semelhante (preto/branco ou marrom), ativando um "gatilho".

    Parâmetros:
        frame (np.ndarray): Imagem (frame) em formato BGR.
        perc_top (float): Porcentagem da altura da imagem para definir a linha superior.
        perc_bottom (float): Porcentagem da altura da imagem para definir a linha inferior.

    Retorna:
        tuple: 
            - trigger_result (bool): True se ambas as regiões tiverem cor marrom, False caso contrário.
            - trigger_top_value (int): 1 se a região superior for marrom, 0 se for preto/branco.
            - trigger_bottom_value (int): 1 se a região inferior for marrom, 0 se for preto/branco.
    """
    frame_height, frame_width = frame.shape[:2]

    # Define as posições das linhas
    line_limit_top = int(frame_height * perc_top)
    line_limit_bottom = int(frame_height * perc_bottom)

    ### TOP - Verificando a média das cores SUPERIORES ###
    mean_color_top = calculate_mean_color(frame, frame_width, line_limit_top, deslocamento_esquerda, deslocamento_direita)

    ### BOTTOM - Verificando a média das cores INFERIORES ###
    mean_color_bottom = calculate_mean_color(frame, frame_width, line_limit_bottom, deslocamento_esquerda, deslocamento_direita)

    # Definir os valores aproximados para preto/branco e marrom
    color_black_white = np.array([128, 128, 128])  # Cinza médio para preto/branco
    color_brown = np.array([42, 42, 165])  # Aproximação de marrom

    # Determinar a cor mais próxima
    trigger_top_value = is_brown(mean_color_top)
    trigger_bottom_value = is_brown(mean_color_bottom)

    # Definir o resultado final: ambos precisam ser preto/branco ou ambos marrom
    if trigger_top_value == 1 and trigger_bottom_value == 1:
        trigger_result = True
    else:
        trigger_result = False

    return frame, trigger_result, trigger_top_value, trigger_bottom_value
    
def trigger_frame(frame, trigger_top_value, trigger_bottom_value, perc_top, perc_bottom, deslocamento_esquerda, deslocamento_direita, box_size, box_distance, box_offset_x):
    """
    Desenha linhas e elementos gráficos sobre a imagem com base no resultado do gatilho.

    Parâmetros:
        frame (np.ndarray): Imagem (frame) em formato BGR.
        trigger_top_value (int): Valor do trigger superior (1 para marrom, 0 para preto/branco).
        trigger_bottom_value (int): Valor do trigger inferior (1 para marrom, 0 para preto/branco).
        perc_top (float): Porcentagem da altura da imagem para definir a linha superior.
        perc_bottom (float): Porcentagem da altura da imagem para definir a linha inferior.

    Retorna:
        frame (np.ndarray): Imagem com sobreposição gráfica indicando o estado dos triggers.
    """
    frame_height, frame_width = frame.shape[:2]

    # Define as posições das linhas
    line_limit_top = int(frame_height * perc_top)
    line_limit_bottom = int(frame_height * perc_bottom)

    ### TOP - Trigger
    frame = frame_trigger_lines(frame, frame_width, trigger_top_value, line_limit_top, deslocamento_esquerda, deslocamento_direita)

    ### BOTTOM - Trigger
    frame = frame_trigger_lines(frame, frame_width, trigger_bottom_value, line_limit_bottom, deslocamento_esquerda, deslocamento_direita)

    # Caso os dois triggers estejam ativados, a bola fica verde, caso não vermelho
    if trigger_top_value == 1 and trigger_bottom_value == 1:
        circle_color_trigger = (0, 255, 0)

    else:
        circle_color_trigger = (0, 0, 255)

    cv2.circle(frame, (frame.shape[1] - 30, 30),15, circle_color_trigger, -1)

    ###### CROPPED
    frame = write_cropped_boxes(frame, frame_width, line_limit_top, line_limit_bottom, box_size, box_distance, box_offset_x)

    return frame