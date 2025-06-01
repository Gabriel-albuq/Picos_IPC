import cv2
import numpy as np

DESC1 = 300
DESC2 = 80

def trigger_test(frame, perc_top, perc_bottom):
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
    x1, y1 = (int((frame_width / 2) - DESC1), line_limit_top)  # Posição do canto superior esquerdo
    x2, y2 = (int((frame_width / 2) - DESC2), line_limit_top + 20)   # Posição do canto inferior direito

    cropped_image_top = frame.copy()[y1:y2, x1:x2]   # Cortando a região de interesse (ROI)
    mean_color_top = cv2.mean(cropped_image_top)[:3]  # Apenas os canais BGR

    ### BOTTOM - Verificando a média das cores INFERIORES ###
    x2, y2 = (int((frame_width / 2) - DESC2), line_limit_bottom,)  # Posição do canto inferior direito
    x1, y1 = (int((frame_width / 2) - DESC1), line_limit_bottom - 20)  # Posição do canto superior esquerdo

    cropped_image_bottom = frame.copy()[y1:y2, x1:x2]   # Cortando a região de interesse (ROI)
    mean_color_bottom = cv2.mean(cropped_image_bottom)[:3]   # Apenas os canais BGR

    # Definir os valores aproximados para preto/branco e marrom
    color_black_white = np.array([128, 128, 128])  # Cinza médio para preto/branco
    color_brown = np.array([42, 42, 165])  # Aproximação de marrom

    # Função para calcular proximidade
    def closest_color(mean_color, color1, color2):
        dist1 = np.linalg.norm(np.array(mean_color) - color1)  # Distância da cor 1

        dist2 = np.linalg.norm(np.array(mean_color) - color2)  # Distância da cor 2

        return 0 if dist1 < dist2 else 1

    # Determinar a cor mais próxima
    trigger_top_value = closest_color(mean_color_top, color_black_white, color_brown)
    trigger_bottom_value = closest_color(mean_color_bottom, color_black_white, color_brown)

    # Definir o resultado final: ambos precisam ser preto/branco ou ambos marrom
    if trigger_top_value == 1 and trigger_bottom_value == 1:
        trigger_result = True
    else:
        trigger_result = False

    return frame, trigger_result, trigger_top_value, trigger_bottom_value


def trigger_frame(frame, trigger_top_value, trigger_bottom_value, perc_top, perc_bottom):
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
    x1, y1 = (int((frame_width / 2) - DESC1), line_limit_top)  # Posição do canto superior esquerdo
    x2, y2 = (int((frame_width / 2) - DESC2), line_limit_top + 20)   # Posição do canto inferior direito
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    ### BOTTOM - Trigger
    x2, y2 = (int((frame_width / 2) - DESC2), line_limit_bottom,)  # Posição do canto inferior direito
    x1, y1 = (int((frame_width / 2) - DESC1), line_limit_bottom - 20)  # Posição do canto superior esquerdo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Copiar a imagem para aplicar as áreas escurecidas
    overlay = frame.copy()
    alpha = 0.5  # Transparência das áreas escurecidas (0 = sem escurecimento, 1 = totalmente escuro)

    # Escurecer a área acima de line_limit_top
    cv2.rectangle(overlay, (0, 0), (frame_width, line_limit_top), (0, 0, 0), -1)

    # Escurecer a área abaixo de line_limit_bottom
    cv2.rectangle(overlay, (0, line_limit_bottom), (frame_width, frame_height), (0, 0, 0), -1)

    # Combinar a imagem com o overlay escurecido
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Desenhar as linhas, a cor depende do trigger
    rectangle_width = 40  # Largura do retângulo
    rectangle_height = 20  # Altura do retângulo
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Superior
    text = str(int(trigger_top_value))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (rectangle_width - text_size[0]) // 2  # Centraliza o texto no retângulo
    text_y = (line_limit_top + text_size[1] // 2)     # Alinha o texto ao centro vertical do retângulo
    
    if trigger_top_value == 1:
        cv2.line(frame, (0, line_limit_top), (frame_width, line_limit_top), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, line_limit_top - rectangle_height // 2), (rectangle_width, line_limit_top + rectangle_height // 2), (0, 255, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    else:
        cv2.line(frame, (0, line_limit_top), (frame_width, line_limit_top), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, line_limit_top - rectangle_height // 2), (rectangle_width, line_limit_top + rectangle_height // 2), (0, 0, 255), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Inferior
    text = str(int(trigger_bottom_value))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (rectangle_width - text_size[0]) // 2  # Centraliza o texto no retângulo
    text_y = (line_limit_bottom + text_size[1] // 2)     # Alinha o texto ao centro vertical do retângulo
    
    if trigger_bottom_value == 1:
        cv2.line(frame, (0, line_limit_bottom), (frame_width, line_limit_bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, line_limit_bottom - rectangle_height // 2), (rectangle_width, line_limit_bottom + rectangle_height // 2), (0, 255, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    else:
        cv2.line(frame, (0, line_limit_bottom), (frame_width, line_limit_bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, line_limit_bottom - rectangle_height // 2), (rectangle_width, line_limit_bottom + rectangle_height // 2), (0, 0, 255), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Caso os dois triggers estejam ativados, a bola fica verde, caso não vermelho
    if trigger_top_value == 1 and trigger_bottom_value == 1:
        circle_color_trigger = (0, 255, 0)

    else:
        circle_color_trigger = (0, 0, 255)

    cv2.circle(frame, (frame.shape[1] - 30, 30),15, circle_color_trigger, -1)

    return frame