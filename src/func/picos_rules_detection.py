import cv2
import numpy as np


def rules_detection(frame, detections_sorted, perc_top, perc_bottom, perc_median, min_score, limit_center):
    height, width = frame.shape[:2]

    # Define as posições das linhas
    line_limit_top = int(height * perc_top)    # Só conta quando a mediana entrar nesse range
    line_limit_bottom = int(height * perc_bottom)   # Só conta quando a mediana entrar nesse range
    dif_limit = line_limit_bottom - line_limit_top

    line_top_median = None  # É calculada a mediana das detecções, e só são contados biscoitos que estão naquela mediana + - um valor de range
    line_bottom_median = None   # É calculada a mediana das detecções, e só são contados biscoitos que estão naquela mediana + - um valor de range

    # Lista para armazenar centros de detecções
    centers = []
    all_centers_y = ([])  # Lista para armazenar as coordenadas y dos centros detectados

    total_detections = 0  # Contador total de detecções

    ### CALCULO DA MEDIANA
    for idx, detection in enumerate(detections_sorted):
        score = detection[1]   # Pontuação de confiança
        x_min, y_min, x_max, y_max = detection[0]  # Coordenadas da caixa - y cresce de cima para baixo

        # Verificar se a pontuação é maior que o limite e se está entre as linhas de contagem
        if (score > min_score
            and y_max > line_limit_top
            and y_min < line_limit_bottom
        ):
            center_y = (y_min + y_max) // 2   # Calcular o centro da caixa de detecção
            all_centers_y.append(center_y)  # Adiciona y à lista de centros

    if all_centers_y:
        median_y = int(np.median(all_centers_y))  # Obtém a mediana
        line_bottom_median = int(median_y + int((height * perc_median) / 2))
        line_top_median = int(median_y - int((height * perc_median) / 2))
        
        cv2.line(
            frame, 
            (640, median_y), 
            (640 + 640, median_y), 
            (255, 0, 0), 
            2,
        )  # Desenhar a linha horizontal na moda

    ### MARCACAO
    if ( all_centers_y and 
        line_limit_bottom > median_y > line_limit_top
    ):   # Se tiver pelo menos uma marcação dentro dos limites e a mediana for dentro dos limites
        for idx, detection in enumerate(detections_sorted):
            score = detection[1]   # Pontuação de confiança
            x_min, y_min, x_max, y_max = detection[0]  # Coordenadas da caixa - y cresce de cima para baixo

            center_x = int((x_min + x_max) // 2)   # Calcular o centro da caixa de detecção
            center_y = int((y_min + y_max) // 2)

            test_score = score > min_score   # Verifica score da deteccao
            test_center = not any(np.linalg.norm(np.array([center_x, center_y]) - np.array(center))< limit_center for center in centers)   # Verifica se está próximo de algum centro
            test_center_x = not any(abs(center_x - center[0]) < limit_center for center in centers)   # Verifica se está próximo de algum x dos centros
            test_median = (y_max > line_top_median and y_min < line_bottom_median)   # Verifica se está na mediana +- range

            if test_score and test_center and test_median:
                total_detections += 1
                # print(f"Deteccao {str(total_detections)} ({idx + 1}): {str(score)} - OK (Score: {test_score} / Center: {test_center} / Center X: {test_center_x} / Median: {test_median})")

                centers.append((center_x, center_y))  # Adiciona o centro à lista

                cv2.circle(frame, (center_x, center_y), 7, (0, 0, 255), -1,)   # Desenhar uma bolinha (círculo) no centro
                cv2.circle(frame, (center_x, center_y), limit_center, (255, 0, 0), 1,)  # Círculo vermelho de limite
                cv2.putText(frame,str(total_detections),(center_x - 6, center_y + 3),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255, 255, 255),1,)   # Colocar o número da marcação dentro da bolinha

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (80, 43, 30), -1)

    cv2.line(frame, (0, line_top_median), (640 + 640, line_top_median), (255, 0, 0), 2)
    cv2.line(frame,(0, line_bottom_median), (640 + 640, line_bottom_median), (255, 0, 0), 2)

    text_position = (0, 0)
    text = f'Total de Biscoitos: {total_detections}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10  # margem esquerda
    text_y = int((40 + text_height) / 2)  # centralizado verticalmente
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return frame, total_detections

def no_rules_detection(frame, detections_sorted, perc_top, perc_bottom, min_score, limit_center):
    total_detections = 0  # Contador total de detecções

    for idx, detection in enumerate(detections_sorted):
        score = detection[1]
        x_min, y_min, x_max, y_max = detection[0]

        center_x = int((x_min + x_max) // 2)
        center_y = int((y_min + y_max) // 2)

        total_detections += 1

        # Marca o centro com um círculo vermelho
        cv2.circle(
            frame,
            (center_x, center_y),
            7,
            (0, 0, 255),
            -1,
        )

        # Círculo azul indicando o raio que antes era o limit_center
        cv2.circle(
            frame,
            (center_x, center_y),
            limit_center,
            (255, 0, 0),
            1,
        )

        # Número da detecção
        cv2.putText(
            frame,
            str(total_detections),
            (center_x - 6, center_y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

    # Área superior de fundo do texto
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 40),
        (80, 43, 30),
        -1,
    )

    # Texto de total
    text = f'Total de Biscoitos: {total_detections}'
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return frame, total_detections