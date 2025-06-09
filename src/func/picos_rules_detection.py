import cv2
import numpy as np


def rules_detection(frame, detections_sorted, perc_top, perc_bottom, perc_median, min_score, limit_center):
    def median_calculate(detections_sorted, min_score, height, perc_median):
        all_centers = ([])  # Lista para armazenar as coordenadas y dos centros detectados
        for idx, detection in enumerate(detections_sorted):
            score = detection[1]   # Pontuação de confiança
            x_min, y_min, x_max, y_max = detection[0]  # Coordenadas da caixa - y cresce de cima para baixo

            # Verificar se a pontuação é maior que o limite e se está entre as linhas de contagem
            if (score > min_score and y_max > line_limit_top and y_min < line_limit_bottom):
                center_y = (y_min + y_max) // 2   # Calcular o centro da caixa de detecção
                all_centers.append(center_y)  # Adiciona y à lista de centros

                if all_centers:
                    median_y = int(np.median(all_centers))  # Obtém a mediana
                    line_bottom_median = int(median_y + int((height * perc_median) / 2))
                    line_top_median = int(median_y - int((height * perc_median) / 2))
                    
                    cv2.line(
                        frame, 
                        (640, median_y), 
                        (640 + 640, median_y), 
                        (255, 0, 0), 
                        2,
                    )  # Desenhar a linha horizontal na moda

                return all_centers, median_y, line_top_median, line_bottom_median
        
        return None, None, None, None

    def filtrar_deteccoes(detections_sorted, centers, median_y, line_limit_top, line_limit_bottom, 
                      line_top_median, line_bottom_median, min_score, limit_center):
        """Filtra as detecções válidas com base em múltiplas condições."""
        valid_centers = []

        if centers and line_limit_bottom > median_y > line_limit_top:
            for idx, detection in enumerate(detections_sorted):
                score = detection[1]
                x_min, y_min, x_max, y_max = detection[0]

                center_x = int((x_min + x_max) // 2)
                center_y = int((y_min + y_max) // 2)
                area = (x_max - x_min) * (y_max - y_min)

                test_score = score > min_score
                test_center = not any(np.linalg.norm(np.array([center_x, center_y]) - np.array(center)) < limit_center for center in valid_centers)
                test_center_x = not any(abs(center_x - center[0]) < limit_center for center in valid_centers) # Redundante
                test_median = (y_max > line_top_median and y_min < line_bottom_median)
                test_area = area >= 2000  # Area precisa ser maior que x (excluir pedacos soltos)

                if test_score and test_median and test_center and test_area:
                        valid_centers.append((center_x, center_y))

        return valid_centers

    def marcar_deteccoes(frame, valid_detections, limit_center):
        """Desenha as detecções válidas no frame e atualiza o total."""
        total_detections = 0  # Contador total de detecções
        for center_x, center_y in valid_detections:
            total_detections += 1

            cv2.circle(frame, (center_x, center_y), limit_center - 1, (0, 0, 255), -1)
            cv2.circle(frame, (center_x, center_y), limit_center, (255, 0, 0), 1)
            cv2.putText(frame, str(total_detections), (center_x - 6, center_y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return total_detections

    height, width = frame.shape[:2]

    # Define as posições das linhas
    line_limit_top = int(height * perc_top)    # Só conta quando a mediana entrar nesse range
    line_limit_bottom = int(height * perc_bottom)   # Só conta quando a mediana entrar nesse range
    dif_limit = line_limit_bottom - line_limit_top

    line_top_median = None  # É calculada a mediana das detecções, e só são contados biscoitos que estão naquela mediana + - um valor de range
    line_bottom_median = None   # É calculada a mediana das detecções, e só são contados biscoitos que estão naquela mediana + - um valor de range

    ### CALCULO DA MEDIANA
    all_centers, median_y, line_top_median, line_bottom_median = median_calculate(detections_sorted, min_score, height, perc_median)

    if all_centers:
        ### FILTRAGEM
        valid_centers = filtrar_deteccoes(detections_sorted, all_centers, median_y, line_limit_top, line_limit_bottom, line_top_median, line_bottom_median, min_score, limit_center)

        ### MARCACAO
        total_detections = marcar_deteccoes(frame, valid_centers, limit_center)

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

    else:
        return frame, 0

    

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