import cv2
import numpy as np

def rules_detection(frame, detections_sorted, perc_top, perc_bottom, perc_median, min_score, limit_center):
    def median_calculate(detections_sorted, min_score, height, perc_median):
        min_score = 0.8
        all_centers = []  # Lista para armazenar as coordenadas y dos centros detectados
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

    def trim_detections_score(valid_detections, no_valid_centers, score_borda):
        """
        Remove detecções das pontas até encontrar a primeira com score > threshold.
        Adiciona os centros das detecções descartadas em no_valid_centers.
        """
        if not valid_detections:
            return [], no_valid_centers

        total = len(valid_detections)
        
        # 1. Encontrar o primeiro índice válido da esquerda para a direita
        idx_inicio = total
        for i in range(total):
            if valid_detections[i][1] >= score_borda:
                idx_inicio = i
                break
            else:
                # Extrai o box e calcula o centro
                box = valid_detections[i][0]
                cx = int((box[0] + box[2]) // 2)
                cy = int((box[1] + box[3]) // 2)
                no_valid_centers.append((cx, cy))

        # 2. Encontrar o primeiro índice válido da direita para a esquerda
        idx_fim = -1
        for i in range(total - 1, idx_inicio - 1, -1):
            if valid_detections[i][1] >= score_borda:
                idx_fim = i
                break
            else:
                # Extrai o box e calcula o centro
                box = valid_detections[i][0]
                cx = int((box[0] + box[2]) // 2)
                cy = int((box[1] + box[3]) // 2)
                no_valid_centers.append((cx, cy))

        # 3. Se ninguém for válido
        if idx_fim == -1 or idx_inicio >= total:
            return [], no_valid_centers

        # 4. Retorna a lista cortada e os centros acumulados
        valid_detections_trimmed = valid_detections[idx_inicio : idx_fim + 1]
        
        return valid_detections_trimmed, no_valid_centers

    def check_nested_centers(valid_detections, valid_centers, no_valid_centers):
        """
        Se um centro estiver dentro do box de outro, só mantém ambos se ambos tiverem score > 0.8.
        Caso contrário, mantém apenas o de maior score.
        """
        indices_para_descartar = set()
        total = len(valid_detections)
        new_valid_detections = []
        new_valid_centers = []

        # 1. Pré-calcula centros para evitar reprocessamento
        centros_todos = []
        for det in valid_detections:
            x1, y1, x2, y2 = det[0]
            centros_todos.append((int((x1 + x2) // 2), int((y1 + y2) // 2)))

        # 2. Compara cada box com cada centro
        for i in range(total):
            if i in indices_para_descartar: continue
            
            box_a = valid_detections[i][0] # [x1, y1, x2, y2]
            score_a = valid_detections[i][1]

            for j in range(total):
                if i == j or j in indices_para_descartar: continue
                
                cx_b, cy_b = centros_todos[j]
                score_b = valid_detections[j][1]

                # 3. Verifica se o centro de B está dentro do box de A
                if (box_a[0] <= cx_b <= box_a[2]) and (box_a[1] <= cy_b <= box_a[3]):
                    
                    # 4. Aplica a Regra de Confiança 0.8
                    # Se algum deles for "fraco" (<= 0.8), precisamos eliminar um
                    if score_a <= 0.8 or score_b <= 0.8:
                        if score_a < score_b:
                            indices_para_descartar.add(i)
                            break # Box A foi descartado, para de testar ele
                        else:
                            indices_para_descartar.add(j)

        # 5. Distribuição final nas listas
        for idx, detection in enumerate(valid_detections):
            centro = centros_todos[idx]
            if idx in indices_para_descartar:
                no_valid_centers.append(centro)
            else:
                new_valid_detections.append(detection)
                new_valid_centers.append(centro)

        return new_valid_detections, new_valid_centers, no_valid_centers

    def check_duplicate_x(valid_detections, no_valid_centers, limit_center):
        """
        Verifica se há coincidência em X. 
        Filtra detecções similares e preenche listas de caixas e seus respectivos centros.
        """
        # Lista para marcar quais índices de valid_detections devem ser descartados
        indices_para_descartar = set()

        # --- FASE 1: Identificação de Duplicidades ---
        new_valid_detections = []
        new_valid_centers = []
        for idx, detection in enumerate(valid_detections):
            if idx in indices_para_descartar:
                continue

            score = detection[1]
            x_min, y_min, x_max, y_max = detection[0]
            center_x = int((x_min + x_max) // 2)
            width_detection = x_max - x_min

            for idx_test, detection_test in enumerate(valid_detections):
                if idx == idx_test or idx_test in indices_para_descartar:
                    continue

                score_test = detection_test[1]
                x_min_test, y_min_test, x_max_test, y_max_test = detection_test[0]
                center_x_test = int((x_min_test + x_max_test) // 2)
                width_detection_test = x_max_test - x_min_test

                # 1. Teste de proximidade horizontal (Coincidência em X)
                if abs(center_x - center_x_test) < limit_center:
                    # 2. Teste de similaridade (Menos de 2x de diferença)
                    
                    is_similar_size = not (width_detection >= 2 * width_detection_test or 
                                        width_detection_test >= 2 * width_detection)
                    
                    if is_similar_size:
                        # Decide pelo score qual índice vai para o descarte
                        if score < score_test:
                            indices_para_descartar.add(idx)
                            break 
                        else:
                            indices_para_descartar.add(idx_test)

        # --- FASE 2: Distribuição e Cálculo de Centros ---
        for idx, detection in enumerate(valid_detections):
            x_min, y_min, x_max, y_max = detection[0]
            
            # Calcula o centro final para cada detecção
            c_x = int((x_min + x_max) // 2)
            c_y = int((y_min + y_max) // 2)
            centro = (c_x, c_y)

            if idx in indices_para_descartar:
                # Adiciona detecção e centro às listas de NÃO VÁLIDOS
                no_valid_centers.append(centro)
            else:
                # Adiciona detecção e centro às listas de VÁLIDOS
                new_valid_detections.append(detection)
                new_valid_centers.append(centro)

        return new_valid_detections, new_valid_centers, no_valid_centers

    def check_color(frame, valid_detections, valid_centers, no_valid_centers):
        """
        Mantém apenas os boxes onde pelo menos 25% dos pixels são marrons.
        Move os centros dos reprovados para no_valid_centers.
        OBS: Apenas aplicado nos primeiro e no ultimo
        """

        if not valid_detections:
            return [], [], no_valid_centers

        total = len(valid_detections)
        
        # IMPORTANTE: Ajuste os limites para o marrom real aqui
        limite_inferior = np.array([5, 40, 20])  
        limite_superior = np.array([30, 255, 240])

        def passa_no_teste_cor(det):
            x_min, y_min, x_max, y_max = map(int, det[0])
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0: return False
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, limite_inferior, limite_superior)
            
            pixels_marrons = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            porcentagem = pixels_marrons / total_pixels
            return porcentagem >= 0.25

        # 1. Encontrar o primeiro índice válido (Esquerda -> Direita)
        idx_inicio = total
        for i in range(total):
            if passa_no_teste_cor(valid_detections[i]):
                idx_inicio = i
                break
            else:
                # Descarte: Calcula centro e move para no_valid
                box = valid_detections[i][0]
                no_valid_centers.append((int((box[0]+box[2])//2), int((box[1]+box[3])//2)))

        # 2. Encontrar o último índice válido (Direita -> Esquerda)
        idx_fim = -1
        for i in range(total - 1, idx_inicio - 1, -1):
            if passa_no_teste_cor(valid_detections[i]):
                idx_fim = i
                break
            else:
                # Descarte: Calcula centro e move para no_valid
                box = valid_detections[i][0]
                no_valid_centers.append((int((box[0]+box[2])//2), int((box[1]+box[3])//2)))

        # 3. Resultado Final
        if idx_fim == -1 or idx_inicio >= total:
            return [], [], no_valid_centers

        # Corta a lista para manter apenas o intervalo entre o primeiro e o último válidos
        new_valid_detections = valid_detections[idx_inicio : idx_fim + 1]
        
        # Calcula os centros dos que sobraram (os válidos)
        new_valid_centers = []
        for det in new_valid_detections:
            box = det[0]
            new_valid_centers.append((int((box[0]+box[2])//2), int((box[1]+box[3])//2)))

        return new_valid_detections, new_valid_centers, no_valid_centers

    def check_color_grey(frame, valid_detections, valid_centers, no_valid_centers, limit_center, sensitivity=3):
        """
        Filtra outliers de cor e saturação baseando-se apenas no círculo central.
        A filtragem ocorre APENAS nas extremidades da fila.
        """
        if not valid_detections:
            return [], [], no_valid_centers

        height, width = frame.shape[:2]
        detections_data = []

        # 1. Coleta de dados (Amostragem Circular Central)
        for det in valid_detections:
            x_min, y_min, x_max, y_max = map(int, det[0])
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

            # Área do círculo de amostragem
            r = int(limit_center)
            y1, y2 = max(0, cy - r), min(height, cy + r)
            x1, x2 = max(0, cx - r), min(width, cx + r)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue

            # Máscara circular para ignorar o que não é o miolo do biscoito
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx - x1, cy - y1), r, 255, -1)

            # Média BGR e Saturação apenas do círculo
            avg_bgr = cv2.mean(roi, mask=mask)[:3]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_saturacao = cv2.mean(hsv_roi, mask=mask)[1]

            detections_data.append({
                'det': det,
                'color': np.array(avg_bgr),
                'sat': avg_saturacao,
                'center': (cx, cy)
            })

        if not detections_data:
            return [], [], no_valid_centers

        # 2. Cálculos Estatísticos Globais do Frame
        all_colors = np.array([d['color'] for d in detections_data])
        global_mean_color = np.mean(all_colors, axis=0)
        distances = np.array([np.linalg.norm(d['color'] - global_mean_color) for d in detections_data])
        
        # Threshold de outlier
        std_dist = np.std(distances)
        threshold_dist = np.mean(distances) + (float(sensitivity) * std_dist) if std_dist > 0 else 999

        # 3. Definição de Biscoito "Bom"
        def eh_biscoito_bom(idx):
            # Passa se tiver saturação (não for cinza) E não for outlier de cor
            teste_sat = detections_data[idx]['sat'] >= 35
            teste_cor = distances[idx] <= threshold_dist
            return teste_sat and teste_cor

        total = len(detections_data)
        idx_inicio = total
        idx_fim = -1

        # 4. BUSCA DA ESQUERDA PARA A DIREITA (Apara a entrada)
        for i in range(total):
            if eh_biscoito_bom(i):
                idx_inicio = i
                break # Encontrou o primeiro bom, PARA de filtrar
            else:
                # É lixo na borda: manda para os centros inválidos
                no_valid_centers.append(detections_data[i]['center'])

        # 5. BUSCA DA DIREITA PARA A ESQUERDA (Apara a saída)
        if idx_inicio < total:
            for i in range(total - 1, idx_inicio - 1, -1):
                if eh_biscoito_bom(i):
                    idx_fim = i
                    break # Encontrou o primeiro bom (vindo da direita), PARA de filtrar
                else:
                    no_valid_centers.append(detections_data[i]['center'])

        # 6. MONTAGEM FINAL (Mantém tudo o que está entre as "âncoras" boas)
        new_valid_detections = []
        new_valid_centers = []

        if idx_fim != -1:
            # Pega o intervalo completo: do primeiro ao último válido
            # O que estiver no meio NÃO é testado, apenas incluído
            for i in range(idx_inicio, idx_fim + 1):
                new_valid_detections.append(detections_data[i]['det'])
                new_valid_centers.append(detections_data[i]['center'])

        return new_valid_detections, new_valid_centers, no_valid_centers

    def check_color_lab_b(frame, valid_detections, valid_centers, no_valid_centers, limit_center, sensitivity=3):
        """
        Filtra a fila de biscoitos apenas nas extremidades.
        Um biscoito interrompe o descarte se: (Cor Boa + Não Outlier) OU (Score > 0.8).
        """
        score_cinza = 0.8

        if not valid_detections:
            return [], [], no_valid_centers

        height, width = frame.shape[:2]
        detections_data = []

        # 1. Amostragem Circular no canal 'b' (LAB)
        for det in valid_detections:
            x_min, y_min, x_max, y_max = map(int, det[0])
            score_ia = det[1]
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

            r = int(limit_center)
            y1, y2 = max(0, cy - r), min(height, cy + r)
            x1, x2 = max(0, cx - r), min(width, cx + r)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue

            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx - x1, cy - y1), r, 255, -1)

            lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
            avg_b_value = cv2.mean(lab_roi, mask=mask)[2]

            detections_data.append({
                'det': det,
                'b_val': avg_b_value,
                'score': score_ia,
                'center': (cx, cy)
            })

        if not detections_data:
            return [], [], no_valid_centers

        # 2. Estatística do Frame para detectar Outliers
        all_b_values = np.array([d['b_val'] for d in detections_data])
        global_mean_b = np.mean(all_b_values)
        b_differences = np.abs(all_b_values - global_mean_b)
        
        std_diff = np.std(b_differences)
        # Threshold dinâmico de "estranheza"
        threshold_dist = np.mean(b_differences) + (float(sensitivity) * std_diff) if std_diff > 0 else 999

        # 3. Critério de Parada (O que define um biscoito como "Válido" para parar o descarte)
        def eh_biscoito_valido(idx):
            score = detections_data[idx]['score']
            # Teste de Cor: Amarelo presente (>140) e dentro da média (Threshold)
            passou_cor = (detections_data[idx]['b_val'] > 140) and (b_differences[idx] <= threshold_dist)
            
            # REGRA: Para o descarte se a cor for boa OU se a IA tiver muita certeza
            return passou_cor or (score > score_cinza)

        total = len(detections_data)
        idx_inicio = total
        idx_fim = -1

        # 4. BUSCA DA ESQUERDA (Apara até o primeiro válido)
        for i in range(total):
            if eh_biscoito_valido(i):
                idx_inicio = i
                break
            else:
                # Não é válido e está na ponta: descarta
                no_valid_centers.append(detections_data[i]['center'])

        # 5. BUSCA DA DIREITA (Apara do fim até o primeiro válido)
        if idx_inicio < total:
            for i in range(total - 1, idx_inicio - 1, -1):
                if eh_biscoito_valido(i):
                    idx_fim = i
                    break
                else:
                    # Não é válido e está na ponta: descarta
                    no_valid_centers.append(detections_data[i]['center'])

        # 6. MONTAGEM DO INTERVALO PRESERVADO
        new_valid_detections = []
        new_valid_centers = []
        if idx_fim != -1:
            # Mantém TUDO o que estiver entre o primeiro e o último válidos
            for i in range(idx_inicio, idx_fim + 1):
                new_valid_detections.append(detections_data[i]['det'])
                new_valid_centers.append(detections_data[i]['center'])

        return new_valid_detections, new_valid_centers, no_valid_centers

    def filtrar_deteccoes(detections_sorted, centers, median_y, line_limit_top, line_limit_bottom, 
                      line_top_median, line_bottom_median, min_score, limit_center):
        """Filtra as detecções válidas com base em múltiplas condições."""
        valid_detections = []
        valid_centers = []
        valid_detections_step1 = []
        valid_centers_step1 = []
        valid_detections_step2 = []
        valid_centers_step2 = []
        no_valid_centers = []

        zone_left = width * 0.20
        zone_right = width * 0.80

        score_borda = 0.5
        score_centro = 0.3

        if centers and line_limit_bottom > median_y > line_limit_top:
            ## Step 1 - Excluindo biscoitos fora da mediana
            for idx, detection in enumerate(detections_sorted):
                score = detection[1]
                x_min, y_min, x_max, y_max = detection[0]

                center_x = int((x_min + x_max) // 2)
                center_y = int((y_min + y_max) // 2)
                area = (x_max - x_min) * (y_max - y_min)

                test_median = (y_max > line_top_median and y_min < line_bottom_median)
                test_area = area >= 0  # Area precisa ser maior que x (excluir pedacos soltos)

                if test_median and test_area:
                    valid_detections_step1.append(detection)
                    valid_centers_step1.append((center_x, center_y))
                else:
                    no_valid_centers.append((center_x, center_y))

            ## Step 2 - Excluindo pelo score de forma dinamica
            valid_detections_trim, no_valid_centers =  trim_detections_score(valid_detections_step1, no_valid_centers, score_borda)
            total_detectados = len(valid_detections_trim)
            for idx, detection in enumerate(valid_detections_trim):
                score = detection[1]
                # print(score)
                x_min, y_min, x_max, y_max = detection[0]

                center_x = int((x_min + x_max) // 2)
                center_y = int((y_min + y_max) // 2)
                area = (x_max - x_min) * (y_max - y_min)

                min_score_din = min_score
                # if center_x < zone_left or center_x > zone_right:
                #         # Zonas Laterais (Exige muita confiança)
                #         min_score_din = 0.5
                if idx < 3 or idx >= (total_detectados - 3):
                    # Extremidades da esteira (Exige maior confiança)
                    min_score_din = score_borda
                else:
                    # Zona Central (Mais permissivo)
                    min_score_din = score_centro

                test_score = score > min_score_din

                if test_score:
                    valid_detections_step2.append(detection)
                    valid_centers_step2.append((center_x, center_y))
                else:
                    no_valid_centers.append((center_x, center_y))

            ## Step 3 - Eliminando duplicidades (centros sobrepostos)
            for idx, detection in enumerate(valid_detections_step2):
                # Pega o centro correspondente que já foi calculado no Step 2
                curr_center = np.array(valid_centers_step2[idx]) 

                # IMPORTANTE: Verificamos contra os centros que já ADICIONAMOS na lista final
                # Se a lista valid_centers estiver vazia, any() retorna False, e test_center vira True
                test_center = not any(
                    np.linalg.norm(curr_center - np.array(prev_center)) < limit_center 
                    for prev_center in valid_centers
                )

                if test_center:
                    valid_detections.append(detection)
                    valid_centers.append(valid_centers_step2[idx])
                else:
                    no_valid_centers.append(valid_centers_step2[idx])


        # valid_detections, valid_centers, no_valid_centers = check_color(frame, valid_detections, valid_centers, no_valid_centers)
        valid_detections, valid_centers, no_valid_centers = check_color_lab_b(frame, valid_detections, valid_centers, no_valid_centers, limit_center)
        valid_detections, valid_centers, no_valid_centers = check_duplicate_x(valid_detections, no_valid_centers, limit_center)
        valid_detections, valid_centers, no_valid_centers = check_nested_centers(valid_detections, valid_centers, no_valid_centers)

        for idx, detection in enumerate(valid_detections):
            score = detection[1]
            x_min, y_min, x_max, y_max = detection[0]

            center_x = int((x_min + x_max) // 2)
            center_y = int((y_min + y_max) // 2)
            area = (x_max - x_min) * (y_max - y_min)

        return valid_detections, valid_centers, no_valid_centers
    
    def marcar_deteccoes(frame, valid_detections, limit_center, cor=(0, 0, 255)):
        """Desenha as detecções válidas (Box e Centro) no frame e atualiza o total."""
        total_detections = 0 
        
        for detection in valid_detections:
            # Extrair dados da detecção
            # d[0] são as coordenadas [x_min, y_min, x_max, y_max]
            coords = detection[0]
            x_min, y_min, x_max, y_max = map(int, coords)
            
            # Calcular centro para o desenho do círculo e texto
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            total_detections += 1

            # 1. Desenhar a Bounding Box (Retângulo)
            # Usamos uma espessura fina (1 ou 2) para não poluir a imagem
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), cor, 2)

            # 2. Desenhar o Centro (Círculo preenchido e borda)
            cv2.circle(frame, (center_x, center_y), limit_center - 1, cor, -1)
            cv2.circle(frame, (center_x, center_y), limit_center, (255, 0, 0), 1)

            # 3. Desenhar o número do índice (ID visual)
            cv2.putText(frame, str(total_detections), (center_x - 6, center_y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Opcional: Mostrar o Score da detecção acima da box
            # score = detection[1]
            # cv2.putText(frame, f'{score:.2f}', (x_min, y_min - 5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, cor, 1)

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

    if all_centers and median_y is not None and line_limit_bottom > median_y > line_limit_top:
        ### FILTRAGEM
        valid_detections, valid_centers, no_valid_centers = filtrar_deteccoes(detections_sorted, all_centers, median_y, line_limit_top, line_limit_bottom, line_top_median, line_bottom_median, min_score, limit_center)
        ### MARCACAO
        total_detections = marcar_deteccoes(frame, valid_detections, limit_center)
        # _ = marcar_deteccoes(frame, no_valid_centers, limit_center, cor = (0, 0, 0))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (80, 43, 30), -1)

        cv2.line(frame, (0, line_top_median), (frame.shape[1], line_top_median), (255, 0, 0), 2)
        cv2.line(frame,(0, line_bottom_median), (frame.shape[1], line_bottom_median), (255, 0, 0), 2)

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

def rules_detection_antiga(frame, detections_sorted, perc_top, perc_bottom, perc_median, min_score, limit_center):
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
        no_valid_centers = []

        if centers and line_limit_bottom > median_y > line_limit_top:
            for idx, detection in enumerate(detections_sorted):
                score = detection[1]
                x_min, y_min, x_max, y_max = detection[0]

                center_x = int((x_min + x_max) // 2)
                center_y = int((y_min + y_max) // 2)
                area = (x_max - x_min) * (y_max - y_min)

                test_score = score > min_score
                test_center = not any(np.linalg.norm(np.array([center_x, center_y]) - np.array(center)) < limit_center for center in valid_centers)
                test_median = (y_max > line_top_median and y_min < line_bottom_median)
                test_area = area >= 0  # Area precisa ser maior que x (excluir pedacos soltos)

                if test_score and test_median and test_center and test_area:
                        valid_centers.append((center_x, center_y))

                else:
                    no_valid_centers.append((center_x, center_y))

        return valid_centers, no_valid_centers
    
    def marcar_deteccoes(frame, valid_detections, limit_center, cor = (0, 0, 255)):
        """Desenha as detecções válidas no frame e atualiza o total."""
        total_detections = 0  # Contador total de detecções
        for center_x, center_y in valid_detections:
            total_detections += 1

            cv2.circle(frame, (center_x, center_y), limit_center - 1, cor, -1)
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

    if all_centers and median_y is not None and line_limit_bottom > median_y > line_limit_top:
        ### FILTRAGEM
        valid_centers, no_valid_centers = filtrar_deteccoes(detections_sorted, all_centers, median_y, line_limit_top, line_limit_bottom, line_top_median, line_bottom_median, min_score, limit_center)
        ### MARCACAO
        total_detections = marcar_deteccoes(frame, valid_centers, limit_center)
        # _ = marcar_deteccoes(frame, no_valid_centers, limit_center, cor = (0, 0, 0))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (80, 43, 30), -1)

        cv2.line(frame, (0, line_top_median), (frame.shape[1], line_top_median), (255, 0, 0), 2)
        cv2.line(frame,(0, line_bottom_median), (frame.shape[1], line_bottom_median), (255, 0, 0), 2)

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

def no_rules_detection(frame, detections_sorted, perc_top, perc_bottom, perc_median, min_score, limit_center):
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