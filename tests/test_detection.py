import os
import re

import cv2
import pytest
import torch

from app.func.picos_load_model import load_model
from app.func.picos_rules_detection import rules_detection
from app.func.picos_run_model import run_model
from app.func.picos_interface import load_settings

# Dispositivo a ser testado
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lista de modelos a serem testados
type_model = 'FRCNN_RN50'
model = load_model(type_model)

# Caminho para a pasta uma acima do diretório atual
pasta = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'data',
    'inputs',
    'test_images',
)

# Carregar configurações do teste
(
    exposure_value,
    perc_top,
    perc_bottom,
    min_score,
    limit_center,
    sec_run_model,
    wait_key,
    save_dir
) = load_settings('tests/config.txt')


def test_detection():
    # Listar arquivos .jpg com caminho completo
    list_images = [
        os.path.join(pasta, f)
        for f in os.listdir(pasta)
        if f.lower().endswith('.jpg')
    ]

    check_ok = 0
    chech_nao_ok = 0
    for image_path in list_images:
        frame = cv2.imread(image_path)  # Carregar a imagem

        # Coletar a quantidade
        match = re.search(r'Imagem(\d+)\s-\s(\d+)\.jpg', image_path)
        if match:
            image_qtd = int(match.group(2))  # Número após o " - "

        # Verificar se a imagem foi carregada corretamente
        assert (
            match is not None
        ), f'Erro ao coletar a quantidade de biscoitos em {image_path}'

        detections_sorted = run_model(torch_device, type_model, model, frame)
        frame_detect, total_detections = rules_detection(
            frame.copy(),
            detections_sorted,
            perc_top,
            perc_bottom,
            min_score,
            limit_center,
        )

        if total_detections == image_qtd:
            check_ok += 1
        else:
            chech_nao_ok += 1

    perc_acertos = check_ok / (check_ok + chech_nao_ok)

    assert (
        perc_acertos >= 0.9
    ), f'A porcentagem de acertos foi menor do que a estipulada para passar no teste (90%): {perc_acertos:.2f}'
