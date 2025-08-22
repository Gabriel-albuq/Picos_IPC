import os

import cv2
import pytest

from app.func.picos_trigger import trigger_test
from app.func.picos_interface import load_settings

# Caminho para a pasta uma acima do diretório atual
pasta = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'data',
    'inputs',
    'test_images',
)

# Listar arquivos .jpg com caminho completo
IMAGES = [
    os.path.join(pasta, f)
    for f in os.listdir(pasta)
    if f.lower().endswith('.jpg')
]

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



@pytest.mark.parametrize(
    'image_path', IMAGES
)  # Passando caminho completo da imagem
def test_trigger_test(image_path):
    frame = cv2.imread(image_path)  # Carregar a imagem

    # Verificar se a imagem foi carregada corretamente
    assert frame is not None, f'Erro ao carregar a imagem {image_path}'

    trigger_result, trigger_top_value, trigger_bottom_value = trigger_test(
        frame, perc_top, perc_bottom
    )

    assert (
        trigger_result is True
    ), f'Erro: O trigger foi ativado incorretamente na imagem {image_path} {trigger_result}'
