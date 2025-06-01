import os
import sys

import pytest
import torch
import torchvision

from app.func.picos_load_model import load_model

# Dispositivo a ser testado
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lista de modelos a serem testados
MODELS = ['YOLO', 'FRCNN_RN50', 'FRCNN_MNV3L', 'FRCNN_MNV3S']


@pytest.mark.parametrize(
    'type_model', MODELS
)   # Permite rodar o mesmo teste para diferentes tipos de modelos.
def test_load_model(type_model):
    """Testa se o modelo carrega corretamente sem erros"""
    model = load_model(type_model)

    assert (
        model is not None
    ), f'Erro: modelo {type_model} não foi carregado corretamente'
    assert isinstance(
        model, torch.nn.Module
    ), f'Erro: {type_model} não é uma instância de torch.nn.Module'
    assert (
        next(model.parameters()).device == torch_device
    ), f'Erro: {type_model} não está no dispositivo correto'
