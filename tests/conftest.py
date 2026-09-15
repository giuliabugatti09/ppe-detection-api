"""
Fixtures compartilhadas entre os arquivos de teste.

O pytest descobre este arquivo automaticamente (não precisa importar
manualmente) e disponibiliza as fixtures aqui definidas para qualquer
teste na mesma pasta ou subpastas.
"""

import pytest

SAMPLE_IMAGE_PATH = "tests/test_images/sample.jpg"


@pytest.fixture
def sample_image_bytes():
    """Bytes crus de uma imagem de teste real, simulando um upload."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()