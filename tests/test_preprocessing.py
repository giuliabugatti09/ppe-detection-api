"""
Testes automatizados para app/services/preprocessing.py.

Cobrem tanto o caminho feliz (imagem válida) quanto os casos de erro
que já validamos manualmente nos primeiros dias do projeto — agora
formalizados como asserções, não mais checagem visual no terminal.
"""

import pytest

from app.services.preprocessing import bytes_to_image, InvalidImageError


def test_bytes_to_image_com_imagem_valida(sample_image_bytes):
    image = bytes_to_image(sample_image_bytes)

    # Verifica que o resultado é um array 3D (altura, largura, canais)
    # e que tem exatamente 3 canais de cor (RGB).
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_bytes_to_image_com_bytes_vazios():
    with pytest.raises(InvalidImageError, match="vazio"):
        bytes_to_image(b"")


def test_bytes_to_image_com_bytes_invalidos():
    with pytest.raises(InvalidImageError, match="decodificar"):
        bytes_to_image(b"isto nao e uma imagem valida")


def test_bytes_to_image_excede_tamanho_maximo():
    # Gera bytes maiores que o limite definido em preprocessing.py,
    # sem precisar criar um arquivo de imagem gigante de verdade.
    from app.services.preprocessing import MAX_FILE_SIZE_BYTES

    fake_large_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1)

    with pytest.raises(InvalidImageError, match="excede"):
        bytes_to_image(fake_large_bytes)