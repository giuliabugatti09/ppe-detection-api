"""
Módulo responsável por transformar bytes crus (vindos de upload HTTP)
em um array numpy válido, pronto para ser passado ao modelo.

Esta camada NÃO sabe nada sobre o modelo YOLO nem sobre FastAPI —
sua única responsabilidade é: bytes entram, array numpy válido sai
(ou uma exceção clara é levantada).
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Tamanho máximo de arquivo aceito, em bytes (5 MB)
# Protege a API contra uploads absurdamente grandes que
# poderiam travar o processo ou ser usados para abuso (DoS simples).
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


class InvalidImageError(Exception):
    """Levantada quando os bytes recebidos não formam uma imagem válida."""
    pass


def bytes_to_image(file_bytes: bytes) -> np.ndarray:
    """
    Decodifica bytes crus de imagem (ex: vindos de UploadFile do FastAPI)
    em um array numpy no formato RGB, pronto para inferência.

    Args:
        file_bytes: conteúdo binário bruto do arquivo enviado

    Returns:
        Array numpy (H, W, 3) em formato RGB

    Raises:
        InvalidImageError: se os bytes estiverem vazios, corrompidos,
            excederem o tamanho máximo, ou não formarem uma imagem válida
    """
    if not file_bytes:
        raise InvalidImageError("Arquivo enviado está vazio.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise InvalidImageError(
            f"Arquivo excede o tamanho máximo permitido "
            f"({MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB)."
        )

    # Converte os bytes crus em um array temporário de 1 dimensão,
    # que o OpenCV consegue interpretar e decodificar como imagem.
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # cv2.imdecode retorna None (não levanta exceção) quando os bytes
    # não formam uma imagem válida — por isso a checagem explícita aqui.
    if image_bgr is None:
        raise InvalidImageError(
            "Não foi possível decodificar o arquivo como imagem. "
            "Verifique se o formato é suportado (JPEG, PNG, etc)."
        )

    # OpenCV decodifica em BGR por padrão; o modelo espera RGB.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    logger.info(f"Imagem decodificada com sucesso: shape={image_rgb.shape}")

    return image_rgb