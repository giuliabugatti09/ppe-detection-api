"""
Módulo responsável por carregar o modelo YOLO e expor uma interface
simples de inferência para o resto da aplicação.

O modelo é carregado UMA VEZ, no momento em que este módulo é
importado (não a cada chamada de predict()). Isso é crítico para
performance: carregar pesos do disco para memória tem custo, e
repetir isso por requisição inviabilizaria a API em produção.
"""

import logging
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Carregamento único do modelo, no nível do módulo ---
# Isso roda apenas na primeira vez que o módulo é importado.
# Chamadas subsequentes de "import model_loader" em outros arquivos
# reutilizam essa mesma instância em memória (comportamento padrão
# do Python para módulos já importados).
try:
    logger.info(f"Carregando modelo de: {settings.weights_path}")
    model = YOLO(str(settings.weights_path))
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    logger.error(f"Falha ao carregar o modelo: {e}")
    raise


def predict(image):
    """
    Roda inferência em uma imagem já decodificada (array numpy, formato BGR/RGB).

    Args:
        image: array numpy representando a imagem (ex: saída do cv2.imdecode)

    Returns:
        Objeto Results do Ultralytics, contendo boxes, classes e scores.
        A conversão para um formato "amigável" (dict/JSON) fica no
        módulo de postprocessing, não aqui — este módulo só sabe
        conversar com o modelo, nada além disso.
    """
    results = model.predict(
        source=image,
        conf=settings.confidence_threshold,
        iou=settings.iou_threshold,
        verbose=False,
    )
    return results[0]  # predict() retorna uma lista; pegamos o resultado da única imagem