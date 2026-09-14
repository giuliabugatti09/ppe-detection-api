"""
Módulo responsável por converter o objeto Results do Ultralytics
em uma estrutura de dados simples e serializável (dict), que pode
virar JSON diretamente numa resposta de API.

Esta camada isola o "contrato" da nossa API do formato interno
do Ultralytics. Se um dia trocarmos de framework de detecção
(ex: para um modelo Keras customizado), só este módulo muda —
o resto da aplicação continua recebendo o mesmo formato de saída.
"""

from app.core.config import CLASS_NAMES


def format_detections(result) -> dict:
    """
    Converte um objeto Results (retornado por model_loader.predict)
    em um dicionário simples, pronto para serialização JSON.

    Args:
        result: objeto Results do Ultralytics (saída de predict())

    Returns:
        dict no formato:
        {
            "detections": [
                {
                    "class_name": str,
                    "confidence": float,
                    "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}
                },
                ...
            ],
            "detection_count": int
        }
    """
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # xyxy = [x1, y1, x2, y2] em coordenadas de pixel da imagem original
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_name": CLASS_NAMES[class_id],
            "confidence": round(confidence, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        })

    return {
        "detections": detections,
        "detection_count": len(detections),
    }