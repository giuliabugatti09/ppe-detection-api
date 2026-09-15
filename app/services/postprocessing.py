"""
Módulo responsável por converter o objeto Results do Ultralytics
em uma estrutura de dados simples e serializável (dict), que pode
virar JSON diretamente numa resposta de API.

Esta camada isola o "contrato" da nossa API do formato interno
do Ultralytics. Se um dia trocarmos de framework de detecção
(ex: para um modelo Keras customizado), só este módulo muda —
o resto da aplicação continua recebendo o mesmo formato de saída.
"""

import cv2
import numpy as np

from app.core.config import settings


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
            "class_name": settings.class_names[class_id],
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


def draw_detections(image_rgb: np.ndarray, result) -> np.ndarray:
    """
    Desenha bounding boxes e labels sobre a imagem, para visualização humana.

    Args:
        image_rgb: array numpy da imagem original, em formato RGB
            (o mesmo formato produzido por preprocessing.bytes_to_image)
        result: objeto Results do Ultralytics (saída de predict())

    Returns:
        Array numpy da imagem anotada, em formato BGR — já pronto
        para ser codificado como JPEG via cv2.imencode.
    """
    # OpenCV desenha e codifica esperando BGR, então convertemos aqui,
    # isoladamente, sem afetar o restante do pipeline (que trabalha em RGB).
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = settings.class_names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Retângulo da detecção
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Label com nome da classe e confiança
        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Fundo sólido atrás do texto, para legibilidade sobre qualquer imagem
        cv2.rectangle(
            image_bgr,
            (x1, y1 - label_h - 6),
            (x1 + label_w, y1),
            color=(0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            image_bgr, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0), thickness=1,
        )

    return image_bgr