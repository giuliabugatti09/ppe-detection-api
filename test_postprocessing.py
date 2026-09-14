"""
Script temporário para validar o pipeline COMPLETO:
bytes crus -> preprocessing -> model_loader -> postprocessing -> JSON

Este é o fluxo exato que o endpoint da API vai executar amanhã.

Rode com: python test_postprocessing.py
"""

import json

from app.services.preprocessing import bytes_to_image, InvalidImageError
from app.core.model_loader import predict
from app.services.postprocessing import format_detections

IMAGE_PATH = "tests/test_images/sample.jpg"

with open(IMAGE_PATH, "rb") as f:
    file_bytes = f.read()

try:
    image = bytes_to_image(file_bytes)
    result = predict(image)
    response = format_detections(result)

    # Isso é literalmente o que a API vai devolver pro cliente
    print(json.dumps(response, indent=2, ensure_ascii=False))

except InvalidImageError as e:
    print(f"Erro de imagem inválida: {e}")