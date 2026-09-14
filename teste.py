"""
Script temporário para validar que o model_loader funciona
corretamente ANTES de construir a API em cima dele.

Rode com: python test_manual.py
(depois pode apagar este arquivo, ele não faz parte da aplicação)
"""

import cv2
from app.core.model_loader import predict
from app.core.model_loader import model

print(model.names)

# Coloque uma imagem de teste em tests/test_images/
IMAGE_PATH = "tests/test_images/sample.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem em {IMAGE_PATH}")

result = predict(image)

print(f"Detecções encontradas: {len(result.boxes)}")
for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(f"  Classe: {class_id} | Confiança: {confidence:.2f}")