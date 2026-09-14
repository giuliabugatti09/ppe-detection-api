"""
Script temporário para validar o pipeline completo:
bytes crus -> preprocessing -> model_loader

Simula o que vai acontecer numa requisição HTTP real, onde a API
recebe bytes de upload, não um caminho de arquivo já pronto.

Rode com: python test_preprocessing.py
"""

from app.services.preprocessing import bytes_to_image, InvalidImageError
from app.core.model_loader import predict
from app.core.config import CLASS_NAMES

IMAGE_PATH = "tests/test_images/sample.jpg"

# Lê o arquivo como bytes crus, simulando o que vem de um UploadFile do FastAPI
with open(IMAGE_PATH, "rb") as f:
    file_bytes = f.read()

try:
    image = bytes_to_image(file_bytes)
    print(f"Imagem decodificada: shape={image.shape}, dtype={image.dtype}")

    result = predict(image)

    print(f"Detecções encontradas: {len(result.boxes)}")
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id]
        confidence = float(box.conf[0])
        print(f"  Classe: {class_name} (id={class_id}) | Confiança: {confidence:.2f}")

except InvalidImageError as e:
    print(f"Erro de imagem inválida: {e}")


# --- Teste extra: confirmar que erros são tratados corretamente ---
print("\n--- Testando bytes inválidos ---")
try:
    bytes_to_image(b"isso nao e uma imagem")
except InvalidImageError as e:
    print(f"Erro corretamente capturado: {e}")