"""
Testes automatizados para app/services/postprocessing.py.

Usa o pipeline real (preprocessing + model_loader) para gerar um
objeto Results de verdade, e valida a estrutura da conversão para
dict/JSON — não fazemos mock do modelo aqui porque o YOLOv8n é leve
o suficiente para rodar em segundos, mesmo em CPU, durante os testes.
"""

from app.services.preprocessing import bytes_to_image
from app.services.postprocessing import format_detections, draw_detections
from app.core.model_loader import predict


def test_format_detections_estrutura_do_retorno(sample_image_bytes):
    image = bytes_to_image(sample_image_bytes)
    result = predict(image)

    response = format_detections(result)

    # Valida a ESTRUTURA da resposta, não um valor específico de
    # detecção — isso evita que o teste quebre só porque o modelo
    # foi retreinado e a confiança mudou de 0.66 para 0.68, por exemplo.
    assert "detections" in response
    assert "detection_count" in response
    assert response["detection_count"] == len(response["detections"])

    for det in response["detections"]:
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det
        assert set(det["bbox"].keys()) == {"x1", "y1", "x2", "y2"}
        assert 0.0 <= det["confidence"] <= 1.0


def test_draw_detections_retorna_imagem_mesmo_shape(sample_image_bytes):
    image = bytes_to_image(sample_image_bytes)
    result = predict(image)

    annotated = draw_detections(image, result)

    # A imagem anotada deve ter as mesmas dimensões da original —
    # só desenhamos por cima, não redimensionamos nada.
    assert annotated.shape == image.shape