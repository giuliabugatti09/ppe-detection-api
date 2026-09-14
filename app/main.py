"""
Ponto de entrada da aplicação FastAPI.

Este arquivo é responsável apenas por ORQUESTRAÇÃO HTTP: receber
requisições, validar formato básico, chamar a lógica de negócio
(que já existe em app/core e app/services), e devolver respostas.

Ele NÃO contém lógica de modelo, nem de processamento de imagem —
essa separação é intencional (mesmo princípio dos dias anteriores).
"""

import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

import cv2

from app.core.model_loader import predict
from app.services.preprocessing import bytes_to_image, InvalidImageError
from app.services.postprocessing import format_detections, draw_detections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPE Detector API",
    description="API de detecção de Equipamentos de Proteção Individual (EPI) usando YOLOv8.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """
    Endpoint simples de verificação de saúde da API.

    Usado para confirmar que o processo está no ar e respondendo —
    útil tanto para testes manuais quanto para orquestradores
    (Docker healthcheck, Kubernetes liveness probe, etc.) verificarem
    automaticamente se o serviço está funcional.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Recebe uma imagem via upload e retorna as detecções de EPI encontradas.

    Fluxo: bytes do upload -> preprocessing -> model_loader -> postprocessing.
    Cada etapa já foi construída e validada isoladamente nos dias anteriores;
    este endpoint apenas orquestra a chamada entre elas.
    """
    file_bytes = await file.read()

    try:
        image = bytes_to_image(file_bytes)
    except InvalidImageError as e:
        # Erro de domínio (imagem inválida) vira erro HTTP 400 (culpa do cliente),
        # não 500 (que indicaria bug do nosso lado).
        raise HTTPException(status_code=400, detail=str(e))

    result = predict(image)
    response = format_detections(result)

    return response


@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Recebe uma imagem via upload e retorna a MESMA imagem, anotada com
    as bounding boxes das detecções desenhadas sobre ela.

    Útil para validação visual humana — em contraste com /predict,
    que devolve os dados estruturados em JSON para consumo programático.
    """
    file_bytes = await file.read()

    try:
        image = bytes_to_image(file_bytes)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = predict(image)
    annotated_image_bgr = draw_detections(image, result)

    # Codifica o array numpy de volta para bytes JPEG, prontos para
    # serem enviados como corpo da resposta HTTP.
    success, encoded_image = cv2.imencode(".jpg", annotated_image_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem de resposta.")

    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")