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

from app.core.model_loader import predict
from app.services.preprocessing import bytes_to_image, InvalidImageError
from app.services.postprocessing import format_detections

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