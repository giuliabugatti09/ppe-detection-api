"""
Ponto de entrada da aplicação FastAPI.

Este arquivo é responsável apenas por ORQUESTRAÇÃO HTTP: receber
requisições, validar formato básico, chamar a lógica de negócio
(que já existe em app/core e app/services), e devolver respostas.

Ele NÃO contém lógica de modelo, nem de processamento de imagem —
essa separação é intencional (mesmo princípio dos dias anteriores).
"""

import logging
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, JSONResponse

import cv2

from app.core.model_loader import predict
from app.services.preprocessing import bytes_to_image, InvalidImageError
from app.services.postprocessing import format_detections, draw_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPE Detector API",
    description="API de detecção de Equipamentos de Proteção Individual (EPI) usando YOLOv8.",
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Intercepta TODA requisição HTTP, registrando método, path, status
    de resposta e tempo total de processamento.

    Isso roda para qualquer endpoint (atual ou futuro) sem precisar
    adicionar logging manualmente em cada um — é a mesma lógica de
    reuso de código que já aplicamos em _validate_and_read_image,
    agora no nível de infraestrutura HTTP.
    """
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> status={response.status_code} "
        f"duration={duration_ms:.1f}ms"
    )

    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Captura qualquer exceção não tratada explicitamente pelos endpoints.

    O erro completo é logado no servidor (para debug), mas o cliente
    recebe apenas uma mensagem genérica — nunca o traceback interno,
    que poderia expor detalhes da implementação ou ser usado de forma
    maliciosa por quem está consumindo a API.
    """
    logger.error(f"Erro não tratado em {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno ao processar a requisição."},
    )


def _validate_and_read_image(file: UploadFile, file_bytes: bytes):
    """
    Validações compartilhadas entre /predict e /predict-image:
    tipo de conteúdo declarado pelo cliente + decodificação dos bytes.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. Envie uma imagem.",
        )

    try:
        return bytes_to_image(file_bytes)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    image = _validate_and_read_image(file, file_bytes)

    inference_start = time.perf_counter()
    result = predict(image)
    inference_ms = (time.perf_counter() - inference_start) * 1000
    logger.info(f"Inferência concluída em {inference_ms:.1f}ms")

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
    image = _validate_and_read_image(file, file_bytes)

    inference_start = time.perf_counter()
    result = predict(image)
    inference_ms = (time.perf_counter() - inference_start) * 1000
    logger.info(f"Inferência concluída em {inference_ms:.1f}ms")

    annotated_image_bgr = draw_detections(image, result)

    # Codifica o array numpy de volta para bytes JPEG, prontos para
    # serem enviados como corpo da resposta HTTP.
    success, encoded_image = cv2.imencode(".jpg", annotated_image_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem de resposta.")

    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")