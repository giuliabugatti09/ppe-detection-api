"""
Ponto de entrada da aplicação FastAPI.

Este arquivo é responsável apenas por ORQUESTRAÇÃO HTTP: receber
requisições, validar formato básico, chamar a lógica de negócio
(que já existe em app/core e app/services), e devolver respostas.

Ele NÃO contém lógica de modelo, nem de processamento de imagem —
essa separação é intencional (mesmo princípio dos dias anteriores).
"""

import logging

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)

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