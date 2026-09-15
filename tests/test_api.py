"""
Testes automatizados para os endpoints da API (app/main.py).

Usa o TestClient do FastAPI, que simula requisições HTTP SEM precisar
do servidor (uvicorn) rodando de verdade — chama a aplicação
diretamente em memória, o que torna os testes rápidos e independentes
de portas de rede ou processos externos.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_com_imagem_valida(sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "detection_count" in data


def test_predict_com_content_type_invalido(sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("arquivo.txt", b"nao e uma imagem", "text/plain")},
    )

    assert response.status_code == 400
    assert "Tipo de arquivo não suportado" in response.json()["detail"]


def test_predict_com_arquivo_vazio():
    response = client.post(
        "/predict",
        files={"file": ("vazio.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "vazio" in response.json()["detail"].lower()


def test_predict_image_retorna_jpeg(sample_image_bytes):
    response = client.post(
        "/predict-image",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # Confirma que o corpo da resposta não está vazio
    assert len(response.content) > 0