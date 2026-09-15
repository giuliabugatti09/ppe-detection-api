# 🦺 PPE Detector API

API de detecção de Equipamentos de Proteção Individual (EPI) construída com YOLOv8, FastAPI, Docker e Streamlit — do treino do modelo até um microsserviço containerizado, pronto para produção.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

## Visão geral

Este projeto evolui um modelo acadêmico de detecção de capacetes/EPIs para uma aplicação de produção real, com foco em engenharia de ML: deploy, empacotamento, API design, observabilidade e testes — não só modelagem.

O modelo (YOLOv8n, treinado via transfer learning) identifica 4 classes de EPI de proteção de cabeça: capacete simples, hard hat, hard hat com visor, e visor isolado.

## Como rodar

Pré-requisito: [Docker](https://www.docker.com/products/docker-desktop/) instalado.

```bash
git clone https://github.com/giuliabugatti09/ppe-detection-api.git
cd ppe-detection-api
docker compose up --build
```

- API: http://localhost:8000/docs (documentação interativa Swagger)
- Interface visual: http://localhost:8501

## Arquitetura

```
Cliente (Streamlit / curl / navegador)
        │
        ▼
   FastAPI (/predict, /predict-image)
        │
        ├─► preprocessing.py   (bytes → array numpy validado, BGR→RGB)
        │
        ├─► model_loader.py    (YOLOv8, carregado uma única vez no startup)
        │
        └─► postprocessing.py  (Results → JSON estruturado ou imagem anotada)
```

Princípios de design:
- **Separação de responsabilidades**: modelo, processamento de imagem e orquestração HTTP em módulos independentes e testáveis isoladamente.
- **Modelo carregado uma vez**: evita o custo de recarregar pesos a cada requisição.
- **Configuração externalizada**: thresholds e paths configuráveis via variáveis de ambiente (`pydantic-settings`), sem precisar alterar código.
- **Dois formatos de resposta**: JSON estruturado (`/predict`) para consumo programático, e imagem anotada (`/predict-image`) para validação visual humana.

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Verificação de saúde da API |
| POST | `/predict` | Retorna detecções em JSON |
| POST | `/predict-image` | Retorna a imagem anotada com bounding boxes |

Limitados a 10 requisições/minuto por IP nos endpoints de inferência.

## Métricas do modelo

| Métrica | Valor |
|---|---|
| mAP@50 | ~0.80 |
| mAP@50-95 | ~0.49 |
| Precision | ~0.87 |
| Recall | ~0.67 |

Análise completa de desempenho por classe, matriz de confusão e limitações conhecidas (incluindo um caso documentado de falso positivo) em [METRICS.md](METRICS.md).

## Stack técnico

- **Modelo**: YOLOv8n (Ultralytics), fine-tuned via transfer learning
- **API**: FastAPI + Uvicorn
- **Processamento de imagem**: OpenCV
- **Interface**: Streamlit
- **Containerização**: Docker (multi-stage build, CPU-only) + Docker Compose
- **Testes**: pytest
- **Configuração**: pydantic-settings

## Testes

```bash
pip install -r requirements.txt
pytest -v
```

## Estrutura do projeto

```
ppe-detection-api/
├── app/
│   ├── core/            # configuração e carregamento do modelo
│   ├── services/        # pré e pós-processamento de imagem
│   └── main.py           # endpoints FastAPI
├── tests/                 # testes automatizados (pytest)
├── models/                # pesos do modelo treinado
├── docs/metrics/          # gráficos de treino e matrizes de confusão
├── streamlit_app.py        # interface de teste visual
├── Dockerfile               # imagem da API
├── Dockerfile.streamlit      # imagem da interface
├── docker-compose.yml         # orquestração dos dois serviços
├── METRICS.md                  # análise detalhada de desempenho
└── TROUBLESHOOTING.md            # problemas reais encontrados e soluções
```

## Dataset

Modelo treinado com o dataset [ppe_detection_with_yolov8](https://universe.roboflow.com/school-xhq28/ppe_detection_with_yolov8) (Roboflow Universe), licenciado sob CC BY 4.0.

## Limitações conhecidas

Ver [METRICS.md](METRICS.md#limitações-gerais-e-próximos-passos) para uma análise honesta de onde o modelo tem pior desempenho e por quê — incluindo a disparidade de dados entre classes e um caso documentado de falso positivo.

## Problemas de ambiente/build

Problemas reais enfrentados durante o desenvolvimento (Docker, WSL2, compatibilidade de versões) estão documentados em [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Licença 

MIT
