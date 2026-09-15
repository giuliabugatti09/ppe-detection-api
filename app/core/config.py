"""
Configurações centrais do projeto.

Manter paths e constantes aqui evita "magic strings" espalhadas
pelo código e facilita trocar o modelo/thresholds no futuro
sem precisar caçar onde cada valor está sendo usado.
"""

from pathlib import Path

# Diretório raiz do projeto (2 níveis acima deste arquivo: app/core/ -> raiz)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Caminho para os pesos do modelo treinado
MODEL_PATH = BASE_DIR / "models" / "ppe_weights.pt"

# Threshold de confiança mínima para considerar uma detecção válida
# (detecções abaixo disso são descartadas como ruído)
CONFIDENCE_THRESHOLD = 0.6

# Threshold de IoU para Non-Maximum Suppression
# (evita múltiplas boxes sobrepostas para o mesmo objeto)
IOU_THRESHOLD = 0.45

# Nomes das classes do dataset (na ordem em que o modelo foi treinado)
# W = capacete simples | WH = hard hat | WHV = hard hat com visor | WV = visor
CLASS_NAMES = ["W", "WH", "WHV", "WV"]