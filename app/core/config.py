"""
Configurações centrais do projeto, carregadas de variáveis de
ambiente (opcionalmente de um arquivo .env) com valores padrão
sensatos para desenvolvimento local.

Usar pydantic-settings garante validação automática de tipos: se uma
variável de ambiente vier com valor inválido (ex: um texto onde se
espera um número), a aplicação falha já na inicialização, com uma
mensagem clara — em vez de quebrar silenciosamente durante uma
requisição real.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Caminho para os pesos do modelo treinado.
    weights_path: Path = BASE_DIR / "models" / "ppe_weights.pt"

    # Threshold de confiança mínima para considerar uma detecção válida.
    confidence_threshold: float = 0.6

    # Threshold de IoU para Non-Maximum Suppression.
    iou_threshold: float = 0.45

    # Nomes das classes do dataset, na ordem em que o modelo foi treinado.
    # W = capacete simples | WH = hard hat | WHV = hard hat com visor | WV = visor
    #
    # Nota: diferente dos valores acima, isto normalmente NÃO deveria ser
    # sobrescrito via ambiente — está atrelado aos pesos específicos do
    # modelo treinado. Mantido aqui centralizado por conveniência, não
    # como um parâmetro pensado para variar entre ambientes.
    class_names: list[str] = ["W", "WH", "WHV", "WV"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PPE_",
        protected_namespaces=(),
    )


# Instância única, importada pelo resto da aplicação — mesma ideia
# de "carregar uma vez" que já usamos para o modelo no Dia 1.
settings = Settings()