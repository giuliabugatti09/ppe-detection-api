# Imagem base "slim": Python oficial, mínima, sem ferramentas desnecessárias.
# Menor tamanho de imagem e menos superfície de ataque de segurança
# comparado a imagens completas (ex: python:3.11 sem "-slim").
FROM python:3.11-slim

# Diretório de trabalho dentro do container — todos os comandos
# seguintes (COPY, RUN, CMD) operam relativos a este caminho.
WORKDIR /app

# Dependências de SISTEMA operacional (não Python) que o OpenCV precisa
# para funcionar, mesmo na versão "headless". Sem isso, o import do
# cv2 falha dentro do container com erro de biblioteca compartilhada ausente.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos APENAS o requirements.txt primeiro, não o código todo.
# Isso é proposital: o Docker cacheia esta camada de instalação de
# dependências. Se você mudar seu código Python depois mas não as
# dependências, o rebuild pula esta etapa lenta (torch/ultralytics
# são pesados) e vai direto para copiar o código atualizado.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Só agora copiamos o código da aplicação e os pesos do modelo —
# a parte que muda com mais frequência durante o desenvolvimento.
COPY app/ ./app/
COPY models/ ./models/

# Documenta a porta que a aplicação usa dentro do container.
# Isso é informativo (não abre a porta sozinho) — o mapeamento real
# de porta acontece no comando "docker run -p" ou no docker-compose.
EXPOSE 8000

# Comando executado quando o container inicia.
# --host 0.0.0.0 é essencial: sem isso, a API só aceitaria conexões
# de dentro do próprio container, ficando inacessível de fora.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]