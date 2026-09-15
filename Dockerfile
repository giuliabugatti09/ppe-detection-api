# ---------- Stage 1: builder ----------
# Este estágio instala TODAS as dependências. Instalamos tudo no
# site-packages padrão da imagem (sem --prefix), para que uma
# instalação "veja" a outra e não tente reinstalar/baixar de novo
# (isso é o que causava o download indevido de bibliotecas CUDA).
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instala PyTorch CPU-only PRIMEIRO, no site-packages padrão da imagem.
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instala o restante das dependências. Como torch/torchvision já estão
# no MESMO site-packages, o pip reconhece que já estão satisfeitos
# e não tenta baixar de novo (nem puxa as dependências CUDA).
RUN pip install --no-cache-dir -r requirements.txt


# ---------- Stage 2: imagem final ----------
# Copiamos o site-packages inteiro (com tudo já instalado) do builder,
# sem levar cache de pip, apt, ou arquivos temporários de instalação.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia os pacotes Python instalados e os executáveis (ex: uvicorn)
# do estágio builder para a imagem final.
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]