"""
Interface Streamlit para testar a PPE Detector API visualmente.

Este arquivo é um CLIENTE da API — ele não importa nada de app/core
ou app/services. Toda comunicação acontece via HTTP, exatamente como
qualquer outro sistema externo faria. Isso mantém a API como o
verdadeiro "produto" do projeto, reutilizável por qualquer interface.
"""

import os

import streamlit as st
import requests

# URL da API, configurável via variável de ambiente.
# - Rodando localmente (fora do Docker): usa o padrão "http://localhost:8000"
# - Rodando dentro do Docker Compose: a variável API_URL é injetada
#   apontando para "http://api:8000" — "api" é o NOME do serviço definido
#   no docker-compose.yml, resolvido automaticamente pela rede interna
#   do Compose, não é um domínio real da internet.
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PPE Detector", page_icon="🦺")

st.title("🦺 PPE Detector")
st.write("Envie uma imagem para detectar Equipamentos de Proteção Individual (EPI).")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem enviada", use_container_width=True)

    if st.button("Detectar EPIs"):
        with st.spinner("Rodando inferência..."):
            # Prepara o arquivo no formato multipart/form-data,
            # o mesmo formato que a API espera (UploadFile do FastAPI).
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if data["detection_count"] == 0:
                        st.warning("Nenhum EPI detectado nesta imagem.")
                    else:
                        st.success(f"{data['detection_count']} detecção(ões) encontrada(s)")

                        # Tabela organizada em vez de linhas soltas de texto.
                        st.subheader("Detecções")
                        for det in data["detections"]:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"**{det['class_name']}**")
                            with col2:
                                # Barra de progresso comunica confiança de forma
                                # mais intuitiva do que só o número percentual.
                                st.progress(
                                    det["confidence"],
                                    text=f"{det['confidence']:.1%} de confiança",
                                )

                    # Segunda chamada, ao endpoint que devolve a imagem anotada,
                    # para exibir visualmente as bounding boxes. Só faz sentido
                    # buscar a imagem anotada se houve alguma detecção.
                    if data["detection_count"] > 0:
                        files_img = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        image_response = requests.post(f"{API_URL}/predict-image", files=files_img, timeout=30)

                        if image_response.status_code == 200:
                            st.subheader("Visualização")
                            st.image(image_response.content, caption="Detecções", use_container_width=True)

                else:
                    st.error(f"Erro da API ({response.status_code}): {response.json().get('detail')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Não foi possível conectar à API. Verifique se ela está rodando "
                    f"em {API_URL}."
                )