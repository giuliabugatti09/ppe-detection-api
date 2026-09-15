"""
Interface Streamlit para testar a PPE Detector API visualmente.

Este arquivo é um CLIENTE da API — ele não importa nada de app/core
ou app/services. Toda comunicação acontece via HTTP, exatamente como
qualquer outro sistema externo faria. Isso mantém a API como o
verdadeiro "produto" do projeto, reutilizável por qualquer interface.
"""
import streamlit as st
import requests

# URL da API. Em desenvolvimento local, aponta para localhost.
# Quando integrarmos via docker-compose (Dia 14), isso vai virar
# o nome do serviço (ex: "http://api:8000"), não mais localhost.
API_URL = "http://localhost:8000"

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

                    st.success(f"{data['detection_count']} detecção(ões) encontrada(s)")

                    for det in data["detections"]:
                        st.write(
                            f"**{det['class_name']}** — confiança: {det['confidence']:.2%}"
                        )

                    # Segunda chamada, ao endpoint que devolve a imagem anotada,
                    # para exibir visualmente as bounding boxes.
                    files_img = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    image_response = requests.post(f"{API_URL}/predict-image", files=files_img, timeout=30)

                    if image_response.status_code == 200:
                        st.image(image_response.content, caption="Detecções", use_container_width=True)

                else:
                    st.error(f"Erro da API ({response.status_code}): {response.json().get('detail')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Não foi possível conectar à API. Verifique se ela está rodando "
                    f"em {API_URL}."
                )