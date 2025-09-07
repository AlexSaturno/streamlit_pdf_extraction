# PASTAS NECESSÁRIAS:
# avaliacao
# uploaded_files
# vectordb
################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain

from datetime import datetime, timedelta
import os
import time
import json
from pydantic import SecretStr
import tiktoken
import streamlit as st
from unidecode import unidecode
import base64
from utils import *

################################################################################################################################
# Ambiente
################################################################################################################################


def define_chain(openai_key):
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_key)

    # Conversão de imagens para base64 para enviar para o modelo
    def load_images(inputs: dict) -> dict:
        """Load multiple images from files and encode them as base64."""
        image_paths = inputs["image_paths"]

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        images_base64 = [encode_image(path) for path in image_paths]
        return {"images": images_base64}

    load_images_chain = TransformChain(
        input_variables=["image_paths"],
        output_variables=["images"],
        transform=load_images,
    )

    def image_model(
        inputs: dict,
    ) -> str | list[str] | dict:
        """Invoke model with images and prompt."""
        image_urls = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            for img in inputs["images"]
        ]

        content = [{"type": "text", "text": inputs["prompt"]}] + image_urls

        msg = llm.invoke([HumanMessage(content=content)])
        return str(msg.content)

    chain = load_images_chain | image_model

    return chain


###############################################################################
# Funcoes auxiliares
def normalize_filename(filename):
    # Mapeamento de caracteres acentuados para não acentuados
    substitutions = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "Á": "A",
        "À": "A",
        "Ã": "A",
        "Â": "A",
        "Ä": "A",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "Í": "I",
        "Ì": "I",
        "Î": "I",
        "Ï": "I",
        "Ó": "O",
        "Ò": "O",
        "Õ": "O",
        "Ô": "O",
        "Ö": "O",
        "Ú": "U",
        "Ù": "U",
        "Û": "U",
        "Ü": "U",
        "Ç": "C",
    }

    # Substitui caracteres especiais conforme o dicionário
    normalized_filename = "".join(substitutions.get(c, c) for c in filename)

    # Remove caracteres não-ASCII
    ascii_filename = normalized_filename.encode("ASCII", "ignore").decode("ASCII")

    # Substitui espaços por underscores
    safe_filename = ascii_filename.replace(" ", "_")

    return safe_filename


def clear_respostas():
    st.session_state["clear_respostas"] = True
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def zera_vetorizacao():
    st.session_state["status_vetorizacao"] = False
    st.session_state["clear_respostas"] = True
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.markdown(
            f'<p class="font-stream">{result}</p>', unsafe_allow_html=True
        )


def get_stream(texto):
    for word in texto.split(" "):
        yield word + " "
        time.sleep(0.01)


# Function to initialize session state
def initialize_session_state():
    if "file_name" not in st.session_state:
        st.session_state["file_name"] = None

    if "status_vetorizacao" not in st.session_state:
        st.session_state["status_vetorizacao"] = False

    if "clear_respostas" not in st.session_state:
        st.session_state["clear_respostas"] = False

    if "data_processamento" not in st.session_state:
        st.session_state["data_processamento"] = None

    if "hora_processamento" not in st.session_state:
        st.session_state["hora_processamento"] = None

    if "tempo_ia" not in st.session_state:
        st.session_state["tempo_ia"] = 0

    if "tempo_vetorizacao" not in st.session_state:
        st.session_state["tempo_vetorizacao"] = 0

    if "pdf_store" not in st.session_state:
        st.session_state["pdf_store"] = True

    if "id_unico" not in st.session_state:
        st.session_state["id_unico"] = True


################################################################################################################################
# UX
################################################################################################################################
# Inicio da aplicação
initialize_session_state()

st.set_page_config(
    page_title="Processamento de documentos",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


################################################################################################################################
# UI
################################################################################################################################


# Inicio da aplicação
def main():
    st.subheader("Análise Automática de Documentos")
    st.write("")

    with st.container(border=True):
        pdf_file = st.file_uploader(
            "Carregamento de arquivo",
            type=["pdf"],
            key="pdf_file",
            on_change=zera_vetorizacao,
        )

        if pdf_file is not None and not st.session_state["status_vetorizacao"]:
            # Se tiver PDFs na pasta quando inicializar a aplicação, apagá-los
            for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
                arquivo.unlink()
            savefile_name = normalize_filename(pdf_file.name)
            with open(PASTA_ARQUIVOS / f"{savefile_name}", "wb") as f:
                f.write(pdf_file.read())

            st.session_state["pdf_store"] = pdf_file.getbuffer()
            st.session_state["file_name"] = pdf_file.name[:-4]

            data_processamento = datetime.now().strftime("%Y-%m-%d")
            hora_processamento = (datetime.now() - timedelta(hours=3)).strftime("%H:%M")
            st.session_state["data_processamento"] = data_processamento
            st.session_state["hora_processamento"] = hora_processamento
            file_name = st.session_state["file_name"]

            id_unico = (
                str(st.session_state["data_processamento"])
                + "_"
                + str(st.session_state["hora_processamento"]).replace(":", "-")
                + "_"
                + unidecode(str(st.session_state["file_name"]).lower())
            )
            st.session_state["id_unico"] = id_unico

            pdf_store_full_path = f"{str(PASTA_ARQUIVOS)}/{id_unico}" + ".pdf"
            pdf_store_full_path = str(PASTA_ARQUIVOS) + "/" + id_unico + ".pdf"

            with open(pdf_store_full_path, "wb") as file:
                file.write(st.session_state["pdf_store"])

            if not st.session_state["status_vetorizacao"]:
                st.session_state["tempo_ia"] = 0
                start_time = time.time()

                with st.spinner("Processando documento..."):
                    # Converter PDF para imagens
                    convert_pdf_to_images(pdf_store_full_path)
                    st.session_state["status_vetorizacao"] = True

                    end_time = time.time()
                    tempo_vetorizacao = end_time - start_time
                    st.session_state["tempo_vetorizacao"] = tempo_vetorizacao
                    st.session_state["tempo_ia"] = 0

    st.write("")
    if st.session_state["status_vetorizacao"]:
        # 1. Campo para chave da OpenAI
        openai_key = st.text_input(
            "Digite sua chave de autenticação da OpenAI",
            type="password",
            key="openai_key",
            help="Sua chave não será armazenada.",
        )

        if not openai_key:
            st.info("Insira sua chave da OpenAI para continuar.")
            st.stop()
        chain = define_chain(openai_key)

        # 2. Campo para perguntas
        perguntas_text = st.text_area(
            "Digite suas perguntas (uma por linha):",
            height=150,
            key="perguntas_text",
            placeholder="Exemplo:\nQual o nome da empresa?\nQual o CNPJ da empresa?",
        )

        if not perguntas_text.strip():
            st.info("Digite ao menos uma pergunta.")
            st.stop()

        perguntas = [p.strip() for p in perguntas_text.splitlines() if p.strip()]

        llm_call = st.button("Perguntar para a IA")
        st.write("")
        ph = st.empty()
        with ph.container():
            if llm_call:
                with st.spinner("Processando perguntas"):
                    for pergunta in perguntas:
                        with get_openai_callback() as cb:
                            id_unico = st.session_state["id_unico"]
                            path_atual = f"{PASTA_IMAGENS}/{id_unico}_images"
                            quantidade_paginas = len(os.listdir(path_atual))
                            response = chain.invoke(
                                {
                                    "image_paths": [
                                        f"{path_atual}/page{n}.jpg"
                                        for n in range(0, quantidade_paginas)
                                    ],
                                    "prompt": pergunta,
                                }
                            )

                        response = response.replace("```json\n", "").replace(
                            "\n```", ""
                        )

                        st.markdown(f"**Pergunta:** {pergunta}")
                        st.markdown(f"**Resposta:** {response}")
                        st.markdown("---")


if __name__ == "__main__":
    main()
