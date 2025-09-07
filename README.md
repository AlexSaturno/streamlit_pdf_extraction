## Como Executar

### 1. Pré-requisitos

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) e poppler-utils instalados no sistema
- Conta e chave de API da OpenAI

### 2. Instalação

Clone o repositório e instale as dependências (recomendação: uv):

```sh
git clone https://github.com/seuusuario/streamlit_pdf_extraction.git
cd streamlit_pdf_extraction
uv venv venv
uv pip install -r [requirements.txt]
```

### 3. Executando a aplicação

streamlit run [main.py]

### 4. Opcional: Usando Docker

docker build -t streamlit-pdf .
docker run -d -p 8501:8501 -v $(pwd):/work streamlit-pdf

### Uso

- Faça upload de um arquivo PDF.
- Aguarde a conversão das páginas em imagens.
- Insira sua chave da OpenAI.
- Digite perguntas sobre o documento (uma por linha).
- Clique em "Perguntar para a IA" e visualize as respostas.

### Observações

- Essa é uma POC simples, a limitação de 50 páginas por documento não está tratada
- Projeto desenvolvido com Langchain, utilizando implementação de chain para tratar imagens
- Utilize modelos multimodais para essa lógica de chain com OCR
