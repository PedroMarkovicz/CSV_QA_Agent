# Use uma imagem base do Python
FROM python:3.11-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o código da aplicação para o diretório de trabalho
COPY . .

# Exponha a porta que o Streamlit usa
EXPOSE 8501

# Comando para iniciar a aplicação
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 