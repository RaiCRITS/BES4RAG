# Usa un'immagine con CUDA se disponibile, altrimenti usa l'immagine Python di base
ARG BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS cuda

FROM python:3.11.11 AS python

# Controlla se CUDA è disponibile
ARG USE_CUDA=1
RUN if [ "$USE_CUDA" = "1" ]; then echo "Using CUDA base image"; else echo "Using Python base image"; fi

# Se si usa l'immagine CUDA, installa Python
RUN if [ "$USE_CUDA" = "1" ]; then apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*; fi

# Imposta la working directory
WORKDIR /app

# Installa git per clonare il repository
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clona il repository BES4RAG da GitHub
RUN git clone https://github.com/RaiCRITS/BES4RAG.git

# Spostati nella cartella del repository
WORKDIR /app/BES4RAG

# Installa le dipendenze da requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mantieni il container attivo permettendo di eseguire comandi interattivi
CMD ["/bin/bash"]
