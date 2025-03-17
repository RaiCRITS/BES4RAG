# Usa un'immagine di base con Python 3.11.11
FROM python:3.11.11

# Imposta il working directory
WORKDIR /app

# Installa git per clonare il repository
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clona il repository BES4RAG da GitHub
RUN git clone https://github.com/tuo-utente/BES4RAG.git

# Spostati nella cartella del repository
WORKDIR /app/BES4RAG

# Installa le dipendenze da requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mantieni il container attivo permettendo di eseguire comandi interattivi
CMD ["/bin/bash"]

