# Usa un'immagine di base con Python 3.11.11
FROM python:3.11.11

# Imposta il working directory
WORKDIR /app

# Installa git per gestire eventuali dipendenze future (opzionale)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copia il contenuto del tuo repository nella cartella di lavoro del container
COPY . /app

# Spostati nella cartella del repository (presumendo che sia nella root del Dockerfile)
WORKDIR /app/BES4RAG

# Installa le dipendenze da requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mantieni il container attivo permettendo di eseguire comandi interattivi
CMD ["/bin/bash"]
