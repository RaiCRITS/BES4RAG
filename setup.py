import os
import subprocess
import sys

def run_command(command):
    """Esegue un comando shell e stampa l'output in tempo reale."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()

def install_dependencies():
    """Installa i pacchetti necessari e configura ColBERT."""
    print("\n🔄 Aggiornamento di pip...")
    run_command("pip install -U pip")

    run_command(f"{sys.executable} -m pip install -r requirements.txt")

    print("\n📥 Clonazione o aggiornamento di ColBERT...")
    if os.path.exists("ColBERT"):
        run_command("git -C ColBERT pull")
    else:
        run_command("git clone https://github.com/stanford-futuredata/ColBERT.git")

    print("\n🔗 Installazione di ColBERT con faiss-gpu...")
    run_command(f"{sys.executable} -m pip install -e ColBERT/['faiss-gpu']")

    print("\n✅ Setup completato!")

if __name__ == "__main__":
    install_dependencies()
