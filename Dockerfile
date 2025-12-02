# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requis
COPY requirements.txt .
COPY app.py .
COPY dataset_final_nettoye.csv .
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/ 

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 5000

# Définir la variable d'environnement
ENV FLASK_APP=app.py

# Lancer l'application
CMD ["python", "app.py"]