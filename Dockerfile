# Étape 6 - Déploiement : l'application Streamlit dans un conteneur Docker.
# Le conteneur contient Python, PyTorch (version CPU), le code, les modèles entraînés
# et les images de test (avec leur vérité terrain) : il se lance de la même façon
# sur n'importe quel ordinateur (Windows, Mac, Linux, processeur Intel ou ARM).
#
#   docker build -t detection-avions .
#   docker run -p 8501:8501 detection-avions        puis ouvrir http://localhost:8501

FROM python:3.12-slim

WORKDIR /app

# 1. Les bibliothèques (couche mise en cache : reconstruite seulement si cette ligne change)
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir streamlit==1.67.0 pandas matplotlib pillow numpy

# 2. Le code de l'application
COPY src/ src/
COPY 08_application.py .

# 3. Les modèles entraînés et leurs réglages (seuil, NMS... choisis sur la validation)
COPY outputs/ outputs/

# 4. Les images de test et leurs annotations XML (exemples + comparaison à la vérité terrain)
COPY data/images_detection/test/ data/images_detection/test/

EXPOSE 8501

# Docker vérifie régulièrement que l'application répond
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "08_application.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
