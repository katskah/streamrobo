import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Titre de l'application
st.title("Roboflow Object Detection - Inférence sur nouvelles images")

# Initialiser le client Roboflow avec la clé API cachée
api_key = st.secrets["ROBOFLOW_API_KEY"]
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Sélection du modèle Roboflow (paramétrable)
model_id = st.selectbox(
    "Choisir le modèle Roboflow",
    ["encre-ferrogallique-2-wy9md/5", "encre-ferrogallique-2-wy9md/3", "encre-ferrogallique-2-wy9md/2"],
    index=0
)

# Upload d'une image depuis le disque dur
uploaded_file = st.file_uploader(
    "Choisir une image à analyser",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Afficher l'image originale
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Image originale",
        use_column_width=True
    )

    # Sélection du seuil de confiance
    seuil_confiance = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.01)

    # Sauvegarder temporairement l'image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name

    # Inférence avec Roboflow
    with st.spinner("Détection en cours..."):
        result = CLIENT.infer(tmp_path, model_id=model_id)

    # Afficher les prédictions brutes (optionnel)
    with st.expander("Voir les prédictions brutes (JSON)"):
        st.json(result)

    # Charger l'image avec OpenCV pour dessiner les boîtes
    img = cv2.imread(tmp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dessiner les boîtes de détection (filtrées par seuil)
    for pred in [p for p in result["predictions"] if p["confidence"] >= seuil_confiance]:
        x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - width/2), int(y - height/2)
        x2, y2 = int(x + width/2), int(y + height/2)
        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image annotée
    st.image(
        img_rgb,
        caption="Résultat de la détection",
        use_column_width=True

    )

    # Afficher les détails des détections
    st.subheader("Détails des détections")
    for pred in [p for p in result["predictions"] if p["confidence"] >= seuil_confiance]:
        x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - width/2), int(y - height/2)
        x2, y2 = int(x + width/2), int(y + height/2)
        st.write(
            f"Classe: {pred['class']}, "
            f"Confiance: {pred['confidence']:.2f}, "
            f"Position: ({x1}, {y1})-({x2}, {y2})"
        )

    # Nettoyer le fichier temporaire
    os.unlink(tmp_path)
