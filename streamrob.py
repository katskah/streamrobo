import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Titre de l'application
st.title("Roboflow Object Detection - Inférence sur nouvelles images")

# Initialiser le client Roboflow
api_key = st.secrets["ROBOFLOW_API_KEY"]
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Sélection du modèle
model_id = st.selectbox(
    "Choisir le modèle Roboflow",
    ["encre-ferrogallique-2-wy9md/5", "encre-ferrogallique-2-wy9md/3", "encre-ferrogallique-2-wy9md/2"],
    index=0
)

# Upload d'une image
uploaded_file = st.file_uploader(
    "Choisir une image à analyser",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Sécurisation de l'ouverture de l'image
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Impossible d’ouvrir l’image : {e}")
        st.stop()

    st.image(
        np.array(image),
        caption="Image originale",
        use_column_width=True
    )

    # Slider seuil de confiance
    seuil_confiance = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.01)

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name

    # Inférence
    with st.spinner("Détection en cours..."):
        try:
            result = CLIENT.infer(tmp_path, model_id=model_id)
        except Exception as e:
            st.error(f"Erreur API Roboflow : {e}")
            st.stop()

    # JSON brut
    with st.expander("Voir les prédictions brutes (JSON)"):
        st.json(result)

    # Annoter image
    img = cv2.imread(tmp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pred in [p for p in result.get("predictions", []) if p["confidence"] >= seuil_confiance]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        label = f"{pred['class']} {pred['confidence']:.2f}"

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher image annotée
    st.image(
        img_rgb,
        caption="Résultat de la détection",
        use_column_width=True
    )

    # Détails
    st.subheader("Détails des détections")
    for pred in [p for p in result.get("predictions", []) if p["confidence"] >= seuil_confiance]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        st.write(
            f"Classe: {pred['class']}, "
            f"Confiance: {pred['confidence']:.2f}, "
            f"Position: ({x1}, {y1})-({x2}, {y2})"
        )

    os.unlink(tmp_path)
