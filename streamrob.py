import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import io

st.set_page_config(page_title="Détection – Encre Ferrogallique", layout="wide")

st.title("🧪 Détection d’encre ferrogallique – Roboflow")
st.write("Analyse automatique de manuscrits anciens")

# --- Roboflow client ---
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)

# --- Upload image ---
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Run inference directly from bytes
    results = client.infer(
        uploaded_file.getvalue(),
        model_id="encre-ferrogallique-2-wy9md/2"
    )

    # Draw boxes manually (without cv2)
    annotated = image.copy()
    pixels = annotated.load()

    for obj in results["predictions"]:
        x0 = int(obj["x"] - obj["width"] / 2)
        y0 = int(obj["y"] - obj["height"] / 2)
        x1 = int(obj["x"] + obj["width"] / 2)
        y1 = int(obj["y"] + obj["height"] / 2)

        # draw a simple red bounding box
        for x in range(x0, x1):
            pixels[x, y0] = (255, 50, 50)
            pixels[x, y1] = (255, 50, 50)
        for y in range(y0, y1):
            pixels[x0, y] = (255, 50, 50)
            pixels[x1, y] = (255, 50, 50)

    # --- Layout side-by-side ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Image annotée")
        st.image(annotated, use_container_width=True)
