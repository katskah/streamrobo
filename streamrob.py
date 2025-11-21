import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(page_title="Détection Roboflow", layout="wide")
st.title("🧪 Détection avec seuil de confiance")

# --- Upload de l'image ---
uploaded_file = st.file_uploader("Choisir une image :", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # --- Slider pour seuil de confiance ---
confidence_threshold = st.slider(
    "Seuil de confiance", 0.0, 1.0, 0.5, 0.01
)
st.write(f"Seuil choisi : {confidence_threshold:.2f}")

    # --- Fake résultat pour exemple ---
result = {
    "predictions": [
        {"x": 150, "y": 100, "width": 100, "height": 50, "confidence": 0.85},
        {"x": 300, "y": 200, "width": 80, "height": 80, "confidence": 0.6},
        {"x": 200, "y": 250, "width": 50, "height": 50, "confidence": 0.3},
    ]
}

    # --- Annotate image selon seuil ---
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
font = ImageFont.load_default()

for pred in result["predictions"]:
    conf = pred["confidence"]
    if conf < confidence_threshold:
        continue  # Ignorer si sous le seuil
    x0 = pred["x"] - pred["width"]/2
    y0 = pred["y"] - pred["height"]/2
    x1 = pred["x"] + pred["width"]/2
    y1 = pred["y"] + pred["height"]/2

    # Boîte de détection
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    draw.text((x0, y0 - 10), f"{conf:.2f}", fill="green", font=font)

# --- Affichage côte à côte ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image originale")
    st.image(image)
with col2:
    st.subheader("Image annotée")
    st.image(annotated)
