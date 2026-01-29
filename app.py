REMEDIES = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial disease causing dark, water-soaked spots on pepper leaves.",
        "remedies": [
            "Remove infected leaves",
            "Avoid overhead watering",
            "Use copper-based bactericides"
        ]
    },
    "Pepper__bell___healthy": {
        "description": "Healthy pepper plant",
        "remedies": ["No action needed"]
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing concentric dark spots on leaves.",
        "remedies": [
            "Remove infected leaves",
            "Apply recommended fungicides",
            "Practice crop rotation"
        ]
    },
    "Potato___Late_blight": {
        "description": "Serious fungal disease affecting leaves and tubers.",
        "remedies": [
            "Destroy infected plants",
            "Avoid excess moisture",
            "Use certified disease-free seeds"
        ]
    },
    "Potato___healthy": {
        "description": "Healthy potato plant",
        "remedies": ["No action needed"]
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial disease causing small dark leaf spots.",
        "remedies": [
            "Remove infected leaves",
            "Apply copper-based sprays",
            "Avoid overhead irrigation"
        ]
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease with dark concentric rings.",
        "remedies": [
            "Apply fungicide",
            "Remove affected leaves",
            "Mulch soil to prevent splash"
        ]
    },
    "Tomato_Late_blight": {
        "description": "Severe fungal disease causing brown lesions.",
        "remedies": [
            "Remove infected plants immediately",
            "Use fungicides",
            "Avoid wet foliage"
        ]
    },
    "Tomato_Leaf_Mold": {
        "description": "Yellow spots and mold growth under leaves.",
        "remedies": [
            "Improve ventilation",
            "Reduce humidity",
            "Apply fungicides"
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease with gray spots and dark borders.",
        "remedies": [
            "Remove infected leaves",
            "Apply fungicide",
            "Avoid overhead watering"
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Pest infestation causing yellow speckling.",
        "remedies": [
            "Use neem oil",
            "Increase humidity",
            "Introduce natural predators"
        ]
    },
    "Tomato__Target_Spot": {
        "description": "Brown spots with concentric rings.",
        "remedies": [
            "Remove affected leaves",
            "Apply fungicide",
            "Improve airflow"
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease causing yellowing and curling.",
        "remedies": [
            "Remove infected plants",
            "Control whiteflies",
            "Use resistant varieties"
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Virus causing mosaic patterns on leaves.",
        "remedies": [
            "Destroy infected plants",
            "Disinfect tools",
            "Avoid handling wet plants"
        ]
    },
    "Tomato_healthy": {
        "description": "Healthy tomato plant",
        "remedies": ["No action needed"]
    }
}


import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Smart Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------- HEADER ---------------- #
st.markdown(
    """
    <h1 style="text-align: center;">🌱 Smart Crop Disease Detection System</h1>
    <p style="text-align: center; color: gray;">
    AI-based leaf image analysis for early crop disease identification
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- UPLOAD SECTION ---------------- #
st.subheader("📤 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Upload a clear image of a single crop leaf (Tomato / Potato / Pepper)",
    type=["jpg", "jpeg", "png"]
)

st.caption(
    "ℹ️ For best results, upload a high-resolution image of a single leaf "
    "with a plain background. Blurry or low-quality images may affect accuracy."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    st.markdown("---")
    st.subheader("🔍 Analysis Result")

    with st.spinner("Analyzing image using AI model..."):
        results = model(
            img_path,
            conf=0.10,
            imgsz=640
        )

    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        st.success("✅ No clear disease detected in the uploaded image.")
        st.info(
            "Try uploading a clearer image with a single leaf and plain background."
        )
    else:
        # ---------------- DRAW BOUNDING BOXES ---------------- #
        annotated_img = results[0].plot()  # BGR image with boxes
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(
            annotated_img,
            caption="Detected Disease Regions",
            use_column_width=True
        )

        # Take highest-confidence detection
        best_box = detections[0]
        class_id = int(best_box.cls)
        confidence = float(best_box.conf)
        label = model.names[class_id]

        st.error("⚠️ Crop Disease Detected")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Disease", label.replace("_", " "))
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")

        # ---------------- REMEDIES SECTION ---------------- #
        info = REMEDIES.get(label)

        if info:
            st.subheader("🩺 Disease Information")
            st.write(info["description"])

            st.subheader("🌿 Recommended Actions")
            for i, remedy in enumerate(info["remedies"], 1):
                st.write(f"{i}. {remedy}")
        else:
            st.info(
                "No specific remedies found for this disease. "
                "Please consult an agricultural expert."
            )

    st.markdown("---")

# ---------------- DISCLAIMER ---------------- #
st.caption(
    "⚠️ Disclaimer: This system is a prototype developed for academic and social "
    "awareness purposes. Predictions depend on image quality and should be "
    "verified by agricultural experts."
)
st.caption(
    "Model used: Open-source YOLO-based pretrained plant disease detection model."
)
