import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

IMG_SIZE = (224, 224)

DAMAGE_CLASSES = [
    "Dent",
    "Scratch",
    "Broken"
]

SEVERITY_CLASSES = [
    "Minor",
    "Moderate",
    "Severe"
]


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "vehicle_damage_resnet50.keras"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚗 Vehicle Damage Detection")

st.write(
    "Upload a vehicle image to detect the damage type "
    "and estimate its severity."
)

st.info(
    "⚠️ This system is an AI-based prototype. "
    "Predictions should be manually verified, especially "
    "for low-confidence results."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Vehicle Image",
        width="stretch"
    )


    # --------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------

    if st.button("🔍 Detect Damage"):

        # ----------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------

        image_resized = image.resize(IMG_SIZE)

        image_array = np.array(
            image_resized,
            dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------

        with st.spinner("Analyzing vehicle..."):

            predictions = model.predict(
                image_array,
                verbose=0
            )


        damage_prediction = predictions[0][0]
        severity_prediction = predictions[1][0]


        # ----------------------------------------------
        # GET PREDICTIONS
        # ----------------------------------------------

        damage_index = np.argmax(
            damage_prediction
        )

        severity_index = np.argmax(
            severity_prediction
        )


        damage = DAMAGE_CLASSES[
            damage_index
        ]

        severity = SEVERITY_CLASSES[
            severity_index
        ]


        damage_confidence = (
            float(damage_prediction[damage_index])
            * 100
        )

        severity_confidence = (
            float(severity_prediction[severity_index])
            * 100
        )


        # ----------------------------------------------
        # RESULTS
        # ----------------------------------------------

        st.success("Analysis Complete!")

        st.subheader("Detection Results")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Damage Type",
                damage
            )

            st.write(
                f"Confidence: **{damage_confidence:.2f}%**"
            )


        with col2:

            st.metric(
                "Severity",
                severity
            )

            st.write(
                f"Confidence: **{severity_confidence:.2f}%**"
            )


        # ----------------------------------------------
        # CONFIDENCE WARNING
        # ----------------------------------------------

        if (
            damage_confidence < 70
            or severity_confidence < 70
        ):

            st.warning(
                "⚠️ One or more predictions have "
                "moderate/low confidence. "
                "Manual inspection is recommended."
            )

        elif (
            damage_confidence < 80
            or severity_confidence < 80
        ):

            st.info(
                "ℹ️ Prediction confidence is moderate. "
                "Manual verification is recommended."
            )

        else:

            st.success(
                "✅ The model has high confidence "
                "in this prediction."
            )


        # ----------------------------------------------
        # PROBABILITY DETAILS
        # ----------------------------------------------

        st.subheader("📊 Prediction Probabilities")


        st.write("Damage Type")

        for i, class_name in enumerate(DAMAGE_CLASSES):

            st.write(
                f"{class_name}: "
                f"{damage_prediction[i] * 100:.2f}%"
            )

            st.progress(
                float(damage_prediction[i])
            )


        st.write("Severity")

        for i, class_name in enumerate(SEVERITY_CLASSES):

            st.write(
                f"{class_name}: "
                f"{severity_prediction[i] * 100:.2f}%"
            )

            st.progress(
                float(severity_prediction[i])
            )


        # ----------------------------------------------
        # FINAL ASSESSMENT
        # ----------------------------------------------

        st.subheader("📋 Assessment")

        st.write(
            f"The model predicts **{damage.lower()}** "
            f"damage with an estimated "
            f"**{severity.lower()}** severity."
        )

        st.caption(
            "Prediction generated using a ResNet50 "
            "transfer-learning model."
        )