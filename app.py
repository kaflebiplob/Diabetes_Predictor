import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered"
)


@st.cache_resource
def load_artifact():
    return joblib.load("./models/diabetes_model.pkl")


artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]
feature_order = artifact["features"]

st.title("🩺 Diabetes Risk Screening")
st.write(
    "Enter a patient's details to get a screening result from a machine learning model "
    "trained on 96,000 patient records."
)
st.warning(
    "This is a screening demo, **not a medical diagnosis**. "
    "Always confirm results with a healthcare professional."
)


with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input(
            "Age (years)", min_value=1, max_value=100, value=45, step=1
        )
        bmi = st.number_input(
            "BMI(Body mass Index)", min_value=10.0, max_value=100.0, value=27.0, step=0.1
        )
        smoking_history = st.selectbox(
            "Smoking history",
            ["never", "No Info", "former", "current", "not current", "ever"],
        )

    with col2:
        hba1c = st.number_input(
            "HbA1c level (%)", min_value=3.0, max_value=10.0, value=5.5, step=0.1
        )
        glucose = st.number_input(
            "Blood glucose level (mg/dL)",
            min_value=50,
            max_value=400,
            value=120,
            step=1,
        )
        hypertension = st.selectbox(
            "Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        heart_disease = st.selectbox(
            "Heart disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button("Check risk", use_container_width=True)

if submitted:
    patient = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": float(age),
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": float(bmi),
                "HbA1c_level": float(hba1c),
                "blood_glucose_level": int(glucose),
            }
        ]
    )[feature_order]

    score = float(model.predict_proba(patient)[0, 1])
    flagged = score >= threshold

    st.divider()
    if flagged:
        st.error("###  Higher risk: follow-up testing recommended")
    else:
        st.success("###  Lower risk")

    st.metric("Model risk score", f"{score:.1%}")
    st.progress(min(score, 1.0))
    st.caption(
        f"A patient is flagged when the score is at or above {threshold:.1%}. "
        "This threshold is set low on purpose so the model catches most diabetic "
        "patients (about 85%). The trade-off is that roughly 1 in 3 flagged patients "
        "is a false alarm. The score is a ranking value, not an exact probability."
    )

with st.expander("About this model"):
    st.markdown("""
- **Model:** HistGradientBoosting inside a scikit-learn pipeline
- **Test performance:** ROC-AUC 0.979, recall 0.85, precision 0.64
- **Training data:** public Kaggle diabetes dataset (96,146 rows after removing duplicates)
- **Limitation:** HbA1c and blood glucose drive most of the prediction, and the
  data may not match real clinical populations.
""")
