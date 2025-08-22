import streamlit as st
import joblib
import sys, os

# Ensure Python can find preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Path to model
MODEL_PATH = "models/toxicity_model.joblib"

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Comment Toxicity Detection", layout="wide")
st.title("📝 Comment Toxicity Detection")
st.write("Enter a comment below to check if it's toxic or not.")

# ---- Single Comment Prediction ----
st.header("🔎 Single Comment")
comment = st.text_area("Type a comment:")

if st.button("Predict"):
    if not comment.strip():
        st.warning("Please enter a comment.")
    else:
        pred = model.predict([comment])[0]
        prob = model.predict_proba([comment])[0, 1]
        st.subheader(f"Prediction: {'Toxic' if pred == 1 else 'Not Toxic'}")
        st.metric("Toxicity Probability", f"{prob*100:.2f}%")
