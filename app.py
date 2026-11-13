from pathlib import Path

import streamlit as st
from fastai.vision.all import load_learner, PILImage

st.set_page_config(page_title="Bear Classifier", page_icon="🐻")


# Load the model once and reuse it for every visitor (much faster).
@st.cache_resource
def get_model():
    return load_learner(Path(__file__).parent / "export.pkl")


learn = get_model()

st.title("🐻 Bear Classifier")
st.write(
    "Upload a photo and I'll guess whether it's a **grizzly**, **black**, or **teddy** bear. "
    "Built by fine-tuning a ResNet-18 with fastai."
)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    img = PILImage.create(uploaded.getvalue())
    st.image(img, width=350)

    pred, idx, probs = learn.predict(img)
    st.subheader(f"Prediction: {pred} bear ({float(probs[idx]):.0%} confident)")

    # Show every class's probability, highest first.
    for label, p in sorted(zip(learn.dls.vocab, probs), key=lambda x: -float(x[1])):
        st.write(f"{label}: {float(p):.1%}")
        st.progress(float(p))
