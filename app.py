import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import load_model

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Signature Verification",
    layout="wide"
)

st.title("Signature Verification System")
st.markdown(
    "Upload two signature images to verify whether they belong to the same person."
)

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def get_model():
    model = load_model("model/model.pth", device)
    return model

model = get_model()

# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((155, 220)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("L")
    image = transform(image)
    image = image.unsqueeze(0)

    return image.to(device)

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def verify_signatures(img1, img2):

    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    with torch.no_grad():

        logits = model(img1, img2)

        probability = torch.sigmoid(logits).item()

        prediction = probability > 0.5

    return prediction, probability

# --------------------------------------------------
# Upload Images
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader(
        "Upload Signature 1",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    uploaded_file2 = st.file_uploader(
        "Upload Signature 2",
        type=["png", "jpg", "jpeg"]
    )

# --------------------------------------------------
# Display Images
# --------------------------------------------------

if uploaded_file1 and uploaded_file2:

    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image1, caption="Signature 1", use_container_width=True)

    with col2:
        st.image(image2, caption="Signature 2", use_container_width=True)

    st.divider()

    if st.button("Verify Signatures", use_container_width=True):

        with st.spinner("Verifying..."):

            prediction, probability = verify_signatures(image1, image2)

        st.subheader("Verification Result")

            
        if prediction:
            st.error("Forged or Different Signatures")
            confidence = probability
        else:
            st.success("Genuine Match")
            confidence = 1 - probability

        st.metric(
                label="Confidence in Result Shown",
                value=f"{confidence*100:.2f}%"
                )

        st.progress(float(confidence))

        st.write(f"**Probability of Genuine:** `{(1-probability)*100:.2f}%`")
        st.write(f"**Probability of Forged:** `{probability*100:.2f}%`")
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Built with PyTorch + Streamlit | Siamese Neural Network for Offline Signature Verification"
)