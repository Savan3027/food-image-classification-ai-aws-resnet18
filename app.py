import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

st.set_page_config(page_title="Food AI", layout="wide")

# --- Premium CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
}

section.main > div {
    max-width: 800px;
    margin: auto;
    padding-top: 50px;
}

h1 {
    color: #ffffff;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    margin-bottom: 40px;
}

.stFileUploader {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
}

.stFileUploader label {
    color: white !important;
    font-weight: 600;
}

.file-info {
    color: #e5e7eb;
    font-size: 14px;
    margin-top: 15px;
}

.prediction-title {
    margin-top: 40px;
    color: white;
    font-size: 26px;
    font-weight: 600;
}

.result-label {
    color: #ffffff;
    font-weight: 500;
}

.stProgress > div > div > div > div {
    background-color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Food Classification AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image and get AI prediction</div>", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("food_model.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["ice_cream", "pizza", "sushi"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    st.markdown(f"<div class='file-info'>File Name: {uploaded_file.name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='file-info'>File Size: {round(uploaded_file.size/1024,2)} KB</div>", unsafe_allow_html=True)

    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    st.markdown("<div class='prediction-title'>Prediction Results</div>", unsafe_allow_html=True)

    for i in range(len(class_names)):
        st.markdown(f"<div class='result-label'>{class_names[i]}: {round(probabilities[i].item()*100,2)}%</div>", unsafe_allow_html=True)
        st.progress(int(probabilities[i].item()*100))