import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import torch.nn as nn
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_NAME = "VisionTransformer"
MODEL_PATH = "best.pth"

CUB_DATASET_PATH = "./CUB_200_2011"
NUM_CLASSES = 200
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(model_name, model_path, num_classes, device):
    """Loads a trained PyTorch model."""
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == "VisionTransformer":
        import timm  # Make sure it's installed (pip install timm)
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported for this app.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() 
    return model

@st.cache_data
def load_class_names(dataset_path):
    """Loads class names from the CUB dataset."""
    classes_txt_path = os.path.join(dataset_path, 'classes.txt')
    df_classes = pd.read_csv(classes_txt_path, sep=' ', names=['class_id', 'class_name'])
    class_names = [name.split('.')[1].replace('_', ' ') for name in df_classes['class_name']]
    return class_names

def preprocess_for_model(pil_image, image_size):
    """Applies transformations to the image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(pil_image).unsqueeze(0)

def preprocess_for_viz(pil_image, image_size):
    """Prepares the image for Grad-CAM visualization."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor() 
    ])
    rgb_img = np.array(pil_image.resize((image_size, image_size))) / 255.0
    return np.float32(rgb_img)

st.set_page_config(page_title="Explainable Bird Identifier", layout="wide")

try:
    model = load_model(MODEL_NAME, MODEL_PATH, NUM_CLASSES, DEVICE)
    class_names = load_class_names(CUB_DATASET_PATH)
except FileNotFoundError:
    st.error(f"Error: Model file not found at '{MODEL_PATH}'.")
    st.stop()

def reshape_transform(tensor, height=14, width=14):
    """
    Reshapes the output from the ViT layer (sequence of tokens) into a 2D feature map.
    """
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

st.title("🐦 Bird Identifier (XAI)")
uploaded_file = st.file_uploader("Select an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Classify'):
        with st.spinner('Analyzing...'):
            input_tensor = preprocess_for_model(pil_image, IMAGE_SIZE).to(DEVICE)
            viz_tensor = preprocess_for_viz(pil_image, IMAGE_SIZE)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
            
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            best_prediction_idx = top5_catid[0][0].item()

            cam_kwargs = {}
            if MODEL_NAME == "ResNet50":
                target_layer = [model.layer4[-1]]
            elif MODEL_NAME == "VisionTransformer":
                target_layer = [model.blocks[-1].norm1]
                cam_kwargs['reshape_transform'] = reshape_transform
            else:
                st.error(f"Grad-CAM not configured for model type: {MODEL_NAME}")
                st.stop()
            
            targets = [ClassifierOutputTarget(best_prediction_idx)]
            
            with GradCAM(model=model, target_layers=target_layer, **cam_kwargs) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            cam_image = show_cam_on_image(viz_tensor, grayscale_cam, use_rgb=True)

        with col2:
            st.image(cam_image, caption="Model Attention Map (Grad-CAM)", use_column_width=True)
        
        st.subheader("Identification Results:")
        best_name = class_names[best_prediction_idx]
        best_prob = top5_prob[0][0].item()
        st.success(f"**Most likely species: {best_name}** ({best_prob:.2%})")

        st.write("Top 5 predictions:")
        for i in range(top5_prob.size(1)):
            prob = top5_prob[0][i].item()
            name = class_names[top5_catid[0][i]]
            st.write(f"{i+1}. {name}: {prob:.2%}")
