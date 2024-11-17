import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from io import BytesIO
from PIL import Image
import requests

@st.cache_resource
def load_models():
    model = VisionEncoderDecoderModel.from_pretrained('./VIT_small_distilgpt')
    feature_extractor = ViTFeatureExtractor.from_pretrained("WinKawaks/vit-small-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_models()

def generate_caption(image):
    image = image.resize((224, 224)).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=20,
            top_k=1000,
            do_sample=False,
            top_p=0.95,
            num_return_sequences=1
        )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

st.title("Image Caption Generator")
st.write("Upload an image to generate a caption")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Generating caption...")
    caption = generate_caption(image)
    st.write("**Generated Caption:**", caption)

st.write("---")
st.write("Or, provide an image URL:")

image_url = st.text_input("Enter Image URL")
if st.button("Generate Caption from URL"):
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("Generating caption...")
            caption = generate_caption(image)
            st.write("**Generated Caption:**", caption)
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")
