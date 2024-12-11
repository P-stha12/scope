import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BertTokenizer, BertModel
from io import BytesIO
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

# Cache models
@st.cache_resource
def load_models():
    model = VisionEncoderDecoderModel.from_pretrained('./VIT_small_distilgpt')
    feature_extractor = ViTFeatureExtractor.from_pretrained("WinKawaks/vit-small-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    return model, feature_extractor, tokenizer

@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, feature_extractor, tokenizer = load_models()
bert_model, bert_tokenizer = load_bert()

stop_words = set(stopwords.words('english'))

def clean(text):
    filtered_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return filtered_text

def generate_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
    return embedding

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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Photo Uploader and Search", "Caption Generator"])

if page == "Photo Uploader and Search":
    st.title("Photo Uploader and Search App")
    st.write("Upload your photos here. You can upload multiple files at once, and the app will display them.")
    
    threshold = 0.85

    uploaded_files = st.file_uploader("Upload Photos", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    st.write("Search for images with a description")
    text = st.text_input("Enter Textual description of image")
    embedding = generate_embeddings(text, bert_model, bert_tokenizer) if text else None

    def process_image(file):
        img = Image.open(file)
        caption = generate_caption(img)
        embedding = generate_embeddings(caption, bert_model, bert_tokenizer)
        return img, caption, embedding

    if uploaded_files:
        st.write("### Uploaded Photos:")
        processed_images = [process_image(file) for file in uploaded_files]
        images, captions, embeddings = zip(*processed_images)

        if text:
            similarities = [cosine_similarity(embedding, e) for e in embeddings]
            matched_images = [images[idx] for idx, sim in enumerate(similarities) if sim > threshold]
            
            if matched_images:
                for img in matched_images:
                    st.image(img, use_container_width=True)
            else:
                st.write("No images match your description.")
        else:
            for img, caption in zip(images, captions):
                st.image(img, caption=caption, use_container_width=True)

elif page == "Caption Generator":
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
