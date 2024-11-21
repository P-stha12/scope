import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BertTokenizer, BertModel
from io import BytesIO
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

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
    
    # embedding = outputs.last_hidden_state
    # return embedding
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


# App title
st.title("Photo Uploader and Search App")

# App description
st.write("""
Upload your photos here. You can upload multiple files at once, and the app will display them.
""")

image_caption = {}
threshold = 0.85

# File uploader
uploaded_files = st.file_uploader(
    "Upload Photos",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

st.write("""Search for images with a description""")
text = st.text_input("Enter Textual description of image")
embedding = generate_embeddings(text, bert_model, bert_tokenizer)
#print(embedding.shape)

def process_image(file, bert_model, bert_tokenizer):
    """Processes an image file to generate a caption and its embedding."""
    img = Image.open(file)
    caption = generate_caption(img)
    embedding = generate_embeddings(caption, bert_model, bert_tokenizer)
    return img, caption, embedding

def display_images_in_grid(images, captions=None, num_columns=3):
    """Displays images in a grid layout using Streamlit."""
    columns = st.columns(num_columns)
    for idx, img in enumerate(images):
        with columns[idx % num_columns]:
            if captions:
                st.image(img, caption=captions[idx], use_container_width=True)
            else:
                st.image(img, use_container_width=True)

def check_similar(text, caption):
    text = clean(text)
    caption = clean(caption)

    for word in text:
        if word not in caption:
            return False
    return True

num_columns = 3

# def similarity(emb1, emb2):
#     sim = -1
#     for i in range(emb1.shape[1]):
#         for j in range(emb1.shape[1]):
#             e1 = emb1[0, i, :].unsqueeze(0)
#             e2 = emb2[0, j, :].unsqueeze(0)
#             s = cosine_similarity(e1, e2)
#             sim = max(sim, s)
#     return sim

# Main logic
if uploaded_files:
    st.write("### Uploaded Photos:")

    # Process all images: generate captions and embeddings
    processed_images = [
        process_image(file, bert_model, bert_tokenizer)
        for file in uploaded_files
    ]
    images, captions, embeddings = zip(*processed_images)  # Unpack results

    if text:
        # Compute cosine similarities for all embeddings with the text embedding
        similarities = [cosine_similarity(embedding, e) for e in embeddings]
        #similarities = [cosine_similarity(embedding, e) for e in embeddings]
        print(similarities)
        matched_images = [
            images[idx] for idx, sim in enumerate(similarities) if (sim > threshold or check_similar(text, captions[idx]))
        ]

        if matched_images:
            display_images_in_grid(matched_images, num_columns=num_columns)
        else:
            st.write("No images match your description.")
    else:
        # Display all images with captions
        display_images_in_grid(images, captions, num_columns=num_columns)


# if uploaded_files:
#     st.write("### Uploaded Photos:")

#     # Define the number of columns for the grid
#     num_columns = 3
#     columns = st.columns(num_columns)
    
#     for idx, file in enumerate(uploaded_files):
#         # Open the uploaded image file
#         img = Image.open(file)
#         caption = generate_caption(img)
#         image_caption[idx] = generate_embeddings(caption, bert_model, bert_tokenizer)

#         if text:
#             print(cosine_similarity(embedding, image_caption[idx]))
#             if cosine_similarity(embedding, image_caption[idx]) > threshold:
#                 with columns[idx % num_columns]:
#                     st.image(img, use_container_width=True)
#         else:
#             with columns[idx % num_columns]:
#                 st.image(img,caption=caption, use_container_width=True)
        
#     if text and not any(cosine_similarity(embedding, image_caption[idx]) > threshold for idx in range(len(uploaded_files))):
#         st.write("No images match your description.")