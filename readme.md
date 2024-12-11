# Image Captioning and Search System

This repository contains an implementation of an efficient image captioning and search system, designed to generate textual descriptions for images and facilitate searching through them using textual queries. The system is optimized for edge devices with reduced model size and computational requirements.

## Features

- **Image Captioning**: Automatically generates captions for images using Vision and Text Transformer models.
- **Search Functionality**: Enables searching for images using textual descriptions or keywords.
- **Visualization**: Provides Grad-CAM visualizations to highlight regions of images influencing caption generation.
- **Efficient Models**: Utilizes quantization to reduce model size for edge device compatibility.
- **User-Friendly Interface**: Built using Streamlit for an easy-to-use web interface.

## Methodology

### Datasets

- **MS COCO**: Large-scale dataset with over 330,000 images and detailed captions.
- **Flickr8k**: Smaller dataset with 8,000 images and high-quality captions.

### Image Captioning Model

- **Image Encoders**:
  - DeiT-Tiny (lightweight transformer-based encoder)
  - ViT (Vision Transformer)
- **Text Decoders**:
  - GPT-2 (Generative Pre-trained Transformer)
  - DistilGPT-2 (distilled version for efficiency)
  - Tiny-GPT2 (ultra-lightweight model)

### Techniques

- **Grad-CAM**: Visualizes regions influencing caption generation.
- **Quantization**: Reduces model size and computational requirements using 4-bit integer weights.
- **Search**: Combines sentence embeddings with cosine similarity for accurate image retrieval.

## Results

- **Model Comparisons**:
  - Large models (ViT + GPT-2) achieve higher accuracy with fewer epochs.
  - Smaller models (DeiT + Tiny-GPT2) balance performance and efficiency.
- **Quantization**: Reduces model size from 3.8GB to 250MB with minimal performance loss.

## Web Interface

The frontend is implemented using Streamlit, allowing users to:

1. Upload images and generate captions.
2. Search for images using textual queries.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SIC-AI-2024/image-captioning-and-search.git
   cd image-captioning-and-search
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the model from the following link and extract in the root directory
   ```bash
   https://drive.google.com/file/d/1dcnSydOCKyKq2pDrqXQsmxWs9YvGgXVg/view?usp=sharing
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run final.py
   ```
5. Open the provided URL in your browser to interact with the system.

## Future Work

- Enhance model performance with additional fine-tuning.
- Improve search functionality for shorter queries.
- Explore further optimization for real-time applications.

## Acknowledgments

- [MS COCO Dataset](https://cocodataset.org/)
- [Flickr8k Dataset](http://cs.stanford.edu/people/karpathy/deepimagesent/)
- Hugging Face Transformers Library
