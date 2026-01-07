# Modi Script-Text-Analyzer
Deep learning–based OCR system for converting Modi Lipi script images to Devanagari text using a CRNN (CNN + BiLSTM) trained on a custom synthetic dataset, with optional multilingual translation via Gemini API and Flask-based web deployment.

# Modi Script Text Analyzer
Modi Lipi to Devanagari OCR with Multilingual Translation
📌 Overview

Script Text Analyzer is a deep learning–based Optical Character Recognition (OCR) system designed to convert Modi Lipi script images into Devanagari text, with optional translation into multiple languages using the Gemini API.
The project aims to support the digitization and accessibility of historical Indian scripts using modern AI techniques.

The system uses a CRNN (Convolutional Recurrent Neural Network) architecture trained on a custom-generated dataset and is deployed via a Flask-based web application.

# ✨ Key Features

OCR for Modi Lipi → Devanagari text

Custom CRNN (CNN + BiLSTM) model trained using CTC Loss

Fully custom dataset generation pipeline

Unicode-aware character mapping and normalization

Gemini API integration for multilingual translation

Flask-based web interface for image upload and inference

Modular and extensible project structure

# 🧠 Model Architecture

The OCR engine is based on a CRNN architecture, consisting of:

CNN layers for visual feature extraction from text images

Bidirectional LSTM layers for sequence modeling

CTC Loss for alignment-free sequence prediction

This architecture allows the model to recognize text sequences of variable length without explicit character segmentation.

# 📂 Custom Dataset Generation

Since real annotated Modi Lipi datasets are scarce, a custom synthetic dataset was created using an automated pipeline.

Dataset Generation Workflow:

Input Text
Marathi / Devanagari text files are used as the source.

Text Conversion
Text is converted to Modi Lipi Unicode using a comprehensive rule-based mapping that supports:

Vowels and consonants

Matras (vowel signs)

Conjuncts and ligatures

Punctuation and numerals

Image Rendering
The converted Modi Lipi text is rendered into synthetic Modi Lipi images using a Modi-compatible font.

Ground Truth Creation
The original Devanagari text is stored as the ground truth label.

Each dataset sample consists of:

A Modi Lipi image

A corresponding Devanagari ground truth text file

# 📊 Training and Performance

The model is trained using PyTorch with the Adam optimizer

Character vocabulary is generated dynamically from the dataset

Training uses CTC Loss for sequence alignment

Results:

✅ 100% accuracy on the synthetic training dataset

❌ Poor performance on unseen or real-world Modi Lipi images

# ⚠️ Known Limitations

The model currently does not generalize well to unseen or real-world data.
This is due to:

Training on fully synthetic data

Fixed font style, spacing, and layout

Absence of real scanned or handwritten Modi Lipi samples

Overfitting to synthetic visual patterns

This limitation reflects a common challenge in OCR systems trained solely on synthetic datasets.

# 🌍 Translation Support

After OCR extraction, the recognized Devanagari text is sent to the Gemini API, enabling translation into multiple languages.
This makes the system useful for accessibility and multilingual understanding of historical content.

# 🌐 Web Application

The project includes a Flask-based web application that allows users to:

Upload a Modi Lipi image

View extracted Devanagari text

Translate the output into other languages

# 🛠️ Tech Stack

Python,
PyTorch,
CNN + BiLSTM (CRNN),
CTC Loss,
Flask,
Gemini API,
PIL / OpenCV,
Unicode Text Processing.

# 📁 Project Structure
Script-Text-Analyzer/
│
├── app.py                  # Flask application
├── model.py                # CRNN model definition
├── dataset.py              # Custom OCR dataset and preprocessing
├── train.py                # Training script
├── prepare_dataset.py      # Dataset preparation utilities
├── data-generate.py        # Custom dataset generation script
├── char_map.json           # Character to index mapping
├── templates/
│   └── index.html          # Web UI
├── train.txt
├── validation.txt
└── test.txt

# 🚀 Future Improvements

Add real scanned and handwritten Modi Lipi documents

Apply data augmentation (noise, blur, distortion)

Train with multiple fonts and layouts

Improve robustness for degraded manuscripts

Extend support to other historical scripts

# ▶️ How to Run the Project :

1️⃣ Clone the Repository
git clone https://github.com/AdityaShegokar01/Script-Text-Analyzer.git
cd Script-Text-Analyzer

2️⃣ Run the Web Application

python app.py


3️⃣ Verify the Model File

Ensure that the trained model is present in the models/ folder

4️⃣ Upload Modi Lipi Image

Upload a Modi Lipi image from the Dataset/ folder

5️⃣ Enable Translation 

To translate the extracted Devanagari text into other languages:

Create a file named api_key.txt in the project root

Paste your Gemini API key inside the file

Save the file and restart the application

api_key.txt
└── YOUR_GEMINI_API_KEY

Once added, the app will enable multilingual translation.

#  Dependencies

🔹 Required Python 

⚙️ Install Dependencies

Run the following command in the project directory:

pip install -r requirements.txt

🔹 Python Libraries:

torch,
torchvision,
flask,
pillow,
numpy,
opencv-python,
unicodedata2,
google-generativeai,
pandas.
