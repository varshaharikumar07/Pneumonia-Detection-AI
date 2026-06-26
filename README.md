#  Pneumonia Detection AI

A Streamlit web app for pneumonia detection using a fine‑tuned ResNet50 model on chest X‑ray images.

##  Requirements

- Python 3.12
- Virtual environment (recommended)
- Dependencies listed in `requirements.txt`

### Install dependencies
```bash
pip install -r requirements.txt

Pneumonia Proj/
│
├── app.py                  # Streamlit app
├── pneumonia_model_finetuned.keras   # Saved fine‑tuned model
├── requirements.txt        # Locked dependencies
└── README.md               # Documentation

Usage:
1  Activate your virtual environment:
   in the terminal ; bash
   .\venv\Scripts\activate

2  Run the Streamlit app:
   in the terminal ;bash
   streamlit run app.py

3  Upload a chest X‑ray image (.jpg, .jpeg,.  png) to get predictions.

Model Information: 
Architecture: Fine‑tuned ResNet50
Input size: 224×224
Threshold: 0.7 (pneumonia probability cutoff)
Performance: ~81% accuracy | ~96% recall

Model is cached (@st.cache_resource) for faster performance.
TensorFlow logs may show oneDNN optimization messages — these are informational, not errors.
To suppress logs, run:
bash
set TF_CPP_MIN_LOG_LEVEL=3