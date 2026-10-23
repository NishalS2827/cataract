# Cataract Detection Clinical Decision Support System (CDSS)

An end-to-end Deep Learning Clinical Decision Support System (CDSS) designed for screening and diagnosing cataracts from fundus imagery.

## 🚀 Overview
- **Deep Learning Pipeline:** Custom CNN / Transfer Learning model for cataract classification.
- **Backend:** FastAPI REST service with automated preprocessing (CLAHE, normalization, square-pad resizing) and inference.
- **Frontend:** Responsive, modern clinical dashboard interface for instant fundus image upload, risk assessment, and decision support visualization.

## 🛠️ Project Structure
```text
cataract-cdss/
├── backend/
│   └── app/
│       └── main.py              # FastAPI server & inference endpoint
├── frontend/
│   ├── index.html               # Clinical Web Dashboard
│   ├── app.js                   # Interactive client logic
│   └── style.css                # Visual presentation & layout
├── dataset_preparation.py       # Data split, balancing & prep
├── extract_odir_cataract.py     # ODIR-5K dataset extractor
├── preprocess_demo.py           # Preprocessing pipeline verification
├── train.py                     # Training script
├── train_local.py               # Local training with checkpoints
├── run.py                       # Unified server runner
└── requirements.txt             # Python dependencies
```

## 📦 Getting Started

### 1. Prerequisites
- Python 3.9+
- Pip / Virtual environment

### 2. Installation
```bash
# Clone the repository
git clone <YOUR-REPO-URL>
cd cataract-cdss

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application
```bash
python run.py
```
Open your browser at `http://localhost:8000` to interact with the CDSS dashboard.
