# 🥚 Duck Egg Fertility Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=flat-square&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat-square&logo=sqlite&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=flat-square)

**AI-powered non-destructive fertility detection system for duck eggs**  
*Sistem Deteksi Fertilitas Telur Bebek Berbasis Deep Learning*

[Features](#-features) · [Pipeline](#-ml-pipeline) · [Installation](#-installation) · [Usage](#-usage) · [Project Structure](#-project-structure) · [Results](#-results)

</div>

---

## 📌 Overview

This project implements a complete AI system for detecting the fertility of duck eggs using **candling images** — a non-destructive inspection method where eggs are illuminated from behind to reveal embryo development.

The system is developed as part of a **PhD dissertation** titled:

> *"Pengembangan Model Deteksi Fertilitas Telur Bebek Menggunakan Segmentasi Citra dan Deep Learning dengan Algoritma Adaptive Weight Clustering"*

### 🐣 Why Duck Eggs?

Most existing research focuses on **chicken eggs**. Duck eggs present unique challenges:

| Property | Chicken Eggs | Duck Eggs |
|----------|-------------|-----------|
| Shell thickness | ~0.30 mm | **~0.35–0.40 mm** |
| Shell color | White / light brown | **Opaque greenish** |
| Light transmission | High | **Low** |
| Detection difficulty | Moderate | **High** |

→ Classical methods (Otsu, Watershed) fail on duck eggs. This system uses **deep learning** to handle the complexity.

---

## ✨ Features

- 🔬 **Advanced preprocessing** — CLAHE + Homomorphic Filtering + Bilateral Denoising
- 🧠 **U-Net segmentation** — lightweight deep learning segmentation (PyTorch)
- 🔀 **Hybrid feature extraction** — Classical (GLCM, LBP, Morphology) + Deep embeddings
- ⚖️ **Adaptive Weight Clustering (AWC)** — novel adaptive classification algorithm
- 🌐 **Web interface** — upload images and get instant results
- 📊 **Dashboard** — prediction history, model performance tracking
- 🔄 **Continuous learning** — users can correct predictions and retrain the model
- 📦 **Model versioning** — automatic versioning of retrained models
- 🏷️ **Annotation tool** — built-in web-based tool for creating segmentation masks

---

## 🔬 ML Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Candling image (duck egg, 5–7 days incubated)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │  [1] PREPROCESSING                 │
          │  CLAHE → Homomorphic → Bilateral   │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │  [2] SEGMENTATION (U-Net)          │
          │  Extract embryo region mask        │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │  [3] FEATURE EXTRACTION            │
          │  Classical: GLCM, LBP, Morphology  │
          │  Deep: U-Net bottleneck embeddings │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │  [4] FEATURE FUSION                │
          │  Normalize + Concatenate (~532-dim)│
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │  [5] CLASSIFICATION (AWC)          │
          │  Adaptive Weight Clustering        │
          └─────────────────┬─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  OUTPUT: FERTILE ✅ / INFERTILE ❌  +  Confidence Score          │
└─────────────────────────────────────────────────────────────────┘
```

### 🏆 Novel Contributions

1. **First application** of U-Net + Hybrid Features + AWC for duck egg fertility detection
2. **Hybrid feature fusion** — classical interpretable features + deep abstract embeddings
3. **Adaptive Weight Clustering** — features weighted by discriminative power (vs. fixed weights in K-Means)

---

## 📊 Target Performance

| Metric | Target | Baseline (K-Means) |
|--------|--------|-------------------|
| Accuracy | **> 90%** | ~85% |
| F1-Score | **> 0.90** | ~0.85 |
| ROC-AUC | **> 0.95** | ~0.90 |
| Dice (segmentation) | **> 0.85** | — |
| IoU (segmentation) | **> 0.75** | — |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0+ |
| Image Processing | OpenCV 4.8, scikit-image |
| Classical ML | scikit-learn, scikit-fuzzy |
| Web Backend | Flask 2.3 |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Frontend | HTML/CSS/JS + Bootstrap 5 |
| Demo UI | Streamlit |
| Deployment | Docker |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Git

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/duck-egg-fertility-detection.git
cd duck-egg-fertility-detection
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch, cv2, sklearn; print('✅ All dependencies installed!')"
```

---

## 🚀 Usage

### Option A: Web Application (Recommended)

```bash
# Start Streamlit demo
streamlit run web/streamlit_app/app.py

# Or start Flask backend
python web/api/app.py
```

Then open `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask)

### Option B: Command Line

```bash
# Step 1: Preprocess images
python scripts/01_preprocess_data.py --input data/raw --output data/preprocessed

# Step 2: Train U-Net segmentation
python scripts/03_train_unet.py --config configs/unet_config.yaml --epochs 100

# Step 3: Extract features
python scripts/04_extract_features.py --model models/unet/unet_best.pth

# Step 4: Train AWC classifier
python scripts/05_train_awc.py --features data/features/hybrid_features.npy

# Step 5: Evaluate all models
python scripts/06_evaluate_models.py --test data/splits/test.txt

# Step 6: Run inference on new image
python scripts/07_inference.py --image path/to/egg.jpg
```

### Option C: Python API

```python
from src.preprocessing import DuckEggPreprocessor
from src.segmentation import UNetModel
from src.features import HybridFeatureExtractor
from src.clustering import AdaptiveWeightClustering
import cv2

# Load image
image = cv2.imread("duck_egg.jpg")

# Preprocess
preprocessor = DuckEggPreprocessor()
preprocessed = preprocessor.preprocess(image)

# Segment
unet = UNetModel.load("models/unet/unet_best.pth")
mask = unet.predict(preprocessed)

# Extract features
extractor = HybridFeatureExtractor()
features = extractor.extract_all(preprocessed, mask)

# Classify
awc = AdaptiveWeightClustering.load("models/awc/awc_model.pkl")
prediction = awc.predict(features.reshape(1, -1))[0]

print("Result:", "FERTILE ✅" if prediction == 1 else "INFERTILE ❌")
```

---

## 📁 Project Structure

```
duck-egg-fertility-detection/
│
├── 📁 data/
│   ├── raw/                    # Original candling images (JPG)
│   │   ├── fertile/            # Fertile egg images
│   │   └── infertile/          # Infertile egg images
│   ├── preprocessed/           # After preprocessing pipeline
│   ├── annotations/
│   │   └── masks/              # Binary segmentation masks (PNG)
│   ├── features/               # Saved feature vectors (.npy)
│   │   ├── hybrid_features.npy
│   │   └── labels.npy
│   ├── splits/                 # train.txt / val.txt / test.txt
│   └── uploads/                # Web app uploads
│       ├── pending/
│       ├── processed/
│       └── training_queue/
│
├── 📁 models/
│   ├── unet/                   # U-Net checkpoints (.pth)
│   ├── awc/                    # AWC model (.pkl) + weights (.npy)
│   ├── baselines/              # K-Means, Fuzzy C-Means
│   └── versions/               # Versioned retrained models
│
├── 📁 src/                     # Core ML source code
│   ├── preprocessing/          # CLAHE, Homomorphic, Bilateral
│   ├── segmentation/           # U-Net architecture & training
│   ├── features/               # Classical + deep feature extraction
│   ├── clustering/             # AWC + baseline clustering
│   ├── evaluation/             # Metrics, visualization
│   ├── utils/                  # Config, logging, file utils
│   └── web/                    # Web service layer
│
├── 📁 web/
│   ├── api/                    # Flask API routes
│   ├── frontend/               # HTML templates + static files
│   └── streamlit_app/          # Streamlit multi-page app
│
├── 📁 notebooks/               # Jupyter experiment notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_experiments.ipynb
│   ├── 03_unet_training.ipynb
│   ├── 04_feature_extraction.ipynb
│   ├── 05_awc_implementation.ipynb
│   ├── 06_baselines_comparison.ipynb
│   └── 07_final_results.ipynb
│
├── 📁 scripts/                 # Executable pipeline scripts
│   ├── 01_preprocess_data.py
│   ├── 03_train_unet.py
│   ├── 04_extract_features.py
│   ├── 05_train_awc.py
│   ├── 06_evaluate_models.py
│   └── 07_inference.py
│
├── 📁 configs/                 # YAML hyperparameter configs
│   ├── preprocessing_config.yaml
│   ├── unet_config.yaml
│   └── awc_config.yaml
│
├── 📁 results/
│   ├── figures/                # Plots & visualizations
│   ├── tables/                 # CSV/Excel result tables
│   └── logs/                   # Training logs
│
├── 📁 tests/                   # Unit & integration tests
├── 📁 database/                # SQLite database files
├── 📁 deployment/              # Docker files
├── 📁 docs/                    # Documentation
├── 📁 papers/                  # Reference papers (PDF)
│
├── annotation_tool.html        # 🏷️ Standalone annotation tool
├── requirements.txt
├── requirements-web.txt
├── .gitignore
├── run.bat                     # Windows startup script
└── README.md
```

---

## 🏷️ Annotation Tool

A standalone web-based tool for creating segmentation masks — **no installation required**.

```bash
# Just open in browser:
annotation_tool.html
```

**Features:**
- 🖌️ Brush, Polygon, Fill, Eraser tools
- ↩️ Undo / Redo support
- 🔍 Zoom & Pan
- 💾 Auto-save between images
- 📦 Export all masks as binary PNG (white = embryo, black = background)
- 📋 Progress tracker for batch annotation

**Mask output format:**
```
data/annotations/masks/
├── egg_001_mask.png    # White pixels = embryo area
├── egg_002_mask.png
└── ...
```

---

## 📈 Development Progress

| Phase | Task | Status |
|-------|------|--------|
| 1A | Preprocessing module | ✅ Done |
| — | Annotation tool | ✅ Done |
| — | Segmentation masks | 🔄 In Progress |
| 1B | U-Net segmentation | ⏳ Pending |
| 1C | Feature extraction | ⏳ Pending |
| 1D | AWC classification | ⏳ Pending |
| 1E | Model evaluation | ⏳ Pending |
| 2 | Flask backend | ⏳ Pending |
| 3 | Web frontend | ⏳ Pending |
| 4 | Integration & testing | ⏳ Pending |
| 5 | Docker deployment | ⏳ Pending |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_preprocessing.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## ⚙️ Configuration

All hyperparameters are stored in `configs/` — no hardcoded values in code.

```yaml
# configs/preprocessing_config.yaml
resize:
  target_size: [256, 256]

clahe:
  clip_limit: 2.0
  tile_grid_size: [8, 8]

homomorphic:
  gamma_low: 0.5
  gamma_high: 1.5
  cutoff: 30

bilateral:
  d: 9
  sigma_color: 75
  sigma_space: 75
```

```yaml
# configs/unet_config.yaml
model:
  input_shape: [256, 256, 1]
  base_filters: 32
  depth: 4

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
```

---

## 📚 References

| # | Paper | Used For |
|---|-------|----------|
| 1 | Ronneberger et al. (2015) — U-Net | Segmentation architecture |
| 2 | Adamyan et al. (2020) — AWC | Classification algorithm |
| 3 | Suhirman et al. (2022) — Otsu for chicken egg detection | Preprocessing reference, baseline |
| 4 | Saifullah (2019) — Watershed + CLAHE-HE | CLAHE-HE hybrid method |

> **Note:** References 3 & 4 use chicken eggs. This project extends the work to **duck eggs** — a key novelty of the dissertation.

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t duck-egg-fertility .

# Run container
docker-compose up -d

# Access web app
open http://localhost:5000
```

---



## 👥 Team

| Role | Person |
|------|--------|
| **Researcher / Supervisor** | Dosen Pembimbing (PhD Candidate) |
| **Developer** | Student Developer |
| **Institution** | Universitas — Program Pascasarjana |

---

## 🙏 Acknowledgments

- Dataset provided by the research supervisor
- U-Net architecture based on Ronneberger et al. (2015)
- AWC algorithm adapted from Adamyan et al. (2020)
- Preprocessing pipeline inspired by Suhirman et al. (2022) and Saifullah (2019)

---

<div align="center">
  <sub>Built with ❤️ for PhD dissertation research on duck egg fertility detection</sub>
</div>