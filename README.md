# OCR Pipeline: Text Detection & Recognition

A simple OCR system using YOLO (detection) + CRNN (recognition).

## Structure

```
ocr-project/
├── src/
│   ├── model.py      # CRNN architecture
│   ├── dataset.py    # Data loading & processing
│   ├── train.py      # Training functions
│   └── utils.py      # Helpers (encoding, metrics, visualization)
│
└── notebooks/
    ├── 01_detection.ipynb    # Train YOLO
    ├── 02_recognition.ipynb  # Train CRNN
    └── 03_evaluate.ipynb     # Compare models
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download dataset
gdown 15bTQg7W2NXg68ERJDSpY1t7EtL8eJ7az
unzip icdar2003.zip -d datasets/

# Run notebooks in order: 01 -> 02 -> 03
jupyter notebook notebooks/
```

## Models Compared

| Model | Trained by Us? | Description |
|-------|----------------|-------------|
| YOLO + CRNN (Ours) | Yes | Custom trained on ICDAR2003 |
| TrOCR | No | Microsoft's pre-trained transformer OCR |
| EasyOCR | No | Pre-trained end-to-end OCR |

Note: Only YOLO + CRNN is trained in this project. TrOCR and EasyOCR are used as pre-trained baselines for comparison only.

## Usage

```python
from src.model import CRNN
from src.utils import OCRPipeline

# Load trained model
crnn = CRNN(vocab_size=37)
crnn.load_state_dict(torch.load('crnn.pt'))

# Create pipeline
ocr = OCRPipeline('runs/detect/train/weights/best.pt', crnn)

# Predict
results = ocr.predict('image.jpg')
for r in results:
    print(f"{r['text']} ({r['confidence']:.2f})")
```

