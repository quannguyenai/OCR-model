"""
Utility functions: encoding, decoding, metrics, visualization.
"""

import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein
from PIL import Image
from torchvision import transforms


# =============================================================================
# CHARACTER ENCODING
# =============================================================================

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
BLANK = "-"

# Create mappings (index 0 = padding, so start from 1)
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(CHARS)


def encode(text, max_len):
    """Encode text to indices."""
    encoded = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    length = len(encoded)
    encoded += [0] * (max_len - len(encoded))
    return torch.LongTensor(encoded), length


def decode(indices):
    """Decode CTC output (with duplicate removal)."""
    result = []
    prev = None
    
    for idx in indices:
        idx = idx.item() if hasattr(idx, 'item') else idx
        if idx != 0:  # Not padding
            char = IDX_TO_CHAR.get(idx, '')
            if char != BLANK and char != prev:
                result.append(char)
            prev = char
    
    return ''.join(result)


def decode_batch(batch_indices):
    """Decode a batch of CTC outputs."""
    return [decode(seq) for seq in batch_indices]


# =============================================================================
# METRICS
# =============================================================================

def char_accuracy(pred, gt):
    """Character accuracy using Levenshtein distance."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    dist = Levenshtein.distance(pred.lower(), gt.lower())
    return max(0, 1 - dist / max(len(pred), len(gt)))


def word_accuracy(pred, gt):
    """Exact word match."""
    return 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0


def calculate_metrics(preds, gts):
    """Calculate average metrics."""
    char_accs = [char_accuracy(p, g) for p, g in zip(preds, gts)]
    word_accs = [word_accuracy(p, g) for p, g in zip(preds, gts)]
    return {
        "char_acc": np.mean(char_accs) * 100,
        "word_acc": np.mean(word_accs) * 100,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_losses(train_losses, val_losses):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.plot(val_losses, label='Val', color='orange')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def show_predictions(model, dataset, device, n=10):
    """Show sample predictions."""
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), n, replace=False)
    
    for ax, idx in zip(axes, indices):
        img, label, _ = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred = decode(output.permute(1, 0, 2).argmax(2)[0])
        
        # Get ground truth
        gt = decode(label)
        
        # Show
        ax.imshow(img.squeeze(), cmap='gray')
        color = 'green' if pred == gt else 'red'
        ax.set_title(f"GT: {gt}\nPred: {pred}", color=color, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

class OCRPipeline:
    """Simple YOLO + CRNN pipeline."""
    
    def __init__(self, yolo_path, crnn_model, device="cuda"):
        from ultralytics import YOLO
        
        self.yolo = YOLO(yolo_path)
        self.crnn = crnn_model.to(device).eval()
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.Resize((100, 420)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    
    def predict(self, image_path, conf=0.3):
        """Run OCR on image."""
        # Detect text regions
        results = self.yolo(image_path, verbose=False, conf=conf)
        detections = json.loads(results[0].to_json())
        
        img = cv2.imread(image_path)
        predictions = []
        
        for det in detections:
            box = det['box']
            x1, y1 = int(box['x1']), int(box['y1'])
            x2, y2 = int(box['x2']), int(box['y2'])
            
            # Crop and recognize
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.crnn(tensor)
                text = decode(output.permute(1, 0, 2).argmax(2)[0])
            
            predictions.append({
                'bbox': (x1, y1, x2-x1, y2-y1),
                'text': text,
                'confidence': det['confidence']
            })
        
        return predictions
