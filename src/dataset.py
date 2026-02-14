"""
Dataset and data processing utilities.
"""

import os
import xml.etree.ElementTree as ET

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# =============================================================================
# XML PARSING
# =============================================================================

def parse_xml(xml_path):
    """
    Parse ICDAR XML file.
    
    Returns:
        image_paths, image_sizes, labels, bboxes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_paths, image_sizes, all_labels, all_bboxes = [], [], [], []

    for image in root:
        labels, bboxes = [], []
        
        for tagged in image.findall("taggedRectangles"):
            for bb in tagged:
                text = bb[0].text
                if not text.isalnum():  # Skip non-alphanumeric
                    continue
                if "é" in text.lower() or "ñ" in text.lower():
                    continue
                
                bboxes.append([
                    float(bb.attrib["x"]),
                    float(bb.attrib["y"]),
                    float(bb.attrib["width"]),
                    float(bb.attrib["height"]),
                ])
                labels.append(text.lower())

        image_paths.append(image[0].text)
        image_sizes.append((int(image[1].attrib["x"]), int(image[1].attrib["y"])))
        all_labels.append(labels)
        all_bboxes.append(bboxes)

    return image_paths, image_sizes, all_labels, all_bboxes


# =============================================================================
# YOLO FORMAT CONVERSION
# =============================================================================

def to_yolo_format(image_paths, image_sizes, bboxes):
    """Convert bboxes to YOLO format: [class, x_center, y_center, w, h] normalized."""
    yolo_data = []
    
    for path, (img_w, img_h), boxes in zip(image_paths, image_sizes, bboxes):
        yolo_boxes = []
        for x, y, w, h in boxes:
            yolo_boxes.append([
                0,  # class id
                (x + w/2) / img_w,
                (y + h/2) / img_h,
                w / img_w,
                h / img_h
            ])
        yolo_data.append((path, yolo_boxes))
    
    return yolo_data


def save_yolo_data(yolo_data, split, save_dir, dataset_dir):
    """Save images and labels in YOLO format."""
    import shutil
    
    img_dir = os.path.join(save_dir, split, "images")
    lbl_dir = os.path.join(save_dir, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path, boxes in yolo_data:
        # Copy image
        name = img_path.replace("/", "_")
        shutil.copy(os.path.join(dataset_dir, img_path), os.path.join(img_dir, name))
        
        # Save labels
        label_name = name.replace(".JPG", ".txt").replace(".jpg", ".txt")
        with open(os.path.join(lbl_dir, label_name), "w") as f:
            for box in boxes:
                f.write(" ".join(map(str, box)) + "\n")


# =============================================================================
# CROP TEXT REGIONS
# =============================================================================

def crop_text_regions(dataset_dir, save_dir="cropped_text"):
    """Crop text regions from images and save them."""
    os.makedirs(save_dir, exist_ok=True)
    
    xml_path = os.path.join(dataset_dir, "words.xml")
    paths, _, all_labels, all_bboxes = parse_xml(xml_path)
    
    cropped_paths, labels = [], []
    
    for img_rel, img_labels, bboxes in zip(paths, all_labels, all_bboxes):
        img_path = os.path.join(dataset_dir, img_rel)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        for i, (bbox, label) in enumerate(zip(bboxes, img_labels)):
            x, y, w, h = map(int, bbox)
            x, y = max(0, x), max(0, y)
            crop = img[y:y+h, x:x+w]
            
            if crop.size == 0:
                continue
            
            name = os.path.basename(img_path).replace(".JPG", f"_{i}.jpg")
            save_path = os.path.join(save_dir, name)
            cv2.imwrite(save_path, crop)
            
            cropped_paths.append(save_path)
            labels.append(label)
    
    return cropped_paths, labels


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class TextDataset(Dataset):
    """Dataset for text recognition."""
    
    def __init__(self, image_paths, labels, char_to_idx, max_len, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # Encode label
        label = self.labels[idx]
        encoded = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
        length = len(encoded)
        encoded += [0] * (self.max_len - len(encoded))  # Pad
        
        return img, torch.LongTensor(encoded), torch.tensor(length)


def get_transforms():
    """Get train and val transforms."""
    return {
        "train": transforms.Compose([
            transforms.Resize((100, 420)),
            transforms.Grayscale(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
        "val": transforms.Compose([
            transforms.Resize((100, 420)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    }
