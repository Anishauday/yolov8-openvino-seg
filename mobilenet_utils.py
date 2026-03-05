#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3 classification utilities for OpenVINO inference.
ImageNet preprocessing, postprocessing, and visualization.
"""

import json
import urllib.request
import cv2
import numpy as np

# ImageNet normalization (same as torchvision)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/tutorials/main/_static/imagenet_class_index.json"
)


def get_imagenet_labels():
    """Load ImageNet 1000 class labels (PyTorch index order). Cached after first load."""
    if not hasattr(get_imagenet_labels, "_labels"):
        try:
            with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as r:
                data = json.loads(r.read().decode())
            get_imagenet_labels._labels = [data[str(i)][1] for i in range(1000)]
        except Exception:
            raise RuntimeError(
                "Failed to load ImageNet labels. Ensure internet access or provide labels locally."
            )
    return get_imagenet_labels._labels


def preproc(img, input_size=(224, 224)):
    """
    ImageNet preprocessing: resize, BGR->RGB, normalize with mean/std.
    Returns tensor [1, 3, H, W] in NCHW order.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    return np.expand_dims(img, axis=0)


def softmax(x):
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def top_k(logits, class_names, k=5):
    """
    Get top-k class indices and probabilities.
    class_names: list of 1000 strings (ImageNet labels)
    Returns list of (class_index, class_name, probability).
    """
    probs = softmax(logits.flatten())
    top_indices = np.argsort(probs)[::-1][:k]
    return [
        (int(i), class_names[i], float(probs[i]))
        for i in top_indices
    ]


def vis_classification(img, top_label, top_conf, font_scale=1.2, thickness=2):
    """
    Draw the top classification result on the image.
    Uses bottom-left placement so the label stays visible when images are cropped or scaled.
    top_label: str (primary class name)
    top_conf: float (primary confidence 0-100)
    """
    img = img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    text = f"{top_label} {top_conf:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 12
    # Bottom-left: avoid top crop in Gradio/notebook displays
    x0, y0 = padding, h - padding
    y_text = y0 - padding
    y_box_top = y_text - th - padding
    # Clamp to image bounds
    y_box_top = max(0, y_box_top)
    x1 = min(w, x0 + tw + padding * 2)
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x0, y_box_top),
        (x1, y0),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (x0 + padding, y_text), font, font_scale, (255, 255, 255), thickness)
    return img
