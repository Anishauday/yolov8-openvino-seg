#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone YOLOX preprocessing and postprocessing utilities.
Compatible with OpenVINO inference - no full YOLOX package required.
"""

import cv2
import numpy as np

# COCO 80 class names (same as yolox.data.datasets.COCO_CLASSES)
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def preproc(img, input_size, swap=(2, 0, 1)):
    """Letterbox resize and normalize for YOLOX model input."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.1, class_agnostic=True):
    """Multiclass NMS - class agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [
                valid_boxes[keep],
                valid_scores[keep, None],
                valid_cls_inds[keep, None],
            ],
            1,
        )
        return dets
    return None


def demo_postprocess(outputs, img_size, p6=False):
    """Decode YOLOX raw output to predictions."""
    outputs = outputs.copy()
    grids = []
    expanded_strides = []
    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


_COLORS = np.array(
    [
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125,
        0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933,
        0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600,
        1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
        0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000,
        0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
        1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000,
        0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500,
        0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500,
        0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500,
        1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500,
        0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000,
        0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
        0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
        0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000,
        1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
        1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
        0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571,
        0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741,
        0.314, 0.717, 0.741, 0.50, 0.5, 0,
    ]
).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, font_scale=1.0, thickness=2):
    """Visualize detection results on image. Larger font for readability."""
    if class_names is None:
        class_names = COCO_CLASSES
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0, y0 = int(box[0]), int(box[1])
        x1, y1 = int(box[2]), int(box[3])
        color = (_COLORS[cls_id % len(_COLORS)] * 255).astype(np.uint8).tolist()
        text = "{} {:.1f}%".format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id % len(_COLORS)]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, max(2, thickness))
        txt_bk_color = (_COLORS[cls_id % len(_COLORS)] * 255 * 0.7).astype(np.uint8).tolist()
        y_top = max(0, y0 - th - 6)
        cv2.rectangle(img, (max(0, x0), y_top), (min(img.shape[1], x0 + tw + 6), y0 + 2), txt_bk_color, -1)
        cv2.putText(img, text, (x0 + 3, y0 - 4), font, font_scale, txt_color, thickness=thickness)
    return img
