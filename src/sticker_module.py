import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
left_model = YOLO(str(resources_path / "models/left_sticker.pt")).to(device)
right_model = YOLO(str(resources_path / "models/right_sticker.pt")).to(device)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def box_inside(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

def resolve_sticker_conflicts(left_boxes, right_boxes, iou_threshold=0.5):
    resolved_left = []
    resolved_right = []
    used_right = set()

    for i, lbox in enumerate(left_boxes):
        best_j = -1
        best_overlap = 0
        for j, rbox in enumerate(right_boxes):
            if j in used_right:
                continue
            score = iou(lbox.xyxy[0].cpu().numpy(), rbox.xyxy[0].cpu().numpy())
            if score > best_overlap:
                best_overlap = score
                best_j = j

        if best_overlap > iou_threshold:
            if lbox.conf[0] > right_boxes[best_j].conf[0]:
                resolved_left.append(lbox)
            else:
                resolved_right.append(right_boxes[best_j])
            used_right.add(best_j)
        else:
            resolved_left.append(lbox)

    for j, rbox in enumerate(right_boxes):
        if j not in used_right:
            resolved_right.append(rbox)

    return resolved_left, resolved_right

def detect_stickers(frame, conf_threshold=0.6, iou_threshold=0.5):
    left_results = left_model.predict(frame, verbose=False)[0].boxes
    right_results = right_model.predict(frame, verbose=False)[0].boxes

    left_filtered = [box for box in left_results if box.conf[0] >= conf_threshold]
    right_filtered = [box for box in right_results if box.conf[0] >= conf_threshold]

    # Resolve overlapping sticker detections
    #resolved_left, resolved_right = resolve_sticker_conflicts(left_filtered, right_filtered, iou_threshold)

    #return resolved_left, resolved_right
    return left_filtered, right_filtered
