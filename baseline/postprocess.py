from __future__ import annotations

import torch


def select_best_box(boxes, scores, labels):
    score_values = torch.as_tensor(scores).detach().cpu().flatten()
    if score_values.numel() == 0:
        return None
    if len(boxes) != score_values.numel() or len(labels) != score_values.numel():
        raise ValueError("boxes, scores and labels must have equal lengths")
    if not torch.isfinite(score_values).all():
        raise ValueError("scores contain NaN or infinity")
    best_idx = int(torch.argmax(score_values).item())
    return {
        "box": boxes[best_idx],
        "score": float(score_values[best_idx].item()),
        "label": str(labels[best_idx]),
    }


def box_cxcywh_to_xyxy(box) -> list[float] | None:
    values = torch.as_tensor(box).detach().cpu().to(torch.float64).flatten()
    if values.shape != (4,):
        raise ValueError(f"box must contain four values, got shape {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError("box contains NaN or infinity")
    cx, cy, w, h = [float(value) for value in values.tolist()]
    x1 = max(0.0, min(1.0, cx - w / 2))
    y1 = max(0.0, min(1.0, cy - h / 2))
    x2 = max(0.0, min(1.0, cx + w / 2))
    y2 = max(0.0, min(1.0, cy + h / 2))
    if x1 >= x2 or y1 >= y2:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def process_prediction(boxes, scores, labels) -> dict | None:
    best = select_best_box(boxes, scores, labels)
    if best is None:
        return None
    bbox = box_cxcywh_to_xyxy(best["box"])
    if bbox is None:
        return None
    return {"bbox": bbox, "score": float(best["score"]), "label": best["label"]}
