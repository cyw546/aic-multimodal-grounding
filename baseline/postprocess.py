import torch

def select_best_box(
    boxes,
    scores,
    labels
):

    if len(scores) == 0:
        return None

    best_idx = torch.argmax(scores)

    return {
        "box": boxes[best_idx],
        "score": scores[best_idx],
        "label": labels[best_idx]
    }

def box_cxcywh_to_xyxy(box):
    """
    GroundingDINO:
        cx,cy,w,h

    转比赛格式:
        x1,y1,x2,y2

    保持归一化坐标
    """

    cx, cy, w, h = box


    x1 = cx - w / 2
    y1 = cy - h / 2

    x2 = cx + w / 2
    y2 = cy + h / 2


    # 裁剪到 [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    # 合法性检查
    if x1 >= x2 or y1 >= y2:
        return None

    return [x1, y1, x2, y2]

def process_prediction(
    boxes,
    scores,
    labels
):

    best_result = select_best_box(
        boxes,
        scores,
        labels
    )

    if best_result is None:
        return None

    xyxy_box = box_cxcywh_to_xyxy(
        best_result["box"]
    )

    if xyxy_box is None:
        return None

    result = {

        "bbox": xyxy_box,

        "score": float(
            best_result["score"]
        ),

        "label": best_result["label"]
    }

    return result