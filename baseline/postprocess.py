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

def box_cxcywh_to_xyxy(
    box,
    image_width,
    image_height
):
    cx, cy, w, h = box


    x1 = (cx - w / 2) * image_width
    y1 = (cy - h / 2) * image_height

    x2 = (cx + w / 2) * image_width
    y2 = (cy + h / 2) * image_height


    return [
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    ]

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


    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2)
    ]

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

    result = {

        "bbox": xyxy_box,

        "score": float(
            best_result["score"]
        ),

        "label": best_result["label"]
    }

    return result
