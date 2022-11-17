from facenet_pytorch import MTCNN
from fastapi import FastAPI
import numpy as np
import base64
import torch
import cv2


mtcnn_model = MTCNN(device="cuda")
mtcnn_model.eval()

prev_points = None
missing_count = 0


def calc_iou(src, dst):
    s_area = (src[2] - src[0]) * (src[3] - src[1])
    d_area = (dst[:, 2] - dst[:, 0]) * (dst[:, 3] - dst[:, 1])

    x_min = np.maximum(src[0], dst[:, 0])
    y_min = np.maximum(src[1], dst[:, 1])
    x_max = np.minimum(src[2], dst[:, 2])
    y_max = np.minimum(src[3], dst[:, 3])

    inter = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    return inter / (s_area + d_area - inter)


@torch.no_grad()
def mtcnn_predict(img, iou_threshold=0.6, max_missing_count=20):
    """
    MTCNNで顔のBoundingBoxを検出

    Args:
        img (np.ndarray, PIL.Image): 入力画像
        iou_threshold (float): 同一人物とみなすためのIOUの閾値
        max_missing_count (int): このフレーム数だけ同一人物を検出できなかった場合に新規の人物を検出する

    Returns:
        cx (int): BoudingBoxの中心座標
        cy (int): BoudingBoxの中心座標
    """
    global prev_points, missing_count

    bboxes, preds = mtcnn_model.detect(img)

    if bboxes is None:
        return list(map(int, prev_points))

    if prev_points is None:
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        x0, y0, x1, y1 = bboxes[np.argmax(areas)]
        missing_count = 0
    else:
        ious = calc_iou(prev_points, bboxes)
        if np.max(ious) < iou_threshold:
            missing_count += 1
            if missing_count > max_missing_count:
                missing_count = 0
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                x0, y0, x1, y1 = bboxes[np.argmax(areas)]
            else:
                x0, y0, x1, y1 = prev_points
        else:
            i = np.argmax(ious)
            x0, y0, x1, y1 = bboxes[i]
            missing_count = 0

    prev_points = (x0, y0, x1, y1)

    return list(map(int, [x0, y0, x1, y1]))


# =====================================================================================================
# Fast API
# =====================================================================================================

app = FastAPI()


@app.get("/")
def index():
    return {"status": "ok"}


@app.get("/reset")
def reset():
    global prev_points, missing_count
    prev_points = None
    missing_count = 0


@app.post("/detect")
def main(request: dict):
    img_data = base64.b64decode(request["img_base64"])
    img_byte = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_byte, cv2.IMREAD_ANYCOLOR)

    x0, y0, x1, y1 = mtcnn_predict(img)

    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
