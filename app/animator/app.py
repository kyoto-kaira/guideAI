from fastapi import FastAPI
import numpy as np
import base64
import torch
import cv2

import sys
sys.path.append("./LIA/")
from networks.generator import Generator


G = Generator(256)
G.load_state_dict(torch.load("LIA/checkpoints/vox.pt")["gen"])
G.to("cuda").eval()

tensor_slider = torch.load("latents/slider.pt").to("cuda")
direction_slider = tensor_slider[1:2] - tensor_slider[0:1]
tensor_wink = torch.load("latents/wink.pt").to("cuda")
directon_wink = tensor_wink[0:1] - tensor_wink[1:2]

yaw_now, yaw_tgt = 0, 0
wink_now, wink_tgt = 0, 1

h, alpha, features = None, None, None


def numpy2tensor(img):
    img = cv2.resize(img, (256, 256))
    tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)
    return tensor.unsqueeze(0).to("cuda")


def tensor2numpy(x):
    x = x[0].to("cpu").permute(1, 2, 0).detach().numpy()
    img = np.uint8(np.clip(x * 127.5 + 127.5, 0, 255))
    return img


def update_move():
    global yaw_now, wink_now, wink_tgt
    if yaw_tgt is None:
        pass
    elif yaw_now < yaw_tgt:
        yaw_now = min(yaw_tgt, yaw_now + 0.05)
    elif yaw_now > yaw_tgt:
        yaw_now = max(yaw_tgt, yaw_now - 0.05)

    # if wink_tgt == 1:
    #     wink_now += 1.3
    #     if wink_now > 1.2:
    #         wink_tgt = 0
    # else:
    #     wink_now -= 1.3
    #     if wink_now < -1.3 * 40:
    #         wink_tgt = 1
    #         wink_now = 0


@torch.no_grad()
def encode_img(img):
    global h, alpha, features
    x = numpy2tensor(img)
    h, features = G.enc.net_app(x)
    alpha = G.enc.fc(h)


@torch.no_grad()
def generate_frame():
    move_alpha = yaw_now * direction_slider + \
                max(wink_now, 0) * directon_wink

    frame = G.synthesis(h, [alpha + move_alpha], features)
    return tensor2numpy(frame)



# =====================================================================================================
# Fast API
# =====================================================================================================

app = FastAPI()


@app.get("/")
def index():
    return {"status": "ok"}


@app.get("/reset")
def reset():
    global yaw_now, wink_now, wink_tgt
    yaw_now, yaw_tgt = 0, 0
    wink_now, wink_tgt = 0, 1


@app.post("/encode")
def encode(request: dict):
    img_data = base64.b64decode(request["img_base64"])
    img_byte = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_byte, cv2.IMREAD_ANYCOLOR)

    encode_img(img)

    return {}


@app.post("/moving")
def main(request: dict):
    global yaw_tgt
    yaw_tgt =  - request["x"] * 0.3
    update_move()

    frame = generate_frame()
    _, data = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(data).decode("utf-8")

    return {"img_base64": img_base64}
