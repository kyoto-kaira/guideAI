import torch
import torchvision
from tqdm import tqdm
import numpy as np
import cv2

import sys
sys.path.append("./LIA/")
from networks.generator import Generator


G = Generator(256)
G.load_state_dict(torch.load("LIA/checkpoints/vox.pt")["gen"])
G.to("cuda").eval()


def numpy2tensor(img, bgr=True):
    if bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (256, 256))
    tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)
    return tensor.unsqueeze(0).to("cuda")


def write_video(frames, name, fps=20.0):
    frames = torch.cat(frames).permute(0, 2, 3, 1)
    frames = torch.clip(frames *127.5 + 127.5, 0, 255)
    torchvision.io.write_video(name, frames.to("cpu"), fps=fps)


@torch.no_grad()
def get_alpha(img):
    img = cv2.resize(img, (256, 256))
    x = numpy2tensor(img)

    alpha = G.enc.enc_motion(x)
    return alpha


def get_slider():
    img = cv2.imread("images/slide.png")

    return torch.cat([
        get_alpha(img),
        get_alpha(img[:, ::-1]),
    ])


def get_wink():
    img_open = cv2.imread("images/opening.png")
    img_close = cv2.imread("images/closing.png")

    return torch.cat([
        get_alpha(img_close),
        get_alpha(img_open),
    ])


@torch.no_grad()
def test_slider(img, direction):
    direction = direction[0:1] - direction[1:2]
    x = numpy2tensor(img)

    h, features = G.enc.net_app(x)
    alpha = G.enc.fc(h)

    frames = []

    for step in tqdm(torch.linspace(-0.2, 0.2, 40)):
        frame = G.synthesis(h, [alpha + step * direction], features)
        frames.append(frame)

    write_video(frames, "videos/slider.mp4")


@torch.no_grad()
def test_wink(img, direction):
    direction = direction[0:1] - direction[1:2]
    x = numpy2tensor(img)

    h, features = G.enc.net_app(x)
    alpha = G.enc.fc(h)

    frames = []

    for step in tqdm(torch.linspace(0, 1.2, 40)):
        frame = G.synthesis(h, [alpha + step * direction], features)
        frames.append(frame)

    write_video(frames, "videos/wink.mp4")



if __name__ == "__main__":
    slider = get_slider()
    torch.save(slider, "latents/slider.pt")

    wink = get_wink()
    torch.save(wink, "latents/wink.pt")

    img = cv2.imread("images/sample.png")
    test_slider(img, slider)
    test_wink(img, wink)


