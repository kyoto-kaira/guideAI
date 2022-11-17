import numpy as np
import argparse
import requests
import base64
import time
import cv2


def encode_img(img):
    _, data = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(data).decode("utf-8")
    return img_base64


def request(args, url):
    img = cv2.imread(args.input)
    img_base64 = encode_img(img)

    t0 = time.time()

    results = requests.post(
        url + "/detect",
        json={"img_base64": img_base64}
    )

    t1 = time.time()

    print("API response:", results.status_code)
    print("Time [ms]: ", t1 - t0)
    print("FPS [ms]: ", 1 / (t1-t0))

    if not results.status_code == 200:
        print(results.json())
        return

    results = results.json()
    x0 = results["x0"]
    y0 = results["y0"]
    x1 = results["x1"]
    y1 = results["y1"]

    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 5)
    cv2.imwrite(args.output, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MTCNN APIのrequestのテスト用")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="APIのIPアドレス")
    parser.add_argument("--port", type=str, default="8006", help="APIのポート")
    parser.add_argument("-i", "--input", type=str, default="sample.png", help="入力画像のパス")
    parser.add_argument("-o", "--output", type=str, default="result.png", help="出力画像のパス")

    args = parser.parse_args()

    url = "http://" + args.host + ":" + args.port
    print("URL:", url)
    print("API status:", requests.get(url).status_code)

    request(args, url)
