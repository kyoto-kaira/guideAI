import argparse
import requests
import json
import time


def request(args, url):
    t0 = time.time()

    query = requests.post(
        url + "/audio_query",
        params={
            "text": args.text,
            "speaker": args.speaker,
        }
    )

    if not query.status_code == 200:
        print(results.json())
        return

    results = requests.post(
        url + "/synthesis",
        params={"speaker": args.speaker},
        data=json.dumps(query.json()),
    )

    t1 = time.time()

    print("API response:", results.status_code)
    print("Time [ms]: ", t1 - t0)
    print("FPS [ms]: ", 1 / (t1-t0))

    if not results.status_code == 200:
        print(results.json())
        return

    with open(args.output, mode="wb") as f:
        f.write(results.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("StableDiffusion APIのrequestのテスト用")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="APIのIPアドレス")
    parser.add_argument("--port", type=str, default="8004", help="APIのポート")
    parser.add_argument("-t", "--text", type=str, default="おはようございます。今日も元気に行きましょう！", help="入力プロンプト")
    parser.add_argument("-s", "--speaker", type=int, default=1, help="speaker id")
    parser.add_argument("-o", "--output", type=str, default="generated.wav", help="出力音声名")

    args = parser.parse_args()

    url = "http://" + args.host + ":" + args.port
    print("URL:", url)

    request(args, url)
