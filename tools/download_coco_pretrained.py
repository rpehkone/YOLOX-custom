from tqdm import tqdm
import requests
import os

def download_file(url):
    response = requests.get(url, stream=True)
    filename = "weights/" + url.split('/')[-1]
    
    if response.status_code == 200:
        total = int(response.headers.get('content-length', 0))
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        print(f"Downloaded '{filename}' successfully.")
    else:
        print(f"Failed to download file: status code {response.status_code}")

os.makedirs("weights", exist_ok=True)
download_file("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth")
download_file("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth")
download_file("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth")
download_file("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth")
