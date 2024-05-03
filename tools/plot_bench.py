import matplotlib.pyplot as plt
from adjustText import adjust_text

models = ["YOLOX-Nano", "YOLOX-Tiny", "YOLOX-s", "YOLOX-m", "YOLOX-l", "YOLOX-Darknet53", "YOLOX-x"]
fps_torch_v100 = [925, 155, 102, 81, 68, 90, 57]
fps_torch_4090 = [43, 50, 49, 45, 42, 49, 40]
fps_torch_jetson_agx_orin = [18, 20, 20, 17, 15, 13, 10]

ms_per_frame_v100 = [1000 / fps for fps in fps_torch_v100]
ms_per_frame_4090 = [1000 / fps for fps in fps_torch_4090]
ms_per_frame_jetson = [1000 / fps for fps in fps_torch_jetson_agx_orin]

map_test = [0.91, 5.06, 40.5, 47.2, 50.1, 48.0, 51.5]

colors = {
    "v100": "#011f05",
    "4090": "#027d12",
    "orin": "#04c41c"
}

plt.figure(figsize=(10, 6))
plt.plot(ms_per_frame_v100, map_test, 'o-', color=colors["v100"], label='Torch V100')
plt.plot(ms_per_frame_4090, map_test, 'o-', color=colors["4090"], label='Torch 4090')
plt.plot(ms_per_frame_jetson, map_test, 'o-', color=colors["orin"], label='Torch Jetson AGX Orin 64GB')

texts = []
for i, model in enumerate(models):
    texts.append(plt.text(ms_per_frame_v100[i], map_test[i], model, color=colors["v100"], ha='center'))
    texts.append(plt.text(ms_per_frame_4090[i], map_test[i], model, color=colors["4090"], ha='center'))
    texts.append(plt.text(ms_per_frame_jetson[i], map_test[i], model, color=colors["orin"], ha='center'))

adjust_text(texts)

plt.xlabel('Milliseconds per Frame')
plt.ylabel('mAP test (0.5:0.95)')
plt.legend()
plt.grid(True)
plt.show()
