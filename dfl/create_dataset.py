import glob
import os

import pandas as pd
import torch

from yolov7.custom_detect import detect

video_base_dir = "./data/raw/train/"
video_file_name_list = glob.glob(os.path.join(video_base_dir, "*.mp4"))


model_weight = "yolov7.pt"

for i, video_path in enumerate(video_file_name_list):
    if i < 5:
        continue
    else:
        with torch.no_grad():
            dst = detect(
                source=video_path,
                weights=model_weight,
                device="0",
                imgsz=640,
            )

        data = pd.DataFrame(dst, columns=["frame", "obj_name", "x", "y", "conf"])
        data = data.assign(obj_name=data["obj_name"].map({0: "person", 32: "ball"}))

        video_id = os.path.basename(video_path).split(".")[0]
        os.makedirs(f"./data/feature/{video_id}/", exist_ok=True)
        data.to_csv(f"./data/feature/{video_id}/detect.csv", index=False)
