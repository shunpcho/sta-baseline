import json
import multiprocessing
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sta_baseline.datasets.short_term_anticipation import PyAVVideoReader

parser = ArgumentParser()
parser.add_argument("path_to_annotations", type=Path)
parser.add_argument("path_to_videos", type=Path)
parser.add_argument("path_to_output", type=Path)
parser.add_argument("--fname_format", type=str, default="{video_uid:s}_{frame_number:07d}.jpg")
parser.add_argument("--jobs", default=8, type=int)
parser.add_argument("--clips", action="store_true")

args = parser.parse_args()

args.path_to_output.mkdir(exist_ok=True, parents=True)

images = []

with Path(args.path_to_annotations / "fho_sta_train.json").open(encoding="utf-8") as f:
    train = json.load(f)
with Path(args.path_to_annotations / "fho_sta_val.json").open(encoding="utf-8") as f:
    val = json.load(f)
with Path(args.path_to_annotations / "fho_sta_test_unannotated.json").open(encoding="utf-8") as f:
    test = json.load(f)

names: list[str] = []
video_ids: list[str] = []
frame_numbers: list[int] = []

for ann in [train, val, test]:
    for x in ann["annotations"]:
        fname = args.fname_format.format(video_uid=x["video_uid"], frame_number=x["frame"])
        names.append(fname)
        if args.clips:
            video_ids.append(x["clip_uid"])
            frame_numbers.append(x["clip_frame"])
        else:
            video_ids.append(x["video_uid"])
            frame_numbers.append(x["frame"])

# images = sorted(images)

print(f"Found {len(names)} frames to extract")

missing: list[int] = []
for idx, im in enumerate(names):
    if not Path(args.path_to_output / im).is_file():
        missing.append(idx)

print(f"Skipping {len(names) - len(missing)} frames already extracted")

names = [names[i] for i in missing]
video_ids = [video_ids[i] for i in missing]
frame_numbers = [frame_numbers[i] for i in missing]

if len(names) == 0:
    sys.exit(0)

df = pd.DataFrame({"video": video_ids, "frame": np.array(frame_numbers).astype(int), "name": names})

groups = df.groupby("video")

all_video_names = []
all_frames = []
names = []

for g in groups:
    vid = g[0]
    frames = g[1]["frame"].to_numpy()
    names.extend(g[1]["name"])

    all_video_names.extend([f"{vid}.mp4"] * len(frames))
    all_frames.extend(frames)

df = pd.DataFrame({"video": all_video_names, "frame": all_frames, "name": names})


def process_video(args: tuple[str, np.ndarray, np.ndarray]) -> None:
    fname, frames, names = args
    vr = PyAVVideoReader(fname)

    video_frames = vr[frames]

    for vf, nam in zip(video_frames, names, strict=True):
        imname = str(args.path_to_output / f"{nam}")
        cv2.imwrite(imname, vf)


params = []
for g in df.groupby("video"):
    vid = g[0]
    fname = str(args.path_to_videos / vid)
    frames = g[1]["frame"].to_numpy()
    names = g[1]["name"].to_numpy()

    params.append((fname, frames, names))

pool = multiprocessing.Pool(processes=args.jobs)

for _ in tqdm(pool.imap_unordered(process_video, params), total=len(params)):
    pass
