"""Check every converted camera segment before starting an overnight training run."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
dataset = root / "outputs/pen_dataset"
info = json.loads((dataset / "meta/info.json").read_text())
split = json.loads((dataset / "split.json").read_text())
assert info["codebase_version"] == "v2.1"
assert set(split["training_episodes"]).isdisjoint(split["validation_episodes"])
reports = []
for index in range(info["total_episodes"]):
    frame = pd.read_parquet(dataset / info["data_path"].format(episode_chunk=index // 1000, episode_index=index))
    assert len(frame) == 1800
    assert frame["episode_index"].eq(index).all()
    assert np.isfinite(np.stack(frame["action"])).all()
    for camera in split["camera_slots"].values():
        path = dataset / info["video_path"].format(episode_chunk=index // 1000, episode_index=index, video_key=camera)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,start_time,duration", "-of", "json", str(path)
        ], text=True))["streams"][0]
        record = {"episode": index, "camera": camera, **probe}
        reports.append(record)
        assert int(probe["nb_read_frames"]) == len(frame), record
        assert probe["r_frame_rate"] == "30/1", record
        assert abs(float(probe["start_time"])) <= 1e-4, record
report = {"passed": True, "episodes": info["total_episodes"], "frames": info["total_frames"],
          "training_episodes": len(split["training_episodes"]),
          "validation_episodes": len(split["validation_episodes"]),
          "checked_video_segments": len(reports), "videos": reports}
(dataset / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "videos"}), flush=True)
