"""Convert successful complete-task demonstrations for the provider's training CLI."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONVERTER_REVISION = "ecfbe20d022e02631946a353142f791939c9725c"
CAMERAS = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pen_dataset")
    parser.add_argument("--converter", type=Path, default=Path("/cache/tools/lerobot-v3-v2-converter"))
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing datasets are preserved.")
    info = json.loads((source / "meta/info.json").read_text())
    if info["fps"] != 30 or info["total_episodes"] != 55:
        raise ValueError("This recipe expects the curated 55 complete-task demonstrations at 30 FPS.")
    if not args.converter.exists():
        args.converter.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/jinnymo/lerobot-v3-v2-converter.git", str(args.converter)], check=True)
    subprocess.run(["git", "-C", str(args.converter), "checkout", CONVERTER_REVISION], check=True)
    subprocess.run([sys.executable, str(args.converter / "v3_to_v2/convert.py"),
                    "--input", str(source), "--output", str(output)], check=True)

    def rename_keys(value):
        return {CAMERAS.get(key, key): item for key, item in value.items()}

    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    features = rename_keys(info["features"])
    nonvisual = {key: value for key, value in features.items() if value["dtype"] != "video"}
    info["features"] = {**nonvisual, **{key: features[key] for key in CAMERAS.values()}}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    stats_path = output / "meta/stats.json"
    if stats_path.exists():
        stats_path.write_text(json.dumps(rename_keys(json.loads(stats_path.read_text())), indent=2) + "\n")
    episode_stats_path = output / "meta/episodes_stats.jsonl"
    if episode_stats_path.exists():
        entries = []
        for line in episode_stats_path.read_text().splitlines():
            entry = json.loads(line)
            entry["stats"] = rename_keys(entry["stats"])
            entries.append(json.dumps(entry))
        episode_stats_path.write_text("\n".join(entries) + "\n")
    for chunk in (output / "videos").glob("chunk-*"):
        for previous, current in CAMERAS.items():
            (chunk / previous).rename(chunk / current)

    indices = list(range(info["total_episodes"]))
    random.Random(20261022).shuffle(indices)
    validation = sorted(indices[:6])
    training = sorted(indices[6:])
    for path in sorted((output / "data").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        for key in ["observation.state", "action"]:
            values = np.stack(frame[key]).astype(np.float32)
            if values.shape[1] != 6 or not np.isfinite(values).all():
                raise ValueError("Invalid joint state/action data: " + str(path))
            if not ((values[:, 5] >= -0.01) & (values[:, 5] <= 100.01)).all():
                raise ValueError("The gripper is not expressed in percent.")
        timestamps = frame["timestamp"].to_numpy()
        if abs(timestamps[0]) > 1e-5 or not np.allclose(np.diff(timestamps), 1 / 30, atol=1e-5):
            raise ValueError("Invalid episode timestamps: " + str(path))
    report = {
        "source": "Curated physical MuJoCo demonstrations; IK teacher, not policy inference.",
        "source_collection": "demonstrations_expanded",
        "source_manifest_sha256": hashlib.sha256((source / "curation.json").read_bytes()).hexdigest(),
        "converter_revision": CONVERTER_REVISION,
        "total_episodes": info["total_episodes"], "total_frames": info["total_frames"],
        "training_episodes": training, "validation_episodes": validation,
        "split_seed": 20261022, "split_unit": "whole episode",
        "fps": 30, "joint_units": "First five joints: physical degrees; gripper: percent.",
        "camera_slots": CAMERAS, "scene": "simulation/scene.xml",
        "task": "Grab pens and place into pen holder",
        "task_success": "All three pens upright in the holder at episode end.",
        "position_jitter_m": 0.012, "yaw_jitter_rad": 0.4,
        "scope": "Original three-pen scene; differs from the reconstructed five-pen scene.",
        "original_dataset_modified": False,
    }
    (output / "split.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
