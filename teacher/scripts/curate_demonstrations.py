"""Exclude episodes with incomplete camera segments using the native LeRobot API."""

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from simulation import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/demonstrations_expanded")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-episodes", type=int,
                        help="Keep this many complete episodes in source order.")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("The requested episode count must be positive.")
    source = args.input / "dataset"
    output = args.output or args.input / "training_dataset"
    if output.exists():
        raise FileExistsError("The curated output directory already exists.")
    info = json.loads((source / "meta/info.json").read_text())
    metadata = pd.concat(pd.read_parquet(path) for path in sorted((source / "meta/episodes").rglob("*.parquet")))
    excluded = set()
    details = []
    for _, row in metadata.iterrows():
        for camera in ("observation.images.top", "observation.images.wrist"):
            prefix = "videos/" + camera + "/"
            frames = round((float(row[prefix + "to_timestamp"]) - float(row[prefix + "from_timestamp"])) * info["fps"])
            if frames != int(row["length"]):
                excluded.add(int(row["episode_index"]))
                details.append({"source_episode_index": int(row["episode_index"]), "camera": camera,
                                "expected_frames": int(row["length"]), "recorded_segment_frames": frames,
                                "reason": "Incomplete recorded camera segment; excluded without synthesizing frames."})
    complete = [index for index in range(info["total_episodes"]) if index not in excluded]
    if args.max_episodes is not None:
        if len(complete) < args.max_episodes:
            raise RuntimeError(
                f"Only {len(complete)} complete successful episodes; collect more in a new directory.")
        for index in complete[args.max_episodes:]:
            excluded.add(index)
            details.append({"source_episode_index": index,
                            "reason": "Excluded by the requested deterministic episode-count limit."})
    print(json.dumps({"excluded_episode_indices": sorted(excluded), "details": details}), flush=True)
    if excluded:
        dataset = LeRobotDataset("local/so101_pens", root=source)
        curated = delete_episodes(dataset, sorted(excluded), output_dir=output, repo_id="local/so101_pens_curated")
        training_episodes = curated.num_episodes
    else:
        shutil.copytree(source, output)
        training_episodes = info["total_episodes"]
    kept = [index for index in range(info["total_episodes"]) if index not in excluded]
    curation = {"source_episodes": info["total_episodes"], "training_episodes": training_episodes,
                "excluded_episode_indices": sorted(excluded), "exclusions": details,
                "episode_mapping": {str(old): new for new, old in enumerate(kept)},
                "native_operation": "lerobot.datasets.dataset_tools.delete_episodes" if excluded else "shutil.copytree; no exclusions needed",
                "synthetic_frames_added": False, "original_dataset_modified": False}
    (output / "curation.json").write_text(json.dumps(curation, indent=2) + "\n")
    print(json.dumps(curation, indent=2), flush=True)


if __name__ == "__main__":
    main()
