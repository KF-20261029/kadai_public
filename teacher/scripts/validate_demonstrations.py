"""Check successful demonstration data and original recordings before training."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from simulation import JOINT_NAMES, ROOT


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def video_stream(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/demonstrations_expanded")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    root = args.input
    report = json.loads((root / "result.json").read_text())
    dataset = args.dataset_root or root / "dataset"
    info = json.loads((dataset / "meta/info.json").read_text())
    stats = json.loads((dataset / "meta/stats.json").read_text())
    successes = [row for row in report["episodes"] if row["task_success"]]
    collected_successes = len(successes)
    curation_path = dataset / "curation.json"
    curation = json.loads(curation_path.read_text()) if curation_path.exists() else None
    if curation:
        successes = [dict(row) for row in successes if row["dataset_episode_index"] not in curation["excluded_episode_indices"]]
        for row in successes:
            row["dataset_episode_index"] = curation["episode_mapping"][str(row["dataset_episode_index"])]
    require(report.get("collection_completed", True), "Collection has not completed.")
    target = report.get("requested_successful_episodes")
    require(target is None or len(successes) >= target, "The requested success count was not reached.")
    require(collected_successes == report["successful_episodes"], "Collected success counts differ.")
    require(len(successes) == info["total_episodes"], "Training and dataset episode counts differ.")
    require(info["fps"] == 30 and info["robot_type"] == "so101_follower", "Unexpected robot or frame rate.")
    require(info["total_frames"] == sum(row["frames"] for row in successes), "Dataset frame count differs.")
    seeds = [row["seed"] for row in report["episodes"]]
    require(len(set(seeds)) == len(seeds), "A collection seed was repeated.")
    require(not set(seeds).intersection(args.evaluation_seeds), "Evaluation seeds leaked into collection.")
    require([row["dataset_episode_index"] for row in successes] == list(range(len(successes))), "Success episode mapping is not contiguous.")
    for key in ("observation.state", "action"):
        require(info["features"][key]["names"] == list(JOINT_NAMES), "Joint order differs.")
        require(info["features"][key]["shape"] == [6], "Joint data is not six-dimensional.")
        for quantile in ("q01", "q99"):
            require(len(stats[key][quantile]) == 6, "Quantile statistics are missing.")
        require(np.isfinite(stats[key]["q01"]).all() and np.isfinite(stats[key]["q99"]).all(), "Quantile statistics are non-finite.")
        require((np.asarray(stats[key]["q01"]) <= np.asarray(stats[key]["q99"])).all(), "Quantile statistics are unordered.")
    data = pd.concat(pd.read_parquet(path) for path in sorted((dataset / "data").rglob("*.parquet")))
    require(len(data) == info["total_frames"], "Parquet rows differ from the frame count.")
    require(np.array_equal(data["index"].to_numpy(), np.arange(len(data))), "Global frame indices are not contiguous.")
    actions = np.stack(data["action"].to_numpy())
    states = np.stack(data["observation.state"].to_numpy())
    require(np.isfinite(actions).all() and np.isfinite(states).all(), "A state or action is non-finite.")
    require(((actions[:, 5] >= 0) & (actions[:, 5] <= 100)).all(), "Gripper commands are outside percent units.")
    require(set(data["task_index"].to_numpy()) == {0}, "Unexpected task indices.")
    tasks = pd.read_parquet(dataset / "meta/tasks.parquet")
    require(report["instruction"] in tasks.index, "The task instruction differs.")
    episode_meta = pd.concat(pd.read_parquet(path) for path in sorted((dataset / "meta/episodes").rglob("*.parquet")))
    require(len(episode_meta) == len(successes), "Episode metadata count differs.")
    camera_files = {}
    for row in successes:
        require(row["placed_pens"] == [True, True, True], "A failed placement entered the training data.")
        require(row["frames"] == 1800, "A successful episode is not 60 seconds long.")
        episode_index = row["dataset_episode_index"]
        episode = data.loc[data["episode_index"] == episode_index]
        require(len(episode) == 1800, "A dataset episode has missing frames.")
        require(np.array_equal(episode["frame_index"].to_numpy(), np.arange(1800)), "Episode frame indices differ.")
        require(np.allclose(episode["timestamp"].to_numpy(), np.arange(1800) / 30, atol=1e-5), "Frame timestamps differ.")
        attempt = seeds.index(row["seed"])
        commands = np.load(root / f"commands_{attempt:03d}.npy", allow_pickle=False)
        require(np.array_equal(np.stack(episode["action"].to_numpy()), commands), "Dataset actions differ from recorded commands.")
        metadata = episode_meta.loc[episode_meta["episode_index"] == episode_index].iloc[0]
        for camera in ("observation.images.top", "observation.images.wrist"):
            prefix = "videos/" + camera + "/"
            relative = info["video_path"].format(video_key=camera, chunk_index=int(metadata[prefix + "chunk_index"]),
                                                file_index=int(metadata[prefix + "file_index"]))
            start, end = float(metadata[prefix + "from_timestamp"]), float(metadata[prefix + "to_timestamp"])
            require(abs(end - start - 60) < 1e-4, "A camera episode segment is not 60 seconds long.")
            camera_files[relative] = max(camera_files.get(relative, 0), end)
    camera_checks = []
    for relative, expected_end in camera_files.items():
        stream = video_stream(dataset / relative)
        require(stream["codec_name"] in ("h264", "av1") and stream["avg_frame_rate"] == "30/1", "Dataset camera encoding differs.")
        require((stream["width"], stream["height"]) == (640, 480), "Dataset camera resolution differs.")
        require(abs(float(stream["duration"]) - expected_end) < 1 / 30 + 1e-4, "Dataset camera duration differs.")
        require(int(stream["nb_read_frames"]) == round(expected_end * 30), "Dataset camera decoding lost frames.")
        camera_checks.append({"file": relative, **stream})
    attempt_checks = []
    for attempt, row in enumerate(report["episodes"]):
        require("video_file" in row, "An attempt has no original recording.")
        path = root / row["video_file"]
        stream = video_stream(path)
        require(stream["codec_name"] == "h264" and stream["avg_frame_rate"] == "30/1", "An attempt recording has unexpected encoding.")
        require(int(stream["nb_read_frames"]) == row["frames"], "An attempt recording lost frames.")
        require(abs(float(stream["duration"]) - row["frames"] / 30) < 1e-4, "An attempt recording has unexpected duration.")
        require((root / f"final_{attempt:03d}.png").is_file(), "An attempt has no final image.")
        attempt_checks.append({"seed": row["seed"], "task_success": row["task_success"], "file": row["video_file"],
                               "frames": row["frames"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    initial_positions = [np.asarray(row["initial_pen_positions_m"]).tobytes() for row in successes]
    require(len(set(initial_positions)) == len(successes), "Identical initial positions were duplicated.")
    validation = {"passed": True, "successful_episodes": len(successes), "dataset_frames": len(data),
                  "collected_successful_episodes": collected_successes,
                  "dataset_directory": str(dataset.relative_to(root)), "curation": curation,
                  "attempted_episodes": len(seeds), "all_attempts_have_original_videos": True,
                  "dataset_actions_match_saved_commands": True, "finite_states_and_actions": True,
                  "unique_successful_initial_positions": len(set(initial_positions)),
                  "evaluation_seeds": args.evaluation_seeds, "evaluation_seed_overlap": [],
                  "normalization_quantiles_checked": True, "dataset_camera_videos": camera_checks,
                  "attempt_recordings": attempt_checks, "increased_data_proves_policy_success": False}
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    from export_demonstration_videos import write_index
    entries = [{"trial_number": index + 1, "seed": row["seed"],
                "original_task_success": row["task_success"], "video": Path(row["file"]).name,
                "final_image": f"../final_{index:03d}.png",
                "video_source": "Original recorded teacher overview, top, and wrist cameras",
                "sha256": row["sha256"]} for index, row in enumerate(attempt_checks)]
    recording_manifest = {"original_successes": collected_successes, "training_episodes": len(successes), "original_attempts": len(seeds),
                          "exported_videos": len(entries), "all_videos_are_original_recordings": True,
                          "trials": entries}
    (root / "recordings/manifest.json").write_text(json.dumps(recording_manifest, indent=2) + "\n")
    write_index(root / "recordings", entries, report)
    print(json.dumps({key: value for key, value in validation.items() if key not in ("dataset_camera_videos", "attempt_recordings")}, indent=2))


if __name__ == "__main__":
    main()
