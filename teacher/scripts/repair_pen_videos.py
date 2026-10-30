"""Export exact original frame ranges when stream-copy conversion shifts timestamps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import pandas as pd
import torch
from torchcodec.decoders import VideoDecoder

from prepare_pen_dataset import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pen_dataset"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_info = json.loads((args.source / "meta/info.json").read_text())
    episodes = pd.concat(pd.read_parquet(path) for path in sorted((args.source / "meta/episodes").rglob("*.parquet")))
    output_info = json.loads((args.output / "meta/info.json").read_text())

    def export(item):
        row, source_camera, destination_camera = item
        episode = int(row["episode_index"])
        prefix = "videos/" + source_camera + "/"
        source_path = args.source / source_info["video_path"].format(
            video_key=source_camera, chunk_index=int(row[prefix + "chunk_index"]),
            file_index=int(row[prefix + "file_index"]))
        destination = args.output / output_info["video_path"].format(
            video_key=destination_camera, episode_chunk=episode // 1000, episode_index=episode)
        temporary = destination.with_suffix(".tmp.mp4")
        first = round(float(row[prefix + "from_timestamp"]) * 30)
        count = int(row["length"])
        decoder = VideoDecoder(str(source_path), device="cpu", num_ffmpeg_threads=1)
        if first + count > len(decoder):
            raise ValueError("A source camera has missing frames.")
        height, width = source_info["features"][source_camera]["shape"][:2]
        process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "30", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", str(temporary),
        ], stdin=subprocess.PIPE)
        try:
            for start in range(first, first + count, 30):
                frames = decoder.get_frames_at(indices=list(range(start, min(start + 30, first + count)))).data
                process.stdin.write(frames.permute(0, 2, 3, 1).contiguous().numpy().tobytes())
        finally:
            process.stdin.close()
            returncode = process.wait()
        if returncode:
            raise RuntimeError("Video encoding failed.")
        temporary.replace(destination)
        print(json.dumps({"episode": episode, "camera": destination_camera, "original_frames_exported": count}), flush=True)

    items = [(row, previous, current) for _, row in episodes.iterrows() for previous, current in CAMERAS.items()]
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(export, items))
    for camera in CAMERAS.values():
        output_info["features"][camera]["info"]["video.codec"] = "h264"
    (args.output / "meta/info.json").write_text(json.dumps(output_info, indent=2) + "\n")
    report = {"segments": len(items), "frame_selection": "Original video frame indices from v3 episode metadata.",
              "synthetic_or_repeated_frames_added": False, "fps": 30, "first_output_timestamp": 0,
              "reason": "Stream-copy conversion produced a nonzero first timestamp and an extra frame."}
    (args.output / "video_conversion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
