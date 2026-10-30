"""Export every teacher attempt and disclose reconstructed missing recordings."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from simulation import ROOT

CAMERAS = ("observation.images.top", "observation.images.wrist")
FPS = 30
HEADER_HEIGHT = 100


def header_image(attempt, result, instruction, reconstructed, total_attempts):
    image = Image.new("RGB", (1280, HEADER_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=19)
    outcome = "SUCCESS" if result["task_success"] else "FAILED"
    source = "RECONSTRUCTION from saved commands; original video was not saved" if reconstructed else "ORIGINAL recorded cameras from the training dataset"
    lines = [
        f"Teacher demonstration | Trial {attempt + 1:02d}/{total_attempts} | Seed {result['seed']} | Original result: {outcome}",
        source,
        "Instruction: " + instruction,
        "Left: top camera | Right: wrist camera | IK teacher, not pi0.5 inference",
    ]
    for index, line in enumerate(lines):
        draw.text((14, index * 24), line, fill="black", font=font)
    return image


def encoding_options(output, frames):
    return [
        "-an", "-frames:v", str(frames), "-r", str(FPS), "-fps_mode", "cfr",
        "-enc_time_base", "1:30", "-c:v", "libx264", "-threads", "4",
        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-map_metadata", "-1", str(output),
    ]


def export_original(dataset, info, episode, header, output, frames):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
    sources = []
    for camera in CAMERAS:
        prefix = "videos/" + camera + "/"
        relative_path = info["video_path"].format(
            video_key=camera,
            chunk_index=int(episode[prefix + "chunk_index"]),
            file_index=int(episode[prefix + "file_index"]),
        )
        start = float(episode[prefix + "from_timestamp"])
        end = float(episode[prefix + "to_timestamp"])
        if abs((end - start) * FPS - frames) > 0.01:
            raise ValueError("The recorded video segment length does not match the trial.")
        command += ["-ss", str(start), "-i", str(dataset / relative_path)]
        sources.append({"camera": camera, "dataset_file": relative_path,
                        "from_timestamp_s": start, "to_timestamp_s": end})
    command += [
        "-loop", "1", "-framerate", str(FPS), "-i", str(header),
        "-filter_complex",
        f"[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];"
        f"[left][right]hstack=inputs=2,pad=1280:580:0:{HEADER_HEIGHT}:color=white[cameras];"
        "[cameras][2:v]overlay=0:0:shortest=1[video]",
        "-map", "[video]",
    ]
    subprocess.run(command + encoding_options(output, frames), check=True)
    return sources


def reconstruct_attempt(commands, result, header, output, report):
    import mujoco
    from simulation import Simulation

    # Match every Teacher.move boundary, including each step of the final hold.
    phase_lengths = (60, 60, 30, 90, 90, 90, 60, 30, 60)
    transfer_frames = len(commands) - 90
    boundaries = set()
    cursor = 0
    while cursor < transfer_frames:
        for length in phase_lengths:
            cursor += length
            if cursor > transfer_frames:
                raise ValueError("The recorded trial ends inside an unknown teacher phase.")
            boundaries.add(cursor)
            if cursor == transfer_frames:
                break
    boundaries.update(range(transfer_frames + 1, len(commands) + 1))
    sim = Simulation()
    process = None
    try:
        sim.reset(result["seed"], position_jitter=report.get("initial_position_jitter_m", 0.008),
                  yaw_jitter=report.get("initial_yaw_jitter_rad", 0.3))
        process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x580",
            "-framerate", str(FPS), "-i", "pipe:0",
        ] + encoding_options(output, len(commands)), stdin=subprocess.PIPE)
        stable = True
        completed = []
        lifted = []
        for index, command in enumerate(commands):
            canvas = Image.new("RGB", (1280, 580), "white")
            canvas.paste(header, (0, 0))
            canvas.paste(Image.fromarray(sim.render("top")), (0, HEADER_HEIGHT))
            canvas.paste(Image.fromarray(sim.render("wrist_cam")), (640, HEADER_HEIGHT))
            process.stdin.write(canvas.tobytes())
            sim.step(command)
            frame = index + 1
            if frame in boundaries:
                mujoco.mj_forward(sim.model, sim.data)
            if frame <= transfer_frames and frame % 570 == 240:
                lifted.append(bool(sim.pen_positions()[frame // 570, 2] >= 0.045))
            if frame <= transfer_frames and frame % 570 == 0:
                completed.append(bool(sim.placement()[frame // 570 - 1]))
            if frame > transfer_frames:
                stable = stable and bool(sim.placement().all())
        positions = sim.pen_positions()
        error = float(np.max(np.abs(positions - np.asarray(result["final_pen_positions_m"]))))
        success = len(completed) == 3 and all(completed) and all(lifted) and stable
        comparison = {
            "reconstructed_task_success": success,
            "reconstructed_placed_pens": sim.placement().tolist(),
            "reconstructed_final_pen_positions_m": positions.tolist(),
            "max_final_position_difference_m": error,
            "final_positions_match_within_1e_minus_6_m": error < 1e-6,
            "outcome_matches_original": success == result["task_success"],
        }
        if error >= 1e-6 or success != result["task_success"]:
            raise RuntimeError("The reconstructed attempt does not match the original result: " + json.dumps(comparison))
        return comparison
    finally:
        if process is not None:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("Reconstruction video encoding failed.")
        sim.close()


def inspect_video(path, expected_frames):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != expected_frames or stream["avg_frame_rate"] != "30/1":
        raise RuntimeError("The exported video has an unexpected frame count or frame rate.")
    if stream["codec_name"] != "h264" or (stream["width"], stream["height"]) not in ((1280, 580), (1280, 540)):
        raise RuntimeError("The exported video has an unexpected encoding or resolution.")
    return stream


def write_index(output, entries, report):
    cards = []
    for entry in entries:
        source = entry["video_source"]
        outcome = "SUCCESS" if entry["original_task_success"] else "FAILED"
        cards.append(
            f'<article><h2>Trial {entry["trial_number"]:02d} / Seed {entry["seed"]} / {outcome}</h2>'
            f'<p>{html.escape(source)}</p>'
            f'<video controls preload="none" poster="{entry["final_image"]}" src="{entry["video"]}"></video>'
            f'<p><a href="{entry["video"]}">Open MP4</a></p></article>'
        )
    reconstructed_count = sum("reconstruction_verification" in entry for entry in entries)
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>SO-101 teacher demonstration trials</title>
<style>body{{font-family:sans-serif;margin:24px;background:#f4f5f7;color:#17202a}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:20px}}article{{background:white;padding:16px;border-radius:8px}}h2{{font-size:19px}}video{{width:100%}}p{{line-height:1.5}}</style>
<h1>SO-101 teacher demonstration trials: {report['successful_episodes']} successes / {report['attempted_episodes']} attempts</h1>
<p>These are IK teacher demonstrations, not pi0.5 inference. Videos are at 30 FPS; each complete successful demonstration is 60 seconds.</p>
<p>Original recordings: {len(entries) - reconstructed_count}. Reconstructions from saved commands: {reconstructed_count}. Each video explicitly identifies its source. Reconstructed final outcomes and object positions were checked against the original records.</p>
<p><a href="manifest.json">Recording sources, results, and verification</a></p><main>
""" + "\n".join(cards) + "\n</main></html>\n"
    (output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/demonstrations")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demonstrations/videos")
    args = parser.parse_args()
    report = json.loads((args.input / "result.json").read_text())
    dataset = args.input / "dataset"
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = pd.concat(pd.read_parquet(path) for path in sorted((dataset / "meta/episodes").rglob("*.parquet")))
    data = pd.concat(pd.read_parquet(path, columns=["episode_index", "action"]) for path in sorted((dataset / "data").rglob("*.parquet")))
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for attempt, result in enumerate(report["episodes"]):
        original_video = args.input / result["video_file"] if "video_file" in result else None
        if original_video is not None and not original_video.is_file():
            raise FileNotFoundError("An original recording declared by the collection report is missing.")
        reconstructed = "dataset_episode_index" not in result and original_video is None
        commands = np.load(args.input / f"commands_{attempt:03d}.npy", allow_pickle=False)
        if commands.shape != (result["frames"], 6):
            raise ValueError("The command array does not match the recorded trial.")
        filename = f"trial_{attempt + 1:02d}_seed_{result['seed']}.mp4"
        output = args.output / filename
        if output.exists():
            raise FileExistsError("Select an empty export directory; existing videos are not overwritten.")
        header = header_image(attempt, result, report["instruction"], reconstructed, report["attempted_episodes"])
        header_path = args.output / f".header_{attempt + 1:02d}.png"
        header.save(header_path)
        entry = {
            "trial_number": attempt + 1, "seed": result["seed"], "video": filename,
            "original_task_success": result["task_success"], "original_placed_pens": result["placed_pens"],
            "video_source": "Reconstructed from saved commands; original video not saved" if reconstructed else "Original recorded top and wrist cameras",
            "not_policy_inference": True,
        }
        try:
            if "dataset_episode_index" in result:
                episode_index = result["dataset_episode_index"]
                episode = episodes.loc[episodes["episode_index"] == episode_index].iloc[0]
                recorded_commands = np.stack(data.loc[data["episode_index"] == episode_index, "action"].to_numpy())
                if not np.array_equal(commands, recorded_commands):
                    raise RuntimeError("The dataset actions do not match the saved trial commands.")
                entry["dataset_episode_index"] = episode_index
                entry["dataset_actions_match_saved_commands"] = True
            if original_video is not None:
                shutil.copyfile(original_video, output)
                entry["video_source"] = "Original recorded teacher overview, top, and wrist cameras"
                entry["original_recording"] = result["video_file"]
            elif reconstructed:
                entry["reconstruction_verification"] = reconstruct_attempt(commands, result, header, output, report)
            else:
                entry["original_camera_segments"] = export_original(dataset, info, episode, header_path, output, len(commands))
            entry["video_verification"] = inspect_video(output, len(commands))
            entry["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            final_image = f"trial_{attempt + 1:02d}_final.png"
            shutil.copyfile(args.input / f"final_{attempt:03d}.png", args.output / final_image)
            entry["final_image"] = final_image
            entries.append(entry)
            manifest = {"instruction": report["instruction"], "teacher": report["teacher"],
                        "original_successes": report["successful_episodes"], "original_attempts": report["attempted_episodes"],
                        "exported_videos": len(entries), "trials": entries}
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            write_index(args.output, entries, report)
            print(json.dumps({"trial": attempt + 1, "video": filename,
                              "source": entry["video_source"], "verified": True}), flush=True)
        finally:
            header_path.unlink(missing_ok=True)
    if len(entries) != report["attempted_episodes"]:
        raise RuntimeError("Some original trials were not exported.")


if __name__ == "__main__":
    main()
