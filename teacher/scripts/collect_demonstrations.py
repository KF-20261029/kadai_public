"""Collect physically successful SO-101 demonstrations with an explicit IK teacher."""

import argparse
import json
from pathlib import Path
import subprocess
import shutil
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw

from simulation import JOINT_NAMES, ROOT, Simulation

TASK = 'Grab pens and place into pen holder'


class Teacher:
    def __init__(self, sim, capture=False):
        self.sim = sim
        self.kinematics = mujoco.MjData(sim.model)
        self.body = sim.model.body('gripper').id
        self.capture = capture
        self.frames = []
        self.commands = []
        self.phase = 'initial'

    def solve(self, point, axis, direction):
        sim = self.sim
        start = sim.data.qpos[sim.qpos_ids[:5]].copy()
        def residual(angles):
            data = self.kinematics
            data.qpos[:] = sim.data.qpos
            data.qpos[sim.qpos_ids[:5]] = angles
            mujoco.mj_forward(sim.model, data)
            rotation = data.xmat[self.body].reshape(3, 3)
            position = data.xpos[self.body] + rotation @ np.array([0, -0.000218, -0.1012])
            return np.r_[(position - point) * 10, (rotation[:, 1] - axis) * 0.5,
                         (rotation[:, 2] - direction) * 0.5, (angles - start) * 0.003]
        low, high = sim.ranges[:5, 0] + 1e-5, sim.ranges[:5, 1] - 1e-5
        result = least_squares(residual, np.clip(start, low, high), bounds=(low, high), max_nfev=150)
        return np.r_[np.rad2deg(result.x), 30.0]

    def move(self, action, seconds):
        sim = self.sim
        initial = sim.state()
        for step in range(int(round(seconds * 30))):
            fraction = (step + 1) / (seconds * 30)
            fraction = fraction * fraction * (3 - 2 * fraction)
            command = (initial * (1 - fraction) + action * fraction).astype(np.float32)
            if self.capture:
                self.frames.append({
                    'observation.images.top': sim.render('top'),
                    'observation.images.wrist': sim.render('wrist_cam'),
                    'observation.state': sim.state().copy(), 'action': command.copy(), 'task': TASK,
                    '_overview': sim.render('overview'),
                })
            self.commands.append(command.copy())
            sim.step(command)
        mujoco.mj_forward(sim.model, sim.data)

    def grasp(self, index):
        sim = self.sim
        position = sim.pen_positions()[index].copy()
        pen_axis = sim.data.xmat[sim.pen_ids[index]].reshape(3, 3)[:, 0]
        axis = -pen_axis if index == 0 else pen_axis
        self.phase = 'approach'
        self.move(self.solve(position + [0, 0, 0.075], axis, np.array([0, 0, 1])), 2)
        self.phase = 'descend'
        self.move(self.solve(position, axis, np.array([0, 0, 1])), 2)
        self.phase = 'close'
        action = sim.state(); action[5] = 3
        self.move(action, 1)
        self.phase = 'lift'
        action = self.solve(position + [0, 0, 0.065], axis, np.array([0, 0, 1])); action[5] = 3
        self.move(action, 3)
        return bool(sim.pen_positions()[index, 2] >= 0.045)

    def place(self, index, position=None):
        sim = self.sim
        position = sim.pen_positions()[index].copy() if position is None else position
        self.phase = 'orient'
        point = np.r_[position[:2], 0.16]
        axis = np.array([0, 0, 1])
        direction = np.r_[-point[:2], 0]; direction /= np.linalg.norm(direction)
        action = self.solve(point, axis, direction); action[5] = 3
        self.move(action, 3)
        self.phase = 'transport'
        holder = sim.data.xpos[sim.holder_id]
        offset = np.array([[0, 0], [-0.012, 0.008], [0.010, 0.005]])[index]
        point = np.r_[holder[:2] + offset, 0.16]
        direction = np.r_[-point[:2], 0]; direction /= np.linalg.norm(direction)
        action = self.solve(point, axis, direction); action[5] = 3
        self.move(action, 3)
        self.phase = 'lower'
        point[2] = 0.105
        action = self.solve(point, axis, direction); action[5] = 3
        self.move(action, 2)
        self.phase = 'release'
        action = sim.state(); action[5] = 20
        self.move(action, 1)
        self.phase = 'retreat'
        point[2] = 0.2
        action = self.solve(point, axis, direction); action[5] = 20
        self.move(action, 2)
        return bool(sim.placement()[index])

    def transfer(self, index):
        position = self.sim.pen_positions()[index].copy()
        if not self.grasp(index):
            return False
        return self.place(index, position)

    def run(self):
        completed = []
        for index in range(3):
            passed = self.transfer(index)
            completed.append(passed)
            print(json.dumps({'pen': index, 'placed': passed,
                              'positions': self.sim.pen_positions().tolist()}), flush=True)
            if not passed:
                break
        self.phase = 'hold'
        stable = True
        for _ in range(90):
            self.move(self.sim.state(), 1 / 30)
            stable = stable and bool(self.sim.placement().all())
        return len(completed) == 3 and all(completed) and stable


def save_video(frames, path, trial_number=None, seed=None, success=None):
    process = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1280x540', '-framerate', '30',
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast',
        '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-map_metadata', '-1', str(path),
    ], stdin=subprocess.PIPE)
    try:
        for index, frame in enumerate(frames):
            canvas = Image.new('RGB', (1280, 540), 'white')
            canvas.paste(Image.fromarray(frame['_overview']), (0, 60))
            canvas.paste(Image.fromarray(frame['observation.images.top']).resize((320, 240)), (640, 60))
            canvas.paste(Image.fromarray(frame['observation.images.wrist']).resize((320, 240)), (960, 60))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 10), 'IK demonstration collection / SO-101 / MuJoCo', fill='black')
            draw.text((20, 30), 'Instruction: ' + TASK + f' | Simulation: {index / 30:.2f} s', fill='black')
            draw.text((660, 340), 'Demonstration teacher uses simulator object poses.', fill='black')
            draw.text((660, 380), 'This video is not pi0.5 inference.', fill='black')
            draw.text((660, 420), 'Actual physical grasp and placement; no weld or teleport.', fill='black')
            if trial_number is not None:
                outcome = 'SUCCESS' if success else 'FAILED'
                draw.text((660, 460), f'Trial {trial_number} | Seed {seed} | Result: {outcome}', fill='black')
                draw.text((660, 490), 'Original recording from this collection run.', fill='black')
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        if process.wait():
            raise RuntimeError('The demonstration video could not be encoded.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--successful-episodes', type=int, default=None,
                        help='Stop after this many successes; --episodes remains the attempt limit.')
    parser.add_argument('--position-jitter', type=float, default=0.008,
                        help='Maximum initial XY displacement per pen in meters.')
    parser.add_argument('--yaw-jitter', type=float, default=0.3,
                        help='Maximum initial yaw angle per pen in radians.')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demonstrations')
    args = parser.parse_args()
    if args.episodes < 1 or (args.successful_episodes is not None and args.successful_episodes < 1):
        parser.error('Episode counts must be positive.')
    if not args.check_only and (args.output / 'dataset').exists():
        parser.error('The dataset directory already exists. Select an empty output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.check_only:
        (args.output / 'recordings').mkdir(exist_ok=True)
    sim = Simulation()
    dataset = None
    results = []
    started = time.perf_counter()

    def save_report(completed):
        report = {'teacher': 'Explicit joint position commands calculated by inverse kinematics',
                  'instruction': TASK, 'physics_only': True, 'objects_teleported_during_episode': False,
                  'grasp_weld_used': False, 'successful_episodes': sum(r['task_success'] for r in results),
                  'attempted_episodes': len(results), 'dataset_written': dataset is not None,
                  'requested_successful_episodes': args.successful_episodes,
                  'maximum_attempts': args.episodes, 'collection_completed': completed,
                  'collection_wall_seconds': time.perf_counter() - started,
                  'initial_position_jitter_m': args.position_jitter,
                  'initial_yaw_jitter_rad': args.yaw_jitter,
                  'all_attempt_videos_recorded': not args.check_only,
                  'episodes': results}
        temporary = args.output / 'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output / 'result.json')
        return report

    try:
        for index in range(args.episodes):
            seed = args.seed + index
            sim.reset(seed, position_jitter=args.position_jitter, yaw_jitter=args.yaw_jitter)
            initial_positions = sim.pen_positions().tolist()
            teacher = Teacher(sim, capture=not args.check_only)
            success = teacher.run()
            result = {'seed': seed, 'task_success': success, 'frames': len(teacher.commands),
                      'initial_pen_positions_m': initial_positions,
                      'placed_pens': sim.placement().tolist(), 'final_pen_positions_m': sim.pen_positions().tolist()}
            results.append(result)
            np.save(args.output / f'commands_{index:03d}.npy', np.array(teacher.commands))
            Image.fromarray(sim.render('overview')).save(args.output / f'final_{index:03d}.png')
            if not args.check_only:
                video_file = f'recordings/trial_{index + 1:03d}_seed_{seed}.mp4'
                save_video(teacher.frames, args.output / video_file, index + 1, seed, success)
                result['video_file'] = video_file
                result['video_source'] = 'Original recording of this attempt; not a reconstruction'
            print(json.dumps(result), flush=True)
            if success and not args.check_only:
                if dataset is None:
                    from lerobot.datasets.lerobot_dataset import LeRobotDataset
                    features = {
                        'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': list(JOINT_NAMES)},
                        'action': {'dtype': 'float32', 'shape': (6,), 'names': list(JOINT_NAMES)},
                        **{key: {'dtype': 'video', 'shape': (480, 640, 3),
                                 'names': ['height', 'width', 'channels']}
                           for key in ('observation.images.top', 'observation.images.wrist')},
                    }
                    dataset = LeRobotDataset.create('local/so101_pens', fps=30, features=features,
                                                    root=args.output / 'dataset', robot_type='so101_follower',
                                                    vcodec='h264', streaming_encoding=True, encoder_threads=4)
                for frame in teacher.frames:
                    dataset.add_frame({key: value for key, value in frame.items() if key != '_overview'})
                result['dataset_episode_index'] = dataset.num_episodes
                dataset.save_episode()
                if sum(item['task_success'] for item in results) == 1:
                    shutil.copyfile(args.output / result['video_file'], args.output / 'demonstration.mp4')
            del teacher
            report = save_report(completed=False)
            print(json.dumps({'attempted_episodes': len(results),
                              'successful_episodes': report['successful_episodes'],
                              'collection_wall_seconds': report['collection_wall_seconds']}), flush=True)
            if args.successful_episodes is not None and report['successful_episodes'] >= args.successful_episodes:
                break
    finally:
        if dataset is not None:
            dataset.finalize()
        sim.close()
    report = save_report(completed=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
