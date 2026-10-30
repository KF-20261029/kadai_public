"""Convert timestamped ROS image messages to a constant-rate MP4."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('output')
    parser.add_argument('--topic', default='/mujoco_camera_plugin/overview/color')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--start-ns', type=int)
    args = parser.parse_args()
    if args.fps <= 0 or Path(args.output).exists():
        raise ValueError('The frame rate must be positive and the output must not exist.')
    if args.seconds is not None and args.seconds <= 0:
        raise ValueError('The requested duration must be positive.')
    limit = None if args.seconds is None else round(args.seconds * args.fps)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    process = None
    first_stamp = args.start_ns
    previous_stamp = None
    previous_frame = None
    frames_written = 0
    images_read = 0
    try:
        while reader.has_next():
            _, data, _ = reader.read_next()
            msg = deserialize_message(data, Image)
            if msg.encoding != 'rgb8' or msg.step < msg.width * 3 or len(msg.data) != msg.height * msg.step:
                raise ValueError('Expected a valid RGB8 image.')
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            frame = pixels[:, :msg.width * 3].tobytes()
            if process is None:
                width, height = msg.width, msg.height
                if first_stamp is None:
                    first_stamp = stamp
                process = subprocess.Popen([
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
                    '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                    '-framerate', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264',
                    '-threads', '2',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.output,
                ], stdin=subprocess.PIPE)
            if (msg.width, msg.height) != (width, height) or (previous_stamp is not None and stamp < previous_stamp):
                raise ValueError('Image dimensions changed or timestamps are out of order.')
            target = max(0, round((stamp - first_stamp) * args.fps / 1_000_000_000))
            if limit is not None:
                target = min(target, limit)
            while previous_frame is not None and frames_written < target:
                process.stdin.write(previous_frame)
                frames_written += 1
            previous_stamp = stamp
            previous_frame = frame
            images_read += 1
            if limit is not None and frames_written >= limit:
                break
        if process is None:
            raise ValueError('No image messages were found.')
        if limit is not None and stamp < first_stamp + round(args.seconds * 1e9) - 500_000_000:
            raise RuntimeError('The recording is shorter than the requested video.')
        end_frame = frames_written + 1 if limit is None else limit
        while frames_written < end_frame:
            process.stdin.write(previous_frame)
            frames_written += 1
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('FFmpeg did not complete successfully.')
        print(f'Images read: {images_read}; video frames: {frames_written}; fps: {args.fps}')
    except BaseException:
        if process is not None and process.poll() is None:
            process.stdin.close()
            process.wait()
        raise


if __name__ == '__main__':
    main()
