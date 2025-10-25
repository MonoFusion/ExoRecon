# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from projectaria_tools.core import calibration, data_provider
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import get_nearest_pose
from projectaria_tools.core.stream_id import StreamId


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--vrs",
        type=str,
        required=True,
        help="Path to VRS file",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to execute",
    )
    
    parser.add_argument(
        "--resize",
        type=int,
        default=0, 
        required=True,
        help="Resize parameter",
    )
    
    parser.add_argument(
        "--debug",
        default=30000000,
        type=int,
        help="Debug parameter",
    )
    
    parser.add_argument(
        "--cam",
        type=str,
        required=True,
        help="Camera parameter",
    )
    
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to the MPS closed loop trajectory file",
    )
    
    parser.add_argument(
        "--points",
        type=str,
        help="Path to the MPS global point file",
    )
    
    parser.add_argument(
        "--eyegaze",
        type=str,
        help="Path to the MPS eye gaze file",
    )
    
    # Hidden options for debugging
    parser.add_argument(
        "--down_sampling_factor", 
        type=int, 
        default=4, 
        help=argparse.SUPPRESS
    )
    
    parser.add_argument(
        "--jpeg_quality", 
        type=int, 
        default=75, 
        help=argparse.SUPPRESS
    )

    return parser.parse_args()


def _resolve_dataset_root(vrs_path: str) -> Path:
    return Path(vrs_path).expanduser().resolve().parent


def _resolve_static_camera_calibrations(dataset_root: Path) -> Optional[Path]:
    """Try a few likely locations for gopro calibration CSV."""

    candidates = []
    trajectory_dir = dataset_root / "trajectory"
    # Search order: cwd, dataset root, <root>/trajectory
    cwd = Path.cwd()
    candidates.append(cwd / "gopro_calibs.csv")
    candidates.append(cwd / "trajectory" / "gopro_calibs.csv")
    candidates.append(dataset_root / "gopro_calibs.csv")
    candidates.append(trajectory_dir / "gopro_calibs.csv")

    seen = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def vis(args):
    """Visualize and process video frames"""
    dataset_root = _resolve_dataset_root(args.vrs)
    rgb_stream_id = StreamId("214-1")
    provider = data_provider.create_vrs_data_provider(args.vrs)

    num_rgb_frames = provider.get_num_data(rgb_stream_id)
    image_data = provider.get_image_data_by_index(rgb_stream_id, 0)
    timestamp_ns = image_data[1].capture_timestamp_ns
    timestamp_sec_0 = timestamp_ns

    frame_aligned_dir = dataset_root / "frame_aligned_videos"
    timestep_path = frame_aligned_dir / "timestep.txt"
    if not timestep_path.exists():
        raise FileNotFoundError(
            f"Missing timestep file: {timestep_path}. Run viewer from a dataset "
            "folder that contains frame_aligned_videos/timestep.txt."
        )

    with open(timestep_path, 'r') as file:
        number_str = file.read().strip()
    start_idx = int(number_str)

    # Process frames
    max_vis_frames = max(0, args.debug)
    if max_vis_frames == 0:
        max_vis_frames = 300
    frame_count = min(300, max_vis_frames)
    end_idx = min(start_idx + frame_count, num_rgb_frames)

    for index in range(start_idx, end_idx):
        image_data = provider.get_image_data_by_index(rgb_stream_id, index)
        timestamp_ns = image_data[1].capture_timestamp_ns
        timestamp_sec = (timestamp_ns - timestamp_sec_0) / 1e9

        # Process each camera folder
        for frames_folder in ['cam01', 'cam02', 'cam03', 'cam04']:
            main_path = dataset_root / 'processed_frames' / frames_folder
            os.makedirs(main_path, exist_ok=True)

            output_image = main_path / f"{index:05d}.jpg"

            # Skip if output image already exists
            if output_image.exists():
                print(f"Skipping {output_image}, already exists.")
                continue

            # Run ffmpeg command
            ffmpeg_command = [
                'ffmpeg',
                '-ss', str(timestamp_sec),
                '-i', str(frame_aligned_dir / f'{frames_folder}.mp4'),
                '-frames:v', '1',
                str(output_image)
            ]
            subprocess.run(ffmpeg_command, check=True)

    calibration_path = _resolve_static_camera_calibrations(dataset_root)
    if not calibration_path:
        print(
            "Skipping image rectification: could not locate gopro_calibs.csv in the "
            "dataset root or trajectory folder."
        )
        return

    static_cameras = mps.read_static_camera_calibrations(str(calibration_path))
    undist_root = dataset_root / 'undist_processed_frames'
    os.makedirs(undist_root, exist_ok=True)

    for frames_folder in ['cam01', 'cam02', 'cam03', 'cam04']:
        output_path = undist_root / f'undist_{frames_folder}'
        os.makedirs(output_path, exist_ok=True)
        main_path = dataset_root / 'processed_frames' / frames_folder
        jpg_files = glob.glob(str(main_path / '*.jpg'))

        src_calib = static_cameras[int(frames_folder[-1]) - 1]

        for file in jpg_files:
            path = output_path / os.path.basename(file)

            # Skip if rectified image already exists
            if path.exists():
                print(f"Skipping {path}, already exists.")
                continue

            # Process the image
            img = cv2.imread(file)
            h, w = img.shape[:2]
            
            rev = calibration.CameraCalibration(
                'name',
                calibration.KANNALA_BRANDT_K3,
                src_calib.intrinsics.astype(np.float32),
                src_calib.transform_world_cam,
                src_calib.width,
                src_calib.height,
                None,
                1,
                src_calib.camera_uid
            )
            
            dst_calib = calibration.get_linear_camera_calibration(
                src_calib.width, src_calib.height, 
                src_calib.intrinsics.astype(np.float64)[0]
            )
            
            print(dst_calib.get_focal_lengths())
            rectified_array = calibration.distort_by_calibration(img, dst_calib, rev)

            # Save the rectified image
            print(f"Saving to {path}")
            cv2.imwrite(str(path), rectified_array)


def matrices(args):
    dataset_root = _resolve_dataset_root(args.vrs)
    vrs_folder_path = dataset_root

    possible_trajectory_file_path = os.path.join(
        vrs_folder_path, "trajectory", "closed_loop_trajectory.csv"
    )
    if not args.trajectory and os.path.exists(possible_trajectory_file_path):
        args.trajectory = possible_trajectory_file_path

    possible_points_file_path = os.path.join(
        vrs_folder_path, "trajectory", "semidense_points.csv.gz"
    )
    if not args.points and os.path.exists(possible_points_file_path):
        args.points = possible_points_file_path

    trajectory_data = None
    if args.cam == 'rgb':
        if not args.trajectory:
            raise FileNotFoundError(
                "Could not locate closed_loop_trajectory.csv. Provide it via --trajectory "
                "or place it under <dataset>/trajectory/."
            )
        trajectory_data = mps.read_closed_loop_trajectory(args.trajectory)

    provider = data_provider.create_vrs_data_provider(args.vrs)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)

    # get all image data by index
    num_data = provider.get_num_data(sensor_stream_id)
    max_samples = max(0, args.debug)
    if max_samples > 0:
        num_data = min(num_data, max_samples)

    if args.cam =='rgb':
        for index in tqdm(range(0, num_data)):
            # Block 1: Image data retrieval and pose calculation

            image_data = provider.get_image_data_by_index(sensor_stream_id, index)
            capture_timestamp_ns = image_data[1].capture_timestamp_ns
            frame_id = index
            pose_info = get_nearest_pose(trajectory_data, capture_timestamp_ns)
            rgb_calib = device_calibration.get_camera_calib("camera-rgb")
            T_world_device = pose_info.transform_world_device.to_matrix()

            T_device_camera = rgb_camera_calibration.get_transform_device_camera().to_matrix()  # Assuming Sophus.SE3
            transform_camera_device = np.linalg.inv(T_device_camera)
            T_device_from_world = np.linalg.inv(T_world_device)
            
            to_be_append_1={'image_id':frame_id, 'T_camera_from_world': np.matmul(transform_camera_device, T_device_from_world)}
            row_str = str(frame_id)+' '
            for row in to_be_append_1['T_camera_from_world']:
                # 创建一个由该行元素组成的字符串，元素之间用空格分隔
                row_str += ' '.join(map(str, row))
                row_str += ' '
            output_file = dataset_root / 'train.matrices.txt'
            with open(output_file, 'a') as file:
                file.write(row_str+'\n')

    elif 'cam0' in args.cam:
        calibration_path = _resolve_static_camera_calibrations(dataset_root)
        if not calibration_path:
            raise FileNotFoundError(
                "Could not locate gopro_calibs.csv for static cameras. Place it under "
                "the dataset root or trajectory folder."
            )
        static_cameras = mps.read_static_camera_calibrations(str(calibration_path))
        src_calib=(static_cameras[int(args.cam[-1])-1])
        output_file = dataset_root / f'{args.cam}_train.matrices.txt'
        for index in tqdm(range(0, num_data)):
            row_str = str(index)+' '
            for row in src_calib.transform_world_cam.to_matrix():
                row_str += ' '.join(map(str, row))
                row_str += ' '
            with open(output_file, 'a') as file:
                file.write(row_str+'\n')
    else:
        raise ValueError(f"Unsupported camera selection: {args.cam}")


def main():
    args = parse_args()
    if args.task=='vis':
        vis(args)
    elif args.task=='matr':
        matrices(args)
    elif args.task=='mate':
        vis(args)
        matrices(args)

if __name__ == "__main__":
    main()
