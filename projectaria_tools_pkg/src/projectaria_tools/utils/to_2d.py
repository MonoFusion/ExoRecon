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
import numpy as np
import os

import rerun as rr

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    filter_points_from_count,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId

from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline, ToTransform3D

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrs",
        type=str,
        required=True,
        help="path to VRS file",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        help="path to the MPS closed loop trajectory file",
    )
    parser.add_argument(
        "--points",
        type=str,
        help="path to the MPS global point file",
    )
    parser.add_argument(
        "--eyegaze",
        type=str,
        help="path to the MPS eye gaze file",
    )

    # Add options that does not show by default, but still accessible for debugging purpose
    parser.add_argument(
        "--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS
    )
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()

    #
    # Gather data input
    # - If MPS data has not been provided we try to find them automatically using default folder hierarchy
    #
    vrs_folder_path = os.path.dirname(args.vrs)

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

    possible_eyegaze_file_path = os.path.join(
        vrs_folder_path, "eye_gaze", "general_eye_gaze.csv"
    )
    if not args.eyegaze and os.path.exists(possible_eyegaze_file_path):
        args.eyegaze = possible_eyegaze_file_path

    # Try load legacy file name
    # (global_points == semidense_points)
    possible_points_file_path = os.path.join(
        vrs_folder_path, "trajectory", "global_points.csv.gz"
    )
    if not args.points and os.path.exists(possible_points_file_path):
        args.points = possible_points_file_path

    # (generalized_eye_gaze == general_eye_gaze)
    possible_eyegaze_file_path = os.path.join(
        vrs_folder_path, "eye_gaze", "generalized_eye_gaze.csv"
    )
    if not args.eyegaze and os.path.exists(possible_eyegaze_file_path):
        args.eyegaze = possible_eyegaze_file_path

    mps_data_available = args.trajectory or args.points or args.eyegaze

    print(
        f"""
    Trying to load the following list of files:
    - vrs: {args.vrs}
    - trajectory/closed_loop_trajectory: {args.trajectory}
    - trajectory/point_cloud: {args.points}
    - eye_gaze/general_eye_gaze: {args.eyegaze}
    """
    )

    #
    # MPS Data loading
    #
    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None
    eyegaze_data = mps.read_eyegaze(args.eyegaze) if args.eyegaze else None

    # Initializing Rerun viewer
    rr.init("MPS Data Viewer", spawn=True)

    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        # Down sample points
        points_data_down_sampled = filter_points_from_count(points_data, 500_000)
        # Retrieve point position
        point_positions = [it.position_world for it in points_data_down_sampled]
        rr.log(
            "world/points",
            rr.Points3D(point_positions, radii=0.006),
            timeless=True,
        )

    #
    # Log device trajectory (reduce sample sample count for display)
    #
    if trajectory_data:
        device_trajectory = [
            it.transform_world_device.translation()[0] for it in trajectory_data
        ][0::10]
        rr.log(
            "world/device_trajectory",
            rr.LineStrips3D(device_trajectory),
            timeless=True,
        )

    #
    # Go over RGB timestamps and
    # - Plot camera pose
    # - Plot user eye gaze
    #

    provider = data_provider.create_vrs_data_provider(args.vrs)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    #
    # Log RGB camera calibration
    #
    if mps_data_available:
        rr.log(
            f"world/device/{rgb_stream_label}",
            rr.Pinhole(
                resolution=[
                    rgb_camera_calibration.get_image_size()[0]
                    / args.down_sampling_factor,
                    rgb_camera_calibration.get_image_size()[1]
                    / args.down_sampling_factor,
                ],
                focal_length=float(
                    rgb_camera_calibration.get_focal_lengths()[0]
                    / args.down_sampling_factor
                ),
            ),
            timeless=True,
        )


    device_calib = provider.get_device_calibration()
    all_sensor_labels = device_calib.get_all_labels()
    print(f"device calibration contains calibrations for the following sensors \n {all_sensor_labels}")

    camera_name = "camera-rgb"
    transform_device_camera = device_calib.get_transform_device_sensor(camera_name).to_matrix()
    transform_camera_device = np.linalg.inv(transform_device_camera)
    print(f"Device calibration origin label {device_calib.get_origin_label()}")
    print(f"{camera_name} has extrinsics of \n {transform_device_camera}")

    rgb_calib = device_calib.get_camera_calib("camera-rgb")
    if rgb_calib is not None:
        # project a 3D point in device frame [camera-slam-left] to rgb camera

        point_in_device = np.array([0, 0, 10])
        point_in_camera = (
            np.matmul(transform_camera_device[0:3,0:3], point_in_device.transpose())
            + transform_camera_device[0:3,3]
        )
            

        maybe_pixel = rgb_calib.project(point_in_camera)
        if maybe_pixel is not None:
            print(
                f"Get pixel {maybe_pixel} within image of size {rgb_calib.get_image_size()}"
            )

    # Configure the loop for data replay
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(rgb_stream_id)  # RGB Stream Id
    rgb_frame_count = provider.get_num_data(rgb_stream_id)



    progress_bar = tqdm(total=rgb_frame_count)
    # Iterate over the data and LOG data as we see fit
    for data in provider.deliver_queued_sensor_data(deliver_option):
        device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
        rr.set_time_nanos("device_time", device_time_ns)
        rr.set_time_sequence("timestamp", device_time_ns)
        progress_bar.update(1)

        # Camera pose
        # 'exp', 'from_matrix', 'from_matrix3x4', 'from_quat_and_translation', 'inverse', 'log', 
        #'rotation', 'to_matrix', 'to_matrix3x4', 'to_quat_and_translation', 'translation']
        if trajectory_data:
            pose_info = get_nearest_pose(trajectory_data, device_time_ns)
            if pose_info:
                T_world_device = pose_info.transform_world_device
                T_device_camera = rgb_camera_calibration.get_transform_device_camera()##Sophus.SE3
                #print(T_device_camera, dir(T_device_camera))
                rr.log(
                    "world/device",
                    ToTransform3D(T_world_device, False),
                )
                rr.log(
                    f"world/device/{rgb_stream_label}",
                    ToTransform3D(T_device_camera, False),
                )

        if data.sensor_data_type() == SensorDataType.IMAGE:
            img = data.image_data_and_record()[0].to_numpy_array()
            if args.down_sampling_factor > 1:
                img = img[:: args.down_sampling_factor, :: args.down_sampling_factor]
            # Note: We configure the QUEUE to return only RGB image, so we are sure this image is corresponding to a RGB frame
            rr.log(
                f"world/device/{rgb_stream_label}",
                rr.Image(img).compress(jpeg_quality=args.jpeg_quality),
            )



if __name__ == "__main__":
    main()
