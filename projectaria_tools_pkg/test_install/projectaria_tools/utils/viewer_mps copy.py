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
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

import argparse
import numpy as np
import os
import cv2
from tqdm import tqdm
import rerun as rr
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    filter_points_from_count,
    get_nearest_pose,
)
from projectaria_tools.core.mps import StaticCameraCalibration
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId
import plotly.graph_objects as go
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline, ToTransform3D
import projectaria_tools.core.mps as mps
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
        "--task",
        type=str,
        required=True,
        help="path to VRS file",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=0, 
        required=True,
        help="path to VRS file",
    )

    parser.add_argument(
        "--debug",
        default=30000000,
        type=int,
        help="path to the MPS closed loop trajectory file",
    )
    parser.add_argument(
        "--cam",
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

def correct_fisheye_image(fisheye_img, h, w, provider=None):
    height, width = fisheye_img.shape[:2]
    corrected_img = np.zeros_like(fisheye_img)
    
    for y in range(height):
        for x in range(width):
            ### for every pixel in rgb fisheye image
            pixel_array = np.array([[x], [y]], dtype=np.float32)
            sensor_name = "camera-rgb"
            device_calib = provider.get_device_calibration()
            src_calib = device_calib.get_camera_calib(sensor_name)
            #print(src_calib)
            ### for uv => project in 3d => 

            ### for [0, 0]          uv
            dst_calib = calibration.get_linear_camera_calibration(h, w, 1220.439291944814, "camera-rgb")
        
            point_3d=src_calib.unproject(pixel_array)
            if point_3d is not None: 
                point_3d=np.array(point_3d).reshape(3,1)
                try: 
                    x_corrected, y_corrected = dst_calib.project(point_3d)
                    #   print(x_corrected)
                    if 0 <= (x_corrected) < width and 0 <= (y_corrected) < height:
                        corrected_img[int(y_corrected), int(x_corrected)] = fisheye_img[int(y), int(x)]  
                except:
                    continue
    return corrected_img


def converter(maybe_pixel):
    maybe_pixel = np.array(maybe_pixel)
    param = [1220.439291944814, 1442.529221859291, 1441.997151419051]
    param_extra=[0.3903350032156671, -0.3628338118352796, -0.0005619143992431268, -0.0009269308782712309, -0.2392777738885203, 1.72968866010095, -2.096873915251233, 0.7499578205635813,  0.002016426267951603, -0.0004207638630777981, 0.0002380609461884488, -0.0005070359350339245]
    #param_extra=[0.3903350032156671,-0.3628338118352796,-0.2392777738885203,1.72968866010095,-2.096873915251233,0.7499578205635813,-0.0005619143992431268,-0.0009269308782712309,0.002016426267951603,-0.0004207638630777981,0.0002380609461884488,-0.0005070359350339245]

    focal_length_x, focal_length_y = param[0], param[0]

    assert len(param_extra) == 12
    principal_point_x, principal_point_y= param[1], param[2]

    camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
            [0, focal_length_y, principal_point_y],
            [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.array(param_extra, dtype=np.float32)  # Replace k1, k2, p1, p2, k3 with your values
    maybe_pixel = cv2.undistortPoints(maybe_pixel, camera_matrix, dist_coeffs)[0][0]
    maybe_pixel[0],  maybe_pixel[1] = focal_length_x*maybe_pixel[0]+principal_point_x, focal_length_y*maybe_pixel[1]+principal_point_y
    return maybe_pixel

def convert_all(img):
    #maybe_pixel = np.array(maybe_pixel)
    param = [1220.439291944814, 1442.529221859291, 1441.997151419051]
    param_extra=[0.3903350032156671, -0.3628338118352796, -0.0005619143992431268, -0.0009269308782712309, -0.2392777738885203, 1.72968866010095, -2.096873915251233, 0.7499578205635813,  0.002016426267951603, -0.0004207638630777981, 0.0002380609461884488, -0.0005070359350339245]
    focal_length_x, focal_length_y = param[0], param[0]

    assert len(param_extra) == 12
    principal_point_x, principal_point_y= param[1], param[2]

    camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
            [0, focal_length_y, principal_point_y],
            [0, 0, 1]], dtype=np.float32)


    dist_coeffs = np.array(param_extra, dtype=np.float32)  # Replace k1, k2, p1, p2, k3 with your values
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    return undistorted_img

def vis(args) :
    import os
    import subprocess
    #
    # Gather data input
    # - If MPS data has not been provided we try to find them automatically using default folder hierarchy
    #


    #sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    #image_data = provider.get_image_data_by_index(sensor_stream_id, 0)
    #image_array = image_data[0].to_numpy_array()
    #print(image_array.shape)
    import glob
    rgb_stream_id = StreamId("214-1")
    provider = data_provider.create_vrs_data_provider(args.vrs)
    #print(provider.get_timestamps_ns(rgb_stream_id))
    print(dir(provider))
    num_rgb_frames = provider.get_num_data(rgb_stream_id)
    image_data = provider.get_image_data_by_index(rgb_stream_id, 0)
    timestamp_ns = image_data[1].capture_timestamp_ns
    timestamp_sec_0 = timestamp_ns #/ 1e9
    with open('./frame_aligned_videos/timestep.txt', 'r') as file:
        number_str = file.read().strip()  

    # Convert the string to an integer
    start_idx = int(number_str)
        
    for index in range(start_idx, start_idx + 300):
        image_data = provider.get_image_data_by_index(rgb_stream_id, index)
        timestamp_ns = image_data[1].capture_timestamp_ns
        timestamp_sec = (timestamp_ns - timestamp_sec_0) / 1e9

        for frames_folder in ['cam01', 'cam02', 'cam03', 'cam04']:
            main_path = './processed_frames/'
            output_path = main_path + 'undist_' + frames_folder
            main_path += frames_folder
            os.makedirs(main_path, exist_ok=True)

            output_image = os.path.join(main_path, f"{index:05d}.jpg")

            # Check if the output image already exists
            if os.path.exists(output_image):
                print(f"Skipping {output_image}, already exists.")
                continue

            # Run ffmpeg command
            ffmpeg_command = [
                'ffmpeg',
                '-ss', str(timestamp_sec),
                '-i', f'frame_aligned_videos/{frames_folder}.mp4',
                '-frames:v', '1',
                output_image
            ]

            # Execute the command
            subprocess.run(ffmpeg_command)

    for frames_folder in ['cam01', 'cam02', 'cam03', 'cam04']:
        os.makedirs('./undist_processed_frames/', exist_ok=True)
        output_path = './undist_processed_frames/' + 'undist_' + frames_folder
        os.makedirs(output_path, exist_ok=True)
        main_path = './processed_frames/' + frames_folder
        jpg_files = glob.glob(os.path.join(main_path, '*.jpg'))
        static_cameras_path = './gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        print(len(static_cameras))
        src_calib = static_cameras[int(frames_folder[-1]) - 1]
        for file in jpg_files:
            # Define the output file path
            path = os.path.join(output_path, os.path.basename(file))

            # Check if the rectified image already exists
            if os.path.exists(path):
                print(f"Skipping {path}, already exists.")
                continue

            # Process the image
            img = cv2.imread(file)
            h, w = img.shape[:2]
            rev = calibration.CameraCalibration(
                'name',  # Camera name or identifier
                calibration.KANNALA_BRANDT_K3,  # Camera model type
                src_calib.intrinsics.astype(np.float32),  # Camera intrinsics
                src_calib.transform_world_cam,  # World to camera transform
                src_calib.width,  # Image width
                src_calib.height,  # Image height
                None,  # Optional float parameter, such as field of view
                1,  # You need to provide this value
                src_calib.camera_uid  # Unique camera identifier
            )
            dst_calib = calibration.get_linear_camera_calibration(
                src_calib.width, src_calib.height, src_calib.intrinsics.astype(np.float64)[0])
            print(dst_calib.get_focal_lengths())
            rectified_array = calibration.distort_by_calibration(img, dst_calib, rev)

            # Save the rectified image
            print(f"Saving to {path}")
            cv2.imwrite(path, rectified_array)







            

    #h, w = image_data[0].get_height(), image_data[0].get_width()
    #device_calib = provider.get_device_calibration()
    #src_calib = device_calib.get_camera_calib(sensor_name)
    #print(dir(src_calib))
    #src_calib.maxSolidAngle
    ##########CameraCalibration(label: camera-rgb, model name: Fisheye624, principal point: [705.015, 704.749], 
    # focal length: [610.22, 610.22], projection params: [610.22, 705.015, 704.749, 0.390335, -0.362834, -0.239278, 
    # 1.72969, -2.09687, 0.749958, -0.000561914, -0.000926931, 0.00201643, -0.000420764, 0.000238061, -0.000507036], 
    # image size (w,h): [1408, 1408], T_Device_Camera:(translation:[-0.00438125, -0.0123959, -0.00536101], 
    # quaternion(x,y,z,w):[0.3293, 0.0417021, 0.0362213, 0.942608]), serialNumber:0450577b730305354401100000000000)

    #'camera_uid', 'end_frame_idx', 'graph_uid', 'height', 'intrinsics', 
    # 'intrinsics_type', 'start_frame_idx', 'transform_world_cam', 'width']
    #print((calibration.get_linear_camera_calibration(h, w, 1220.439291944814)))
   #* @param maybeValidRadius [optional] radius of a circular mask that represents the valid area on
   #* the camera's sensor plane. Pixels out of this circular region are considered invalid. Setting
   #* this to nullopt means the entire sensor plane is valid.
   #* @param maxSolidAngle an angle theta representing the FOV cone of the camera. Rays out of
   #* [-theta, +theta] will be rejected during projection.
   #* @param serialNumber The serial number of the camera

    #dst_calib = calibration.get_linear_camera_calibration(h//4, w//4, src_calib.intrinsics.astype(np.float64)[0])

    #dst_calib = calibration.get_linear_camera_calibration(h, w, 1220.439291944814)

    #dst_calib = calibration.get_linear_camera_calibration(h//4, w//4, 1220.439291944814/4)
    #print(dir(dst_calib))
    # distort image


    #rectified_array = correct_fisheye_image(image_array, h, w, provider=provider)

    #rectified_array=convert_all(rectified_array)
    # visualize input and results

def point_7_gaussian(args):
    args = parse_args()
    debug=args.debug

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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]
    point_positions = np.array(point_positions)
    print(point_positions.shape)
    data=np.zeros((len(point_positions), 4))
    for i, line in enumerate(range(len(point_positions))):
        parts = [255, 0, 0]
        # point position
        data[i, 0] = int(parts[0])
        data[i, 1] = int(parts[1])
        data[i, 2] = int(parts[2])
        data[i, 3] = 1
    result = np.concatenate((point_positions, data), axis=1)
    print(result.shape)
    np.savez("init_pt_cld.npz", data=result)
    return data
    
    to_be_append_0={'id': id0, 'position':position,'frames':prepare} 
    line = to_point_3d(to_be_append_0)
    with open('3d_points.txt', 'a') as file:  # 使用 'a' 模式以追加内容
        file.write(line)

def transform_3d_points(transform, points):
    N = len(points)
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed_points_h = (transform @ points_h.T).T
    transformed_points = transformed_points_h[:, :-1]
    return transformed_points
def transform_points(points_camera, R, t):
    """
    Transforms points from the camera coordinate system to the world coordinate system.
    
    Parameters:
    - points_camera: A tensor of shape (N, 3), where N is the number of points.
    - R: A 3x3 rotation matrix.
    - t: A translation vector of shape (3,).
    
    Returns:
    - A tensor of shape (N, 3) containing points in the world coordinate system.
    """
    # Convert points_camera to homogeneous coordinates (N, 4)
    R=R.reshape(3,3)
    t=t.reshape(3)
    T = torch.zeros(4, 4)
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    points_world = (transform_3d_points(T, points_camera))
    return points_world

def initialize_params():
    init_pt_cld = np.load(f"init_pt_cld.npz")["data"]
    return init_pt_cld[:, :3], init_pt_cld[:, 3:6]
import open3d as o3d
def SLAM(args):
    import numpy as np
    args = parse_args()
    debug=args.debug
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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None


    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]

    provider = data_provider.create_vrs_data_provider(args.vrs)

    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    
    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    num_data = provider.get_num_data(sensor_stream_id)
    from PIL import Image
    
    all_matrices = []
    for index in range(1477, 1777):
        image_data = provider.get_image_data_by_index(sensor_stream_id, index)
        capture_timestamp_ns = image_data[1].capture_timestamp_ns
        frame_id = index
        h, w = image_data[0].get_height(), image_data[0].get_width()
        device_calib = provider.get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)
        focal=src_calib.get_focal_lengths()[0] 


        dst_calib = calibration.get_linear_camera_calibration(h, w, focal)
        undistort_image = calibration.distort_by_calibration(cv2.resize(image_data[0].to_numpy_array(), (h, w)), dst_calib, src_calib, InterpolationMethod.BILINEAR) #image_data[0].to_numpy_array()#
        if args.resize:
            new_w = args.resize
            new_h = int(h * (new_w / w))
            h, w = new_h, new_w

        
        print(dst_calib)
        undistort_image = cv2.resize(undistort_image, (h, w))
        path = './ims'
        filename = os.path.join(path, f'{index:05d}.jpg')
        undistort_image = cv2.cvtColor(undistort_image, cv2.COLOR_BGR2RGB)
        undistort_image = cv2.rotate(undistort_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filename,(undistort_image))
        pose_info = get_nearest_pose(trajectory_data, capture_timestamp_ns)
        T_world_device = pose_info.transform_world_device.to_matrix() ## transform_world_device  
        T_device_camera = rgb_camera_calibration.get_transform_device_camera().to_matrix()  # transform_device_camera

        #T_world_device = np.linalg.inv(T_world_device) # transform_device_world_
        #T_device_camera = np.linalg.inv(T_device_camera) # transform_camera_device_
        T_camera_from_world = np.dot(T_world_device, T_device_camera)
        #T_camera_from_world = np.linalg.inv(T_camera_from_world)

        all_matrices.append(T_camera_from_world)

        vrs_folder_path = os.path.dirname(args.vrs)

        # T_world_rgb_camera = T_world_device @ T_device_rgb_camera       transform_world_cam inv

        tx, ty, tz = T_camera_from_world[:3, 3]
        rotation_matrix = T_camera_from_world[:3, :3]

        rotation = R.from_matrix(rotation_matrix)
        rt=[tx, ty, tz]
        quaternion = list(rotation.as_quat())
        rt.extend(quaternion)

        #print(quaternion)
        camera_data = np.array(rt).reshape(1, -1)
        import json
        aaa = []
        # Assuming 'all_matrices' is a list of numpy arrays
        aaa.append([matrix.tolist() for matrix in all_matrices])  # Convert each ndarray to a list
        aaa = {'ex': aaa}

        with open(f'{path}/transformation_matrices.json', 'w') as f:
            json.dump(aaa, f, indent=4)




def Depth(args):
    import numpy as np
    args = parse_args()
    debug=args.debug
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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None


    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]
    position_dict = {point_id[i]: point_positions[i] for i in range(len(point_id))}
    provider = data_provider.create_vrs_data_provider(args.vrs)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    
    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    num_data = provider.get_num_data(sensor_stream_id)
    from PIL import Image
    
    all_matrices = []
    for index in range(1400):
        # Block 1: Image data retrieval and pose calculation

        image_data = provider.get_image_data_by_index(sensor_stream_id, index)
        capture_timestamp_ns = image_data[1].capture_timestamp_ns
        frame_id = index
        h, w = image_data[0].get_height(), image_data[0].get_width()
        device_calib = provider.get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)
        print(src_calib)
        focal=src_calib.get_focal_lengths()[0] 

        focal *= 1.1
            #index=70
        print(h, w)
        if args.resize:
            new_w = args.resize
            new_h = int(h * (new_w / w))
            print('f', (new_w / w))
            print('f', src_calib.get_focal_lengths()[0]  * (new_w / w))
            print('p', 703.5 * (new_w / w))
            focal = focal * (new_w / w)
            h, w = new_h, new_w
        dst_calib = calibration.get_linear_camera_calibration(h, w, 670)
        print(dst_calib, dir(dst_calib), dst_calib.get_focal_lengths(), dst_calib.get_principal_point())
        undistort_image = calibration.distort_by_calibration(cv2.resize(image_data[0].to_numpy_array(), (h, w)), dst_calib, src_calib, InterpolationMethod.BILINEAR)
        #undistort_image = cv2.rotate(undistort_image, cv2.ROTATE_90_CLOCKWISE)
        filename = os.path.join('images', f'{index:06d}.jpg')
        cv2.imwrite(filename,(undistort_image)[:,:,::-1])
        undistort_image = cv2.resize(undistort_image, (h, w))


        #print(dir(image_data[0]), dir(image_data[1]))

        

        pose_info = get_nearest_pose(trajectory_data, capture_timestamp_ns)
        
        rgb_calib = device_calibration.get_camera_calib("camera-rgb")
        TTT=dir(rgb_calib)
        print(TTT)
        # transform_device_camera: get_transform_device_sensor
        T_world_device = pose_info.transform_world_device.to_matrix() ## transform_world_device  
        T_device_camera = rgb_camera_calibration.get_transform_device_camera().to_matrix()  # transform_device_camera

        #T_world_device = np.linalg.inv(T_world_device) # transform_device_world_
        #T_device_camera = np.linalg.inv(T_device_camera) # transform_camera_device_
        T_camera_from_world = np.dot(T_world_device, T_device_camera)
        #T_camera_from_world = np.linalg.inv(T_camera_from_world)

        all_matrices.append(T_camera_from_world)

        vrs_folder_path = os.path.dirname(args.vrs)
        '''        init_pc=np.load('init_pt_cld.npz')['data']
                ##print(init_pc.shape) 我需要transform_world_cam


                # T_world_rgb_camera = T_world_device @ T_device_rgb_camera       transform_world_cam inv

                _, colors = init_pc[:, :3], init_pc[:, 3:6]
                # Camera data 
                print(T_camera_from_world)

                tx, ty, tz = T_camera_from_world[:3, 3]
                rotation_matrix = T_camera_from_world[:3, :3]

                rotation = R.from_matrix(rotation_matrix)
                rt=[tx, ty, tz]
                quaternion = list(rotation.as_quat())
                rt.extend(quaternion)'''

###   world -> cam 
        points_in_camera = transform_3d_points(T_camera_from_world, point_positions)

        depth_map = np.full((h, w), np.inf)

        binary_mask = np.zeros((h, w), dtype=np.uint8) 
        for point in (points_in_camera):
            maybe_pixel=dst_calib.project(point)    
            if maybe_pixel is not None:       
                x_pixel, y_pixel = int(maybe_pixel[0]), int(maybe_pixel[1])
                if 0 <= x_pixel < w and 0 <= y_pixel < h:
                    current_depth = point[2] 
                    depth_map[y_pixel, x_pixel] = current_depth                                                                                                                                                                                                                                                                                                                                                                                                     
                    binary_mask[y_pixel, x_pixel] = 1 

        valid_depths = depth_map[depth_map != np.inf]
        depth_map[depth_map==np.inf] = 0
        depth_map = depth_map.astype(float)*255
        depth_image = Image.fromarray(depth_map.astype(np.uint8))
        depth_image.save(f'./depth_ffola/depth_map_{index}.png')

        

        #print(quaternion)
        #camera_data = np.array(rt).reshape(1, -1)
    import json
    aaa = []
    # Assuming 'all_matrices' is a list of numpy arrays
    aaa.append([matrix.tolist() for matrix in all_matrices])  # Convert each ndarray to a list


    print(np.array(aaa).shape)
    aaa = {'ex': aaa}

    with open('transformation_matrices.json', 'w') as f:
        json.dump(aaa, f, indent=4)



    args = parse_args()
    debug=args.debug

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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None

    import numpy as np
    import open3d as o3d
    import pandas as pd
    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]
    point_positions=np.array(point_positions)

    position_dict = {point_id[i]: point_positions[i] for i in range(len(point_id))}
    final={}
    for id in point_id:
        final[(id)]=[]
    def pixels_to_ndc_new(w, h, depthggg=None):
        depthggg=torch.tensor(depthggg)
        y, x = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            indexing="ij",
        )
        #x = (x - c_x) * depth / f_x
        #y = (y - c_y) * depth / f_x
        try:
            return torch.stack([x, y, depthggg], dim=-1)
        except:
            return torch.stack([x, y, depthggg], dim=-1)

    fig = go.Figure()
    for ix in range(4):
        cam = ['cam01', 'cam02',  'cam03', 'cam04'][ix]
        #rgb_path=f'/Users/ha1o/Desktop/CMU/cam_{int(cam[-1])}_00000.png'
        #image=cv2.imread(rgb_path)
        #print(image.shape)
        static_cameras_path = '/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        #print(static_cameras)
        src_calib=(static_cameras[int(cam[-1])-1])
        h, w = src_calib.height, src_calib.width
        focal=src_calib.intrinsics.astype(np.float64)[0]
        if args.resize:
            new_w = args.resize
            new_h = int(h * (new_w / w))
            focal = focal * (new_w / w)
            h, w = new_h, new_w 
        print('RRRRRRRattio', (new_w / w))
        dst_calib = calibration.get_linear_camera_calibration(w, h, focal)
        point_positions = np.array(point_positions)
        T_device_from_camera = src_calib.transform_world_cam.to_matrix()
        
        T_camera_from_device = np.linalg.inv(T_device_from_camera)
        print(cam, T_device_from_camera, T_camera_from_device)
        points_in_camera = transform_3d_points(T_camera_from_device, point_positions)

        binary_mask = np.zeros((h, w), dtype=np.uint8)
        depth_map = np.full((h, w), np.inf)  # 初始化深度图，使用无限大表示初始深度

        for point, id in zip(points_in_camera, point_id):
            maybe_pixel = dst_calib.project(point)    
            if maybe_pixel is not None:       
                x_pixel, y_pixel = int(maybe_pixel[0]), int(maybe_pixel[1])
                if 0 <= x_pixel < w and 0 <= y_pixel < h:
                    current_depth = point[2] 
                    # 更新深度图和二值掩码时，确保保存的是最近的深度值
                    if current_depth < depth_map[y_pixel, x_pixel]:
                        depth_map[y_pixel, x_pixel] = current_depth
                        binary_mask[y_pixel, x_pixel] = 1 

        valid_depths = depth_map[depth_map != np.inf]
        depth_map[depth_map==np.inf] = 0


        depth_map = depth_map.astype(float)
        if args.resize==512:
            np.savez_compressed(f'{cam}_depth_512.npz', depth_map=depth_map)
            np.savez_compressed(f'{cam}_mask_512.npz', binary_mask=binary_mask)
        else:
            np.savez_compressed(f'{cam}_depth.npz', depth_map=depth_map)
            np.savez_compressed(f'{cam}_mask.npz', binary_mask=binary_mask)   



def draw_camera_plotly(position, orientation, color):
    ratio = 300
    focal_length = 210   / ratio  # Adjusted focal length
    sensor_width = 256 / ratio  # Adjusted sensor width
    sensor_height = 144 / ratio  # Adjusted sensor height, keeping aspect ratio 3:2

    # Define corners of the camera in camera frame
    corners = np.array([
        [0, 0, 0],  # Camera origin
        [-sensor_width / 2, -sensor_height / 2, focal_length],
        [sensor_width / 2, -sensor_height / 2, focal_length],
        [sensor_width / 2, sensor_height / 2, focal_length],
        [-sensor_width / 2, sensor_height / 2, focal_length]
    ])

    # Convert orientation from quaternion to rotation matrix
    rotation_matrix = R.from_quat(orientation).as_matrix()

    # Transform corners to the world frame
    corners_world = np.dot(corners, rotation_matrix.T) + position

    # Collecting lines to draw the camera frustum
    lines = []
    for i in range(1, 5):
        lines.append(go.Scatter3d(x=[position[0], corners_world[i, 0]],
                                y=[position[1], corners_world[i, 1]],
                                z=[position[2], corners_world[i, 2]],
                                mode='lines',
                                line=dict(color=color, width=5)))
    # Connecting the corners to form the sensor plane
    sensor_edges = [[1, 2], [2, 3], [3, 4], [4, 1]]
    for edge in sensor_edges:
        lines.append(go.Scatter3d(x=[corners_world[edge[0], 0], corners_world[edge[1], 0]],
                                y=[corners_world[edge[0], 1], corners_world[edge[1], 1]],
                                z=[corners_world[edge[0], 2], corners_world[edge[1], 2]],
                                mode='lines',
                                line=dict(color=color, width=5)))
    return lines


def Align(args):
    import numpy as np
    args = parse_args()
    debug=args.debug
    import plotly.graph_objects as go
    import numpy as np
    from scipy.spatial.transform import Rotation as R
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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None

    import numpy as np
    import open3d as o3d
    import pandas as pd
    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]
    point_positions=np.array(point_positions)

    position_dict = {point_id[i]: point_positions[i] for i in range(len(point_id))}
    final={}
    for id in point_id:
        final[(id)]=[]

    iddddd=[]
    #colors = np.zeros(len(point_positions), 3, 3)
    for ix in range(4):
        cam = ['cam01', 'cam02',  'cam03', 'cam04'][ix]
        #rgb_path=f'/Users/ha1o/Desktop/CMU/cam_4_00000.png'
        rgb_path=f'/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/masked_cmu_bike/{int(cam[-1])}/filled_box_image.jpg'
        
        image=cv2.imread(rgb_path)
        #print(image.shape)
        static_cameras_path = '/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        #print(static_cameras)
        src_calib=(static_cameras[int(cam[-1])-1])
        
        h, w = src_calib.height, src_calib.width
        focal=src_calib.intrinsics.astype(np.float64)[0]
        if args.resize:
            new_w = args.resize
            new_h = int(h * (new_w / w))
            #focal = focal * (new_w / w)
            h, w = new_h, new_w 
        
        dst_calib = calibration.get_linear_camera_calibration(w, h, focal   )
        print(dst_calib)
        point_positions = np.array(point_positions)
        T_device_from_camera = src_calib.transform_world_cam.to_matrix()
        T_camera_from_device = np.linalg.inv(T_device_from_camera)
        points_in_camera = transform_3d_points(T_camera_from_device, point_positions)
        ###
        # id_position, 
        
        for point, id in zip(points_in_camera, point_id):
            maybe_pixel=dst_calib.project(point)    
            if maybe_pixel is not None:       
                m_x, m_y = maybe_pixel
                #print(m_x, m_y)
                x, y = w, h
                if (-0.5<=(m_x)<=x and -0.5<=(m_y)<=y):  
                    color = image[int(m_y), int(m_x), :][::-1]
                    #print((color))
                    iddddd.append(id)
                    final[(id)].append(color)
    revised={}
    
    print('before_value',   len(point_positions))
    print('final_value',   len(iddddd), len(set(iddddd)))
    ### no empty
    revised = {k: v for k, v in final.items() if len(v)!=0} 
    
    ### 
    revised_new = {k: np.array(v).mean(axis=0) for k, v in revised.items()}
    colors = np.array(list(revised_new.values()))/255
    green_colors = np.tile([0, 1, 0], (colors.shape[0], 1))


    ### filttered
    result = [key for key in position_dict if key in revised]
    seen_point = np.array([position_dict[key] for key in result])

    #fig.add_trace(go.Scatter3d(x=point_positions[:, 0], y=point_positions[:, 1], z=point_positions[:, 2], mode='markers', marker=dict(color='blue', size=2)))
    fig = go.Figure()
    '''fig.add_trace(go.Scatter3d(x=seen_point[:, 0], y=seen_point[:, 1], z=seen_point[:, 2], mode='markers', marker=dict(
            size=1,  # Adjust marker size according to your preference
            color=green_colors,  # Set colors of markers
            opacity=0.9
            )))'''

    def initialize_params():
        init_pt_cld = np.load(f"./da_pt_cld.npz")["data"]
        return init_pt_cld[:, :3], init_pt_cld[:, 3:6],

    seen_point, colors=initialize_params()


    fig.add_trace(go.Scatter3d(x=seen_point[:, 0], y=seen_point[:, 1], z=seen_point[:, 2], mode='markers', marker=dict(
    size=1,  # Adjust marker size according to your preference
    color=colors,  # Set colors of markers
    opacity=1.0
    )))


    # Update layout for a better view
    fig.update_layout(scene=dict(aspectmode='data'), width=800, height=600)

    # Show the interactive plot
    fig.show()






def pointsRGB(args):
    import numpy as np
    args = parse_args()
    debug=args.debug
    import plotly.graph_objects as go
    import numpy as np
    from scipy.spatial.transform import Rotation as R



    # Depth ...  init /  fx
    # Gather data input
    # - If MPS data has not been provided we try to find them automatically using default folder hierarchy
    #
    vrs_folder_path = os.path.dirname(args.vrs)
    '''    init_pc=np.load('final_pt_cld.npz')['data']
        #init_pc=np.load('init_pt_cld.npz')['data']
        ##print(init_pc.shape)
        seen_point, colors = init_pc[:, :3], init_pc[:, 3:6]
        # Camera data 
        camera_data = np.array([
            [0.374366	,1.441324	,-0.057627,	-0.19374	,-0.754455	,0.594122	,0.200704],
            [-1.42594, 1.021887, -0.088919, 0.224576, -0.702463, 0.64663, -0.194888],
            [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
            [-1.547741, -1.348028, -0.099894, -0.7248, 0.265416, -0.236691, 0.590082], 
            [-0.11952237039804459,
    0.16671882569789886,
    0.1641380786895752,
    -0.40006671677065453,
    0.6798872641118899,
    0.3536430534231149,
    0.5026296061677764],
    [-0.98248363,
    -0.00900431,
    0.07934562,
    0.8759966824488735,
    0.23074015531879166,
    -0.4076241996567296,
    0.11502740941287898]


        ])

        # Initialize Plotly figure
        fig = go.Figure()

        # Draw each camera
        for iiiiii, camera in enumerate(camera_data):
            if iiiiii>3:
                col='red'

            else:
                col='black'

            for line in draw_camera_plotly(camera[:3], camera[3:], col):
                fig.add_trace(line)

        #seen_point = np.array([v  for k, v  in position_dict.items() if k in iddddd])
        #fig.add_trace(go.Scatter3d(x=point_positions[:, 0], y=point_positions[:, 1], z=point_positions[:, 2], mode='markers', marker=dict(color='blue', size=2)))

        fig.add_trace(go.Scatter3d(x=seen_point[:, 0], y=seen_point[:, 1], z=seen_point[:, 2], mode='markers', marker=dict(
        size=1,  # Adjust marker size according to your preference
        color=colors,  # Set colors of markers
        opacity=0.7
        )))

        # Update layout for a better view
        fig.update_layout(scene=dict(aspectmode='data'), width=800, height=600)

        # Show the interactive plot
        fig.show()'''

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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None

    import numpy as np
    import open3d as o3d
    import pandas as pd
    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        #points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]
    point_positions=np.array(point_positions)

    position_dict = {point_id[i]: point_positions[i] for i in range(len(point_id))}
    final={}
    for id in point_id:
        final[(id)]=[]

    iddddd=[]
    #colors = np.zeros(len(point_positions), 3, 3)
    for ix in range(4):
        cam = ['cam01', 'cam02',  'cam03', 'cam04'][ix]
        #rgb_path=f'/Users/ha1o/Desktop/CMU/cam_4_00000.png'
        rgb_path=f'/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/masked_cmu_bike/{int(cam[-1])}/filled_box_image.jpg'
        
        image=cv2.imread(rgb_path)
        #print(image.shape)
        static_cameras_path = '/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        #print(static_cameras)
        src_calib=(static_cameras[int(cam[-1])-1])
        
        h, w = src_calib.height, src_calib.width
        focal=src_calib.intrinsics.astype(np.float64)[0]
        if args.resize:
            new_w = args.resize
            new_h = int(h * (new_w / w))
            focal = focal * (new_w / w)
            h, w = new_h, new_w 
        
        dst_calib = calibration.get_linear_camera_calibration(w, h, focal   )
        print(dst_calib, dir(dst_calib))
        point_positions = np.array(point_positions)
        T_device_from_camera = src_calib.transform_world_cam.to_matrix()
        T_camera_from_device = np.linalg.inv(T_device_from_camera)
        points_in_camera = transform_3d_points(T_camera_from_device, point_positions)
        ###
        # id_position, 
        
        for point, id in zip(points_in_camera, point_id):
            maybe_pixel=dst_calib.project(point)    
            if maybe_pixel is not None:       
                m_x, m_y = maybe_pixel
                #print(m_x, m_y)
                x, y = w, h
                if (-0.5<=(m_x)<=x and -0.5<=(m_y)<=y):  
                    color = image[int(m_y), int(m_x), :][::-1]
                    #print((color))
                    iddddd.append(id)
                    final[(id)].append(color)
    revised={}
    
    print('before_value',   len(point_positions))
    print('final_value',   len(iddddd), len(set(iddddd)))
    ### no empty
    revised = {k: v for k, v in final.items() if len(v)!=0} 
    
    ### 
    revised_new = {k: np.array(v).mean(axis=0) for k, v in revised.items()}
    colors = np.array(list(revised_new.values()))/255

    ### filttered
    result = [key for key in position_dict if key in revised]
    seen_point = np.array([position_dict[key] for key in result])

    #seen_point= np.array([value for key, value in position_dict.items() if key in revised])
    #print(len(seen_point))
    open3d=False
    from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

    #your code
    def quat_to_rot_matrix(quat):
        # Ensure the quaternion is in the format [x, y, z, w]
        x, y, z, w = quat
        # Compute the rotation matrix elements
        xx, xy, xz, xw = x*x, x*y, x*z, x*w
        yy, yz, yw = y*y, y*z, y*w
        zz, zw = z*z, z*w

        rot_matrix = np.array([[1 - 2*(yy + zz),     2*(xy - zw),     2*(xz + yw)],
                            [    2*(xy + zw), 1 - 2*(xx + zz),     2*(yz - xw)],
                            [    2*(xz - yw),     2*(yz + xw), 1 - 2*(xx + yy)]])
        return rot_matrix

    camera_data = pd.read_csv('/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv')
    import numpy as np
    camera_positions = camera_data[['tx_world_cam', 'ty_world_cam', 'tz_world_cam']].values[:]
    camera_orientations_quat = camera_data[['qx_world_cam', 'qy_world_cam', 'qz_world_cam', 'qw_world_cam']].values[:]
    vissssss=True
    if vissssss:
        if open3d is False:
            import plotly.graph_objects as go
            import numpy as np
            from scipy.spatial.transform import Rotation as R



            # Camera data
            camera_data = np.array([
                [-1.42594, 1.021887, -0.088919, 0.224576, -0.702463, 0.64663, -0.194888],
                [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
                [-1.547741, -1.348028, -0.099894, -0.7248, 0.265416, -0.236691, 0.590082]
            ])

            # Initialize Plotly figure
            fig = go.Figure()

            # Draw each camera
            for iiiiii, camera in enumerate(camera_data):
                if iiiiii==1:
                    col='red'

                else:
                    col='black'
    
                for line in draw_camera_plotly(camera[:3], camera[3:], col):
                    fig.add_trace(line)

            seen_point = np.array([v  for k, v  in position_dict.items() if k in iddddd])
            #fig.add_trace(go.Scatter3d(x=point_positions[:, 0], y=point_positions[:, 1], z=point_positions[:, 2], mode='markers', marker=dict(color='blue', size=2)))

            fig.add_trace(go.Scatter3d(x=seen_point[:, 0], y=seen_point[:, 1], z=seen_point[:, 2], mode='markers', marker=dict(
            size=1,  # Adjust marker size according to your preference
            color=colors,  # Set colors of markers
            opacity=0.7
        )))

            # Update layout for a better view
            fig.update_layout(scene=dict(aspectmode='data'), width=800, height=600)

            # Show the interactive plot
            fig.show()

    
    data=np.zeros((len(seen_point), 7))
    data[:, :3], data[:, 3:6] = seen_point, colors
    data[:, 6] = np.ones((len(seen_point)))


    np.savez("init_pt_cld.npz", data=data)
    print('SAVED!')

    return data

from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

def points3d(args):
    args = parse_args()
    debug=args.debug

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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None


    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        import numpy as np
        point_positions_np=np.array(point_positions)
        
        np.save('./points.npy', point_positions_np)
        print('saved!')
        point_id = [it.uid for it in points_data_down_sampled]
    return 
    position_dict = {point_id[i]: point_positions[i] for i in range(len(point_id))}
    #print(f'Filtered to #3dPoints:{len(point_positions)}')
    ######### What need to do now ######### 1. intrinsics  2. extrinsics 3. proj 

    # Go over RGB timestamps and
    # - Plot camera pose
    # - Plot user eye gaze
    #

    ### to_be_append={'frame_id':frame_id, 'uid': ids, 'uv':uvs}
    provider = data_provider.create_vrs_data_provider(args.vrs)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)

    num_data = provider.get_num_data(sensor_stream_id)
    from PIL import Image

    resuls=[]
    results=[]
    #num_data=min(num_data, debug)
    if not os.path.exists('images'):
        os.makedirs('images')

    if args.cam=='rgb':
        for index in tqdm(range(0, num_data)):
            # Block 1: Image data retrieval and pose calculation

            image_data = provider.get_image_data_by_index(sensor_stream_id, index)
            h, w = image_data[0].get_height(), image_data[0].get_width()
            device_calib = provider.get_device_calibration()
            src_calib = device_calib.get_camera_calib(sensor_name)

            focal=1220.439291944814

            dst_calib = calibration.get_linear_camera_calibration(h, w, focal)
            print(dst_calib)
            undistort_image = calibration.distort_by_calibration(cv2.resize(image_data[0].to_numpy_array(), (h, w)), dst_calib, src_calib, InterpolationMethod.BILINEAR)
            if args.resize:
                new_w = args.resize
                new_h = int(h * (new_w / w))
                focal = focal * (new_w / w)
                print('f', 666 * (new_w / w))
                print('p', 703.5 * (new_w / w))
                h, w = new_h, new_w
            dst_calib = calibration.get_linear_camera_calibration(h, w, focal)
            undistort_image = cv2.resize(undistort_image, (h, w))
            #print(src_calib)
            assert undistort_image.shape[0] == h
            filename = os.path.join('images', f'{index:05d}.jpg')
            #print(dir(image_data[0]), dir(image_data[1]))
            cv2.imwrite(filename, np.transpose(undistort_image.transpose(), (1, 2, 0)))
            capture_timestamp_ns = image_data[1].capture_timestamp_ns
            frame_id = index
            pose_info = get_nearest_pose(trajectory_data, capture_timestamp_ns)
            
            rgb_calib = device_calibration.get_camera_calib("camera-rgb")
            
            T_world_device = pose_info.transform_world_device.to_matrix()
            T_device_camera = rgb_camera_calibration.get_transform_device_camera().to_matrix()  # Assuming Sophus.SE3
            transform_camera_device = np.linalg.inv(T_device_camera)
            T_device_from_world = np.linalg.inv(T_world_device)

            point_positions = np.array(point_positions)
            points_in_world_homogeneous = np.hstack((point_positions, np.ones((point_positions.shape[0], 1))))
            points_in_device_homogeneous = np.dot(points_in_world_homogeneous, T_device_from_world.T)
            points_in_device = points_in_device_homogeneous[:, :3]
            
            points_in_device_homogeneous = np.hstack((points_in_device, np.ones((points_in_device.shape[0], 1))))
            points_in_camera_homogeneous = np.dot(points_in_device_homogeneous, transform_camera_device.T)
            points_in_camera = points_in_camera_homogeneous[:, :3]
            frame_plus_id=[]
            for point, id in zip(points_in_camera, point_id):
                maybe_pixel=dst_calib.project(point)
                if maybe_pixel is not None:
                    m_x, m_y = maybe_pixel
                    x, y = w, h
                    if (-0.5<=(m_x)<=x-0.5 and -0.5<=(m_y)<=y-0.5):  
                        frame_plus_id.append(id) 
            results.append((index, frame_plus_id)) 
        ##############################
        #### results: each frame: [each point: id]
        # ########       (Num_frame, id)     
        #    

        id_to_frames = {}
        for num_frame, ids in (results):
            for id in ids:
                # 如果id已经在字典中，则添加当前的Num_frame
                if id in id_to_frames:
                    id_to_frames[id].append(num_frame)
                # 如果id不在字典中，创建一个新的条目
                else:
                    id_to_frames[id] = [num_frame]
        for id0, frame0 in id_to_frames.items():
            prepare=[]
            for frame in frame0:
                prepare.append([frame, id0])
            position=position_dict[id0]
            to_be_append_0={'id': id0, 'position':position,'frames':prepare} 
            line = to_point_3d(to_be_append_0)
            with open('3d_points.txt', 'a') as file:  # 使用 'a' 模式以追加内容
                file.write(line)

    elif 'cam0' in args.cam:
        directory_path = f'/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/frame_aligned_videos/{args.cam}'
        static_cameras_path = '/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        src_calib=(static_cameras[int(args.cam[-1])-1])
        for index in tqdm(range(0, 1400)):    
            h, w = src_calib.height, src_calib.width
            focal=src_calib.intrinsics.astype(np.float64)[0]
            if args.resize:
                new_w = args.resize
                new_h = int(h * (new_w / w))
                focal = focal * (new_w / w)
                h, w = new_h, new_w

            dst_calib = calibration.get_linear_camera_calibration(w, h, focal)
            point_positions = np.array(point_positions)
            T_device_from_camera = src_calib.transform_world_cam.to_matrix()
            T_camera_from_device = np.linalg.inv(T_device_from_camera)
            points_in_camera = transform_3d_points(T_camera_from_device, point_positions)
            ###
            frame_plus_id=[]
            for point, id in zip(points_in_camera, point_id):
                maybe_pixel=dst_calib.project(point)
                if maybe_pixel is not None:
                    m_x, m_y = maybe_pixel
                    x, y = w, h
                    if (-0.5<=(m_x)<=x-0.5 and -0.5<=(m_y)<=y-0.5):  
                        frame_plus_id.append(id) 
            results.append((index, frame_plus_id)) 

        id_to_frames = {}
        for num_frame, ids in (results):
            for id in ids:
                # 如果id已经在字典中，则添加当前的Num_frame
                if id in id_to_frames:
                    id_to_frames[id].append(num_frame)
                # 如果id不在字典中，创建一个新的条目
                else:
                    id_to_frames[id] = [num_frame]
        for id0, frame0 in id_to_frames.items():
            prepare=[]
            for frame in frame0:
                prepare.append([frame, id0])
            position=position_dict[id0]
            to_be_append_0={'id': id0, 'position':position,'frames':prepare} 
            line = to_point_3d(to_be_append_0)
            with open(f'./3d_points_{args.cam}.txt', 'a') as file:  # 使用 'a' 模式以追加内容
                file.write(line)

def to_gaussian_cam(args):
    args = parse_args()
    debug=args.debug

    to_be_append_1={'image_id':frame_id, 'T_camera_from_world': np.matmul(transform_camera_device, T_device_from_world), 'points_2d': points_2d}
    line = to_image(to_be_append_1)
    with open('images.txt', 'a') as file:  # 使用 'a' 模式以追加内容
        file.write(line)

def to_images(args):
    args = parse_args()
    debug=args.debug
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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )
    points_data = mps.read_global_point_cloud(args.points) if args.points else None


    #
    # Log Point Cloud (reduce point count for display)
    #
    if points_data and len(points_data) > 0:
        # Filter out low confidence points
        points_data = filter_points_from_confidence(points_data)
        
        # Down sample points
        points_data_down_sampled=points_data
        #points_data_down_sampled = filter_points_from_count(points_data, 50_000)
        # Retrieve point position
        point_positions = [(it.position_world) for it in points_data_down_sampled]
        point_id = [it.uid for it in points_data_down_sampled]
    position_dict = {point_id[i]: point_positions[i] for i in range(len(point_id))}
    #print(f'Filtered to #3dPoints:{len(point_positions)}')
    ######### What need to do now ######### 1. intrinsics  2. extrinsics 3. proj 

    # Go over RGB timestamps and
    # - Plot camera pose
    # - Plot user eye gaze
        #

    ### to_be_append={'frame_id':frame_id, 'uid': ids, 'uv':uvs}
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
    from PIL import Image
    resuls=[]
    results=[]

    #num_data=min(num_data, debug)
    if args.cam=='rgb':
        for index in tqdm(range(0, num_data)):
            # Block 1: Image data retrieval and pose calculation

            image_data = provider.get_image_data_by_index(sensor_stream_id, index)
            capture_timestamp_ns = image_data[1].capture_timestamp_ns
            frame_id = index
            h, w = image_data[0].get_height(), image_data[0].get_width()
            device_calib = provider.get_device_calibration()
            src_calib = device_calib.get_camera_calib(sensor_name)
            focal=1220.439291944814
            if args.resize:
                new_w = args.resize
                new_h = int(h * (new_w / w))
                focal = focal * (new_w / w)
                h, w = new_h, new_w
            dst_calib = calibration.get_linear_camera_calibration(h, w, focal)

            print(dst_calib)
            pose_info = get_nearest_pose(trajectory_data, capture_timestamp_ns)
            
            rgb_calib = device_calibration.get_camera_calib("camera-rgb")
            T_world_device = pose_info.transform_world_device.to_matrix()

            #print(dir(rgb_camera_calibration))
            T_device_camera = rgb_camera_calibration.get_transform_device_camera().to_matrix()  # Assuming Sophus.SE3
            transform_camera_device = np.linalg.inv(T_device_camera)
            T_device_from_world = np.linalg.inv(T_world_device)

            point_positions = np.array(point_positions)
            points_in_world_homogeneous = np.hstack((point_positions, np.ones((point_positions.shape[0], 1))))
            points_in_device_homogeneous = np.dot(points_in_world_homogeneous, T_device_from_world.T)
            points_in_device = points_in_device_homogeneous[:, :3]
            points_in_device_homogeneous = np.hstack((points_in_device, np.ones((points_in_device.shape[0], 1))))
            points_in_camera_homogeneous = np.dot(points_in_device_homogeneous, transform_camera_device.T)
            points_in_camera = points_in_camera_homogeneous[:, :3]

            points_2d=[]
            for point, id in zip(points_in_camera, point_id):
                maybe_pixel=dst_calib.project(point)

                if maybe_pixel is not None:
                    m_x, m_y = maybe_pixel
                    x, y = w, h
                    if (-0.5<=(m_x)<=x-0.5 and -0.5<=(m_y)<=y-0.5):  
                        points_2d.append([m_x, m_y, id])            
    
                    #print(ins, outs, ins+outs, len(point_id))
            ### (world_coord,(image_id, point_id))
            ### (frame_id, T, (u, v, id))
            to_be_append_1={'image_id':frame_id, 'T_camera_from_world': np.matmul(transform_camera_device, T_device_from_world), 'points_2d': points_2d}
            line = to_image(to_be_append_1)
            with open('images.txt', 'a') as file:  # 使用 'a' 模式以追加内容
                file.write(line)

    elif 'cam0' in args.cam:
        directory_path = f'/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/frame_aligned_videos/{args.cam}'
        static_cameras_path = '/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        src_calib=(static_cameras[int(args.cam[-1])-1])
        for index in tqdm(range(0, 1400)):
            h, w = src_calib.height, src_calib.width
            focal=src_calib.intrinsics.astype(np.float64)[0]
            if args.resize:
                new_w = args.resize
                new_h = int(h * (new_w / w))
                focal = focal * (new_w / w)
                h, w = new_h, new_w

            dst_calib = calibration.get_linear_camera_calibration(w, h, focal)
            print(dst_calib, dst_calib.get_principal_point(), dst_calib.get_focal_lengths())

            point_positions = np.array(point_positions)
            T_device_from_camera = src_calib.transform_world_cam.to_matrix()
            T_camera_from_device = np.linalg.inv(T_device_from_camera)
            points_in_camera = transform_3d_points(T_camera_from_device, point_positions)
            points_2d=[]
            for point, id in zip(points_in_camera, point_id):
                maybe_pixel=dst_calib.project(point)
                if maybe_pixel is not None:
                    m_x, m_y = maybe_pixel
                    x, y = w, h
                    if (-0.5<=(m_x)<=x-0.5 and -0.5<=(m_y)<=y-0.5):
                        points_2d.append([m_x, m_y, id])            

            to_be_append_1={'image_id':index, 'T_camera_from_world': src_calib.transform_world_cam.to_matrix(), 'points_2d': points_2d}
            line = to_image(to_be_append_1)
            with open(f'./images_{args.cam}.txt', 'a') as file:  # 使用 'a' 模式以追加内容
                file.write(line)

def to_image(dic):
    frame_id = dic['image_id']
    
    camera_id=1
    NAME = '{:05d}.jpg'.format(int(frame_id))
    T = dic['T_camera_from_world']
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    T = np.array(dic['T_camera_from_world'])

    tx, ty, tz = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()

    # 四元数的元素顺序通常是[x, y, z, w]，需要调整为[w, x, y, z]
    qw, qx, qy, qz = quaternion[3], quaternion[0], quaternion[1], quaternion[2]

    first_line = f"{frame_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {NAME}"

    second_line = ''
    ####usefuL changes
    for u, v, id in dic['points_2d']:
        second_line+=f"{u} {v} -1 "
        #second_line+=f"{u} {v} {id} "

    return first_line+'\n'+second_line+'\n'



def to_point_3d(dic):
    #{'id': id, 'position':position,'frames':frames} 
    point_id=dic['id']
    position=dic['position']
    frames=dic['frames']
    x, y, z = position
    r, g, b = 255, 255, 255
    error = 0.0
    first_line = f"{point_id} {x} {y} {z} {r} {g} {b} {error} " 
    for frame in frames:
        first_line+=f'{frame[0]} {frame[1]} '
    first_line += '\n'
    return first_line


def matrices(debug=3000000):
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

    trajectory_data = (
        mps.read_closed_loop_trajectory(args.trajectory) if args.trajectory else None
    )

    ### to_be_append={'frame_id':frame_id, 'uid': ids, 'uv':uvs}
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
    from PIL import Image

    resuls=[]
    results=[]
    #num_data=min(num_data, debug)

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
            with open('train.matrices.txt', 'a') as file:  # 使用 'a' 模式以追加内容
                file.write(row_str+'\n')




    elif 'cam0' in args.cam:
        directory_path = f'/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/frame_aligned_videos/{args.cam}'
        static_cameras_path = '/Users/ha1o/Desktop/CMU_SCS_RI/Capstone/newnewnew/takes/cmu_bike03_4/trajectory/gopro_calibs.csv'
        static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
        src_calib=(static_cameras[int(args.cam[-1])-1])
        for index in tqdm(range(0, num_data)):
            row_str = str(index)+' '
            for row in src_calib.transform_world_cam.to_matrix():
                # 创建一个由该行元素组成的字符串，元素之间用空格分隔
                row_str += ' '.join(map(str, row))
                row_str += ' '
            with open(f'{args.cam}_train.matrices.txt', 'a') as file:  # 使用 'a' 模式以追加内容
                file.write(row_str+'\n')
def main():
    args = parse_args()
    if args.task=='imgs':
        to_images(args)
    elif args.task=='points':
        points3d(args)
    elif args.task=='vis':
        vis(args)
    elif args.task=='matr':
        matrices(args)        
    elif  args.task=='Gaussian':
        point_7_gaussian(args)
    elif args.task=='rgb':
        pointsRGB(args)
    elif args.task=='depth':
        Depth(args)
    elif args.task=='slam':
        SLAM(args)
    elif args.task=='align':
        Align(args)

if __name__ == "__main__":
    main()