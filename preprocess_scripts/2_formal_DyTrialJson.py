import json
import torch
import numpy as np
import pandas as pd
import argparse
import os
import glob


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices using only PyTorch."""
    if quaternions.shape[-1] != 4:
        raise ValueError('Quaternions should have shape (..., 4).')

    quaternions = quaternions / torch.linalg.norm(
        quaternions, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternions.dtype).eps)

    w, x, y, z = torch.unbind(quaternions, dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    rotation = torch.stack(
        [m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=-1
    ).reshape(quaternions.shape[:-1] + (3, 3))
    return rotation

# Add argument parser to accept seq from terminal
parser = argparse.ArgumentParser(description='Generate DyTrial metadata for a specific sequence.')
parser.add_argument('--seq', type=str, required=True, help='Sequence name inside the raw-data root')
parser.add_argument('--raw-data-root', type=str, default='../_raw_data',
                    help='Directory containing all raw sequences (default: ../_raw_data)')
args = parser.parse_args()

# Assign seq_name from argument
seq_name = args.seq
raw_data_root = os.path.abspath(args.raw_data_root)
seq_root = os.path.join(raw_data_root, seq_name)

if not os.path.isdir(seq_root):
    raise FileNotFoundError(f"Sequence directory not found: {seq_root}")

# Your existing code to generate k_multi_cam, w2c_multi_cam, h, w, generated_list, and cam_ids
# ...
def intrinsic_matrix(fx, fy, cx, cy, dtype=torch.float32, device='cpu'):
    """
    Create the camera intrinsic matrix using PyTorch.

    Parameters:
    fx (float): Focal length along the x-axis in pixels.
    fy (float): Focal length along the y-axis in pixels.
    cx (float): The x-coordinate of the principal point in pixels.
    cy (float): The y-coordinate of the principal point in pixels.
    dtype (torch.dtype): Data type of the returned tensor.
    device (str or torch.device): The device on which the tensor will be allocated.
    
    Returns:
    torch.Tensor: The 3x3 camera intrinsic matrix.
    """
    return torch.tensor([[fx,  0, cx], 
                         [ 0, fy, cy],
                         [ 0,  0,  1]], dtype=dtype, device=device)

def construct_extrinsic_matrix(Q, T):
    # Ensure Q is a 3x3 matrix and T is a 3x1 vector
    assert Q.size() == (3, 3), "Q must be a 3x3 matrix"
    assert T.size() == (3,) or T.size() == (3, 1), "T must be a 3x1 vector"
    
    # Reshape T to ensure it is a 3x1 column vector if it's not already
    T = T.view(3, 1)

    # Concatenate Q and T to form the top part of the matrix
    top_part = torch.cat((Q, T), dim=1)

    # Create the bottom part of the matrix [0, 0, 0, 1]
    bottom_part = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    # Combine the top and bottom parts to form the 4x4 extrinsic matrix
    extrinsic_matrix = torch.cat((top_part, bottom_part), dim=0)
    return extrinsic_matrix#torch.inverse(extrinsic_matrix)

h, w = 2160, 3840

# Update the path to use seq_name from the argument
path = os.path.join(seq_root, 'undist_processed_frames', 'undist_cam01', '*.jpg')
paths = glob.glob(path)
sorted_paths = sorted(paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
print(path, sorted_paths)
min_ = int(os.path.basename(sorted_paths[0]).split('.')[0])
max_= int(os.path.basename(sorted_paths[-1]).split('.')[0])

frames = max_
# Reading the CSV file using pandas
trajectory_root = os.path.join(seq_root, 'trajectory')



df = pd.read_csv(os.path.join(trajectory_root, 'gopro_calibs.csv'))[:]
df_clp = pd.read_csv(os.path.join(trajectory_root, 'closed_loop_trajectory.csv'))[:]
intrinsics =np.array(df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist())
to_cat=[]
hw=[]
### seq: ego_cam,  stat_cam, 2*slam
intrinsics =np.array(df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist())
to_cat=[]
hw=[]
# fx, fy, cx, cy =610.21964597, 610.21964597, 703.5, 703.5 real image
fx, fy, cx, cy =670, 670, 703.5, 703.5  # no edge img

for _ in range(1):
  to_cat.append(intrinsic_matrix(fx, fy, cx, cy))
  hw.append([1408, 1408])

##stat h, w = 2160, 3840
for _ in range(len(df)):
  hw.append([2160, 3840])

for item in intrinsics:
  _, _, fx, fy, cx, cy = item
  cx -= 0.5
  cy -= 0.5
  ratio = 1
  fx *= ratio
  fy *= ratio
  cx *= ratio
  cy *= ratio
  print(ratio)
  to_cat.append(intrinsic_matrix(fx, fy, cx, cy))

fx, fy, cx, cy = 300, 300, 95.5, 127.5

k_multi_cam=torch.stack(to_cat, dim=0).repeat(frames, 1, 1, 1)

q_values = df[[ 'qw_world_cam', 'qx_world_cam', 'qy_world_cam', 'qz_world_cam']]
t_values = df[['tx_world_cam', 'ty_world_cam', 'tz_world_cam']]
top_q_values = q_values.head(len(df))
top_t_values = t_values.head(len(df))

q_list = top_q_values.values.tolist()
t_list = top_t_values.values.tolist()
to_cat=[]

######### WARNING THIS IS A PLACEHOLDER #############
for q, t in zip(q_list[:1], t_list[:1]):
  q_values_tensor = torch.tensor(q, dtype=torch.float64)
  t_values_tensor = torch.tensor(t, dtype=torch.float64)
  R = quaternion_to_matrix(q_values_tensor.unsqueeze(0)).squeeze(0)  
  T = t_values_tensor
  to_cat.append(construct_extrinsic_matrix(R, torch.tensor(t)))

for q, t in zip(q_list, t_list):
  q_values_tensor = torch.tensor(q, dtype=torch.float64)
  t_values_tensor = torch.tensor(t, dtype=torch.float64)
  R = quaternion_to_matrix(q_values_tensor.unsqueeze(0)).squeeze(0)  
  T = t_values_tensor
  to_cat.append(construct_extrinsic_matrix(R, torch.tensor(t)))

w2c_multi_cam=torch.stack(to_cat, dim=0).repeat(frames, 1, 1, 1)

generated_list=[[f"{i}/{num:06d}.jpg" for i in range(1)] for num in range(frames)]
generated_list_2=[[f"{i}/filled_box_image.jpg"] for i in range(1,5)]
generated_list =  generated_list+ generated_list_2
generated_list += [[f"{i}/{num:06d}.jpg" for i in range(5,6)] for num in range(frames)]
generated_list += [[f"{i}/{num:06d}.jpg" for i in range(6,7)] for num in range(frames)]
generated_list=(np.array(generated_list).reshape(1,-1)).tolist()

generated_list=np.array([[f"{i}/{num:06d}.jpg" for i in range(1) for gg in range(1)] for num in range(frames)])
generated_list_2=np.array([[f"undist_data/undist_cam0{i}/{num:05d}.jpg" for i in range(1, len(df)+1)] for num in range(frames)])

generated_list = np.concatenate((generated_list, generated_list_2), axis=1).tolist()
cam_ids=[[i for i in range(len(df)+1)] for num in range(frames)]

k_multi_cam_list = k_multi_cam.tolist()
w2c_multi_cam_list = w2c_multi_cam.tolist()

print(len(df_clp))
json_list = {
    'hw': hw,
    'k': k_multi_cam_list,
    'w2c': w2c_multi_cam_list,
    'fn': generated_list,
    'cam_id': cam_ids
}

for k, v in json_list.items():
  try:
    print(k, np.array(v).shape)
  except:
    print(len(v))
    continue 

save_path = trajectory_root

with open(os.path.join(save_path, 'Dy_train_meta.json'), 'w') as file:
    json.dump(json_list, file)
