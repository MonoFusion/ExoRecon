import os
import glob
import argparse
from PIL import Image


def parse_args():
    """Collect CLI overrides so the data root can live anywhere."""
    parser = argparse.ArgumentParser(description='Resize undistorted images for a single sequence.')
    parser.add_argument(
        '--sequence', '--seq',
        required=True,
        dest='sequence',
        help='Name of the take inside the raw-data root (e.g., indiana_music_14_3).'
    )
    parser.add_argument(
        '--raw-data-root',
        default='../_raw_data',
        help='Directory that contains the per-sequence folders (default: ../_raw_data).'
    )
    parser.add_argument(
        '--target-root',
        default='../data/images',
        help='Destination root for resized images (default: ../data/images).'
    )
    parser.add_argument(
        '--source-subdir',
        default='undist_processed_frames',
        help='Subdirectory inside the take that holds the undistorted camera folders.'
    )
    parser.add_argument(
        '--camera-glob',
        default='undist_cam0*',
        help='Glob that selects the camera folders to process inside the source subdirectory.'
    )
    parser.add_argument('--width', type=int, default=512, help='Output width (default: 512).')
    parser.add_argument('--height', type=int, default=288, help='Output height (default: 288).')
    return parser.parse_args()


def describe_sequence(seq_name):
    """Return the scene token (2nd chunk) when available, otherwise the raw name."""
    parts = seq_name.split('_')
    if len(parts) >= 2:
        return parts[1]
    return seq_name


def main(args):
    new_width, new_height = args.width, args.height
    base_directory = os.path.abspath(args.raw_data_root)
    target_root_folder = os.path.abspath(args.target_root)
    sequence_name = args.sequence
    scene_token = describe_sequence(sequence_name)

    if not os.path.isdir(base_directory):
        raise FileNotFoundError(f"Raw data directory not found: {base_directory}")

    sequence_directory = os.path.join(base_directory, sequence_name)
    if not os.path.isdir(sequence_directory):
        raise FileNotFoundError(f"Sequence directory not found: {sequence_directory}")

    source_root = os.path.join(sequence_directory, args.source_subdir)
    if not os.path.isdir(source_root):
        raise FileNotFoundError(
            f"Source subdirectory '{args.source_subdir}' not found inside {sequence_directory}"
        )

    camera_pattern = os.path.join(source_root, args.camera_glob)
    camera_directories = sorted(
        d for d in glob.glob(camera_pattern) if os.path.isdir(d)
    )

    if not camera_directories:
        raise FileNotFoundError(
            f"No camera folders matched pattern '{args.camera_glob}' under {source_root}"
        )

    os.makedirs(target_root_folder, exist_ok=True)

    for cam_dir in camera_directories:
        cam_name = os.path.basename(os.path.normpath(cam_dir))
        tgt_subdir_path = os.path.join(target_root_folder, f"{scene_token}_{cam_name}")
        os.makedirs(tgt_subdir_path, exist_ok=True)

        for filename in sorted(os.listdir(cam_dir)):
            input_file_path = os.path.join(cam_dir, filename)

            if not os.path.isfile(input_file_path):
                continue

            try:
                with Image.open(input_file_path) as img:
                    img_resized = img.resize((new_width, new_height))
                    tgt_file_path = os.path.join(tgt_subdir_path, filename)
                    img_resized.save(tgt_file_path)
                    print(f"Processed and saved: {tgt_file_path}")
            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
