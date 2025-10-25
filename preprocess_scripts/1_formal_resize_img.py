import os
import argparse
from PIL import Image


def parse_args():
    """Collect CLI overrides so the data root can live anywhere."""
    parser = argparse.ArgumentParser(description='Resize all images in the raw dataset tree.')
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
    parser.add_argument('--width', type=int, default=512, help='Output width (default: 512).')
    parser.add_argument('--height', type=int, default=288, help='Output height (default: 288).')
    return parser.parse_args()


def main(args):
    new_width, new_height = args.width, args.height
    base_directory = os.path.abspath(args.raw_data_root)
    target_root_folder = os.path.abspath(args.target_root)

    if not os.path.isdir(base_directory):
        raise FileNotFoundError(f"Raw data directory not found: {base_directory}")

    # Ensure the target root folder exists
    os.makedirs(target_root_folder, exist_ok=True)

    # Loop through each folder in the base directory
    for root_folder in os.listdir(base_directory):
        # Check if the folder starts with "Processinfg_"
        # if root_folder.startswith('Processing_cmu_soc'):
        if len(root_folder.split('_')) > 2:
          input_directory = os.path.join(base_directory, root_folder)
          tgt_name = root_folder.split('_')[2]
          
          # Loop through each subdirectory in the input directory
          for subdir in os.listdir(input_directory):
              input_subdir_path = os.path.join(input_directory, subdir)
              
              # Check if it's a directory
              if os.path.isdir(input_subdir_path):
                  tgt_subdir_path = os.path.join(target_root_folder, tgt_name + '_' + subdir)
                  
                  # Ensure the target subdirectory exists
                  os.makedirs(tgt_subdir_path, exist_ok=True)
                  
                  # Loop through each file in the subdirectory
                  for filename in os.listdir(input_subdir_path):
                      input_file_path = os.path.join(input_subdir_path, filename)
                      
                      # Check if it's a file
                      if os.path.isfile(input_file_path):
                          try:
                              # Open the image
                              with Image.open(input_file_path) as img:
                                  # Resize the image
                                  img_resized = img.resize((new_width, new_height))
                                  
                                  # Save the resized image to the target directory
                                  tgt_file_path = os.path.join(tgt_subdir_path, filename)
                                  img_resized.save(tgt_file_path)
                                  
                                  print(f"Processed and saved: {tgt_file_path}")
                          except Exception as e:
                              print(f"Error processing {input_file_path}: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
