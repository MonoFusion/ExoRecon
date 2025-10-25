# EgoExo Processing Pipeline

This repository captures the minimum set of instructions and scripts we need to process EgoExo takes locally. It packages the vendored Project Aria Tools build that `push_all_data.sh` uses plus the conda environment definition that keeps every machine consistent.

## 1. Prepare and download data

1. Register for access at https://docs.ego-exo4d-data.org/ and obtain a license key.
2. Follow the official download guide to fetch the EgoExo dataset (≈18 TB). For smaller experiments visit https://visualize.ego4d-data.org/login, pick a scene, note its `take_id`, and download the matching subset, for example:
   ```bash
   egoexo -o ./raw_data/ \
     --uids ae3e392f-5db9-471e-b910-796a9d65e3ac \
     --parts take_vrs takes -y
   ```
   After the download finishes you should have folders such as `./raw_data/indiana_piano_14_4/` containing `ariaXX.vrs`, `frame_aligned_videos/*.mp4`, and the `trajectory/` artifacts.
3. Copy the take you want to process into this repository (e.g., `indiana_music_14_3/`). Keep the heavy data out of git—the `.gitignore` already excludes these paths.

## 2. Create the Python environment

All binaries we ship were compiled against Python 3.9. Create the conda env exactly once per machine:

```bash
conda env create -f egorecon.yml
conda activate egorecon
```

You can update the environment later with `conda env update -f egorecon.yml --prune`.

## 3. Install the vendored Project Aria Tools

Install the package in editable mode so CLI entry points stay in sync with our sources:

```bash
python -m pip install --upgrade pip
python -m pip install -e projectaria_tools_pkg
```

The `viewer_mps` CLI now exposes a `mate` task that performs the custom EgoExo workflow (frame extraction + matrix generation) expected by `push_all_data.sh`.

## 4. Required dataset contents

Before running the processing script, ensure the selected take contains:

- `aria01.vrs` (or the appropriate VRS file).
- `frame_aligned_videos/` with `cam01.mp4`–`cam04.mp4` plus `timestep.txt`.
- `trajectory/closed_loop_trajectory.csv` for RGB pose export.
- `gopro_calibs.csv` either in the take root or under `trajectory/` for the static camera rectification step.

## 5. Process a take

Set `LOCAL_BASE_PATH` inside `push_all_data.sh` if the data lives elsewhere, then run:

```bash
./push_all_data.sh
```

The script will `cd` into the take directory, call `viewer_mps --task mate --vrs aria01.vrs --cam cam01 --resize 512`, and generate:

- Extracted JPEG frames under `processed_frames/cam0X`.
- Optional undistorted frames under `undist_processed_frames/` (if `gopro_calibs.csv` is present).
- Pose matrices in `train.matrices.txt` (RGB) or `cam0X_train.matrices.txt`.

## 6. Git hygiene

- `indiana_music_14_3/` and other raw-data folders remain untracked by default. Keep checkpoints inside those directories if needed; they will be ignored.
- Commit and push everything else (scripts, documentation, vendored package) from `/Users/ha1o/Downloads/takes/takes`.

## 7. Run the preprocessing scripts (sequential)

The folder `/Users/ha1o/Downloads/takes/takes/preprocess_scripts` contains two numbered scripts that must be executed in order right after `push_all_data.sh` finishes. They assume you already ran the Project Aria pipeline so that `processed_frames/` and `undist_processed_frames/` exist.

1. **Resize + organize RGB images** (Pillow is already included through `egorecon.yml`):
   ```bash
   python preprocess_scripts/1_formal_resize_img.py \
     --raw-data-root ./raw_data \
     --target-root ./data/images \
     --width 512 --height 288
   ```
   - Iterates over every take inside `./raw_data` (or the path you pass) and writes resized JPEGs under `./data/images/<take>/<camera>/`.
   - Override the defaults if your raw data lives somewhere else.

2. **Generate DyTrial JSON metadata** (requires `torch`, `pytorch3d`, `pandas`, etc.—install via `python -m pip install -r requirements.txt` if they are missing):
   ```bash
   python preprocess_scripts/2_formal_DyTrialJson.py \
     --seq indiana_music_14_3 \
     --raw-data-root ./raw_data
   ```
   - `--seq` should match the folder name created by step 1 (e.g., `indiana_piano_14_4`).
   - The script expects per-frame images under `undist_processed_frames/undist_cam01` and calibration CSVs under `<take>/trajectory/` (produced earlier by `push_all_data.sh`).
   - A `Dy_train_meta.json` file is emitted in the take’s `trajectory/` folder, ready for downstream training code.

Re-run both scripts whenever you update the underlying raw data; they are idempotent and overwrite outputs in place.

## Troubleshooting

- If `viewer_mps` reports `ModuleNotFoundError: _core_pybinds`, double-check that you are using the conda environment created from `egorecon.yml` (Python 3.9) and that `pip install -e projectaria_tools_pkg` succeeded inside that environment.
- When `gopro_calibs.csv` is missing, the script will still extract frames but skip rectification—copy the calibration file next to the take if you need undistorted images or GoPro matrices.
