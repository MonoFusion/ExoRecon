# EgoExo Processing Pipeline

This repository captures the minimum set of instructions and scripts we need to process EgoExo takes locally. It packages the vendored Project Aria Tools build that `push_all_data.sh` uses plus the conda environment definition that keeps every machine consistent.

All commands assume you start inside `MonoFusion/preproc/ExoRecon`, where raw takes live in `../../raw_data` and derived assets in `../../data`.

## 1. Prepare and download data

1. Register for access at https://docs.ego-exo4d-data.org/ and obtain a license key.
2. Follow the official download guide to fetch the EgoExo dataset (≈18 TB). For smaller experiments visit https://visualize.ego4d-data.org/login, pick a scene, note its `take_id`, and download the matching subset, for example:
   ```bash
   egoexo -o ../../ \
     --uids ae3e392f-5db9-471e-b910-796a9d65e3ac \
     --parts take_trajectory take_vrs takes -y
   ```
   Adjust `--uids` as needed; the `--parts` list pulls a synchronized copy of the trajectories, VRS, and take archive in a single call.
3. Flatten the CLI output into `../../raw_data` and capture the first valid timestep for each sequence:
   ```bash
   mv ../../raw_data/takes/* ../../raw_data/
   rm -rf ../../raw_data/takes
   echo 1487 > ../../raw_data/indiana_music_14_3/timestep.txt
   ```
   Replace `1487` and `indiana_music_14_3` with the right frame index and folder name for your take. Keep all heavy data inside `../../raw_data`—the `.gitignore` already excludes these paths.

## 2. Environment and dataset prerequisites (merged from §§2–4)

All binaries we ship were compiled against Python 3.9. Create the conda env and install the vendored Project Aria Tools exactly once per machine:

```bash
conda env create -f egorecon.yml
conda activate egorecon
python -m pip install --upgrade pip
python -m pip install -e projectaria_tools_pkg
```

The editable install keeps the `viewer_mps` CLI in sync with this checkout. Before running any processing script, make sure each take under `../../raw_data/<take_id>/` contains:

- `aria01.vrs` (or the appropriate VRS file).
- `frame_aligned_videos/` with `cam01.mp4`–`cam04.mp4` plus your freshly written `timestep.txt`.
- `trajectory/closed_loop_trajectory.csv` for RGB pose export.
- `gopro_calibs.csv` either in the take root or under `trajectory/` so the static camera rectification step can run.

## 5. Process a take

Set `LOCAL_BASE_PATH` inside `push_all_data.sh` if the data lives elsewhere, then run:

```bash
./push_all_data.sh
```

`push_all_data.sh` now drives the entire pipeline: it drops into the take folder, runs `viewer_mps`, and immediately follows up with the two preprocessing scripts so you do not have to call them manually. Add (or keep) a block similar to the snippet below toward the end of the script:

```bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT="$SCRIPT_DIR/../../data"
RAW_ROOT="$SCRIPT_DIR/raw_data"

viewer_mps --vrs aria01.vrs --task mate --resize 512 --cam cam01

python "$SCRIPT_DIR/preprocess_scripts/1_formal_resize_img.py" \
  --raw-data-root "$RAW_ROOT" \
  --target-root "$DATA_ROOT/images" \
  --width 512 --height 288

python "$SCRIPT_DIR/preprocess_scripts/2_formal_DyTrialJson.py" \
  --seq "$FOLDER_NAME" \
  --raw-data-root "$RAW_ROOT"
```

With that block in place the script produces:

- Extracted JPEG frames under `processed_frames/cam0X`.
- Optional undistorted frames under `undist_processed_frames/` (if `gopro_calibs.csv` is present).
- Pose matrices in `train.matrices.txt` (RGB) or `cam0X_train.matrices.txt`.
- Resized RGB dumps under `../../data/images/<take>/<camera>/`.
- `Dy_train_meta.json` inside `<take>/trajectory/`.

## 7. Optional: Run the preprocessing scripts manually

`push_all_data.sh` already chains these helpers, so you only need the commands below when debugging or when you want to re-run a single stage without extracting frames again. The `preprocess_scripts/` directory contains two numbered scripts that must be executed in order.

1. **Resize + organize RGB images** (Pillow is already included through `egorecon.yml`):
   ```bash
   python preprocess_scripts/1_formal_resize_img.py \
     --sequence indiana_music_14_3 \
     --raw-data-root ../../raw_data \
     --target-root ../../data/images \
     --width 512 --height 288
   ```
   - Scans `undist_processed_frames/undist_cam0*` inside the specified take and writes resized JPEGs under `../../data/images/<scene-token>_undist_camXX/` (the scene token is the 2nd chunk of the take name, e.g., `indiana_music_14_3 → music`).
   - Override `--camera-glob` or `--source-subdir` if your folder layout differs from the EgoExo defaults.

2. **Generate DyTrial JSON metadata** (requires `torch`, `pytorch3d`, `pandas`, etc.—install via `python -m pip install -r requirements.txt` if they are missing):
   ```bash
   python preprocess_scripts/2_formal_DyTrialJson.py \
     --seq indiana_music_14_3 \
     --raw-data-root ../../raw_data
   ```
   - `--seq` should match the folder name created by step 1 (e.g., `indiana_piano_14_4`).
   - The script expects per-frame images under `undist_processed_frames/undist_cam01` and calibration CSVs under `<take>/trajectory/` (produced earlier by `push_all_data.sh`).
   - A `Dy_train_meta.json` file is emitted in the take’s `trajectory/` folder, ready for downstream training code.

Re-run both scripts whenever you update the underlying raw data; they are idempotent and overwrite outputs in place.
