# EgoExo Processing — Short Guide

All commands start inside `MonoFusion/preproc/ExoRecon`. Raw takes live in `../../raw_data`, processed assets in `../../data`.

## 1. Grab a take
Register for EgoExo access at https://docs.ego-exo4d-data.org/, accept the license, then use the portal at https://visualize.ego4d-data.org/login to find the `<take_uuid>` you want.
```bash
egoexo -o ../../ \
  --uids <take_uuid> \
  --parts take_trajectory take_vrs takes -y
mv ../../raw_data/takes/* ../../raw_data/
rm -rf ../../raw_data/takes
```
Add the first valid RGB frame index to each take folder:
```bash
echo <first_frame_idx> > ../../raw_data/<take_id>/timestep.txt
```

## 2. One-time environment setup
```bash
conda env create -f egorecon.yml
conda activate egorecon
python -m pip install --upgrade pip
python -m pip install -e projectaria_tools_pkg
```

## 3. Process everything
Edit `LOCAL_BASE_PATH` inside `push_all_data.sh` if paths differ, then:
```bash
./push_all_data.sh
```
This runs `viewer_mps` plus both preprocessing scripts for every take under `../../raw_data`.

## 4. Desired layout after `push_all_data.sh`
```
../../raw_data/<take_id>/
  aria01.vrs
  frame_aligned_videos/
    cam01.mp4 … cam04.mp4
  processed_frames/
    cam01/...
  undist_processed_frames/
    undist_cam01/...
  train.matrices.txt
  cam0X_train.matrices.txt
  trajectory/
    closed_loop_trajectory.csv
    gopro_calibs.csv
    Dy_train_meta.json
  timestep.txt
../../data/images/<take_id>/
  cam01/...
  cam02/...
  cam03/...
  cam04/...
```
If any folder is missing, re-run `./push_all_data.sh` or the numbered scripts inside `preprocess_scripts/` for that take.
