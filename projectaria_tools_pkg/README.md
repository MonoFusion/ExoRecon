# projectaria-tools (vendored build)

This directory packages the exact Project Aria Tools bits that are already present in the current environment under `site-packages/projectaria_tools`.  It keeps the full Python sources, type stubs, pybind-based modules, and the prebuilt `_core_pybinds` extension plus the `.dylibs` the tools expect at runtime.  The modified `viewer_mps` utility now includes the custom `mate` workflow that `push_all_data.sh` expects (extract RGB-aligned camera frames and write calibration matrices in a single invocation) plus a more robust search for static camera calibrations.

## Installing from source in a fresh environment

1. Create or activate a Python **3.9** environment (the packaged `_core_pybinds` wheel is compiled for 3.9). For example:
   ```bash
   conda create -n egoexo python=3.9 -y
   conda activate egoexo
   ```
2. From the repository root run `python -m pip install --upgrade pip` once per environment.
3. Install this vendored package directly: `python -m pip install -e projectaria_tools_pkg`.

The editable install keeps the CLI entry points (`viewer_mps`, etc.) in sync with the sources here, so re-installing the environment automatically picks up the customizations without manual edits under `site-packages`.

If you prefer a wheel, you can still build one locally:

```bash
cd /Users/ha1o/Downloads/takes/takes/projectaria_tools_pkg
python -m pip install --upgrade build
python -m build
python -m pip install dist/projectaria_tools-1.3.2.post0-*.whl
```

## Requirements for `push_all_data.sh`

The helper script simply `cd`s into the dataset folder (for example `indiana_music_14_3`) and invokes `viewer_mps --task mate --cam cam01 ...`.  For that to work every time you bootstrap a new **Python 3.9** environment:

- The dataset directory must contain `frame_aligned_videos/` with `cam01.mp4` through `cam04.mp4` and a `timestep.txt` file (the first frame index to export).
- Place `gopro_calibs.csv` either in the dataset root or under `<dataset>/trajectory/`.  The CLI will look in both locations before falling back to the current working directory.  Without this file the `matrices` step for GoPro cameras cannot run.
- Place `trajectory/closed_loop_trajectory.csv` next to the VRS file if you plan to generate RGB-camera matrices (the CLI raises a readable error if it cannot find the file).
- Ensure `ffmpeg` is available on the `PATH` because `viewer_mps` shells out to it for frame extraction.

With those files in place you can simply:

```bash
cd /Users/ha1o/Downloads/takes/takes
./push_all_data.sh
```

The script calls the new `mate` task, which first extracts/undistorts frames (honoring `--debug` to cap the number of frames) and then emits the per-frame matrices for `cam01` as expected by the downstream pipeline.
