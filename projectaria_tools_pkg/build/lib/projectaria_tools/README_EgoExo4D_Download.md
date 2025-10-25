# Ego-Exo4D Dataset Access & Download Guide

This guide summarizes the official instructions from [ego-exo4d-data.org](https://ego-exo4d-data.org) and the Ego-Exo4D documentation so you can register, obtain credentials, and download exactly the takes you need.

## 1. Request Access (required)
- **License portal:** https://ego4ddataset.com/egoexo-license/
- **What happens:** You (or an authorized institutional signatory) complete the electronic license. Approval typically takes ~48 hours.
- **Credentials:** Once approved you receive AWS credentials by email. They are valid for 14 days, so plan to download locally before they expire. Renewals use the same portal and are instant after a prior approval.

## 2. Prepare Your Environment
- **Install / upgrade the Ego4D tools:**
  ```bash
  pip install --upgrade ego4d
  ```
  Ego-Exo4D downloads require `ego4d` version 1.7.1 or newer.
- **Install the AWS CLI (if not already available):** `pip install awscli`
- **Configure AWS with the emailed keys:**
  ```bash
  aws configure --profile egoexo
  # paste the Access Key ID and Secret Access Key
  # press Enter to accept the default region and output format
  ```
  Use `--s3_profile egoexo` with the downloader if you store the keys under a non-default profile name.

## 3. (Optional) Browse Before Downloading
- Read the "Overview" and "Data" sections at https://docs.ego-exo4d-data.org to understand available modalities.
- The small `metadata` part (< 100 MB) contains `captures.json` and `takes.json`, useful for discovering `take_uid` values before downloading larger assets.

## 4. Run the CLI Downloader
The `egoexo` CLI pulls data from `s3://ego4d-consortium-sharing/egoexo-public`. Basic usage downloads the ~14 TiB recommended subset:

```bash
egoexo -o /path/to/output
```

Key flags (can be combined):
- `--parts <p1> <p2> ...` – select dataset parts (`annotations`, `takes`, `take_point_cloud`, `take_vrs`, `captures`, etc.). See the [CLI docs](https://docs.ego-exo4d-data.org/download/) for the full list and sizes.
- `--splits train|val|test` – restrict to splits.
- `--views ego|exo` – restrict to egocentric or exocentric views.
- `--benchmarks <name>` – limit to benchmark-specific data.
- `--universities <u1> <u2>` – filter by contributing institution.
- `--uids <uid1> <uid2> ...` – download only the takes or captures you list.
- `-y` – skip interactive confirmation, `--num_workers` – adjust parallelism.

## 5. Downloading with a Take List File
1. **Fetch metadata once** (if you have not already) so you can inspect take IDs:
   ```bash
   egoexo -o /data/egoexo --parts metadata
   jq '.takes[] | {take_uid, activity, capture_uid} | select(.activity=="guitar")' \
     /data/egoexo/metadata/takes.json | head
   ```
   Replace the `jq` filter with criteria relevant to your project.
2. **Create a text file** containing one `take_uid` per line, for example `take_uids.txt`:
   ```text
   take_a302c11b-105f-4a9a-a526-2b9f65ba45f6
   take_9f12192b-0eb4-4d14-a1dc-0f7fa85a4fd2
   ```
3. **Run the downloader for those takes** (here downloading frame-aligned videos and matching trajectories as an example):
   ```bash
   egoexo -o /data/egoexo \
     --parts takes take_trajectory \
     --uids $(tr '\n' ' ' < take_uids.txt)
   ```
   - Use `--parts take_vrs` to grab VRS files instead, or add other parts (e.g. `take_point_cloud`, `take_eye_gaze`).
   - The `tr` command converts the newline-separated list into the space-separated format that `--uids` expects. On Windows PowerShell, replace it with ``$(Get-Content take_uids.txt)``.

## 6. Verify & Resume Downloads
- The CLI resumes interrupted downloads automatically. Re-run the same command; the tool skips completed files unless you pass `--force`.
- Use `--delete` to prune files that are no longer part of the selected release.
- Keep an eye on remaining credential validity (14 days). Renew via the license portal if needed.

## 7. Helpful Links
- License portal: https://ego4ddataset.com/egoexo-license/
- Getting Started guide: https://docs.ego-exo4d-data.org/getting-started/
- CLI download reference: https://docs.ego-exo4d-data.org/download/
- Data overview & take structure: https://docs.ego-exo4d-data.org/data/takes/

Contact the dataset maintainers at [info@ego4d-data.org](mailto:info@ego4d-data.org) for account or access issues.

