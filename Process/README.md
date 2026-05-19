# Data Preprocessing Pipelines

`Process/` contains dataset-specific preprocessing utilities used to convert raw images, videos, camera calibration, and ground-truth trajectories into the unified iWorldBench / CityWorld-style assets.

The common target is:

- **Segmented videos**: fixed-length clips, usually 81 frames per segment.
- **Camera trajectory TXT files**: one synchronized camera-control record per frame or segment frame.
- **Unified camera-motion representation**: raw dataset poses are converted into a common locomotion/action format used by the benchmark.

These scripts are reference preprocessing utilities. Different datasets have different raw layouts, image resolutions, frame rates, calibration files, and pose conventions, so each dataset folder may contain slightly different steps.

## Unified trajectory format

The preprocessing pipeline normalizes ground-truth trajectories into a CityWorld-style camera-control format. This format represents both the camera motion direction and the image-space displacement that should be reflected during video generation.

Raw trajectories commonly appear in one of three forms:

| Raw pose style | Description | Purpose in preprocessing |
|---|---|---|
| 6-DoF pose | Translation and rotation pose records | Compact source pose format. |
| 7-element pose | Translation plus quaternion-style rotation | Intermediate normalized pose format. |
| Extrinsic matrix | Camera/world transform matrix | Matrix form for consistent geometric parsing. |

The final per-frame TXT action record is a 19-value row:

| Columns | Meaning |
|---|---|
| 1 | Timestamp. |
| 2-5 | Four camera intrinsic values, typically `fx`, `fy`, `cx`, `cy`. |
| 6-7 | Two distortion coefficients. |
| 8-19 | Flattened `3 x 4` camera matrix. |

The first 7 values provide timing and camera calibration metadata. The last 12 values encode the flattened `3 x 4` matrix used by downstream camera-control and metric code.

## Main subdirectories

The exact filenames vary by dataset, but the major roles are consistent.

| Directory | Role |
|---|---|
| `fix/` | Converts raw dataset pose files into the project coordinate convention. For NCLT-style data, this step reads raw ground-truth CSV files and writes per-frame 6-DoF pose files under CityWorld-style pose directories such as `pose/<view>/<sequence>/six_Dof.txt`. |
| `city_txt/` | Builds `path.txt` image lists in the CityWorld directory structure. Each row points to one image frame and should align with one pose row after synchronization. |
| `transfer/` | Converts existing `six_Dof.txt` pose files into additional pose formats such as `seven_element.txt` and `extrinsics_matrix.txt`, keeping the same number of rows. This makes heterogeneous dataset formats easier to parse through one downstream path. |
| `reflect/` or `reflection/` | Aligns image paths, pose rows, and calibration files, then exports segmented videos and camera TXT files. Segments are typically fixed at 81 consecutive frames. Dataset-specific resolution handling and compression constants are defined in the corresponding scripts. Debug variants are used for camera-only regeneration or inspection. |
| `cam_para/` | Utilities for generating or adapting camera intrinsic files, especially when image orientation or effective resolution changes. |
| `pic_pro/` | Optional image preprocessing helpers, such as orientation fixes or quick video previews from image sequences. Some operations may rewrite source images in place, so inspect scripts before running them. |
| `process/` | Dataset-specific staging utilities, including frame extraction, path-list construction, and pose preparation for datasets whose raw packages are not already in CityWorld-style layout. |
| `auto_pipeline/` | Optional orchestration scripts that chain pose preparation, coordinate fixing, data staging, and transfer steps for supported datasets. |

## Recommended raw-to-unified flow

The following describes the logical data flow. Adjust root paths and constants inside each dataset script before running.

### 1. Raw data

Typical inputs are:

- **Images or videos**: raw image sequences or source videos from each dataset.
- **Ground-truth poses**: dataset-provided pose files, such as timestamped CSV files, 6-DoF records, quaternions, or extrinsic matrices.
- **Calibration**: camera intrinsics, image resolution, and distortion coefficients when available.

### 2. Optional image preprocessing

Use dataset-specific `pic_pro/` or equivalent utilities only when raw image orientation, resolution, or preview generation needs adjustment.

Before running any script that rewrites images, verify:

- The input/output path constants.
- Whether files are modified in place.
- Whether calibration files need to be updated after rotation or resizing.

### 3. Convert poses into CityWorld structure with `fix/`

The `fix/` step converts raw ground-truth pose files into the project coordinate convention.

Typical output:

```text
<pose_root>/<view>/<sequence>/six_Dof.txt
```

The output is usually a headerless, whitespace-separated per-frame 6-DoF pose file.

### 4. Build image lists with `city_txt/` or dataset-specific staging scripts

This step writes one image path per line:

```text
<rgb_root>/<view>/<sequence>/path.txt
```

The intended invariant is that each row in `path.txt` corresponds to the same frame index as the pose files. If raw frame rates differ, synchronize or downsample before final export.

### 5. Generate multiple pose formats with `transfer/`

For every sequence containing `six_Dof.txt`, generate the additional pose files needed by downstream code:

```text
seven_element.txt
extrinsics_matrix.txt
```

These files should have the same row count as `six_Dof.txt`.

### 6. Prepare camera intrinsics

Final export requires readable camera intrinsics and image-size metadata. Some `reflect/` scripts can write dataset-default calibration values when `calibration.txt` is missing, but custom calibration is recommended when images are rotated, cropped, or resized.

The calibration file must be compatible with the parser used by the corresponding export script.

### 7. Align frames and export final assets with `reflect/`

The final export stage aligns:

- `path.txt` image rows.
- Pose rows from `six_Dof.txt`, `seven_element.txt`, or `extrinsics_matrix.txt`.
- `calibration.txt` camera metadata.

A common strategy is to use `path.txt` as the reference row count. If pose files contain more rows, scripts may downsample pose files and save backups before export.

Final output is typically:

```text
<output_root>/<dataset_namespace>/videos/*.mp4
<output_root>/<dataset_namespace>/cameras/*.txt
```

The exported videos and camera TXT files are segmented into consecutive clips, commonly with:

```text
FRAME_COUNT_PER_SEG = 81
```

Resolution and compression behavior are dataset-specific and defined by constants in each `reflect` script.

## Dataset-specific notes

- **NCLT**: includes `fix/`, `city_txt/`, `transfer/`, `reflect/`, `cam_para/`, and `pic_pro/` utilities for converting raw image sequences and timestamped ground-truth poses into segmented video/camera assets.
- **SpatialVID**: includes a broader staging pipeline with `process/`, `fix/`, `transfer/`, `reflection/`, and optional `auto_pipeline/` orchestration.
- **TartanAir, 7-Scenes, Princeton365, realestate10k, NuScenes-style folders**: contain dataset-specific variants for path-list generation, pose conversion, frame/video handling, or export. Inspect script constants and input assumptions before use.

## Practical invariants

Before using generated data downstream, verify:

- Every final video segment has continuous frames and the expected segment length.
- Each camera TXT corresponds to the same segment index and frame order as its video.
- Pose rows and image rows are synchronized before segmentation.
- The final TXT rows follow the 19-value unified action format.
- Calibration values match the effective exported image resolution.
