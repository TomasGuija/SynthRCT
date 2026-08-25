# Dataset

This repository does not include the medical imaging data used for training.

The code expects a local HDF5 file prepared from a 4DCT dataset. In the original project, the data were prepared from the [DIR-Lab/Emory 4DCT dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html).


Users should obtain the original data from the official source and prepare their own local HDF5 file. Preprocessed HDF5 files, CT volumes, masks, and patient-derived arrays are not redistributed with this repository.

## Expected HDF5 structure

The dataloader expects one `.h5` file with axial 3D volumes. The expected structure is:

```text
dataset.h5
├── attrs
│   ├── target_shape = (D, H, W)
│
├── case_id             # int array of shape (N,)
├── time_id             # int array of shape (N,)
├── valid_axis_start    # int array of shape (N,)
├── valid_axis_end      # int array of shape (N,)
│
├── images/
│   ├── 0000000         # float32 array of shape (D, H, W)
│   ├── 0000001
│   └── ...
│
└── anatomy_masks/      # optional
    ├── 0000000         # uint8 or bool array of shape (D, H, W)
    ├── 0000001
    └── ...
```

The current public dataloader assumes axial slicing. Therefore, `D` is the axial/depth dimension.


## Image normalization

The dataloader does not perform intensity normalization online.

All images stored under `images/` are expected to be already preprocessed and normalized, typically to the range `[0, 1]`. If raw or non-normalized CT volumes are stored in the HDF5 file, the model will train/infer on those values directly.


## Example NIfTI preprocessing script

This repository includes a small example script for preparing one CT NIfTI volume for demos:

```bash
python src/rctsynth/data/preprocess_ct.py \
  --input-nii data/raw/example_ct.nii.gz \
  --output-dir data/demo \
  --target-shape 207 256 256 \
  --spacing-mm 1.5 \
  --norm-mode clip
```

The script writes:

```text
preprocessed_ct.nii.gz   # normalized CT in model DHW shape
body_mask.nii.gz         # heuristic body/anatomy mask
lung_mask.nii.gz         # heuristic lung-air mask
```

The body and lung masks are threshold-based heuristics intended for cropping, demos, and optional loss support. They are not clinical segmentations. The output files use a simple isotropic affine for model inputs and do not preserve the full source scanner-space affine.


## Required fields

### `attrs["target_shape"]`

Common image shape after preprocessing:

```text
(D, H, W)
```

All images in the HDF5 file are expected to have this shape.

### `case_id`

Integer array of shape `(N,)`.

`case_id[i]` identifies the patient/case to which image `i` belongs. The dataset uses this field to construct intra-patient moving/fixed pairs only.

Example:

```text
case_id = [0, 0, 0, 1, 1, 1]
```

means:

```text
images/0000000, images/0000001, images/0000002 -> patient 0
images/0000003, images/0000004, images/0000005 -> patient 1
```

The dataset assumes that `case_id[i]` corresponds to `images/{i:07d}`.

### `time_id`

Integer array of shape `(N,)`.

`time_id[i]` identifies the respiratory phase/order of image `i` within its patient. The dataset sorts images from the same patient by `time_id` before constructing training pairs.

Example:

```text
case_id = [0, 0, 0, 0]
time_id = [0, 1, 2, 3]
```

With `min_scan_distance = 2`, possible phase pairs include:

```text
0 -> 2
0 -> 3
1 -> 3
```

Directly consecutive phases are excluded in this example.

### `valid_axis_start` and `valid_axis_end`

Integer arrays of shape `(N,)`.

These define the valid axial range for each volume:

```text
[valid_axis_start[i], valid_axis_end[i])
```

The dataset uses these values to sample local two-slab regions only where both moving and fixed images contain valid anatomical support.

For a moving/fixed pair, the shared valid range is computed as:

```text
shared_start = max(valid_axis_start[moving_idx], valid_axis_start[fixed_idx])
shared_end   = min(valid_axis_end[moving_idx], valid_axis_end[fixed_idx])
```

Only slab centers that fit inside this shared range are sampled.

### `images/`

HDF5 group containing normalized CT volumes.

Each image is stored with a zero-padded integer key:

```text
images/0000000
images/0000001
images/0000002
...
```

Each image must be a `float32` array with shape:

```text
(D, H, W)
```

The dataset converts this internally to:

```text
(D, H, W, 1)
```

before collation.

## Optional fields

### `anatomy_masks/`

Optional HDF5 group containing binary anatomical masks.

Each mask should use the same key as the corresponding image:

```text
anatomy_masks/0000000
anatomy_masks/0000001
...
```

Each mask must have shape:

```text
(D, H, W)
```

If available, the dataset builds a loss mask for each moving/fixed pair as the overlap between the two anatomical masks:

```text
loss_mask = moving_anatomy_mask AND fixed_anatomy_mask
```

This mask is used by the training code to restrict the image-similarity loss to relevant anatomy.

If `anatomy_masks/` is not present, the dataloader returns `loss_mask = None`, and the image-similarity loss is computed over the full sampled region.

## Pair construction

Training samples are formed intra-patient. The dataset groups images by `case_id`, sorts them by `time_id`, and constructs moving/fixed phase pairs from the same patient.

For each candidate pair, the dataset:

1. Computes the shared valid axial range.
2. Checks that a two-slab region fits inside that range.
3. Stores the pair and the valid interval for sampling slab centers.

At runtime, each sample returns one moving/fixed pair and one or more two-slab center pairs.

The returned dictionary has the form:

```python
{
    "moving": moving_image,      # array of shape (D, H, W, 1)
    "fixed": fixed_image,        # array of shape (D, H, W, 1)
    "center": centers,           # array of shape (R, 2)
    "loss_mask": loss_mask,      # array of shape (D, H, W, 1), or None
}
```

where `R = num_sampled_regions`.

For each sampled region, `center[r, 0]` is the center of the first slab and `center[r, 1]` is the center of the second slab. The dataset generates them as:

```text
center[r, 1] = center[r, 0] + stitch_stride
```

The Lightning module then extracts the corresponding slabs from the full moving, fixed, and optional mask volumes.

## Original data source

The experiments in the original project used the DIR4DCT dataset, which contains thoracic 4DCT images from 10 patients. For each patient, 10 CT volumes are available, corresponding to different respiratory phases acquired during the same imaging session.

The original data can be requested from the official [DIR-Lab/Emory 4DCT dataset download page](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html).


Please follow the terms and instructions provided by the official data source. This repository does not redistribute the original or preprocessed medical imaging data.

## Original preprocessing used in this project

In the original project, the DIR4DCT data were converted and preprocessed before training.

The main preprocessing steps were:

1. Convert the original `.img` files to NIfTI volumes.
2. Resample all volumes to isotropic voxel spacing of 1.5 mm.
3. Estimate an anatomical foreground/support mask for each volume.
4. Estimate a common center of mass per patient from the anatomical support of the respiratory phases.
5. Crop or pad all phases around this common center to a fixed shape of 256 x 256 x 207 voxels.
6. Normalize image intensities to the range [0, 1].
7. Compute binary anatomical masks from image intensity values.
8. Store the processed images, metadata, valid axial bounds, and optional masks in the HDF5 format described above.

The foreground/anatomical masks were obtained using thresholding-based heuristics. This was necessary because the original data do not always follow the typical CT Hounsfield-unit intensity scale expected from standard clinical CT volumes.

Training pairs were formed intra-patient by selecting two respiratory phases from the same subject: one as the moving/reference image and the other as the fixed/target image. This avoids introducing inter-patient correspondence assumptions.

During training and evaluation, the two selected phases were required to be separated by a minimum distance in the respiratory phase sequence. This reduces the number of near-identity training cases, where moving and fixed images are already highly aligned.


## References

[1] Castillo, E., Castillo, R., Martinez, J., Shenoy, M., & Guerrero, T. (2010). Four-dimensional deformable image registration using trajectory modeling. *Physics in Medicine and Biology, 55*(1), 305–327. https://doi.org/10.1088/0031-9155/55/1/018

[2] Castillo, R., Castillo, E., Guerra, R., Johnson, V. E., McPhail, T., Garg, A. K., & Guerrero, T. (2009). A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. *Physics in Medicine and Biology, 54*(7), 1849–1870. https://doi.org/10.1088/0031-9155/54/7/001


## Questions

For questions about the expected HDF5 format, preprocessing assumptions, or reproducing the dataset preparation, please contact the main author of this repository.
