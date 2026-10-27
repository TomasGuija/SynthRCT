# SynthRCT: Scalable Conditional Deformation Synthesis for Synthetic Repeat CT Generation

[![arXiv](https://img.shields.io/badge/arXiv-2609.03956-b31b1b.svg)](https://arxiv.org/abs/2609.08627)
[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-0055A4.svg)](https://papers.miccai.org/miccai-2026-sat/MIART_010.html)

This is the official repository for the paper "**SynthRCT: Scalable Conditional Deformation Synthesis for Synthetic Repeat CT Generation**".
RCTSynth learns a distribution of plausible anatomical deformations from longitudinal CT images. Given one moving CT volume, the model can sample latent deformation codes, decode them into dense deformation fields, and generate multiple anatomically consistent repeat CT volumes.

<img src="./doc/Overview.svg"
     alt="RCTSynth method overview"
     width="100%">

## Method

RCTSynth is a conditional variational registration model. During training, a posterior encoder observes a fixed/moving CT pair and estimates `q(z | fixed, moving)`, while an anatomy-conditioned prior estimates `p(z | moving)` from the moving image alone. A latent-conditioned decoder predicts stationary velocity fields (SVFs) over overlapping axial slabs. The slab predictions are stitched and refined, integrated into a dense deformation field, and used to warp the input CT.

This design supports:

- patient-specific sampling from a single input CT;
- full-volume synthesis using a memory-aware slab decoder;
- smooth latent-space traversals and interpolation;
- export of warped CT volumes, SVFs, and displacement fields.

## Open H 4D phase-conditioned extension

We have extended the work described in the SynthRCT paper with explicit
respiratory phase conditioning. This development supports a proposal for the
[Open H 4D initiative](https://github.com/open-h/open-h-4d) and takes
inspiration from [MONAI Physio](https://github.com/Project-MONAI/monai-physio).
The phase-conditioned training and inference modes were introduced after the
paper and are therefore not described or evaluated in the publication.

The original model generates plausible repeat CTs without assigning them an
explicit respiratory state. The extension adds a scalar target phase
`s in [0, 1)` to make that state controllable. Following MONAI Physio, this
implementation considers ten respiratory phases, represented by `s=0.0`,
`s=0.1`, ..., `s=0.9`.

Training and inference use the same path:

1. One phase is selected as the reference CT for every subject (the supplied
   configuration uses `s=0.7`). During training, a target phase from the same
   subject is sampled and used as the registration target.
2. The prior encoder receives only the reference CT and the requested `s`, and
   produces the deformation latent code. The posterior path is not used.
3. The decoder receives the reference-image features, latent code, and `s`,
   then predicts the deformation that warps the reference CT toward the
   requested respiratory state.
4. At inference, the same reference CT can be reused with different values of
   `s` to synthesize a full respiratory cycle. In this mode, the predicted
   prior mean is used directly as the latent code during both training and
   inference; no random latent samples are drawn.

Setting `model.condition_on_phase: true` automatically removes the posterior
path. With phase conditioning disabled, the original paper model remains the
default.

This deterministic formulation is intended as a proof of concept: it isolates
whether the target phase can control the predicted motion without introducing
sampling variability. A future extension could restore a variational
formulation, sampling latent codes during training and using the learned mean
for deterministic inference, to represent multiple plausible motions for the
same anatomy and respiratory phase.

## Installation

Python 3.10 or newer is recommended. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/TomasGuija/RCTSynth.git
cd RCTSynth

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

That command installs RCTSynth and the dependencies required for training and inference. To run the Jupyter notebook and export GIFs, install the notebook extras instead:

```bash
pip install -e ".[notebook]"
```

If you need a specific CUDA-enabled PyTorch build, install the appropriate PyTorch version first and then run `pip install -e .`. CPU inference is supported but full-volume decoding can be slow.

## Pretrained weights

The pretrained RCTSynth weights are openly available from
[Hugging Face](https://huggingface.co/TomasGuija/RCTSynth). Public downloads do
not require authentication.

Install the Hugging Face Hub client and download the model configuration and
checkpoint into the path used by the inference notebook. The configuration
records the released architecture, checkpoint format, and weight checksum.

```bash
pip install huggingface_hub
hf download TomasGuija/RCTSynth \
  config.json rctsynth.ckpt \
  --local-dir checkpoints
```

The same files can be downloaded and the checkpoint loaded from Python:

```python
from huggingface_hub import hf_hub_download
from rctsynth.utils.model_loading import load_vae

config_path = hf_hub_download(
    repo_id="TomasGuija/RCTSynth",
    filename="config.json",
    local_dir="checkpoints",
)

checkpoint_path = hf_hub_download(
    repo_id="TomasGuija/RCTSynth",
    filename="rctsynth.ckpt",
    local_dir="checkpoints",
)

module, model = load_vae(checkpoint_path, device="cuda")
```

## Data preparation

Training data are not distributed with this repository. Training expects a local HDF5 file containing normalized CT volumes, case and respiratory/time identifiers, and valid axial ranges. The complete schema is documented in [`src/rctsynth/data/README.md`](src/rctsynth/data/README.md).

Update `data.h5_path` and `data.val_case_ids` in [`config/vae.yaml`](config/vae.yaml) before training. Images in the HDF5 file must already be preprocessed and normalized, normally to `[0, 1]`.

For inference on a single NIfTI CT, the included preprocessing utility resamples, center-crops or pads, normalizes the image, and creates heuristic body and lung masks:

```bash
python src/rctsynth/data/preprocess_ct.py \
  --input-nii /path/to/input_ct.nii.gz \
  --output-dir data/demo \
  --target-shape 207 256 256 \
  --spacing-mm 1.5 \
  --norm-mode clip
```

The masks produced by this script are threshold-based aids for demos and cropping; they are not clinical segmentations.

## Training

After preparing the HDF5 dataset and updating the configuration, start training from the repository root:

```bash
python -m rctsynth.training.train fit --config config/vae.yaml
```

To resume a run from a Lightning checkpoint:

```bash
python -m rctsynth.training.train fit \
  --config config/vae.yaml \
  --ckpt_path /path/to/last.ckpt
```

To train the phase-conditioned, prior-only extension from scratch, run:

```bash
python -m rctsynth.training.train fit \
  --config config/vae_phase_prior_only.yaml
```

This uses [`config/vae_phase_prior_only.yaml`](config/vae_phase_prior_only.yaml).

## Inference

The recommended inference entry point is [`notebooks/inference_demo.ipynb`](notebooks/inference_demo.ipynb).

The notebook demonstrates how to load the checkpoint, sample the anatomy-conditioned prior, synthesize a random repeat CT, and save the warped volume. It also samples the patient-specific latent space, performs a principal-component traversal around the encoded identity deformation, and exports intermediate warped images and displacement fields as NIfTI files.

The 2D interpolation GIF uses Pillow. The optional surface-rendered lung GIF additionally requires a local [3D Slicer](https://www.slicer.org/) installation and a valid `SLICER_EXECUTABLE` path.

<table width="100%" cellpadding="12" cellspacing="8" bgcolor="#EAF6FF">
  <tr>
    <td width="40%" align="center" valign="middle" bgcolor="#D9EEFF">
      <span style="color:#174A6E"><strong>Axial view</strong></span><br>
      <img src="doc/pc1_axial.gif" alt="Axial interpolation" height="240">
    </td>
    <td width="60%" rowspan="2" align="center" valign="middle" bgcolor="#D9EEFF">
      <span style="color:#174A6E"><strong>3D lung deformation</strong></span><br>
      <img src="doc/pc1_lung_3d.gif" alt="3D lung interpolation" height="500">
    </td>
  </tr>
  <tr>
    <td align="center" valign="middle" bgcolor="#D9EEFF">
      <span style="color:#174A6E"><strong>Coronal view</strong></span><br>
      <img src="doc/pc1_coronal.gif" alt="Coronal interpolation" height="240">
    </td>
  </tr>
</table>

### Phase-conditioned inference

The [`phase_conditioned_inference.ipynb`](notebooks/phase_conditioned_inference.ipynb)
notebook loads one source CT and generates the complete ten-state respiratory
cycle. It saves each predicted volume and deformation field, creates axial and
coronal GIFs, and can optionally render a 3D lung animation with 3D Slicer.

<table width="100%" cellpadding="12" cellspacing="8" bgcolor="#EAF6FF">
  <tr>
    <td width="50%" align="center" valign="middle" bgcolor="#D9EEFF">
      <span style="color:#174A6E"><strong>Axial breathing cycle</strong></span><br>
      <img src="doc/phase_cycle_axial.gif" alt="Phase-conditioned axial breathing cycle" height="220">
    </td>
    <td width="50%" align="center" valign="middle" bgcolor="#D9EEFF">
      <span style="color:#174A6E"><strong>Coronal breathing cycle</strong></span><br>
      <img src="doc/phase_cycle_coronal.gif" alt="Phase-conditioned coronal breathing cycle" height="220">
    </td>
  </tr>
</table>

## License

The RCTSynth software and released model weights are available under the
[MIT License](LICENSE). The third-party demo data under `data/demo/` are not
covered by that license; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance and terms.

## Citation

```bibtex
@InProceedings{GuiTom_SynthRCT_MICCAISAT2026,
  author    = {Guija-Valiente, Tomás and Malpica, Norberto and Rodriguez-Gonzalez, Blanca},
  title     = {SynthRCT: Scalable Conditional Deformation Synthesis for Synthetic Repeat CT Generation},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 17274},
  month     = {pending},
  pages     = {pending}
}
```

## References

[1] Ma, J., et al. (2020). *COVID-19 CT Lung and Infection Segmentation Dataset*. [https://doi.org/10.5281/zenodo.3757476](https://doi.org/10.5281/zenodo.3757476).

[2] Castillo, E., Castillo, R., Martinez, J., Shenoy, M., & Guerrero, T. (2010). Four-dimensional deformable image registration using trajectory modeling. *Physics in Medicine and Biology, 55*(1), 305–327. https://doi.org/10.1088/0031-9155/55/1/018

[3] Castillo, R., Castillo, E., Guerra, R., Johnson, V. E., McPhail, T., Garg, A. K., & Guerrero, T. (2009). A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. *Physics in Medicine and Biology, 54*(7), 1849–1870. https://doi.org/10.1088/0031-9155/54/7/001
