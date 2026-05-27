# Tactile Hide and Seek Baselines

Official offline baseline code for the ICRA 2026 paper:

**Tactile Hide and Seek: Bimanual Object Blind Search and Retrieval Via Tactile-Only Feedback**

This repository provides a minimal public release for loading the Hugging Face dataset, creating temporal windows, training a readable PyTorch dual-head baseline, and evaluating checkpoints. 

## Links

- Dataset: https://huggingface.co/datasets/StanFu/Hide-and-Seek-v1
- Project page: [tactile-hide-and-seek](https://tactile-hide-and-seek.github.io/)
- Paper: (Coming soon)

## Dataset

The dataset is frame-wise and contains:

- train: 777,169 examples
- validation: 221,195 examples
- test: 121,752 examples
- total: 1,120,116 frame-wise tactile examples

The baseline trains on temporal windows with default `window_length=100` and `stride=40`. Windows are grouped by `episode_index` and never cross episode boundaries.

```python
from datasets import load_dataset

dataset = load_dataset("StanFu/Hide-and-Seek-v1")
```

## Installation

```bash
conda create -n thas python=3.10 -y
conda activate thas
pip install -e ".[dev]"
```

On Linux or macOS, use `source activate thas` instead of the Windows activation command.


## Training

Full M1-M5 baseline:

```bash
python scripts/train.py --config configs/baseline_m12345.yaml --output_dir checkpoints/m12345
```

M1-M3 baseline:

```bash
python scripts/train.py --config configs/baseline_m123.yaml --output_dir checkpoints/m123
```

You can also choose modalities directly:

```bash
python scripts/train.py --modalities M23 --batch_size 32 --learning_rate 1e-4 --epochs 20
```

## Evaluation

```bash
python scripts/eval.py --checkpoint checkpoints/m12345/best_model.pt --split test
```

Optional object confusion matrix export:

```bash
python scripts/eval.py --checkpoint checkpoints/m12345/best_model.pt --split test --confusion_matrix outputs/object_confusion.csv
```

> Trained checkpoints are not shipped with this release. The `checkpoints/` directory is created locally when you run `scripts/train.py` with `--output_dir`. Re-run the training commands above to reproduce the weights and `last_metrics.json` files referenced below.

## Results

The table below reports validation-set accuracy (%) for the two reference configurations in this public release, alongside the corresponding rows from Table I of the paper.

The public pipeline is a cleaned-up release version of the internal training code used for the paper. Due to post-submission implementation updates, including differences in data loading, normalization, and training configuration, the released pipeline may produce results that differ from the published numbers.

| Modalities | Source        | Object Acc | Weight Acc | Joint Acc |
| ---------- | ------------- | ---------- | ---------- | --------- |
| M123       | Paper Table I | 90.8       | 83.2       | 79.0      |
| M123       | This release  | **93.47**  | **87.50**  | **84.62** |
| M12345     | Paper Table I | 91.1       | 83.1       | 79.6      |
| M12345     | This release  | **93.83**  | **88.55**  | **85.62** |

Release numbers were produced by `scripts/train.py` with `configs/baseline_m123.yaml` and `configs/baseline_m12345.yaml` using the default seed, and read from `checkpoints/<run>/last_metrics.json` after training.

The trained checkpoints and generated `last_metrics.json` files are not included in this repository. To reproduce these release numbers, please re-run the corresponding training commands.

Your own runs may differ slightly depending on hardware, PyTorch/CUDA versions, random seed behavior, and Hugging Face dataset cache state.

## Labels

`data/label_mapping.json` and `data/label_mapping.yaml` contain:

- `classification_labels`: 61 object-weight labels, including `negative`
- `object_labels`: 34 object labels, including `negative`
- `weight_labels`: 4 labels: `none`, `light`, `medium`, `heavy`

Examples:

- `negative -> object=negative, weight=none`
- `bag_medium -> object=bag, weight=medium`
- `tomatosoup_light -> object=tomatosoup, weight=light`

The training loss is:

```text
object_cross_entropy + 0.5 * weight_cross_entropy
```

Metrics are object accuracy, weight accuracy, and joint accuracy.

## Repository Layout

```text
configs/      Baseline and dataset configs
data/         Label mapping and optional normalization stats
docs/         Dataset schema and repository audit notes
has_dataset/  Hugging Face loading, windowing, transforms, collate
models/       Point cloud, temporal, fusion, and dual-head model modules
scripts/      Training, evaluation, inspection, and stats commands
```

The older internal pipeline is retained under `legacy/` for maintainer review. New public users should start with `scripts/train.py`, `scripts/eval.py`, `has_dataset/`, and `models/`.

## Citation

```text
@inproceedings{fu2026tactilehideandseek,
  title     = {Tactile Hide and Seek: Bimanual Object Blind Search and Retrieval Via Tactile-Only Feedback},
  author    = {Fu, Xiangyu and Xing, Hao and Armleder, Simon and Shen, Wenlan and Wang, Fengyi and Guadarrama-Olvera, Julio Rogelio and Cheng, Gordon},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## Contact
[Xiangyu Fu](mailto:xiangyu.fu@example.com)
