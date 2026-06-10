# Dataset Schema

The public Hugging Face dataset is loaded as:

```python
from datasets import load_dataset
dataset = load_dataset("TUM-ICS/Hide-and-Seek")
```

Expected fields per frame:

| Field | Shape | Type | Notes |
| --- | --- | --- | --- |
| `timestamp` | scalar | float64 | Frame timestamp |
| `episode_index` | scalar | int64 | Episode identifier |
| `obs_point_cloud_skin_contact` | 88 x 6 | float64 | Tactile contact point cloud |
| `mask_point_cloud_skin_contact` | 88 | bool | Valid point mask |
| `obs_pose_left` | 7 | float64 | Left pose |
| `obs_skin_prox_left` | 44 | float64 | Raw skin proximity |
| `obs_skin_force_left` | 44 | float64 | Raw skin force |
| `obs_skin_dist_left` | 44 | float64 | Raw skin distance |
| `obs_dami_force_left` | 6 | float64 | Virtual skin force wrench |
| `obs_dami_prox_left` | 6 | float64 | Virtual skin proximity wrench |
| `obs_ft_left` | 6 | float64 | Wrist force/torque wrench |
| `classification_label` | scalar | int64 | 61-way object-weight label |

Frame-wise examples are converted into windows by `HASWindowDataset`.

- Default window length: 100 frames
- Default stride: 40 frames
- Grouping key: `episode_index`
- Episode boundaries are never crossed
- The window label is read from the final frame in the window

Returned PyTorch keys include all input fields plus:

- `classification_label`
- `object_label`
- `weight_label`
- `episode_index`
- `window_start`
