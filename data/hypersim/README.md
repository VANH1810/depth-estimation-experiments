# Hypersim Data

Use this directory for Hypersim data.

```text
data/hypersim/
  raw/<scene_id>/          Raw HyperSim scene with images/*_hdf5 folders.
  samples/
    rgb/*.png             Extracted RGB images.
    depth/*.npy           Extracted metric depth in meters.
    intrinsics.json       Optional per-frame 3x3 intrinsics.
```

Extraction command:

```bash
python scripts/extract_hypersim.py --scene_root data/hypersim/raw/ai_001_001 --output_root data/hypersim/samples
```

Do not commit raw scenes, extracted arrays, archives, or model outputs.
