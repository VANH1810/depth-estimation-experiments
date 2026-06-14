# Depth_Test Data

Place local Depth_Test recordings here.

```text
data/depth_test/
  recording_*/
    color.avi
    camera_parameters.json
    depth_metadata.json
    depth_data/*.npz
```

The loader converts `uint16` depth values to meters using
`camera_parameters.json["depth_scale"]`.

Do not commit recordings, `.npz` batches, videos, or generated evaluation
outputs.

