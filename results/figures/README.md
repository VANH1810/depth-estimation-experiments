# Figure Captions

These figures use raw metric-depth predictions from UniDepth, ZoeDepth, and DA-V2 Metric Base.
Depth Anything V2 Small relative outputs and aligned variants are excluded from the main figures.

## depth_test_metric_summary_grid.png
Caption:
Metric comparison on depth_test using raw metric-depth predictions from UniDepth, ZoeDepth, and DA-V2 Metric Base.

## depth_test_qualitative_samples.png
Caption:
Qualitative results on depth_test. Each pair of consecutive rows corresponds to one test sample. The odd row shows the RGB image and error-colored prediction maps using coolwarm based on absolute relative error. The even row shows GT depth and predicted depth. The last column shows the colormap ranges for depth and error.

## depth_test_scatter_pred_vs_gt.png
Caption:
Predicted depth versus GT depth scatter plots for the three metric-depth models on depth_test. The diagonal line indicates perfect metric prediction.

## hypersim_metric_summary_grid.png
Caption:
Metric comparison on hypersim using raw metric-depth predictions from UniDepth, ZoeDepth, and DA-V2 Metric Base. DA-V2 Metric Base is included as a reference, not as a main few-shot baseline, if its checkpoint has Hypersim train/fine-tune overlap.

## hypersim_qualitative_samples.png
Caption:
Qualitative results on hypersim using UniDepth, ZoeDepth, and DA-V2 Metric Base. DA-V2 Metric Base is shown only as a reference if the checkpoint has Hypersim train/fine-tune overlap.

## hypersim_scatter_pred_vs_gt.png
Caption:
Predicted depth versus GT depth scatter plots for the three metric-depth models on hypersim. DA-V2 Metric Base should be interpreted carefully if it has train/fine-tune overlap with Hypersim.

## depth_hypersim_qualitative_samples.png
Caption:
Combined qualitative results for depth_test and hypersim using raw metric-depth predictions from UniDepth, ZoeDepth, and DA-V2 Metric Base. The figure contains one representative sample from depth_test and one representative sample from hypersim while keeping the same qualitative layout. Columns show RGB/GT, UniDepth, ZoeDepth, DA-V2 Metric Base, and the colormap ranges for absolute relative error and depth. Dataset labels mark each dataset group, a light horizontal separator distinguishes the datasets, and a subtle vertical separator separates the RGB/GT column from the model prediction columns.

## Reproducibility
```bash
python -m src.visualization.make_report_figures \
  --results-root results \
  --datasets depth_test hypersim \
  --output-dir results/figures \
  --figures metric_summary_grid scatter_pred_vs_gt qualitative_samples qualitative_samples_combined \
  --models unidepth zoedepth depth_anything_v2_metric_indoor_base \
  --num-qualitative-samples 2 \
  --max-points 100000 \
  --overwrite
```
