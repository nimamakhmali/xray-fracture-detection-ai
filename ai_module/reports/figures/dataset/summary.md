# Dataset figures — computed summary

- generated: 2026-09-04T19:29:35
- manifest: `/home/nimamakhmali/Documents/Ai Project /xray-fracture-detection-ai/ai_module/data/processed/manifest.csv`
- manifest sha256: `723371399d98c0af38a326593dbc875cd8526e8742a2ec9062b5b3266524ffe5`
- rows: 24,337

## Source × split × label

| source / split | positive | negative | total |
|---|---:|---:|---:|
| FracAtlas/train | 499 | 2,307 | 2,806 |
| FracAtlas/val | 106 | 492 | 598 |
| FracAtlas/test | 108 | 498 | 606 |
| GRAZPEDWRI-DX/train | 9,395 | 4,782 | 14,177 |
| GRAZPEDWRI-DX/val | 2,054 | 1,034 | 3,088 |
| GRAZPEDWRI-DX/test | 2,101 | 961 | 3,062 |

## Labels (parsed from YOLO .txt)

- label files found / missing: 24,337 / 0
- malformed lines: 0
- boxes parsed: 19,005  per split: {'train': 13173, 'val': 2918, 'test': 2914}
- class ids: {'0': 19005}
- rows with manifest.num_boxes ≠ parsed: 0

## patient_id / leakage (recomputed from manifest)

- FracAtlas: coverage 0.0%, unique patients —, patients in >1 split: —
- GRAZPEDWRI-DX: coverage 100.0%, unique patients 6091, patients in >1 split: 0
- cross-split hash overlap: {'train_val': 0, 'train_test': 0, 'val_test': 0}; duplicate hashes in manifest: 0

## Exclusion accounting

| source | raw | orphan | dup | dup(+) | integrity | other | expected | manifest | reconciled | positives lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| FracAtlas | 4084 | 1 | 15 | 4 | 58 | 0 | 4010 | 4010 | True | 0 |
| GRAZPEDWRI-DX | 20327 | 0 | 0 | 0 | 0 | 0 | 20327 | 20327 | True | 0 |
- duplicate label conflicts: 0

| split | pre-drop | post-drop | Δ | pos Δ |
|---|---:|---:|---:|---:|
| train | 17024 | 16983 | -41 | +0 |
| val | 3697 | 3686 | -11 | +0 |
| test | 3674 | 3668 | -6 | +0 |
- total Δ = -58

## FracAtlas integrity locality

- dropped index range: [4028, 4308], re-synced range: [4027, 4310]

## Re-synced images — dimension check

- checked 27: width/height mismatch vs manifest = 0

## Image hash verification (md5 on disk vs manifest)

- checked 24,337: mismatches 0, missing 0

## Provenance note — manifest hash sync

- 27 image hashes were re-synced on 20260904T072559Z (files changed on disk after manifest creation).
- sources: {'fracatlas': 27}
- These files must be listed in the report as *modified after preparation* (e.g. Ultralytics in-place JPEG repair).

## Figures

- `fig01_split_counts`
- `fig02_label_balance`
- `fig03_source_counts`
- `fig04_source_by_split`
- `fig05_label_by_split`
- `fig06_label_by_source`
- `fig07_source_split_label_table`
- `fig08_boxes_per_image`
- `fig09_box_area_distribution`
- `fig10_box_shape`
- `fig11_box_center_heatmap`
- `fig12_image_resolution`
- `fig13_annotation_status`
- `fig14_patient_id_coverage`
- `fig15_class_ids_in_labels`
- `fig16_exclusion_funnel`
- `fig16b_split_pre_post_drop`
- `fig17_summary_table`
- `fig18_fracatlas_integrity_locality`