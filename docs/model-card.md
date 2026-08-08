# FoodVision model card

## Model details

- **Task:** Food-101 image classification
- **Classes:** 101
- **Architecture:** Populate after the final experiment
- **Fine-tuning strategy:** Populate after the final experiment
- **License:** Verify both the selected pretrained weights and dataset terms before redistribution

## Intended use

This model is an educational food-image classifier and portfolio demonstration. It can be used for experimentation, dataset exploration, and non-critical consumer prototypes.

## Out-of-scope use

Do not use predictions for allergy, nutrition, food-safety, medical, religious, or regulatory decisions. A visual category cannot reliably establish ingredients or preparation methods.

## Evaluation

Populate Top-1, Top-5, macro F1, expected calibration error, latency, throughput, and per-class failures using generated artifacts. Never publish placeholder numbers as measured results.

## Limitations

Food may be occluded, combined with other dishes, photographed under unusual lighting, or outside the 101 known categories. The classifier is closed-set and will still assign an in-distribution label to unknown food unless an explicit rejection mechanism is added.

