# Posts Annotation Report

- Annotation source: `data/annotations/posts_distress_annotation_completed (1).csv`
- Samples: 300
- Agreement: 0.700
- Disagreement: 0.300
- Cohen kappa: 0.000 (Slight agreement)
- 95% bootstrap CI, agreement: [0.643, 0.747]
- 95% bootstrap CI, kappa: [0.000, 0.000]

## Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| ambiguous | 0.0000 | 0.0000 | 0.0000 | 84 |
| distress | 0.7000 | 1.0000 | 0.8235 | 210 |
| non_distress | 0.0000 | 0.0000 | 0.0000 | 6 |

## Interpretation
The validation estimates how closely automated distress labels align with completed human annotation. Disagreements should be inspected as candidate boundary cases for the paper discussion.
