# Comments Annotation Report

- Annotation source: `data/annotations/comments_interaction.annotation.csv`
- Samples: 300
- Agreement: 0.177
- Disagreement: 0.823
- Cohen kappa: 0.102 (Slight agreement)

## Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| advice_guidance | 0.2800 | 0.3043 | 0.2917 | 23 |
| blame_criticism | 0.1600 | 1.0000 | 0.2759 | 4 |
| crisis_escalation | 0.1200 | 0.4286 | 0.1875 | 7 |
| dismissive_minimizing | 0.0000 | 0.0000 | 0.0000 | 0 |
| encouragement_motivation | 0.2000 | 0.5556 | 0.2941 | 9 |
| humor_meme_coping | 0.0000 | 0.0000 | 0.0000 | 4 |
| information_resources | 0.0000 | 0.0000 | 0.0000 | 0 |
| neutral_discussion | 0.9200 | 0.1070 | 0.1917 | 215 |
| personal_experience | 0.2000 | 0.2381 | 0.2174 | 21 |
| self_disclosure | 0.0000 | 0.0000 | 0.0000 | 0 |
| support_empathy | 0.1200 | 0.2308 | 0.1579 | 13 |
| toxic_abusive | 0.1200 | 0.7500 | 0.2069 | 4 |

## Interpretation
The comments task is more fine-grained than post distress detection, so lower agreement is expected when categories overlap semantically.
