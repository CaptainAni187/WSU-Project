# Data

**The dataset is not distributed in this repository.** It consists of public Reddit posts and
comments that include usernames and first-person mental-health disclosures, so raw text,
post/comment IDs, and per-document labels are kept out of version control (see `.gitignore`).

The scripts expect the following local layout (all paths are relative to the repo root):

```
data/
├── raw/
│   ├── phase1.csv, phase2.csv, phase3.csv, extra.csv   # scraped posts (candidate set)
│   └── reddit_comments.csv                             # 164,367 comments linked by post_id
├── processed/
│   ├── posts_merged_final.csv                          # 70,425 de-duplicated posts
│   ├── posts_merged_final_preprocessed.csv             # + cleaned_text / phrase_text
│   └── comments_preprocessed.csv                       # written by scripts/comments/preprocess_comments.py
├── embeddings/
│   ├── posts/{model}_embeddings.npy                    # 70,425 x 768/1024
│   └── comments/{model}_embeddings.npy                 # 138,989 x 768/1024
└── annotations/
    ├── posts_distress_annotation.csv                   # 300 posts, human-labelled
    └── comments_interaction_annotation.csv             # 300 comments, human-labelled
```

## Corpus summary

| Stage | Count |
|---|---|
| Posts initially collected | 207,546 |
| Distress-candidate posts (after two-stage filtering) | 71,873 |
| Posts after de-duplication (209 by ID, 1,239 by exact text) | **70,425** |
| Comments collected (on 8,863 parent posts) | 164,367 |
| Comments after cleaning (deleted/removed/bot/short removed) | **138,989** |

Posts span 11 communities mapped to five lifecycle stages:

| Stage | Posts | Communities |
|---|---:|---|
| Competitive examination | 35,189 | r/JEENEETards, r/CATpreparation, r/UPSC, r/GATEtard (349) |
| Undergraduate | 17,646 | r/Btechtards, r/EngineeringStudents, r/Indian_Academia |
| Career | 9,176 | r/developersIndia, r/GATEtard (1,719), r/IndianEngineers |
| School | 7,267 | r/CBSE |
| Higher education | 1,147 | r/Indians_StudyAbroad |

(Counts are over the full 70,425-post corpus; the topic-level analysis covers the 69,845 posts
assigned to a topic.)

Researchers who need the data for replication can contact the authors.
