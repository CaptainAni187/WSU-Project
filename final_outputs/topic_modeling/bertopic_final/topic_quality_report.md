# Final BERTopic Model — Quality Report

Selected configuration: **`u_nn50_c10__h_mcs300_ms10_leaf`**

## Before / after vs the previous 98-topic model

Both models measured through the IDENTICAL metric pipeline on their final
(all-documents-assigned) labels, so the comparison is apples-to-apples.

| Metric | 98-topic baseline | Final | Verdict |
|---|---:|---:|---|
| Topics | 98 | 41 | fewer — fragmentation resolved (P1) |
| c_v (coherence) | 0.6258 | 0.6518 | higher is better |
| c_npmi | 0.1356 | 0.1419 | higher is better |
| Topic diversity | 0.6408 | 0.7195 | higher is better |
| Tiny topics (<200) | 4 | 0 | fewer is better (P5) |
| Smallest / largest topic | 149 / 2636 | 498 / 4770 | — |
| Redundant pairs (centroid >0.90) | 280 | 128 | fewer is better |
| Inter-topic keyword sim (c-TF-IDF) | 0.1457 | 0.2156 | trade-off (broader topics) |
| Intra-topic sim (compactness) | 0.7432 | 0.7285 | trade-off (broader topics) |
| Outlier rate (after reduction) | 0.0003 | 0.0082 | both near-zero |

> Embedding-centroid `separation` / `inter_sim_emb` are confounded with topic
> count and full-assignment (corr(n_topics, inter_sim_emb) = -0.88), so they were
> excluded from selection and are omitted here. Selection was anchored on
> count-independent quality (coherence, stability) and semantic interpretability.

## Configuration

- UMAP: {'n_neighbors': 50, 'n_components': 10, 'min_dist': 0.0, 'metric': 'cosine'}
- HDBSCAN: {'min_cluster_size': 300, 'min_samples': 10, 'cluster_selection_method': 'leaf'}
- Representation: corpus-level stopword+bigram vectorizer + KeyBERTInspired + MMR
- Outlier reduction: c-TF-IDF strategy, threshold 0.05

## Final topics

| Topic | Size | Keywords |
|---:|---:|---|
| 2 | 4770 | din, baad, saal, mei, baat, agar, bc, itna, hain, ghar |
| 0 | 3398 | dsa, leetcode, coding, learn, learning, code, web, python, projects, development |
| 10 | 3366 | cat, mba, iim, profile, job, work, gap, percentile, college, calls |
| 37 | 3184 | engineering, semester, classes, school, ve, don, degree, class, work, gpa |
| 39 | 2908 | jee, school, parents, started, didn, class, coaching, life, don, study |
| 4 | 2763 | lpa, company, role, offer, work, job, companies, switch, salary, experience |
| 3 | 2758 | upsc, prelims, attempt, exam, preparation, job, mains, preparing, clear, don |
| 12 | 2704 | boards, marks, board, paper, maths, exam, english, hindi, pre, cbse |
| 5 | 2485 | branch, cse, college, nit, ece, core, branches, colleges, rank, placements |
| 8 | 2432 | sir, maths, hain, physics, lectures, backlog, chem, chemistry, aata, karna |
| 16 | 2322 | drop, jee, college, percentile, mains, jee mains, partial, attempt, ile, dropper |
| 7 | 2250 | internship, resume, job, experience, summer, internships, work, skills, engineering, gpa |
| 1 | 2119 | gate, mtech, exam, gate exam, job, cse, college, da, prepare, iit |
| 13 | 1960 | india, abroad, masters, university, germany, myqualifications, visa, job, uk, degree |
| 35 | 1880 | physics, chemistry, maths, chapters, class, chem, pyqs, questions, syllabus, backlog |
| 9 | 1815 | cat, varc, mocks, dilr, quants, mock, questions, qa, score, xat |
| 26 | 1781 | friends, life, college, don, talk, friend, group, lonely, social, study |
| 34 | 1533 | sleep, study, hours, anxiety, stress, don, exam, studying, feeling, brain |
| 6 | 1510 | bsdk, padhle, padhle bsdk, stress, suicidal, depressed, depression, anxiety, shear, stresses |
| 23 | 1469 | company, interview, hr, ghosted, round, offer, job, manager, companies, interviews |
| 15 | 1420 | modules, books, book, cengage, hcv, allen, jee, pdf, physics, pyqs |
| 17 | 1370 | results, cbse, result, release, date, tomorrow, today, digilocker, anxiety, news |
| 21 | 1350 | commerce, science, stream, pcm, maths, humanities, school, don, subjects, choose |
| 18 | 1306 | advanced, jee advanced, jee, adv, jee adv, mains, advance, jee advance, fucked, jee mains |
| 28 | 1288 | engineering, computer, electrical, major, ee, mechanical, degree, coding, cs, software |
| 25 | 1222 | coaching, online, offline, batch, allen, pw, offline coaching, jee, teachers, online coaching |
| 11 | 1133 | wasted, poll, view poll, view, barbaad, jee, canon, syllabus, canon event, event |
| 36 | 1107 | cgpa, sem, semester, college, backlogs, placements, backlog, tier, placement, exams |
| 31 | 1026 | neet, bsc, pcb, mbbs, career, myquals, don, doctor, parents, college |
| 33 | 995 | exam, card, admit card, admit, centre, form, center, invigilator, certificate, paper |
| 29 | 962 | ece, cse, vs, btech, btech cse, core, cs, eee, ai, specialization |
| 14 | 959 | hopium, copium, prodijee, percentile, marks, dedo, shift, lelo, ile, hopium dedo |
| 32 | 856 | improvement, compartment, improvement exam, exam, cbse, nios, pcm, board, marks, subjects |
| 24 | 836 | cooked, marks, chat, shift, maths, boards, jee, physics, exam, percentile |
| 38 | 794 | bitsat, test series, series, test, mathongo, jee, tests, mains, mathongo test, allen |
| 27 | 767 | partial, partial drop, drop, dropper, double, double dropper, droppers, double droppers, jee, double drop |
| 20 | 748 | tcs, nqt, tcs nqt, ninja, digital, joining, offer, prime, tcs ninja, tcs digital |
| 19 | 649 | p1, p2, interview, told, answered, m1, answer, m2, questions, sir |
| 30 | 608 | chud, guru, lag, rassi, barbaad, bc, dimag kharab, kharab, dimag, laude lag |
| 22 | 544 | daddy, alecc, alecc daddy, anup, anup daddy, sir, nishant, alakh, pw, sub |
| 40 | 498 | fiitjee, aits, teachers, centre, fiitjee aits, aakash, delhi, noida, test, closed |
