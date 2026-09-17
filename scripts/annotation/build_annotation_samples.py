#!/usr/bin/env python3
"""Build the stratified annotation samples (written to data/annotations/).

Human labels were then added to the `human_annotation` column; agreement is computed by
scripts/annotation/annotation_agreement.py."""
import pandas as pd, numpy as np, json, os, re, warnings
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
ANNOT_DIR = os.path.join(BASE, "data", "annotations")

print("=" * 70)
print("ANNOTATION DATASET MODIFICATION")
print("=" * 70)

# ================================================================
# 1. POSTS DISTRESS
# ================================================================
print("\n[1] Posts Distress — ~300 stratified samples...")
df_posts = pd.read_csv(os.path.join(BASE, "data", "processed", "posts_merged_final_preprocessed.csv"),
                       usecols=["id", "subreddit", "full_text", "lifecycle_stage", "is_distress", "distress_confidence"])

def conf_bucket(c):
    if c >= 0.8: return "high"
    elif c >= 0.5: return "medium"
    elif c <= 0.2: return "low"
    else: return "borderline"

df_posts["_bucket"] = df_posts["distress_confidence"].apply(conf_bucket)
np.random.seed(42)
sampled = []
for stage in df_posts["lifecycle_stage"].unique():
    for bucket in ["high", "medium", "low", "borderline"]:
        sub = df_posts[(df_posts["lifecycle_stage"] == stage) & (df_posts["_bucket"] == bucket)]
        if len(sub) > 0:
            sampled.append(sub.sample(min(15, len(sub)), random_state=42))

already = set()
for s in sampled: already.update(s["id"].tolist())
need = 300 - sum(len(s) for s in sampled)
if need > 0:
    rem = df_posts[~df_posts["id"].isin(already)]
    if len(rem) > 0:
        sampled.append(rem.sample(min(need, len(rem)), random_state=42))

posts = pd.concat(sampled).drop_duplicates("id").reset_index(drop=True)
posts = posts[["id", "subreddit", "full_text", "lifecycle_stage", "distress_confidence"]]
posts["assistant_annotation"] = posts["is_distress"].map({True: "distress", False: "non_distress"})
posts["human_annotation"] = ""
posts = posts[["id", "subreddit", "full_text", "lifecycle_stage", "distress_confidence", "assistant_annotation", "human_annotation"]]
posts.to_csv(os.path.join(ANNOT_DIR, "posts_distress_annotation.csv"), index=False)
print(f"  Saved: {len(posts)} rows")

# ================================================================
# 2. RETRIEVAL RELEVANCE (250 rows: 25 queries × 10 docs)
# ================================================================
print("\n[2] Retrieval Relevance — 25 queries × 10 docs...")

QUERIES = [
    ("jee mains advanced preparation stress", "exams"),
    ("neet biology physics chemistry preparation", "exams"),
    ("academic burnout can not study anymore", "burnout"),
    ("padhai nahi ho rahi hai exhausted", "burnout"),
    ("drop year waste of life regret", "drop_year"),
    ("double dropper partial dropper failure", "drop_year"),
    ("low cgpa backlogs academic probation", "cgpa"),
    ("semester failure cgpa khatam", "cgpa"),
    ("no job offers placement season failure", "placements"),
    ("tier 3 college no placement package", "placements"),
    ("no internship experience coding skills", "internships"),
    ("tech layoffs recession job loss", "layoffs"),
    ("fresher layoff company shut down", "layoffs"),
    ("leetcode dsa unable to solve easy", "coding"),
    ("competitive programming beginner stuck", "coding"),
    ("branch regret cse mech civil", "branch_regret"),
    ("hostel life mess food terrible", "hostel"),
    ("lonely no friends college isolated", "loneliness"),
    ("depression therapy medication need help", "depression"),
    ("anxiety panic attacks before exam", "anxiety"),
    ("want to end my life suicide", "suicidal_ideation"),
    ("mar jana hai sab khatam hopeless", "suicidal_ideation"),
    ("sab barbaad ho gaya kuch nahi bacha", "implicit_distress"),
    ("nta unfair result delay cheating", "nta"),
    ("coaching institute wasted money time", "coaching"),
]

embeddings = np.load(os.path.join(BASE, "data", "embeddings", "posts", "bge_m3_embeddings.npy")).astype(np.float32)
df_text = pd.read_csv(os.path.join(BASE, "data", "processed", "posts_merged_final_preprocessed.csv"),
                      usecols=["id", "subreddit", "full_text"])
texts = df_text["full_text"].fillna("").astype(str).tolist()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir="model_cache", trust_remote_code=True)
model = AutoModel.from_pretrained("BAAI/bge-m3", cache_dir="model_cache", trust_remote_code=True)
model.eval().to(DEVICE)

q_texts = [q[0] for q in QUERIES]
q_emb = np.zeros((len(q_texts), 1024), dtype=np.float32)
for i in range(0, len(q_texts), 16):
    batch = q_texts[i:i+16]
    inp = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.inference_mode():
        out = model(**inp)
        q_emb[i:i+len(batch)] = out.last_hidden_state[:, 0, :].cpu().numpy()

sims = cosine_similarity(q_emb, embeddings)

def kwset(text): return set(re.findall(r'[a-z_0-9]+', text.lower()))

rows = []
for qi, (query, cat) in enumerate(QUERIES):
    qw = kwset(query)
    top10 = np.argsort(sims[qi])[::-1][:10]
    for rank, idx in enumerate(top10, 1):
        dt = texts[idx]
        ol = len(qw & kwset(dt))
        rel = "relevant" if ol >= 3 else ("partially_relevant" if ol >= 1 else "not_relevant")
        rows.append({
            "query": query, "category": cat, "doc_rank": rank,
            "doc_id": df_text.iloc[idx]["id"], "subreddit": df_text.iloc[idx]["subreddit"],
            "similarity_score": round(float(sims[qi][idx]), 6),
            "text_preview": dt[:400], "keyword_overlap": ol,
            "assistant_annotation": rel, "human_annotation": "",
        })

rel_df = pd.DataFrame(rows)
rel_df.to_csv(os.path.join(ANNOT_DIR, "retrieval_relevance_annotation.csv"), index=False)
print(f"  Saved: {len(rel_df)} rows")

del model, tok
if DEVICE.type == "mps": torch.mps.empty_cache()

# ================================================================
# 3. TOPIC LABELS (~294 rows, add quality columns)
# ================================================================
print("\n[3] Topic Labels — adding quality columns...")

tk = pd.read_csv(os.path.join(BASE, "final_outputs", "posts_benchmark", "bge_m3", "topic_keywords.csv"))
topics = pd.read_csv(os.path.join(BASE, "final_outputs", "posts_benchmark", "bge_m3", "topics.csv"))

ARTIFACT_PATS = [r"format=png", r"format=pjpg", r"auto=webp", r"x200b", r"urn%3a", r"s=[a-f0-9]{32,}", r"[a-f0-9]{40,}", r"preview\.redd", r"redd\.it"]
STOPWORDS = {"hai","ke","se","aur","nahi","ka","bhi","tha","ho","ki","to","my","the","and","in","for","of","is","it","you","that","he","was","on","are","as","with","his","they","i","at","be","this","have","from","or","one","had","by","word","but","not","what","all","were","we","when","your","can","said","there","each","which","she","do","how","their","if","will","up","other","about","out","many","then","them","these","so","some","her","would","make","like","into","him","time","has","two","more","very","after","words","just","where","most","know","get","through","back","much","before","go","good","new","write","our","me","too","any","day","same","right","look","think","also","around","another","came","come","work","three","must","because","does","part","even","place","well","such","here","take","why","help","put","different","away","again","off","went","old","number","great","tell","men","say","small","every","found","still","between","name","should","home","big","give","air","line","set","own","under","read","last","never","us","left","end","along","while","might","next","sound","below","saw","something","thought","both","few","those","always","show","large","often","together","asked","house","dont","world","going","want","school","important","until","form","food","keep","children","feet","land","side","without","boy","once","animal","life","enough","took","sometimes","four","head","above","kind","began","almost","live","page","got","girl","sometimes","talk","soon","body","dog","family","direct","leave","song","measure","door","product","black","short","numeral","class","wind","question","happen","complete","ship","area","half","rock","order","fire","south","problem","piece","told","knew","pass","since","top","whole","king","space","heard","best","hour","better","during","hundred","am","remember","step","early","hold","west","ground","interest","reach","fast","verb","sing","listen","six","table","travel","less","morning","ten","simple","several","vowel","toward","war","lay","against","pattern","slow","center","love","person","money","serve","appear","road","map","rain","rule","govern","pull","cold","notice","voice","unit","power","town","fine","certain","fly","fall","lead","cry","dark","machine","note","wait","plan","figure","star","box","noun","field","rest","correct","able","pound","done","beauty","drive","stood","contain","front","teach","week","final","gave","green","oh","quick","develop","ocean","warm","free","minute","strong","special","mind","behind","clear","tail","produce","fact","street","inch","multiply","nothing","course","stay","wheel","full","force","blue","object","decide","surface","deep","moon","island","foot","system","busy","test","record","boat","common","gold","possible","plane","stead","dry","wonder","laugh","thousand","ago","ran","check","game","shape","equate","hot","miss","brought","heat","snow","tire","bring","yes","distant","fill","east","paint","language","among","grand","ball","yet","wave","drop","heart","am","present","heavy","dance","engine","position","arm","wide","sail","material","size","vary","settle","speak","weight","general","ice","matter","circle","pair","include","divide","syllable","felt","perhaps","pick","sudden","count","square","reason","length","represent","art","subject","region","energy","hunt","probable","bed","brother","egg","ride","cell","believe","fraction","forest","sit","race","window","store","summer","train","sleep","prove","lone","leg","exercise","wall","catch","mount","wish","sky","board","joy","winter","sat","written","wild","instrument","kept","glass","grass","cow","job","edge","sign","visit","past","soft","fun","bright","gas","weather","month","million","bear","finish","happy","hope","flower","clothe","strange","gone","jump","baby","eight","village","meet","root","buy","raise","solve","metal","whether","push","seven","paragraph","third","shall","held","hair","describe","cook","floor","either","result","burn","hill","safe","cat","century","consider","type","law","bit","coast","copy","phrase","silent","tall","sand","soil","roll","temperature","finger","industry","value","fight","lie","beat","excite","natural","view","sense","ear","else","quite","broke","case","middle","kill","son","lake","moment","scale","loud","spring","observe","child","straight","consonant","nation","dictionary","milk","speed","method","organ","pay","age","section","dress","cloud","surprise","quiet","stone","tiny","climb","cool","design","poor","lot","experiment","bottom","key","iron","single","stick","flat","twenty","skin","smile","crease","hole","trade","melody","trip","office","receive","row","mouth","exact","symbol","die","least","trouble","shout","except","wrote","seed","tone","join","suggest","clean","break","lady","yard","rise","bad","blow","oil","blood","touch","grew","cent","mix","team","wire","cost","lost","brown","wear","garden","equal","sent","choose","fell","fit","flow","fair","bank","collect","save","control","decimal","gentle","woman","captain","practice","separate","difficult","doctor","please","protect","noon","whose","locate","ring","character","insect","caught","period","indicate","radio","spoke","atom","human","history","effect","electric","expect","crop","modern","element","hit","student","corner","party","supply","bone","rail","imagine","provide","agree","thus","capital","wont","chair","danger","fruit","rich","thick","soldier","process","operate","guess","necessary","sharp","wing","create","neighbor","wash","bat","rather","crowd","corn","compare","poem","string","bell","depend","meat","rub","tube","famous","dollar","stream","fear","sight","thin","triangle","planet","hurry","chief","colony","clock","mine","tie","enter","major","fresh","search","send","yellow","gun","allow","print","dead","spot","desert","suit","current","lift","rose","continue","block","chart","hat","sell","success","company","subtract","event","particular","deal","swim","term","opposite","wife","shoe","shoulder","spread","arrange","camp","invent","cotton","born","determine","quart","nine","truck","noise","level","chance","gather","shop","stretch","throw","shine","property","column","molecule","select","wrong","gray","repeat","require","broad","prepare","salt","nose","plural","anger","claim","continent","oxygen","sugar","death","pretty","skill","women","season","solution","magnet","silver","thank","branch","match","suffix","especially","fig","afraid","huge","sister","steel","discuss","forward","similar","guide","experience","score","apple","bought","led","pitch","coat","mass","card","band","rope","slip","win","dream","evening","condition","feed","tool","total","basic","smell","valley","nor","double","seat","arrive","master","track","parent","shore","division","sheet","substance","favor","connect","post","spend","chord","fat","glad","original","share","station","dad","bread","charge","proper","bar","offer","segment","slave","duck","instant","market","degree","populate","chick","dear","enemy","reply","drink","occur","support","speech","nature","range","steam","motion","path","liquid","log","meant","quotient","teeth","shell","neck"}

def assess_quality(kw_str):
    words = [w.strip() for w in kw_str.split(",")]
    for w in words:
        for pat in ARTIFACT_PATS:
            if re.search(pat, w, re.I):
                return "artifact"
    non_stop = [w for w in words if w not in STOPWORDS]
    if len(non_stop) <= 2:
        return "mixed"
    return "coherent"

qmap = {row["topic_id"]: assess_quality(row["keywords"]) for _, row in tk.iterrows()}

t_rows = []
for _, row in tk.iterrows():
    tid = row["topic_id"]
    didx = topics[topics["topic"] == tid].index[:3]
    for idx in didx:
        if idx < len(df_posts):
            doc = df_posts.iloc[idx]
            t_rows.append({
                "topic_id": tid, "keywords": row["keywords"],
                "doc_id": doc["id"], "subreddit": doc["subreddit"],
                "lifecycle_stage": doc["lifecycle_stage"],
                "text_preview": str(doc["full_text"])[:300],
                "assistant_annotation": f"topic_{tid}", "human_annotation": "",
                "assistant_topic_quality": qmap.get(tid, "coherent"), "human_topic_quality": "",
            })

tl = pd.DataFrame(t_rows)
tl = tl.groupby("topic_id").head(3).reset_index(drop=True)
tl.to_csv(os.path.join(ANNOT_DIR, "topic_labels_annotation.csv"), index=False)
print(f"  Saved: {len(tl)} rows ({tl['topic_id'].nunique()} topics)")

# ================================================================
# 4. COMMENTS INTERACTION — ~300 balanced
# ================================================================
print("\n[4] Comments Interaction — ~300 balanced samples...")

df_comm = pd.read_csv(os.path.join(BASE, "final_outputs", "comments_analysis", "comments_taxonomy.csv"))
sampled = []
for cat, group in df_comm.groupby("primary_category"):
    sampled.append(group.sample(min(25, len(group)), random_state=42))
ci = pd.concat(sampled).reset_index(drop=True)
ci = ci[["comment_id", "body", "primary_category", "confidence"]].copy()
ci["assistant_annotation"] = ci["primary_category"]
ci["human_annotation"] = ""
ci = ci[["comment_id", "body", "confidence", "assistant_annotation", "human_annotation"]]
ci.to_csv(os.path.join(ANNOT_DIR, "comments_interaction_annotation.csv"), index=False)
print(f"  Saved: {len(ci)} rows ({ci['assistant_annotation'].nunique()} categories)")

# ================================================================
# 5. DELETE LIFECYCLE
# ================================================================
print("\n[5] Deleting lifecycle_annotation.csv...")
lp = os.path.join(ANNOT_DIR, "lifecycle_annotation.csv")
if os.path.exists(lp): os.remove(lp); print("  Deleted.")
else: print("  Already gone.")

# ================================================================
# 6. THEME VALIDATION — ~150 posts
# ================================================================
print("\n[6] Theme Validation — ~150 posts with discovered themes...")

DOMAIN_KW = {
    "JEE Preparation": ["jee", "mains", "advanced", "drop", "droppers", "tard", "coaching", "kota", "allen", "pw", "physics", "chemistry", "maths", "inorganic", "organic", "physical"],
    "NEET & Medical": ["neet", "biology", "medical", "mbbs", "doctor", "aiims", "nursing"],
    "Board Exams": ["boards", "cbse", "10th", "12th", "result", "results", "percentage", "marks"],
    "College & Branch": ["college", "branch", "cse", "ece", "mechanical", "civil", "specialization", "tier", "nit", "iit", "iiit", "vit", "btech"],
    "Placement & Jobs": ["placement", "lpa", "package", "salary", "offer", "company", "ctc", "role", "campus"],
    "Internship & Career": ["internship", "intern", "resume", "job", "career", "fresher", "experience", "hiring"],
    "IT & Coding": ["tcs", "nqt", "leetcode", "coding", "dsa", "programming", "developer", "software"],
    "Mental Health": ["depression", "anxiety", "therapy", "suicide", "mental", "health", "therapist", "psychiatrist", "medication", "panic", "stress"],
    "Academic Burnout": ["burnout", "exhausted", "tired", "sleep", "study", "routine", "padhai", "padhne", "motivation"],
    "Drop Year & Failure": ["drop", "year", "failure", "failed", "waste", "regret", "backlog", "backlogs", "cgpa"],
    "Layoffs & Economy": ["layoff", "layoffs", "recession", "fired", "unemployment", "economy", "market"],
    "Coaching & Exam Bodies": ["coaching", "nta", "unfair", "delay", "cheating", "result", "exam"],
    "Social & Loneliness": ["lonely", "friends", "isolated", "hostel", "mess", "social", "relationship"],
    "Meme & Slang": ["cooked", "meme", "joke", "funny", "lol", "shitpost", "tard", "removed", "deleted"],
    "General Hinglish": ["hai", "nahi", "bhai", "yaar", "hain", "tha", "ho", "ki"],
}

def infer_theme(kw_str):
    words = [w.strip() for w in kw_str.split(",")]
    scores = {t: sum(1 for w in words if w in kws) for t, kws in DOMAIN_KW.items()}
    best = max(scores, key=scores.get) if scores else "General"
    if scores.get(best, 0) == 0:
        non_stop = [w for w in words if w not in STOPWORDS and len(w) > 2]
        return f"{' '.join(non_stop[:2]).title()}" if non_stop else "General Discussion"
    return best

theme_map = {row["topic_id"]: infer_theme(row["keywords"]) for _, row in tk.iterrows()}

t_counts = topics["topic"].value_counts()
valid = [t for t in t_counts.index if t >= 0]
np.random.seed(42)

samp = []
for tid in valid:
    n = max(1, min(3, int(t_counts[tid] / 100)))
    didx = topics[topics["topic"] == tid].index
    if len(didx) > 0:
        chosen = np.random.choice(didx, min(n, len(didx)), replace=False)
        for idx in chosen:
            if idx < len(df_posts):
                doc = df_posts.iloc[idx]
                samp.append({
                    "post_id": doc["id"], "subreddit": doc["subreddit"],
                    "lifecycle_stage": doc["lifecycle_stage"],
                    "full_text": str(doc["full_text"])[:500],
                    "discovered_topic": f"topic_{tid}",
                    "discovered_theme": theme_map.get(tid, "General"),
                    "assistant_annotation": theme_map.get(tid, "General"),
                    "human_annotation": "",
                })

tv = pd.DataFrame(samp)
if len(tv) > 150:
    tv = tv.sample(150, random_state=42).reset_index(drop=True)
tv.to_csv(os.path.join(ANNOT_DIR, "theme_validation_annotation.csv"), index=False)
print(f"  Saved: {len(tv)} rows ({tv['discovered_theme'].nunique()} themes)")

# ================================================================
# 7. UPDATE GUIDELINES
# ================================================================
print("\n[7] Updating annotation_guidelines.md...")

guide = """# Annotation Guidelines

## Overview

This document provides guidelines for human annotation of 5 validation datasets. Each dataset addresses a specific aspect of the pipeline quality.

**Time estimate:** ~12-18 hours total across all 5 datasets (2-4 hours each).

---

## 1. Posts Distress Annotation (`posts_distress_annotation.csv`)

**Goal:** Validate the automated distress classification of posts.

**Dataset:** ~300 posts, stratified across all 5 lifecycle stages and 4 confidence buckets.

**Columns:** `id`, `subreddit`, `full_text`, `lifecycle_stage`, `distress_confidence`, `assistant_annotation`, `human_annotation`

**Labels:**
- `distress` — Clearly expresses academic distress
- `non_distress` — Neutral, supportive, informational, humorous
- `ambiguous` — Cannot determine; mixed signals

**Guidelines:**
- Focus on emotional tone, not just keywords
- Hinglish expressions of distress count ("sab khatam", "mar jana hai")
- Ignore `assistant_annotation` when deciding

**Expected agreement:** Kappa ≥ 0.80

---

## 2. Comments Interaction Annotation (`comments_interaction_annotation.csv`)

**Goal:** Validate the 12-category taxonomy of comment responses.

**Dataset:** ~300 comments, balanced across all 12 categories (~25 each).

**Labels (12):**
- Supportive: `support_empathy`, `advice_guidance`, `personal_experience`, `encouragement_motivation`, `information_resources`, `humor_meme_coping`
- Neutral: `neutral_discussion`
- Harmful: `dismissive_minimizing`, `toxic_abusive`, `blame_criticism`, `self_disclosure`, `crisis_escalation`

**Expected agreement:** Kappa ≥ 0.75

---

## 3. Retrieval Relevance Annotation (`retrieval_relevance_annotation.csv`)

**Goal:** Validate whether retrieved documents are actually relevant to the query.

**Dataset:** 250 rows (25 queries × 10 top-retrieved documents). Retrieved by BGE-m3 dense similarity.

**Labels:**
- `relevant` — Genuinely about the query topic
- `partially_relevant` — Related but not a strong match
- `not_relevant` — About something else entirely

**Expected distribution:** ~40% relevant, ~30% partially, ~30% not

**Evaluation:** Will compute Precision, Recall, MAP, MRR, nDCG using human labels.

---

## 4. Topic Labels Annotation (`topic_labels_annotation.csv`)

**Goal:** Validate BERTopic cluster coherence and assign meaningful names.

**Dataset:** ~294 rows (3 docs per topic × 98 topics).

**Columns:**
- `assistant_annotation` — Pre-filled topic_id
- `human_annotation` — **Manually assigned topic name** (e.g., "JEE Advanced Failure")
- `assistant_topic_quality` — Pre-filled: `coherent`, `mixed`, `artifact`
- `human_topic_quality` — **Your quality assessment**

**Quality Labels:**
- `coherent` — Clear, meaningful theme; all 3 docs fit
- `mixed` — Combines unrelated themes
- `artifact` — Contaminated by formatting/URLs/non-semantic tokens

**Expected:** 80%+ coherent, <10% artifact

---

## 5. Theme Validation Annotation (`theme_validation_annotation.csv`)

**Goal:** Validate whether automatically discovered BERTopic themes accurately describe posts.

**Dataset:** ~150 posts, sampled from BGE-m3 topic assignments with inferred theme names.

**Columns:** `post_id`, `subreddit`, `lifecycle_stage`, `full_text`, `discovered_topic`, `discovered_theme`, `assistant_annotation`, `human_annotation`

**Labels:**
- `correct` — Theme accurately describes the post
- `partially_correct` — Related but not the best fit
- `incorrect` — Theme does not describe the post at all

**Expected:** ≥75% correct

**Evaluation:** Reports percentage of correctly/partially/incorrectly themed posts.

---

## General Best Practices

1. Work in batches of 50-100 items
2. Take breaks every 30 minutes
3. When uncertain, mark `ambiguous` or `partially_relevant`
4. Don't look at assistant_annotation before deciding
5. Track time per dataset

*Generated: 2026-07-03*
"""

with open(os.path.join(ANNOT_DIR, "annotation_guidelines.md"), "w") as f:
    f.write(guide)
print("  Saved: annotation_guidelines.md")

# ================================================================
print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
for f in sorted(os.listdir(ANNOT_DIR)):
    fp = os.path.join(ANNOT_DIR, f)
    if os.path.isfile(fp):
        print(f"  {f:45s} | {os.path.getsize(fp):>8,} bytes")
