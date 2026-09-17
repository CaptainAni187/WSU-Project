#!/usr/bin/env python3
"""Compute c_v and c_npmi coherence for a single comments model."""
import argparse, json, os, re, warnings, time
import pandas as pd
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')

CULTURAL_PHRASES = ["chud_gaye","padhle_bsdk","tier_3","chud_gaye_guru","nta_ki_mkc",
    "mar_jana_hai","jeene_ka_mann_nahi","sab_khatam","zindagi_barbaad","chud_gaye",
    "kuch_nahi_ho_sakta","padhle_bhai","padhle_bsdc","bhosdike","madarchod","bhenchod",
    "chutiya","bkl","mc","bc","jee_mains","jee_advanced","dropper_life","dropper",
    "coaching_adda","allen_kota","physics_wallah","pw_vip","arre_bhai","bhai_yaar",
    "mains_ki_tayari","board_exam","cbse_board","general_category","obc_category",
    "meme_distress","cooked","canon_event","joever"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    
    OUTPUT_DIR = "final_outputs/comments_benchmark"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_slug = args.model.lower().replace('-', '_').replace('/', '_')
    out_path = f"{OUTPUT_DIR}/{model_slug}_coherence.json"
    if os.path.exists(out_path):
        print(f"Already done: {out_path}")
        return
    
    path = f"final_outputs/comments_benchmark/{model_slug}"
    keywords_csv = pd.read_csv(f"{path}/topic_keywords.csv")
    
    df_text = pd.read_csv("final_outputs/comments_analysis/comments_preprocessed.csv")
    texts = df_text["clean_body"].fillna("").tolist()
    
    SAMPLE_SIZE = 20000
    if len(texts) > SAMPLE_SIZE:
        np.random.seed(42)
        indices = np.random.choice(len(texts), SAMPLE_SIZE, replace=False)
        sample_texts = [texts[i] for i in indices]
    else:
        sample_texts = texts
    
    tokenized = [re.findall(r'\b[a-z_0-9]+\b', t.lower()) for t in sample_texts]
    
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    
    topic_words = []
    for _, row in keywords_csv.iterrows():
        words = [w.strip() for w in str(row['keywords']).split(',')[:10] if w.strip()]
        valid = [w for w in words if w in dictionary.token2id]
        if valid:
            topic_words.append(valid)
    
    print(f"{args.model}: Dictionary size={len(dictionary)}, Topics={len(topic_words)}")
    
    t0 = time.time()
    cm_cv = CoherenceModel(topics=topic_words, texts=tokenized, dictionary=dictionary, coherence='c_v', processes=1, topn=10)
    coherence_cv = cm_cv.get_coherence()
    
    cm_npmi = CoherenceModel(topics=topic_words, texts=tokenized, dictionary=dictionary, coherence='c_npmi', processes=1, topn=10)
    coherence_npmi = cm_npmi.get_coherence()
    
    elapsed = time.time() - t0
    
    result = {
        'model': args.model,
        'c_v': round(float(coherence_cv), 4),
        'c_npmi': round(float(coherence_npmi), 4),
        'n_topics': len(topic_words),
        'dictionary_size': len(dictionary),
        'sample_size': len(sample_texts),
        'elapsed_seconds': round(elapsed, 1),
    }
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"{args.model}: c_v={result['c_v']:.4f}, c_npmi={result['c_npmi']:.4f} ({elapsed:.1f}s)")
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
