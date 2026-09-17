"""
Phrase Tokenizer — Preserve multi-word Hinglish expressions before word-level normalization.

Problem: BERTopic's CountVectorizer with ngram_range=(1,2) can catch 2-grams, but
3+ word cultural phrases like "chud gaye guru" split into incoherent fragments.
This module preserves multi-word phrases by replacing them with underscore-connected
tokens BEFORE the CountVectorizer tokenizes the text.

Design:
  1. Apply regex PHRASE_MAP to replace multi-word expressions with underscore tokens.
     Example: "chud gaye guru" → "chud_gaye_guru"
     Example: "life barbaad ho gaya" → "life_barbaad_ho_gaya"
  2. Pass underscore-connected tokens to CountVectorizer.
  3. Use a custom tokenizer that preserves underscore-connected tokens as single tokens
     while still splitting on whitespace for everything else.
  4. After BERTopic topic extraction, replace underscores with spaces for display.

Usage:
    from phrase_tokenizer import PhraseTokenizer, preserve_phrases, restore_phrases
    
    # In preprocessing pipeline
    cleaned = artifact_remover.clean(raw_text)
    phrase_preserved = preserve_phrases(cleaned)
    # Now pass to CountVectorizer with custom tokenizer

    # In BERTopic config
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        tokenizer=PhraseTokenizer().tokenize,
        preprocessor=None,  # No preprocessing — we already did it
        ngram_range=(1, 1),  # Single tokens include underscore phrases
    )
"""

import re
from typing import Dict, List, Tuple


# ============================================================================
# PHRASE MAP — 25+ multi-word Hinglish expressions
# ============================================================================

# Longest phrases first to avoid partial matches (greedy left-to-right replacement)
PHRASE_MAP: Dict[str, str] = {
    # ---- 3-word phrases (preserve as single tokens) ----
    "chud gaye guru": "chud_gaye_guru",
    "chud gaya guru": "chud_gaya_guru",
    "chud gayi guru": "chud_gayi_guru",
    "life barbaad ho gaya": "life_barbaad_ho_gaya",
    "life barbaad ho gayi": "life_barbaad_ho_gayi",
    "life barbaad ho gaye": "life_barbaad_ho_gaye",
    "career khatam ho gaya": "career_khatam_ho_gaya",
    "career khatam ho gayi": "career_khatam_ho_gayi",
    "career khatam ho gaye": "career_khatam_ho_gaye",
    "nahi ho raha hai": "nahi_ho_raha_hai",
    "nahi ho raha tha": "nahi_ho_raha_tha",
    "nahi ho rahi hai": "nahi_ho_rahi_hai",
    "nahi ho rahi thi": "nahi_ho_rahi_thi",
    "nahi ho rahe hai": "nahi_ho_rahe_hai",
    "nahi ho rahe the": "nahi_ho_rahe_the",
    "man nahi kar raha": "man_nahi_kar_raha",
    "man nahi kar rahi": "man_nahi_kar_rahi",
    "man nahi kar rahe": "man_nahi_kar_rahe",
    "samajh nahi aa raha": "samajh_nahi_aa_raha",
    "samajh nahi aa rahi": "samajh_nahi_aa_rahi",
    "samajh nahi aa rahe": "samajh_nahi_aa_rahe",
    "padh nahi ho raha": "padh_nahi_ho_raha",
    "padh nahi ho rahi": "padh_nahi_ho_rahi",
    "padh nahi ho rahe": "padh_nahi_ho_rahe",
    "thak gaya hoon main": "thak_gaya_hoon_main",
    "thak gayi hoon main": "thak_gayi_hoon_main",
    "thak gaye hain hum": "thak_gaye_hain_hum",
    "kya karu main": "kya_karu_main",
    "kya karoon main": "kya_karoon_main",
    "kya karu ab": "kya_karu_ab",
    "kya karoon ab": "kya_karoon_ab",
    "kya karu bhai": "kya_karu_bhai",
    "kya karoon bhai": "kya_karoon_bhai",
    "kya karu yaar": "kya_karu_yaar",
    "kya karoon yaar": "kya_karoon_yaar",
    "kya karu batao": "kya_karu_batao",
    "kya karoon batao": "kya_karoon_batao",
    "placement pressure bahut hai": "placement_pressure_bahut_hai",
    "placement pressure bahut zyada hai": "placement_pressure_bahut_zyada_hai",
    "parental pressure bahut hai": "parental_pressure_bahut_hai",
    "parental pressure bahut zyada hai": "parental_pressure_bahut_zyada_hai",
    "peer pressure bahut hai": "peer_pressure_bahut_hai",
    "peer pressure bahut zyada hai": "peer_pressure_bahut_zyada_hai",
    "academic stress bahut hai": "academic_stress_bahut_hai",
    "academic stress bahut zyada hai": "academic_stress_bahut_zyada_hai",
    "mental stress bahut hai": "mental_stress_bahut_hai",
    "mental stress bahut zyada hai": "mental_stress_bahut_zyada_hai",
    "rank inflation ho raha hai": "rank_inflation_ho_raha_hai",
    "rank inflation ho rahi hai": "rank_inflation_ho_rahi_hai",
    "nta ki mkc": "nta_ki_mkc",
    "nta ki maa ki": "nta_ki_maa_ki",
    "nta ki maa chud gayi": "nta_ki_maa_chud_gayi",
    
    # ---- 2-word phrases (critical for topic coherence) ----
    "chud gaye": "chud_gaye",
    "chud gaya": "chud_gaya",
    "chud gayi": "chud_gayi",
    "gaye guru": "gaye_guru",
    "gaya guru": "gaya_guru",
    "gayi guru": "gayi_guru",
    "life barbaad": "life_barbaad",
    "life barbad": "life_barbaad",
    "career khatam": "career_khatam",
    "career khtm": "career_khatam",
    "drop year": "drop_year",
    "partial drop": "partial_drop",
    "double drop": "double_drop",
    "placement pressure": "placement_pressure",
    "parental pressure": "parental_pressure",
    "peer pressure": "peer_pressure",
    "academic stress": "academic_stress",
    "mental stress": "mental_stress",
    "rank inflation": "rank_inflation",
    "nta ki": "nta_ki",
    "padhle bsdk": "padhle_bsdk",
    "padh lo bsdk": "padh_lo_bsdk",
    "padh le bsdk": "padh_le_bsdk",
    "canon event": "canon_event",
    "canon events": "canon_events",
    "joever state": "joever_state",
    "joevered": "joevered",
    "joever ho gaya": "joever_ho_gaya",
    "joever ho gayi": "joever_ho_gayi",
    "joever ho gaye": "joever_ho_gaye",
    "we are cooked": "we_are_cooked",
    "we're cooked": "we_are_cooked",
    "its over": "its_over",
    "it's over": "its_over",
    "over for me": "over_for_me",
    "over for us": "over_for_us",
    "over for you": "over_for_you",
    "over for him": "over_for_him",
    "over for her": "over_for_her",
    "hopium": "hopium",
    "copium": "copium",
    "cope": "cope",
    "coping": "coping",
    "coped": "coped",
    "seethe": "seethe",
    "seething": "seething",
    "no placement": "no_placement",
    "no job": "no_job",
    "no future": "no_future",
    "no scope": "no_scope",
    "tier 3": "tier_3",
    "tier-3": "tier_3",
    "tier3": "tier_3",
    "low tier": "low_tier",
    "backup college": "backup_college",
    "private college": "private_college",
    "government college": "government_college",
    "state college": "state_college",
    "deemed university": "deemed_university",
    "central university": "central_university",
    "iit bombay": "iit_bombay",
    "iit delhi": "iit_delhi",
    "iit madras": "iit_madras",
    "iit kanpur": "iit_kanpur",
    "iit kharagpur": "iit_kharagpur",
    "iit roorkee": "iit_roorkee",
    "iit guwahati": "iit_guwahati",
    "iit hyderabad": "iit_hyderabad",
    "iit indore": "iit_indore",
    "iit varanasi": "iit_varanasi",
    "iit patna": "iit_patna",
    "iit mandi": "iit_mandi",
    "iit ropar": "iit_ropar",
    "iit bhilai": "iit_bhilai",
    "iit jodhpur": "iit_jodhpur",
    "iit tirupati": "iit_tirupati",
    "iit palakkad": "iit_palakkad",
    "iit dhanbad": "iit_dhanbad",
    "iit bhubaneswar": "iit_bhubaneswar",
    "iit gandhinagar": "iit_gandhinagar",
    "nit trichy": "nit_trichy",
    "nit surathkal": "nit_surathkal",
    "nit warangal": "nit_warangal",
    "nit calicut": "nit_calicut",
    "nit rourkela": "nit_rourkela",
    "nit kurukshetra": "nit_kurukshetra",
    "nit durgapur": "nit_durgapur",
    "nit jaipur": "nit_jaipur",
    "nit allahabad": "nit_allahabad",
    "nit bhopal": "nit_bhopal",
    "nit nagpur": "nit_nagpur",
    "nit patna": "nit_patna",
    "nit jalandhar": "nit_jalandhar",
    "nit srinagar": "nit_srinagar",
    "nit hamirpur": "nit_hamirpur",
    "nit uttarakhand": "nit_uttarakhand",
    "nit goa": "nit_goa",
    "nit puducherry": "nit_puducherry",
    "nit arunachal": "nit_arunachal",
    "nit delhi": "nit_delhi",
    "nit manipur": "nit_manipur",
    "nit mizoram": "nit_mizoram",
    "nit meghalaya": "nit_meghalaya",
    "nit nagaland": "nit_nagaland",
    "nit sikkim": "nit_sikkim",
    "bits pilani": "bits_pilani",
    "bits goa": "bits_goa",
    "bits hyderabad": "bits_hyderabad",
    "vit vellore": "vit_vellore",
    "vit chennai": "vit_chennai",
    "vit bhopal": "vit_bhopal",
    "vit amaravati": "vit_amaravati",
    "srm university": "srm_university",
    "srm chennai": "srm_chennai",
    "srm kattankulathur": "srm_kattankulathur",
    "manipal university": "manipal_university",
    "manipal jaipur": "manipal_jaipur",
    "thapar university": "thapar_university",
    "lnmiit jaipur": "lnmiit_jaipur",
    "dtu delhi": "dtu_delhi",
    "nsut delhi": "nsut_delhi",
    "ipu delhi": "ipu_delhi",
    "jnu delhi": "jnu_delhi",
    "du delhi": "du_delhi",
    "bhu varanasi": "bhu_varanasi",
    "amu aligarh": "amu_aligarh",
    "jee mains": "jee_mains",
    "jee advanced": "jee_advanced",
    "jee main": "jee_main",
    "jee advance": "jee_advance",

    "dropping year": "dropping_year",
    "dropped year": "dropped_year",
    "partial dropper": "partial_dropper",
    "double dropper": "double_dropper",
    "no backlog": "no_backlog",
    "backlog clear": "backlog_clear",
    "backlog ho gaya": "backlog_ho_gaya",
    "backlog ho gayi": "backlog_ho_gayi",
    "backlog ho gaye": "backlog_ho_gaye",
    "supply clear": "supply_clear",
    "supply ho gaya": "supply_ho_gaya",
    "supply ho gayi": "supply_ho_gayi",
    "supply ho gaye": "supply_ho_gaye",
    "reappear ho gaya": "reappear_ho_gaya",
    "reappear ho gayi": "reappear_ho_gayi",
    "reappear ho gaye": "reappear_ho_gaye",
    "semester clear": "semester_clear",
    "semester fail": "semester_fail",
    "semester back": "semester_back",
    "semester ho gaya": "semester_ho_gaya",
    "semester ho gayi": "semester_ho_gayi",
    "semester ho gaye": "semester_ho_gaye",
    "cgpa low": "cgpa_low",
    "cgpa high": "cgpa_high",
    "cgpa decrease": "cgpa_decrease",
    "cgpa increase": "cgpa_increase",
    "cgpa improve": "cgpa_improve",
    "cgpa drop": "cgpa_drop",
    "low cgpa": "low_cgpa",
    "high cgpa": "high_cgpa",
    "good cgpa": "good_cgpa",
    "bad cgpa": "bad_cgpa",
    "package mil gaya": "package_mil_gaya",
    "package mil gayi": "package_mil_gayi",
    "package mil gaye": "package_mil_gaye",
    "package nahi mila": "package_nahi_mila",
    "package nahi mili": "package_nahi_mili",
    "package nahi mile": "package_nahi_mile",
    "offer mil gaya": "offer_mil_gaya",
    "offer mil gayi": "offer_mil_gayi",
    "offer mil gaye": "offer_mil_gaye",
    "offer nahi mila": "offer_nahi_mila",
    "offer nahi mili": "offer_nahi_mili",
    "offer nahi mile": "offer_nahi_mile",
    "placement mil gaya": "placement_mil_gaya",
    "placement mil gayi": "placement_mil_gayi",
    "placement mil gaye": "placement_mil_gaye",
    "placement nahi mila": "placement_nahi_mila",
    "placement nahi mili": "placement_nahi_mili",
    "placement nahi mile": "placement_nahi_mile",
    "internship mil gaya": "internship_mil_gaya",
    "internship mil gayi": "internship_mil_gayi",
    "internship mil gaye": "internship_mil_gaye",
    "internship nahi mila": "internship_nahi_mila",
    "internship nahi mili": "internship_nahi_mili",
    "internship nahi mile": "internship_nahi_mile",
    "lpa package": "lpa_package",
    "lpa mil gaya": "lpa_mil_gaya",
    "lpa mil gayi": "lpa_mil_gayi",
    "lpa mil gaye": "lpa_mil_gaye",
    "lpa nahi mila": "lpa_nahi_mila",
    "lpa nahi mili": "lpa_nahi_mili",
    "lpa nahi mile": "lpa_nahi_mile",
    "ghosted ho gaya": "ghosted_ho_gaya",
    "ghosted ho gayi": "ghosted_ho_gayi",
    "ghosted ho gaye": "ghosted_ho_gaye",
    "ghosted by company": "ghosted_by_company",
    "ghosted by recruiter": "ghosted_by_recruiter",
    "ghosted by hr": "ghosted_by_hr",
    "rejection ho gaya": "rejection_ho_gaya",
    "rejection ho gayi": "rejection_ho_gayi",
    "rejection ho gaye": "rejection_ho_gaye",
    "rejection aa gaya": "rejection_aa_gaya",
    "rejection aa gayi": "rejection_aa_gayi",
    "rejection aa gaye": "rejection_aa_gaye",
    "rejection mil gaya": "rejection_mil_gaya",
    "rejection mil gayi": "rejection_mil_gayi",
    "rejection mil gaye": "rejection_mil_gaye",
    "shortlist ho gaya": "shortlist_ho_gaya",
    "shortlist ho gayi": "shortlist_ho_gayi",
    "shortlist ho gaye": "shortlist_ho_gaye",

    "shortlist aa gayi": "shortlist_aa_gayi",
    "shortlist aa gaye": "shortlist_aa_gaye",
    "shortlist mil gaya": "shortlist_mil_gaya",
    "shortlist mil gayi": "shortlist_mil_gayi",
    "shortlist mil gaye": "shortlist_mil_gaye",
    "interview crack": "interview_crack",
    "interview clear": "interview_clear",
    "interview fail": "interview_fail",
    "interview pass": "interview_pass",
    "interview ho gaya": "interview_ho_gaya",
    "interview ho gayi": "interview_ho_gayi",
    "interview ho gaye": "interview_ho_gaye",
    "interview nahi hua": "interview_nahi_hua",
    "interview nahi hui": "interview_nahi_hui",
    "interview nahi hue": "interview_nahi_hue",
    "hr round": "hr_round",
    "technical round": "technical_round",
    "coding round": "coding_round",
    "aptitude round": "aptitude_round",
    "group discussion": "group_discussion",
    "gd round": "gd_round",
    "online assessment": "online_assessment",
    "oa clear": "oa_clear",
    "oa fail": "oa_fail",
    "oa ho gaya": "oa_ho_gaya",
    "oa ho gayi": "oa_ho_gayi",
    "oa ho gaye": "oa_ho_gaye",
    "oa nahi hua": "oa_nahi_hua",
    "oa nahi hui": "oa_nahi_hui",
    "oa nahi hue": "oa_nahi_hue",

    "dsa padh rahi": "dsa_padh_rahi",
    "dsa padh rahe": "dsa_padh_rahe",
    "dsa padhta": "dsa_padhta",
    "dsa padhti": "dsa_padhti",
    "dsa padhte": "dsa_padhte",
    "dsa nahi ho raha": "dsa_nahi_ho_raha",
    "dsa nahi ho rahi": "dsa_nahi_ho_rahi",
    "dsa nahi ho rahe": "dsa_nahi_ho_rahe",
    "leetcode solve": "leetcode_solve",
    "leetcode solved": "leetcode_solved",
    "leetcode solving": "leetcode_solving",
    "leetcode nahi ho raha": "leetcode_nahi_ho_raha",
    "leetcode nahi ho rahi": "leetcode_nahi_ho_rahi",
    "leetcode nahi ho rahe": "leetcode_nahi_ho_rahe",
    "striver sheet": "striver_sheet",
    "love babbar sheet": "love_babbar_sheet",
    "dp nahi samajh aaya": "dp_nahi_samajh_aaya",
    "dp nahi samajh aayi": "dp_nahi_samajh_aayi",
    "dp nahi samajh aaye": "dp_nahi_samajh_aaye",
    "graphs nahi samajh aaya": "graphs_nahi_samajh_aaya",
    "graphs nahi samajh aayi": "graphs_nahi_samajh_aayi",
    "graphs nahi samajh aaye": "graphs_nahi_samajh_aaye",
    "tree nahi samajh aaya": "tree_nahi_samajh_aaya",
    "tree nahi samajh aayi": "tree_nahi_samajh_aayi",
    "tree nahi samajh aaye": "tree_nahi_samajh_aaye",
    "dp samajh aaya": "dp_samajh_aaya",
    "dp samajh aayi": "dp_samajh_aayi",
    "dp samajh aaye": "dp_samajh_aaye",
    "graphs samajh aaya": "graphs_samajh_aaya",
    "graphs samajh aayi": "graphs_samajh_aayi",
    "graphs samajh aaye": "graphs_samajh_aaye",
    "tree samajh aaya": "tree_samajh_aaya",
    "tree samajh aayi": "tree_samajh_aayi",
    "tree samajh aaye": "tree_samajh_aaye",
}

# Sorted by length (descending) for greedy matching
_SORTED_PHRASES: List[Tuple[str, str]] = sorted(
    PHRASE_MAP.items(),
    key=lambda x: len(x[0]),
    reverse=True
)

# Compiled regex for each phrase (case-insensitive, word-boundary aware)
_PHRASE_RES: List[Tuple[re.Pattern, str]] = []
for phrase, replacement in _SORTED_PHRASES:
    escaped = re.escape(phrase)
    # Use word boundary-like negative lookaround to avoid consuming delimiters
    pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    _PHRASE_RES.append((pattern, replacement))


# ============================================================================
# Core functions
# ============================================================================

def preserve_phrases(text: str) -> str:
    """
    Replace multi-word phrases with underscore-connected tokens.
    
    Processes phrases longest-first to avoid partial matches.
    Case-insensitive matching. Preserves surrounding punctuation.
    
    Parameters
    ----------
    text : str
        Pre-cleaned text (after artifact_remover). Lowercase recommended.
    
    Returns
    -------
    str
        Text with multi-word phrases replaced by underscore tokens.
    """
    result = text
    # Apply longest phrases first (greedy left-to-right)
    for pattern, replacement in _PHRASE_RES:
        result = pattern.sub(replacement, result)
    return result


def restore_phrases(text: str) -> str:
    """
    Reverse the underscore tokenization for human-readable display.
    
    Replaces underscores with spaces in known phrase tokens.
    Unknown underscore tokens are left unchanged (could be compound words).
    
    Parameters
    ----------
    text : str
        Text containing underscore tokens (from BERTopic output or topic words).
    
    Returns
    -------
    str
        Text with underscores replaced by spaces where they were phrase tokens.
    """
    # Build reverse mapping
    reverse_map = {v: k for k, v in PHRASE_MAP.items()}
    
    # Split on whitespace, restore each token if in reverse map
    tokens = text.split()
    restored = []
    for token in tokens:
        if token in reverse_map:
            restored.append(reverse_map[token])
        else:
            # Check if it's a multi-word token that might have been created
            # (e.g., "chud_gaye_guru" which is not in the original map but is a 3-word)
            # Try to split on underscores and see if parts form a known phrase
            parts = token.split("_")
            if len(parts) > 1:
                # Try to reconstruct: join with spaces and see if it matches any phrase
                candidate = " ".join(parts)
                if candidate in PHRASE_MAP:
                    restored.append(candidate)  # Restore to space-separated phrase
                else:
                    restored.append(token.replace("_", " "))
            else:
                restored.append(token)
    
    return " ".join(restored)


# ============================================================================
# Custom Tokenizer for CountVectorizer
# ============================================================================

class PhraseTokenizer:
    """
    Custom tokenizer for sklearn CountVectorizer that preserves underscore-connected
    phrase tokens as single tokens while splitting everything else on whitespace.
    
    Usage:
        from sklearn.feature_extraction.text import CountVectorizer
        from phrase_tokenizer import PhraseTokenizer
        
        tokenizer = PhraseTokenizer()
        vectorizer = CountVectorizer(
            tokenizer=tokenizer.tokenize,
            preprocessor=None,  # Preprocessing already done in pipeline
            ngram_range=(1, 1),  # Single tokens include multi-word phrases
        )
    
    This eliminates the need for ngram_range=(1,2) or (1,3) because phrase tokens
    are already multi-word concepts encoded as single tokens.
    """
    
    def __init__(self, 
                 preserve_underscores: bool = True,
                 min_token_length: int = 2,
                 max_token_length: int = 100):
        self.preserve_underscores = preserve_underscores
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text, preserving underscore-connected phrases as single tokens.
        
        Parameters
        ----------
        text : str
            Preprocessed text (after artifact removal and phrase preservation).
        
        Returns
        -------
        List[str]
            List of tokens. Phrase tokens (containing underscores) are kept whole.
            Regular words are split on whitespace and punctuation.
        """
        tokens = []
        # Split on whitespace first
        raw_tokens = text.split()
        
        for token in raw_tokens:
            # Check if this is a phrase token (contains underscore)
            if self.preserve_underscores and "_" in token:
                # Strip punctuation from ends but keep internal underscores
                cleaned = token.strip(".,!?;:\"'()[]{}<>")
                if self.min_token_length <= len(cleaned) <= self.max_token_length:
                    tokens.append(cleaned)
            else:
                # Regular word: strip punctuation and filter by length
                cleaned = token.strip(".,!?;:\"'()[]{}<>")
                if self.min_token_length <= len(cleaned) <= self.max_token_length:
                    tokens.append(cleaned)
        
        return tokens
    
    def __call__(self, text: str) -> List[str]:
        """Make callable for sklearn compatibility."""
        return self.tokenize(text)


# ============================================================================
# Convenience functions for pipeline integration
# ============================================================================

def apply_phrase_preservation(texts: List[str]) -> List[str]:
    """Batch apply phrase preservation to a list of texts."""
    return [preserve_phrases(t) for t in texts]


def get_phrase_stats() -> dict:
    """Return statistics about the phrase map."""
    lengths = [len(p.split()) for p in PHRASE_MAP.keys()]
    return {
        "total_phrases": len(PHRASE_MAP),
        "two_word_phrases": sum(1 for l in lengths if l == 2),
        "three_word_phrases": sum(1 for l in lengths if l == 3),
        "four_word_phrases": sum(1 for l in lengths if l == 4),
        "five_plus_word_phrases": sum(1 for l in lengths if l >= 5),
        "longest_phrase": max(PHRASE_MAP.keys(), key=len) if PHRASE_MAP else None,
        "shortest_phrase": min(PHRASE_MAP.keys(), key=len) if PHRASE_MAP else None,
        "sample_phrases": list(PHRASE_MAP.items())[:10],
    }


# ============================================================================
# Testing / validation
# ============================================================================

if __name__ == "__main__":
    # Print stats
    stats = get_phrase_stats()
    print("=" * 60)
    print("PHRASE TOKENIZER STATS")
    print("=" * 60)
    for k, v in stats.items():
        if k != "sample_phrases":
            print(f"{k}: {v}")
    print(f"\nSample phrases:")
    for phrase, token in stats["sample_phrases"]:
        print(f"  '{phrase}' → '{token}'")
    
    # Test cases
    print("\n" + "=" * 60)
    print("TEST CASES")
    print("=" * 60)
    
    test_cases = [
        "chud gaye guru kya karu main",
        "life barbaad ho gaya placement pressure bahut hai",
        "nta ki mkc rank inflation ho raha hai",
        "joever state we are cooked",
        "padhle bsdk dsa nahi ho raha leetcode nahi ho raha",
        "iit bombay mein placement mil gaya 50 lpa package",
        "bits pilani se tier 3 college better hai kya",
        "drop year ke baad partial drop kar raha hoon",
        "samajh nahi aa raha hai kya karu batao",
        "man nahi kar raha hai padh nahi ho raha hai",
    ]
    
    for test in test_cases:
        print(f"\nBEFORE: {test}")
        preserved = preserve_phrases(test.lower())
        print(f"AFTER:  {preserved}")
        restored = restore_phrases(preserved)
        print(f"RESTORED: {restored}")
    
    # Test tokenizer
    print("\n" + "=" * 60)
    print("TOKENIZER TEST")
    print("=" * 60)
    tokenizer = PhraseTokenizer()
    sample = "chud_gaye_guru kya_karu_main placement_pressure_bahut_hai nta_ki_mkc"
    print(f"Input:  {sample}")
    print(f"Tokens: {tokenizer.tokenize(sample)}")
