"""
Artifact Remover — Reddit post artifact stripper.

Removes platform-specific artifacts that pollute BERTopic topics:
  - [removed], [deleted] markers
  - view poll, poll comments
  - image processing failed, processing failed
  - ampx200b (common Reddit encoding artifact)
  - URLs and markdown formatting
  - Reddit crosslinks, permalinks, subreddit references
  - Upvote/downvote metadata leakage

Expected input:  raw combined_text (title + body, concatenated by build_combined_text).
Expected output:  cleaned text with artifacts replaced by single space or removed.

Usage:
    from artifact_remover import ArtifactRemover
    remover = ArtifactRemover()
    cleaned = remover.clean(text)
"""

import re
import warnings
from typing import Optional, Union, List


class ArtifactRemover:
    """
    Configurable artifact stripper for Reddit social-media text.
    
    All regex patterns are compiled once at import time for performance.
    The removal order is carefully designed to avoid cascading false positives.
    """
    
    # ---- Hardcoded patterns that MUST be removed (not normalized) ----
    
    # Reddit-specific content markers
    _REMOVED_DELETED_PATTERNS = [
        r"\[removed\]",
        r"\[deleted\]",
        r"\[removed by.*\]",
        r"\[deleted by.*\]",
    ]
    
    # Platform metadata / UI artifacts
    _UI_ARTIFACT_PATTERNS = [
        r"view poll\b",
        r"poll comments\b",
        r"image processing failed\b",
        r"processing failed\b",
        r"ampx200b",
        r"x200b\b",              # Zero-width space character
        r"amp;\s*",          # HTML entity ampersand leakage
        r"&amp;\s*",         # Another common encoding artifact
        r"highlightedupdate\S*",  # LinkedIn image URL fragment prefix
        r"urn%3a\S*",            # URL-encoded LinkedIn URN fragments
        r"urn%3Ali%3A\S*",       # Full LinkedIn URN pattern
    ]
    
    # Image CDN / URL parameter artifacts (Reddit, LinkedIn image links)
    _IMAGE_CDN_PATTERNS = [
        r"format=png\b",
        r"format=pjpg\b",
        r"format=jpg\b",
        r"format=jpeg\b",
        r"format=gif\b",
        r"format=webp\b",
        r"auto=webp\b",
        r"auto=png\b",
        r"auto=jpg\b",
        r"auto=jpeg\b",
        r"auto=format\b",
        r"w=\d+",              # Width parameter
        r"h=\d+",              # Height parameter
        r"s=\b",               # Image size/scaling parameter prefix
        r"s=[a-f0-9]{32,}",    # Hash strings in image URLs (s=...)
        r"[a-f0-9]{40,}",      # Long hex/hash strings (40+ chars)
        r"preview\b",
        r"thumbnail\b",
        r"cdn\b",
        r"imgur\b",
        r"i\.redd\.it\b",
        r"preview\.redd\.it\b",
        r"external-preview\.redd\.it\b",
    ]
    
    # Reddit structural / crosslink patterns
    _REDDIT_LINK_PATTERNS = [
        r"/r/\w+",           # Subreddit references
        r"/u/\w+",           # User references
        r"reddit\.com/r/\w+",
        r"reddit\.com/u/\w+",
        r"permalink\b",
        r"crosspost\b",
        r"x-post\b",
        r"xpost\b",
    ]
    
    # Engagement metadata (these appear in some crawled datasets)
    _ENGAGEMENT_PATTERNS = [
        r"upvote[s]?\b",
        r"downvote[s]?\b",
        r"\d+\s*(upvotes?|downvotes?|points?)",
        r"score:\s*\d+",
        r"edited\b",
        r" award[s]?\b",
    ]
    
    # URL patterns (comprehensive)
    _URL_PATTERN = re.compile(
        r"https?://"                    # http:// or https://
        r"(?:[-\w.])+(?:[:\d]+)?"      # domain (optional port)
        r"(?:/[^\s]*)?"                # path
        r"|www\.\w+\.\w+(?:/[^\s]*)?"  # www URLs without protocol
        r"|[^\s]+\.com/[^\s]*"          # bare .com links
        r"|[^\s]+\.in/[^\s]*"           # bare .in links
        r"|[^\s]+\.org/[^\s]*"          # bare .org links
    )
    
    # Markdown formatting artifacts
    _MARKDOWN_PATTERNS = [
        r"!\[.*?\]\(.*?\)",   # Images ![alt](url)
        r"\[.*?\]\(.*?\)",     # Links [text](url)
        r"`{1,3}.*?`{1,3}",    # Inline code / code blocks
        r"\*{1,2}(.*?)\*{1,2}", # Bold/italic markers (capture inner text)
        r"_{1,2}(.*?)_{1,2}",   # Underscore emphasis
        r">{1,3}\s*",           # Blockquote markers
        r"#{1,6}\s*",           # Heading markers
        r"---+",                # Horizontal rules
        r"\|\s*",               # Table cell separators
        r"- - -",
    ]
    
    # HTML entity artifacts
    _HTML_ENTITY_PATTERNS = [
        r"&lt;", r"&gt;", r"&quot;", r"&apos;",
        r"&nbsp;", r"&ndash;", r"&mdash;",
    ]
    
    # ---- Configuration ----
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_markdown: bool = True,
        remove_reddit_refs: bool = True,
        remove_engagement_meta: bool = True,
        remove_image_cdn: bool = True,  # NEW
        aggressive_mode: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize artifact remover with configurable removal policies.
        
        Parameters
        ----------
        remove_urls : bool
            Strip all URLs from text.
        remove_markdown : bool
            Strip markdown formatting while preserving inner text where possible.
        remove_reddit_refs : bool
            Strip /r/ and /u/ references, permalink strings, crosspost markers.
        remove_engagement_meta : bool
            Strip upvote/downvote/score metadata.
        aggressive_mode : bool
            If True, also strips short sequences of non-ASCII characters
            that are likely encoding artifacts (e.g., \u200b, \ufeff).
        verbose : bool
            Print warnings about heavily-artifacted posts.
        """
        self.remove_urls = remove_urls
        self.remove_markdown = remove_markdown
        self.remove_reddit_refs = remove_reddit_refs
        self.remove_engagement_meta = remove_engagement_meta
        self.remove_image_cdn = remove_image_cdn  # NEW
        self.aggressive_mode = aggressive_mode
        self.verbose = verbose
        
        # Compile all regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns into single combined regexes for speed."""
        # Combine removed/deleted into one
        self._removed_re = re.compile(
            "|".join(self._REMOVED_DELETED_PATTERNS),
            re.IGNORECASE
        )
        
        # Combine UI artifacts
        self._ui_re = re.compile(
            "|".join(self._UI_ARTIFACT_PATTERNS),
            re.IGNORECASE
        )
        
        # Combine image CDN artifacts
        self._image_cdn_re = re.compile(
            "|".join(self._IMAGE_CDN_PATTERNS),
            re.IGNORECASE
        )
        
        # Combine reddit refs
        self._reddit_re = re.compile(
            "|".join(self._REDDIT_LINK_PATTERNS),
            re.IGNORECASE
        )
        
        # Combine engagement
        self._engagement_re = re.compile(
            "|".join(self._ENGAGEMENT_PATTERNS),
            re.IGNORECASE
        )
        
        # Markdown: compile individually because some need substitution not deletion
        self._markdown_res = [
            (re.compile(p, re.IGNORECASE), "delete") for p in self._MARKDOWN_PATTERNS
        ]
        # Special handling for bold/italic: keep inner text
        self._markdown_res[4] = (re.compile(r"\*{1,2}(.*?)\*{1,2}"), "keep_inner")
        self._markdown_res[5] = (re.compile(r"_{1,2}(.*?)_{1,2}"), "keep_inner")
        
        # HTML entities: replace with ASCII equivalents
        self._html_entity_re = re.compile(
            "|".join(self._HTML_ENTITY_PATTERNS)
        )
        self._html_entity_map = {
            "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'",
            "&nbsp;": " ", "&ndash;": "-", "&mdash;": "—",
        }
        
        # Aggressive: zero-width and invisible characters
        if self.aggressive_mode:
            self._invisible_re = re.compile(
                r"[\u200b\u200c\u200d\u2060\ufeff\u00ad\u200e\u200f]+"
            )
        
        # Detect posts that are >50% artifacts after removal
        self._artifact_density_re = re.compile(
            r"\[removed\]|\[deleted\]|view poll|image processing failed|ampx200b|"
            r"format=png|format=pjpg|auto=webp|x200b|highlightedupdate|urn%3a"
        )
    
    # ---- Core cleaning methods ----
    
    def clean(self, text: Union[str, None]) -> str:
        """
        Clean a single text string, removing all configured artifacts.
        
        Parameters
        ----------
        text : str or None
            Raw text to clean.
        
        Returns
        -------
        str
            Cleaned text. Empty string if input is None or whitespace-only.
        """
        if text is None or not isinstance(text, str):
            return ""
        
        # Work on a copy
        cleaned = text
        
        # Step 1: Remove [removed] / [deleted] — these are the most harmful
        cleaned = self._removed_re.sub(" ", cleaned)
        
        # Step 2: Remove UI artifacts (view poll, ampx200b, etc.)
        cleaned = self._ui_re.sub(" ", cleaned)
        
        # Step 2b: Remove image CDN artifacts (format=png, auto=webp, etc.)
        if self.remove_image_cdn:
            cleaned = self._image_cdn_re.sub(" ", cleaned)
        
        # Step 3: Remove URLs (before markdown so [text](url) is handled properly)
        if self.remove_urls:
            cleaned = self._URL_PATTERN.sub(" ", cleaned)
        
        # Step 4: Handle markdown
        if self.remove_markdown:
            cleaned = self._process_markdown(cleaned)
        
        # Step 5: Remove Reddit structural references
        if self.remove_reddit_refs:
            cleaned = self._reddit_re.sub(" ", cleaned)
        
        # Step 6: Remove engagement metadata
        if self.remove_engagement_meta:
            cleaned = self._engagement_re.sub(" ", cleaned)
        
        # Step 7: Decode HTML entities
        cleaned = self._html_entity_re.sub(
            lambda m: self._html_entity_map.get(m.group(0), m.group(0)),
            cleaned
        )
        
        # Step 8: Aggressive invisible character removal
        if self.aggressive_mode:
            cleaned = self._invisible_re.sub("", cleaned)
        
        # Step 9: Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)
        
        # Optional: warn about posts that were mostly artifacts
        if self.verbose and self._is_mostly_artifacts(text):
            warnings.warn(
                f"Post was mostly artifacts: original length {len(text)}, "
                f"cleaned length {len(cleaned)}, sample: {cleaned[:100]}..."
            )
        
        return cleaned
    
    def clean_series(self, texts: List[Union[str, None]]) -> List[str]:
        """Batch clean a list of texts."""
        return [self.clean(t) for t in texts]
    
    def clean_dataframe_column(self, df, column: str, new_column: str = "cleaned_text") -> object:
        """
        Clean a pandas DataFrame column in-place.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing text column.
        column : str
            Name of column to clean.
        new_column : str
            Name of output column. If same as column, overwrites in-place.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with cleaned column added (or overwritten).
        """
        import pandas as pd
        df[new_column] = df[column].apply(self.clean)
        
        # Report statistics
        n_empty = (df[new_column] == "").sum()
        n_original_empty = df[column].isna().sum()
        if self.verbose:
            print(f"Artifact removal: {n_empty} empty results "
                  f"({n_original_empty} were originally null/NaN)")
        
        return df
    
    # ---- Internal helpers ----
    
    def _process_markdown(self, text: str) -> str:
        """Process markdown formatting, keeping inner text for emphasis."""
        result = text
        for pattern, mode in self._markdown_res:
            if mode == "delete":
                result = pattern.sub(" ", result)
            elif mode == "keep_inner":
                # Keep the text between emphasis markers
                result = pattern.sub(r"\1", result)
        return result
    
    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters and strip ends."""
        # Replace tabs, newlines, multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _is_mostly_artifacts(self, original: str) -> bool:
        """Check if original text was predominantly artifact content."""
        if not original or len(original) < 20:
            return False
        artifact_count = len(self._artifact_density_re.findall(original))
        # If more than 3 artifact matches in a short text, flag it
        return artifact_count >= 3 and len(original) < 200


# ---- Convenience functions for one-off usage ----

def quick_clean(text: str) -> str:
    """One-shot clean with default settings."""
    return ArtifactRemover().clean(text)


def clean_dataframe(df, text_column: str = "combined_text", 
                    output_column: str = "cleaned_text",
                    **remover_kwargs) -> object:
    """Clean a dataframe column with a fresh ArtifactRemover."""
    remover = ArtifactRemover(**remover_kwargs)
    return remover.clean_dataframe_column(df, text_column, output_column)


# ---- Validation / testing helpers ----

def validate_artifact_removal(texts: List[str]) -> dict:
    """
    Run validation checks on a batch of texts, returning diagnostic metrics.
    
    Returns dict with:
        - n_posts_total: total posts checked
        - n_posts_with_artifacts: posts containing known artifacts
        - n_posts_now_empty: posts that became empty after cleaning
        - artifact_types_found: dict of {pattern_name: count}
        - sample_artifacts: list of (index, original, cleaned) tuples
    """
    remover = ArtifactRemover(verbose=False)
    
    results = {
        "n_posts_total": len(texts),
        "n_posts_with_artifacts": 0,
        "n_posts_now_empty": 0,
        "artifact_types_found": {},
        "sample_artifacts": [],
    }
    
    # Check each text
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        
        has_artifacts = False
        cleaned = remover.clean(text)
        
        # Check for specific artifact types in original
        if "[removed]" in text.lower():
            results["artifact_types_found"]["removed"] = \
                results["artifact_types_found"].get("removed", 0) + 1
            has_artifacts = True
        if "[deleted]" in text.lower():
            results["artifact_types_found"]["deleted"] = \
                results["artifact_types_found"].get("deleted", 0) + 1
            has_artifacts = True
        if "view poll" in text.lower():
            results["artifact_types_found"]["view_poll"] = \
                results["artifact_types_found"].get("view_poll", 0) + 1
            has_artifacts = True
        if "ampx200b" in text.lower():
            results["artifact_types_found"]["ampx200b"] = \
                results["artifact_types_found"].get("ampx200b", 0) + 1
            has_artifacts = True
        if "image processing failed" in text.lower():
            results["artifact_types_found"]["image_failed"] = \
                results["artifact_types_found"].get("image_failed", 0) + 1
            has_artifacts = True
        
        if has_artifacts:
            results["n_posts_with_artifacts"] += 1
            if len(results["sample_artifacts"]) < 5:
                results["sample_artifacts"].append((i, text[:200], cleaned[:200]))
        
        if cleaned == "":
            results["n_posts_now_empty"] += 1
    
    return results


if __name__ == "__main__":
    # Quick sanity test
    test_cases = [
        "[removed] [deleted] what do I do? view poll",
        "My life is ruined https://reddit.com/r/Indian_Academia/comments/abc123",
        "**Placement season** is here [link text](https://example.com) upvote please!",
        "ampx200b ampx200b ampx200b processing failed",
        "I got 99.5 percentile in JEE. Joever state. Padhle bsdk.",
    ]
    
    remover = ArtifactRemover(verbose=True)
    for test in test_cases:
        print("=" * 60)
        print(f"BEFORE: {test[:80]}...")
        cleaned = remover.clean(test)
        print(f"AFTER:  {cleaned[:80]}...")
        print()
