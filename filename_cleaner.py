"""
filename_cleaner.py — Converts a raw caption into a clean, filename-safe slug.

Pipeline:
  caption → lowercase → remove stopwords → remove punctuation
          → replace spaces with hyphens → truncate → done
"""

import re
import unicodedata

# Common English stopwords to strip (keeps technical terms like "aws", "vpc", etc.)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "their", "there", "here", "as", "into",
    "through", "during", "including", "until", "against", "among",
    "throughout", "despite", "towards", "upon", "concerning", "about",
    "shown", "showing", "view", "screen", "screenshot", "image", "photo",
    "picture", "display", "page", "window", "panel",
}

MAX_FILENAME_LENGTH = 60  # characters (excluding extension)
LOW_SIGNAL_PHRASES = {
    "a screenshot of",
    "screenshot of a",
    "image of a",
    "image of",
    "photo of",
    "picture of",
    "computer screen",
    "text on the screen",
    "text on screen",
    "close up of",
    "close up",
    "black screen",
    "white screen",
}
LOW_SIGNAL_WORDS = {
    "computer",
    "screen",
    "screenshot",
    "image",
    "photo",
    "picture",
    "display",
    "monitor",
    "desktop",
    "close",
    "view",
}
GENERIC_UI_WORDS = {
    "settings",
    "setting",
    "search",
    "dashboard",
    "dashboards",
    "menu",
    "home",
    "page",
    "tab",
    "tabs",
    "button",
    "dialog",
    "modal",
    "form",
    "table",
    "list",
    "results",
}
GENERIC_VENDOR_WORDS = {
    "aws",
    "azure",
    "gcp",
    "cloud",
    "google",
    "microsoft",
    "amazon",
}


def caption_to_filename(caption: str, extension: str = "") -> str:
    """
    Convert a human-readable caption into a safe, descriptive filename.

    Args:
        caption:   Raw caption string, e.g. "AWS VPC console showing route tables"
        extension: File extension including dot, e.g. ".png"

    Returns:
        Full filename string, e.g. "aws-vpc-route-tables.png"
    """
    slug = _slugify(caption)
    if not slug:
        slug = "unnamed-screenshot"
    return slug + extension


def is_low_signal_caption(caption: str) -> bool:
    """
    Detect captions that are too generic to safely use as filenames.
    """
    normalized = unicodedata.normalize("NFKD", caption)
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return True

    if normalized in {"unknown screenshot", "unknown-screenshot", "untitled"}:
        return True

    if any(phrase in normalized for phrase in LOW_SIGNAL_PHRASES):
        return True

    words = re.findall(r"[a-z0-9]+", normalized)
    if len(words) < 2:
        return True

    informative_words = [word for word in words if word not in STOPWORDS]
    if not informative_words:
        return True

    if all(word in LOW_SIGNAL_WORDS for word in informative_words):
        return True

    if _has_repeated_stems(informative_words):
        return True

    if len(informative_words) <= 3 and _is_generic_ui_phrase(informative_words):
        return True

    return False


def is_low_signal_filename(filename: str) -> bool:
    """
    Detect filenames that are still too generic after slug generation.
    """
    stem = filename.rsplit(".", 1)[0].strip("-")
    if not stem:
        return True

    words = [word for word in stem.split("-") if word]
    if len(words) < 2:
        return True

    if _has_repeated_stems(words):
        return True

    if _is_generic_ui_phrase(words):
        return True

    return False


def _slugify(text: str) -> str:
    # 1. Normalize unicode → ASCII-safe
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 2. Lowercase
    text = text.lower()

    # 3. Replace common separators with spaces
    text = re.sub(r"[-_/\\|]+", " ", text)

    # 4. Remove all punctuation except spaces and alphanumerics
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 5. Tokenize
    words = text.split()

    # 6. Remove stopwords (but keep if result would be empty)
    filtered = [w for w in words if w not in STOPWORDS]
    if not filtered:
        filtered = words  # fallback: keep all words

    # 7. Remove numeric tokens so filenames stay word-based only
    filtered = [w for w in filtered if not any(ch.isdigit() for ch in w)]

    # 8. Remove very short tokens (single chars that aren't meaningful)
    filtered = [w for w in filtered if len(w) > 1]

    # 9. Join with hyphens
    slug = "-".join(filtered)

    # 10. Truncate to max length at a word boundary
    if len(slug) > MAX_FILENAME_LENGTH:
        slug = slug[:MAX_FILENAME_LENGTH]
        # Cut at last hyphen to avoid partial words
        last_hyphen = slug.rfind("-")
        if last_hyphen > 20:  # only cut if we have a reasonable length left
            slug = slug[:last_hyphen]

    # 11. Strip leading/trailing hyphens
    slug = slug.strip("-")

    return slug


def _normalize_stem(word: str) -> str:
    if word.startswith("screenshot"):
        return "screenshot"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _has_repeated_stems(words: list[str]) -> bool:
    stems = [_normalize_stem(word) for word in words]
    return len(set(stems)) <= max(1, len(stems) // 2)


def _is_generic_ui_phrase(words: list[str]) -> bool:
    normalized = [_normalize_stem(word) for word in words]
    generic_count = sum(1 for word in normalized if word in GENERIC_UI_WORDS or word in LOW_SIGNAL_WORDS)
    vendor_count = sum(1 for word in normalized if word in GENERIC_VENDOR_WORDS)

    if generic_count == len(normalized):
        return True

    if len(normalized) <= 3 and generic_count >= 1 and generic_count + vendor_count == len(normalized):
        return True

    return False


def make_unique_filename(desired: str, existing: set) -> str:
    """
    If desired filename already exists in the existing set, append -02, -03, etc.

    Args:
        desired:  Desired filename, e.g. "aws-vpc-route-tables.png"
        existing: Set of already-used filenames (strings)

    Returns:
        A unique filename string.
    """
    if desired not in existing:
        return desired

    # Split off extension
    path_parts = desired.rsplit(".", 1)
    if len(path_parts) == 2:
        stem, ext = path_parts[0], "." + path_parts[1]
    else:
        stem, ext = path_parts[0], ""

    counter = 2
    while True:
        candidate = f"{stem}-{counter:02d}{ext}"
        if candidate not in existing:
            return candidate
        counter += 1
