"""
WebShop Harness — H0/H1/H2/H3/H4/H5

H0  Task Parser        — one-time requirement extraction per episode
H1  Page State Tracker — page type detection + shopping state bookkeeping
H2  Action Gate        — validity check + fuzzy matching + repeat-click block
H3  Tool Description   — static shopping strategy hint embedded in tool descriptions
H4  Shopping Monitor   — search loop / price-over-budget / product-stall detection
                         + step-budget warn/force (forced Buy Now)
                         + product-category sanity check
H5  Goal-directed Hint — page-type-aware per-step guidance (attribute checklist)

Generalization notes:
  - All page-type detection uses environment API signals (has_search_bar, observation
    structure), which are identical across train/test splits.
  - Attribute checklist in H5 is derived purely from the task instruction + current
    page clickables — no training statistics are used.
  - H4 budget force-buy only triggers when "buy now" is already in clickables
    (admissible guard), so it never issues an invalid action.
"""

import copy
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Page type constants
# ─────────────────────────────────────────────────────────────────────────────

PAGE_HOME           = "HOME"
PAGE_SEARCH_RESULTS = "SEARCH_RESULTS"
PAGE_PRODUCT_DETAIL = "PRODUCT_DETAIL"
PAGE_UNKNOWN        = "UNKNOWN"

# Clickable values that are navigation controls, not attribute options
_NAV_CLICKABLES = {
    "back to search", "next >", "< prev", "description",
    "features", "reviews", "buy now", "search",
}

# Size tokens (commonsense, not training statistics)
_SIZE_TOKENS = {
    "xs", "s", "m", "l", "xl", "xxl", "xxxl",
    "x-small", "small", "medium", "large", "x-large", "xx-large", "xxx-large",
    "xsmall", "xlarge", "xxlarge", "xxxlarge",
    "2x-large", "3x-large", "4x-large",
    "one size", "one-size",
    # Bed/furniture sizes
    "twin", "full", "queen", "king",
    # Clothing sizes (even numbers)
    "0", "2", "4", "6", "8", "10", "12", "14", "16",
    # Shoe sizes (whole and half)
    "5", "5.5", "6", "6.5", "7", "7.5", "8", "8.5", "9", "9.5",
    "10", "10.5", "11", "11.5", "12", "12.5", "13", "13.5", "14",
}

# Measurement/spec pattern: matches option values like "8 ounce", "6.6 feet",
# "18 inch", "32gb", "28 in x 63 in", "3.3 feet", "1 pound (pack of 1)" etc.
# Also matches hyphenated forms like "13-ounce", "2-pack", "6-inch".
# Used to classify product-page "compound" or measurement options.
_MEASUREMENT_RE = re.compile(
    r"(\d[\d.,]*)\s*-?\s*"
    r"(oz|fl\.?\s*oz|ounce|lb|lbs|pound|kg|gram|g\b|"
    r"inch|inches|in\b|ft\b|feet|foot|cm|mm|meter|"
    r"gb|tb|mb|ghz|mhz|"
    r"ml|liter|litre|gallon|qt|quart|"
    r"pack|count|ct\.?|piece|pcs|pair|set)",
    re.IGNORECASE,
)

# Word-form numbers that appear in size contexts (e.g. "size five" → "size 5")
_WORD_NUMS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
}

# Core color tokens — always used for color extraction from task instructions.
# These words are unambiguously colors in all contexts.
_CORE_COLOR_TOKENS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "grey", "gray", "silver", "gold", "navy", "beige",
    "tan", "maroon", "ivory", "cream", "charcoal", "teal", "turquoise",
    "coral", "lavender", "magenta", "cyan", "lime", "indigo", "violet",
    "rose", "khaki", "olive", "burgundy", "aqua", "fuchsia",
    # Extended fashion colors (unambiguous as color even standalone)
    "bronze", "copper", "rust", "sage", "mint", "peach", "plum", "taupe",
    "wine", "blush", "nude", "emerald", "cobalt", "scarlet", "crimson",
    "mustard", "amber", "champagne", "mahogany", "onyx", "slate", "smoke",
    "multicolor", "multicolored", "multi-color", "multi-colored",
    # Additional unambiguous fashion/product colors seen in WebShop tasks
    "peony", "periwinkle", "lilac", "mauve", "sienna", "ecru", "rainbow",
}

# Ambiguous color words — only used when appearing as "X colored", "X color",
# or "X shade" in the instruction. Standalone, they more often name flavors,
# materials, or product types (e.g. "chocolate extract", "natural ingredients",
# "honey roasted", "camel leather", "espresso machine").
_AMBIGUOUS_COLOR_TOKENS = {
    "honey", "chocolate", "caramel", "mocha", "espresso",
    "camel", "sand", "denim", "natural", "walnut", "carbon", "ash",
    # Ambiguous: can be color names when followed by "color/colored/shade"
    "sunstone", "bisque", "flax", "stone", "moss", "pine", "fern",
    "blueberry", "watermelon", "cloud", "snow",
}

# Combined for use in product-page option classification (clickables).
# On the page, options like "chocolate" are unambiguously color/style choices.
_COLOR_TOKENS = _CORE_COLOR_TOKENS | _AMBIGUOUS_COLOR_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WebShopHarnessConfig:
    enabled: bool = False
    h2_enabled: bool = True
    h3_enabled: bool = True
    h4_enabled: bool = True
    h5_enabled: bool = True

    # H2
    h2_click_similarity_threshold: float = 0.80  # higher than ALFWorld to avoid ASIN mis-match
    h2_repeat_click_block_after: int = 2          # block if same click N times consecutively

    # H4
    h4_search_loop_threshold: int = 4     # back_to_search_count triggers recovery
    h4_duplicate_search_threshold: int = 2  # same query repeated N times triggers recovery
    h4_product_stall_turns: int = 3        # turns on same product page without progress
    h4_hint_max_words: int = 60            # word cap for H4-E per-step hints
    h4_warn_threshold: int = 5   # H4-D: remaining steps < N on SEARCH_RESULTS → urgency hint
    h4_force_threshold: int = 3  # H4-D: remaining steps < N with buy now available → force

    # Price
    price_tolerance: float = 0.05  # 5% over budget still accepted (rounding)

    # H5 skill retrieval
    h5_top_k: int = 2              # max skills injected at cold-start
    h5_cold_start_max_words: int = 50  # word cap per skill text
    h5_score_threshold: float = 0.0  # min BM25 score; 0 = no threshold


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval helpers (used by H5 cold-start skill matching)
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _skill_doc_tokens(skill: Dict[str, Any]) -> List[str]:
    parts = [skill.get("text", "")] + list(skill.get("keywords", []))
    return _bm25_tokenize(" ".join(parts))


def _bm25_scores(
    query_tokens: List[str],
    docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    if not docs:
        return []
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / n_docs if n_docs else 1
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    scores: List[float] = []
    unique_q = list(dict.fromkeys(query_tokens))
    for doc in docs:
        dl = len(doc)
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in unique_q:
            if t not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            f = tf[t]
            denom = f + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * f * (k1 + 1.0) / denom
        scores.append(score)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# WebShop task type detection (for skill filtering)
# ─────────────────────────────────────────────────────────────────────────────

# Task types — broad product categories that determine which skills apply.
# Detected from the instruction text (user's first message).
TASK_TYPE_APPAREL   = "apparel"      # clothing, shoes, accessories
TASK_TYPE_BEAUTY    = "beauty"       # skincare, cosmetics, hair products
TASK_TYPE_FOOD      = "food"         # food, drinks, supplements
TASK_TYPE_TECH      = "tech"         # electronics, gadgets, cables
TASK_TYPE_HOME      = "home"         # furniture, decor, bedding, rugs
TASK_TYPE_GENERAL   = "general"      # catch-all


def _detect_task_type(instruction: str) -> str:
    """Classify WebShop task into a broad product category from instruction text."""
    text = instruction.lower()
    # Apparel
    if re.search(
        r'\b(shirts?|dress(?:es)?|pants|jeans|jackets?|hoodies?|sweaters?|coats?|blous(?:es?)|shorts|skirts?|'
        r'sneakers?|shoes?|boots?|sandals?|heels?|slippers?|socks?|leggings?|cardigans?|sweatshirts?|'
        r'vests?|polos?|tank\s+tops?|bikinis?|bras?|underwear|boxers?|pajamas?|robes?|scarv(?:es)?|hats?|caps?|'
        r'gloves?|belts?|ties?|watch\s+bands?|wigs?|bottoms?|trunks?|briefs?)\b',
        text,
    ):
        return TASK_TYPE_APPAREL
    # Beauty
    if re.search(
        r'\b(shampoo|conditioner|lotion|cream|serum|moisturizer|sunscreen|lipstick|'
        r'mascara|foundation|concealer|eyeshadow|nail\s+polish|perfume|cologne|'
        r'cleanser|toner|scrub|mask|body\s+wash|soap|deodorant|toothpaste|'
        r'hair\s+dye|hair\s+color|lash|eyelash|brow)\b',
        text,
    ):
        return TASK_TYPE_BEAUTY
    # Food
    if re.search(
        r'\b(chocolate|cookie|candy|gum|snack|coffee|tea\b|protein|vitamin|'
        r'supplement|syrup|sauce|spice|seasoning|butter|oil|honey|sugar|flour|'
        r'cereal|granola|bar|drink|juice|water|soda|wine|beer|nut|pecan|almond)\b',
        text,
    ):
        return TASK_TYPE_FOOD
    # Tech
    if re.search(
        r'\b(phone|tablet|laptop|computer|keyboard|mouse|monitor|speaker|headphone|'
        r'earphone|earbud|charger|cable|usb|hdmi|adapter|case|cover|screen\s+protector|'
        r'camera|projector|printer|router|smart\s*watch|fitness\s+tracker|'
        r'battery|power\s+bank|led|bulb|light)\b',
        text,
    ):
        return TASK_TYPE_TECH
    # Home
    if re.search(
        r'\b(rugs?|curtains?|pillows?|blankets?|towels?|sheets?|bedding|mattress(?:es)?|furniture|chairs?|'
        r'tables?|desks?|shelv(?:es)?|cabinets?|lamps?|candles?|vases?|frames?|mirrors?|clocks?|'
        r'organizers?|baskets?|storage|kitchen|bathroom|garden)\b',
        text,
    ):
        return TASK_TYPE_HOME
    return TASK_TYPE_GENERAL


# ─────────────────────────────────────────────────────────────────────────────
# Skill definitions (used by H5 cold-start)
#
# Each skill has:
#   id         — unique identifier
#   task_types — product categories this skill applies to (hard filter)
#   keywords   — BM25 ranking terms (combined with text for scoring)
#   text       — the actual hint injected into agent context
# ─────────────────────────────────────────────────────────────────────────────

_ALL_TASK_TYPES = [
    TASK_TYPE_APPAREL, TASK_TYPE_BEAUTY, TASK_TYPE_FOOD,
    TASK_TYPE_TECH, TASK_TYPE_HOME, TASK_TYPE_GENERAL,
]

WEBSHOP_SKILLS: List[Dict[str, Any]] = [
    # ── Tool-contract skills: non-obvious interface constraints ───────────────
    # These address behaviors the model cannot infer from tool descriptions or
    # real-time hints (H3/H4). Strategy skills that duplicate H3 (tool desc)
    # or H4 (step guidance / attribute checklist) have been removed.
    {
        "id": "price_slightly_over",
        "task_types": _ALL_TASK_TYPES,
        "keywords": [
            "price", "budget", "over", "above", "exceed",
            "expensive", "cost", "afford", "close", "lower", "dollars",
        ],
        "text": (
            "If a product is only slightly above budget (about 5%), buy it. "
            "A near match is better than no purchase. Do not spend many turns hunting "
            "for cheaper options that may not exist. Skip only clearly overpriced items "
            "(for example above 20%)."
        ),
    },
    {
        "id": "compound_color_matching",
        "task_types": [TASK_TYPE_APPAREL, TASK_TYPE_HOME, TASK_TYPE_BEAUTY],
        "keywords": [
            "light", "dark", "sky", "navy", "rose", "baby",
            "royal", "hot", "deep", "bright", "pale", "neon",
            "fuchsia", "dusty", "salmon", "pearl", "champagne",
            "wine", "maroon", "turquoise", "teal", "coral", "burgundy",
            "charcoal", "aqua", "blue", "green", "pink", "grey", "gray", "gold",
        ],
        "text": (
            "For compound colors like 'light blue' or 'fuchsia pink', first look for "
            "an exact compound option. If only a partial option exists (for example "
            "'blue'), choose the closest partial match instead of skipping color selection."
        ),
    },
    {
        "id": "size_with_fit_modifier",
        "task_types": [TASK_TYPE_APPAREL],
        "keywords": [
            "petite", "tall", "plus", "regular", "short",
            "size", "fit", "length",
        ],
        "text": (
            "For sizes with fit modifiers (like 'petite small' or 'tall large'), "
            "select the combined option (for example 'small petite'), not only "
            "the base size."
        ),
    },
    {
        "id": "spec_measurement_matching",
        "task_types": [TASK_TYPE_TECH, TASK_TYPE_HOME, TASK_TYPE_BEAUTY, TASK_TYPE_FOOD],
        "keywords": [
            "ounce", "inch", "feet", "pound", "gram", "pack",
            "count", "piece", "gb", "tb", "ml", "liter",
        ],
        "text": (
            "When the task specifies measurements (like '8 ounce', '6.6 feet', "
            "or '32 gb'), select that value on the product page before buying. "
            "It may appear under 'size' or another option group."
        ),
    },
    {
        "id": "food_variant_selection",
        "task_types": [TASK_TYPE_FOOD],
        "keywords": [
            "flavor", "variant", "type", "roasted", "chocolate",
            "honey", "sugar", "spice", "option", "pack",
        ],
        "text": (
            "Food items often require explicit flavor or variant selection. "
            "Match the task wording as closely as possible (for example "
            "'honey roasted' or 'dark chocolate'). Treat them as clickable options, "
            "not just description text."
        ),
    },
    {
        "id": "pack_count_selection",
        "task_types": [TASK_TYPE_FOOD, TASK_TYPE_BEAUTY, TASK_TYPE_GENERAL],
        "keywords": [
            "pack", "count", "set", "piece", "dozen",
            "pair", "bundle", "quantity", "ounce", "oz",
            "assortment",
        ],
        "text": (
            "Food and beauty products often have pack-count options such as "
            "'pack of 1' or 'pack of 3'. Select the option that matches the required "
            "quantity and size. These are real attributes, not labels."
        ),
    },
    {
        "id": "dimension_size_selection",
        "task_types": [TASK_TYPE_HOME, TASK_TYPE_GENERAL],
        "keywords": [
            "feet", "foot", "ft", "inch", "inches",
            "rug", "curtain", "poster", "frame", "mat",
            "area", "window", "panel",
        ],
        "text": (
            "For rugs, curtains, and posters, pick the dimension option that matches "
            "the task (for example '3x5 feet' or '16 x 24 in'). "
            "If the exact size is unavailable, choose the closest one."
        ),
    },
    {
        "id": "coded_option_matching",
        "task_types": [TASK_TYPE_APPAREL, TASK_TYPE_HOME, TASK_TYPE_TECH],
        "keywords": [
            "#", "01", "02", "06", "a-", "v-",
            "hoodie", "shirt", "case", "cover", "phone",
            "sweatshirt", "pullover",
        ],
        "text": (
            "Some options include code prefixes like '01# black' or 'v366-pink'. "
            "Ignore the code and match the color or style text after it. "
            "If the task asks for black, selecting '01# black' is correct."
        ),
    },
    {
        "id": "closest_available_option",
        "task_types": _ALL_TASK_TYPES,
        "keywords": [
            "closest", "nearest", "similar", "approximate",
            "unavailable", "alternative", "substitute",
            "extension", "extender", "dreadlock", "clip",
        ],
        "text": (
            "If the exact attribute value is not available, choose the closest match "
            "instead of leaving the attribute unselected. A good partial match "
            "(for example 'pink' for 'fuchsia pink') is better than no selection."
        ),
    },
]


def retrieve_webshop_skills(
    task_type: str,
    query: str,
    top_k: int = 2,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Two-layer skill retrieval for WebShop:
      1. Hard filter — keep only skills whose task_types include task_type.
      2. BM25 ranking — rank filtered candidates by relevance to query.
      3. Score threshold — drop skills below score_threshold (0 = no threshold).
    Returns up to top_k results.
    """
    candidates = [s for s in WEBSHOP_SKILLS if task_type in s.get("task_types", [])]
    if not candidates:
        return []
    if len(candidates) <= top_k and score_threshold <= 0.0:
        return candidates
    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return candidates[:top_k]
    docs = [_skill_doc_tokens(s) for s in candidates]
    scores = _bm25_scores(query_tokens, docs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    if score_threshold > 0.0:
        ranked = [(sc, c) for sc, c in ranked if sc >= score_threshold]
    return [c for _, c in ranked[:top_k]]


def _truncate_to_word_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# H0 — Task Requirements Parser
# ─────────────────────────────────────────────────────────────────────────────

# Stopwords excluded from instruction keyword extraction for option matching
_TASK_STOPWORDS = {
    "price", "lower", "than", "dollars", "looking", "would", "like", "need",
    "want", "find", "some", "that", "with", "from", "this", "have", "which",
    "made", "come", "easy", "high", "good", "best", "very", "also", "more",
    "into", "each", "most", "only", "both", "does", "make", "your", "their",
    "should", "here", "look", "please", "order", "under", "just", "help",
    "must", "will", "been", "they", "them", "these", "those", "such", "size",
    "color", "price", "item", "product", "brand", "type", "style", "model",
    "piece", "pair", "pack", "pack", "inch", "feet", "ounce", "pound",
    "quality", "great", "perfect", "ideal", "nice", "right", "well",
    # Task-instruction meta-verbs: about the shopping act, not the product
    "interested", "buying", "wanting", "ordering", "needing", "choosing",
    "picking", "searching", "getting", "looking", "shopping", "purchase",
    "trying", "going", "using", "asking", "seeking",
}


@dataclass
class WebShopTaskRequirements:
    item_keywords: str                  # core item noun phrase
    color: Optional[str]                # e.g. "pink"
    color_alt: Optional[str]            # alternative color in "X or Y" expressions (e.g. "ivory" in "lavender or ivory")
    size: Optional[str]                 # e.g. "large", "xx-large"
    material: Optional[str]             # e.g. "cotton", "leather"
    price_max: Optional[float]          # numeric upper bound
    quantity: Optional[str]             # e.g. "1 dozen", "3x", "pack of 2"
    style_keywords: List[str]           # remaining descriptors (e.g. ["dust proof", "wireless"])
    specs: List[str]                    # measurement/tech specs (e.g. ["8 ounce", "32gb", "6.6 feet"])
    task_keywords: List[str]            # significant words from instruction for option matching
    task_type: str                      # product category for skill filtering
    raw_instruction: str


def parse_task_requirements(instruction: str) -> WebShopTaskRequirements:
    """H0: Extract structured shopping requirements from the task instruction."""
    # Strip WebShop observation formatting artifacts (e.g. "[SEP]", "WebShop")
    instruction = re.sub(r'\[SEP\]', ' ', instruction, flags=re.IGNORECASE)
    instruction = re.sub(r'\bWebShop\b', '', instruction, flags=re.IGNORECASE)
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    text = instruction.lower().strip()

    # ── Multi-word size normalization (before token matching) ─────────────────
    # Normalize multi-word size expressions to their hyphenated canonical forms
    # so that _SIZE_TOKENS (which uses hyphens) can match them.
    # Order matters: longer patterns first.
    _SIZE_MULTI_ALIASES = [
        (r'\bextra\s+extra\s+extra\s+large\b', '3x-large'),
        (r'\bextra\s+extra\s+large\b',         'xx-large'),
        (r'\bextra\s+large\b',                  'x-large'),
        (r'\bextra\s+small\b',                  'x-small'),
        (r'\b(\d)\s*x\s+large\b',               r'\1x-large'),   # "4x large" → "4x-large"
        (r'\b(\d)\s*x\s*large\b',               r'\1x-large'),   # "4xlarge"
        (r'\bx\s+large\b',                      'x-large'),
        (r'\bx\s+small\b',                      'x-small'),
        (r'\bxx\s+large\b',                     'xx-large'),
        (r'\bxxx\s+large\b',                    'xxx-large'),
    ]
    for _pat, _repl in _SIZE_MULTI_ALIASES:
        text = re.sub(_pat, _repl, text)

    # ── Multiplication spec expansion ─────────────────────────────────────────
    # "15*10ft", "96" w x 90"", "5x7 inch" → expand so both dimensions are
    # captured by _MEASUREMENT_RE as separate spec entries.
    # Replace "NxM unit" or "N*M unit" with "N unit x M unit".
    def _expand_mult_spec(m: re.Match) -> str:
        n1, n2, unit = m.group(1), m.group(2), m.group(3)
        return f"{n1} {unit} x {n2} {unit}"
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*[x×*]\s*(\d+(?:\.\d+)?)\s*'
        r'(ft\b|feet\b|foot\b|inch(?:es)?\b|in\b|cm\b|mm\b|meter\b)',
        _expand_mult_spec,
        text,
    )

    # Convert word-form numbers in size context: "size five" → "size 5",
    # "size nine boots" → "size 9 boots", etc.
    # Only replace in "size X" contexts to avoid "one pair" → "1 pair" affecting quantity
    text = re.sub(r'\bsize\s+(' + '|'.join(_WORD_NUMS.keys()) + r')\b',
                  lambda m: f"size {_WORD_NUMS[m.group(1)]}", text)

    # Price: "price lower than $X", "price < $X", "less than $X.XX", "under $X"
    price_max: Optional[float] = None
    price_m = re.search(
        r"(?:price\s+)?(?:lower than|less than|under|below|<)\s*\$?([\d,]+(?:\.\d+)?)",
        text,
    )
    if price_m:
        price_max = float(price_m.group(1).replace(",", ""))

    # Quantity: "1 dozen", "pack of N", "set of N", "Nx" (e.g. "3x large")
    quantity: Optional[str] = None
    qty_m = re.search(
        r"(\d+\s*(?:dozen|pack|set|pairs?|pieces?|count|ct\.?))|"
        r"(?:pack|set|lot)\s+of\s+(\d+)|"
        r"(\d+)\s*x\b",
        text,
    )
    if qty_m:
        quantity = qty_m.group(0).strip()

    # Color — also detect OR alternatives ("lavender or ivory")
    # Strategy:
    #   1. Core colors: match anywhere in the instruction.
    #   2. Ambiguous colors (chocolate/honey/natural/...): only match when followed by
    #      "colored", "color", "shade", or "hue" — never as a standalone product descriptor.
    color: Optional[str] = None
    color_alt: Optional[str] = None

    def _is_color_in_text(c: str, t: str) -> bool:
        """Returns True if `c` is a valid color reference in text `t`."""
        if c not in _AMBIGUOUS_COLOR_TOKENS:
            return bool(re.search(r'\b' + re.escape(c) + r'\b', t))
        # Ambiguous: require "X colored", "X color", "X shade", "X hue"
        return bool(re.search(
            r'\b' + re.escape(c) + r'\s*(?:color(?:ed)?|shade|hue)\b', t, re.IGNORECASE
        ))

    # First try OR patterns
    all_color_pat = '|'.join(re.escape(c) for c in sorted(_COLOR_TOKENS, key=len, reverse=True))
    or_m = re.search(
        r'\b(' + all_color_pat + r')\s+or\s+(' + all_color_pat + r')\b',
        text,
    )
    if or_m:
        c1, c2 = or_m.group(1), or_m.group(2)
        if _is_color_in_text(c1, text) and _is_color_in_text(c2, text):
            color = c1
            color_alt = c2
        elif _is_color_in_text(c1, text):
            color = c1
        elif _is_color_in_text(c2, text):
            color = c2
    if color is None:
        for c in sorted(_COLOR_TOKENS, key=len, reverse=True):
            if _is_color_in_text(c, text):
                color = c
                break

    # Compound color: capture multi-word color shades like "sky blue", "rose gold",
    # "elephant grey", "bubble pink".  Two strategies:
    #   1. Scan known compound patterns first (highest precision).
    #   2. Look for "modifier + color_token" where modifier is in _COLOR_MODIFIERS.
    # This ensures _match_option_in_list can prefer the full compound option.
    _COLOR_MODIFIERS = {
        "sky", "light", "dark", "pale", "deep", "bright", "baby", "powder",
        "neon", "army", "forest", "royal", "hot", "bubble", "mint", "lime",
        "ocean", "steel", "smoke", "stone", "ash", "slate", "nude", "blush",
        "blood", "fire", "ice", "pearl", "golden", "electric", "elephant",
        "dusty", "earthy", "bold", "wine", "smoky", "heather",
        # Color tokens that commonly appear as modifiers of another color
        "rose",   # rose gold, rose silver
        "navy",   # navy blue
        "olive",  # olive green
        "coral",  # coral red
        "peach",  # peach gold
    }
    if color and color not in {"multicolor", "multicolored", "multi-color", "multi-colored"}:
        # Strategy 1: modifier BEFORE the color token ("sky blue", "light grey")
        compound_m = re.search(r'\b([a-z]+)\s+' + re.escape(color) + r'\b', text)
        if compound_m:
            modifier = compound_m.group(1)
            if (modifier in _COLOR_MODIFIERS
                    and modifier not in _TASK_STOPWORDS
                    and modifier not in _SIZE_TOKENS):
                color = f"{modifier} {color}"
        # Strategy 2: the color itself is a modifier for another color token AFTER it
        # e.g. color='rose' found first, then "rose gold" → compound = "rose gold"
        if len(color.split()) == 1:  # only if not already compound
            compound_m2 = re.search(re.escape(color) + r'\s+([a-z]+)\b', text)
            if compound_m2:
                suffix = compound_m2.group(1)
                if (suffix in _COLOR_TOKENS
                        and suffix != color
                        and suffix not in _TASK_STOPWORDS):
                    color = f"{color} {suffix}"

    # Size (prefer longer tokens like "xx-large" over "large")
    # Numbers are only treated as sizes when NOT:
    #   (a) part of a decimal number (e.g. "6" in "6.6")
    #   (b) followed by a measurement unit (e.g. "8 ounce")
    #   (c) followed by a hyphen+unit (e.g. "13-ounce")
    #   (d) immediately preceded by an apostrophe (e.g. "m" in "i'm")
    size: Optional[str] = None
    size_candidates = []
    for s in _SIZE_TOKENS:
        pat = r'\b' + re.escape(s) + r'\b'
        m_s = re.search(pat, text)
        if m_s:
            start, end = m_s.start(), m_s.end()
            # Reject if part of a decimal number (e.g. "6" in "6.6").
            # Only reject when the '.' is followed by a digit — a trailing period
            # (end of sentence: "size 8. buy") must NOT be rejected.
            if end < len(text) and text[end] == '.' and end + 1 < len(text) and text[end + 1].isdigit():
                continue
            if start > 0 and text[start-1] == '.' and start >= 2 and text[start-2].isdigit():
                continue
            # Reject if preceded by an apostrophe (e.g. "m" in "i'm", "s" in "it's")
            if start > 0 and text[start-1] in "'\u2019\u2018":
                continue
            # Reject bare numbers followed by a measurement unit (with optional hyphen)
            suffix = text[end:end+25].strip()
            if re.match(r"^-?\s*(?:oz|fl|ounce|lbs?|pound|kg|gram|inch|in\b|ft\b|feet|foot|cm|mm|gb|tb|mb|ml|liter|gallon|pack|count|pcs?|piece|pair)", suffix, re.IGNORECASE):
                continue
            # Reject "medium" when used as an adjective modifying a color or
            # descriptor: "medium brown" (shade), "medium heel" (product type),
            # "medium weight", "medium sized", "medium wash".  "medium" is uniquely
            # ambiguous as an adjective — unlike "small"/"large".
            _MEDIUM_ADJ_FOLLOWERS = (
                _CORE_COLOR_TOKENS | _AMBIGUOUS_COLOR_TOKENS | {
                    "heel", "heeled", "weight", "duty", "length", "rise",
                    "wash", "sized", "width", "height", "strength",
                    "rare", "well", "done", "roast", "roasted", "bodied",
                }
            )
            if s == "medium":
                # Don't reject if preceded by "size" — that's an explicit size context
                pre_text = text[:start].rstrip()
                if not pre_text.endswith("size"):
                    next_word_m = re.match(r'\s+([a-z]+)', text[end:])
                    if next_word_m:
                        nw = next_word_m.group(1)
                        if nw in _MEDIUM_ADJ_FOLLOWERS:
                            continue
            size_candidates.append(s)
    if size_candidates:
        size = max(size_candidates, key=len)

    # Size fit modifiers: "petite" and "tall" are cut/fit qualifiers that appear
    # ALONGSIDE standard size tokens.  "petite small" or "small petite" in the
    # instruction means the agent must select the "small petite" button, not plain
    # "small".  Append the modifier so H5 can fuzzy-match the compound option.
    # Only fires when a standard size was already found (guards against "tall lamp").
    _SIZE_FIT_MODIFIERS = ("petite", "tall")
    if size:
        for _mod in _SIZE_FIT_MODIFIERS:
            if re.search(r'\b' + _mod + r'\b', text):
                size = f"{size} {_mod}"
                break

    # Material
    material: Optional[str] = None
    _MATERIALS = [
        "cotton", "polyester", "nylon", "wool", "leather", "suede", "canvas",
        "denim", "linen", "silk", "fleece", "spandex", "rubber", "plastic",
        "metal", "stainless steel", "aluminum", "wood", "bamboo", "ceramic",
    ]
    for m in _MATERIALS:
        if re.search(r'\b' + re.escape(m) + r'\b', text):
            material = m
            break

    # Item keywords: strip away constraint phrases to get the core noun
    item_text = re.sub(
        r"(?:price\s+)?(?:lower than|less than|under|below|<)\s*\$?[\d,]+(?:\.\d+)?",
        "", text
    )
    item_text = re.sub(r"\b(?:i need|i want|find|buy|get|look for|search for)\b", "", item_text)
    item_text = re.sub(r"\b(?:and|that|which|with|for|the|a|an|some|of)\b", " ", item_text)
    item_text = re.sub(r"\s+", " ", item_text).strip().rstrip(".,;")
    item_keywords = item_text[:80]  # cap to avoid over-long queries

    # Style keywords: descriptors that are not color/size/material
    style_keywords: List[str] = []
    _STYLE_PATTERNS = [
        r"wireless", r"bluetooth", r"waterproof", r"water[\s-]?resistant",
        r"dust[\s-]?proof", r"rechargeable", r"portable", r"adjustable",
        r"foldable", r"lightweight", r"heavy[\s-]?duty", r"non[\s-]?slip",
        r"organic", r"natural", r"eco[\s-]?friendly", r"hypoallergenic",
        r"machine[\s-]?washable", r"odor[\s-]?resistant",
    ]
    for pat in _STYLE_PATTERNS:
        if re.search(pat, text):
            style_keywords.append(re.sub(r"[\[\]]", "", pat).replace(r"[\s-]?", "-"))

    # Specs: measurement/tech values (e.g. "8 ounce", "32 gb", "6.6 feet", "18 inch")
    # Collect all distinct measurement mentions in the instruction.
    # Normalize: remove hyphens between number and unit ("13-ounce" → "13 ounce").
    specs: List[str] = []
    seen_specs: set = set()
    for m in _MEASUREMENT_RE.finditer(text):
        spec = m.group(0).strip()
        # Normalize hyphen between digit and unit word
        spec = re.sub(r'(\d)\s*-\s*([a-z])', r'\1 \2', spec)
        spec = re.sub(r"\s+", " ", spec)
        norm = spec.lower()
        if norm not in seen_specs:
            seen_specs.add(norm)
            specs.append(spec)

    # Task keywords: significant words for instruction-grounded option matching.
    # Used by H5 to match "other" page options that aren't color/size/spec but
    # ARE semantically relevant (e.g. "huckleberry" matches "huckleberry syrup" option).
    #
    # Apostrophe normalization: re.findall with \b word boundaries silently drops
    # possessive/gendered words — "men's" → boundary breaks at apostrophe, leaving
    # "men" (3 chars, below {4,} threshold) and "s" (1 char).  Normalize first so
    # "men's" → "mens" (4 chars, included), "women's" → "womens" (6 chars, included).
    text_for_kw = re.sub(r"'([a-z])", r'\1', text)   # strip apostrophes: men's→mens
    task_keywords: List[str] = [
        w for w in re.findall(r'\b[a-z]{4,}\b', text_for_kw)
        if w not in _TASK_STOPWORDS
    ]
    # Deduplicate while preserving order
    seen_kw: set = set()
    task_keywords_dedup: List[str] = []
    for w in task_keywords:
        if w not in seen_kw:
            seen_kw.add(w)
            task_keywords_dedup.append(w)

    return WebShopTaskRequirements(
        item_keywords=item_keywords,
        color=color,
        color_alt=color_alt,
        size=size,
        material=material,
        price_max=price_max,
        quantity=quantity,
        style_keywords=style_keywords,
        specs=specs,
        task_keywords=task_keywords_dedup,
        task_type=_detect_task_type(instruction),
        raw_instruction=instruction,
    )


def requirements_summary(req: WebShopTaskRequirements) -> str:
    """Short human-readable summary of requirements for use in hints."""
    parts = [req.item_keywords[:40]]
    if req.color:
        color_str = f"{req.color}/{req.color_alt}" if req.color_alt else req.color
        parts.append(f"color={color_str}")
    if req.size:
        parts.append(f"size={req.size}")
    if req.material:
        parts.append(f"material={req.material}")
    if req.specs:
        parts.append(f"spec={'/'.join(req.specs[:2])}")
    if req.price_max is not None:
        parts.append(f"price<${req.price_max:.0f}")
    if req.quantity:
        parts.append(f"qty={req.quantity}")
    return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# H1 — Page State Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WebShopPageState:
    page_type: str = PAGE_UNKNOWN
    search_queries: List[str] = field(default_factory=list)
    back_to_search_count: int = 0
    current_asin: Optional[str] = None
    asins_visited: List[str] = field(default_factory=list)
    # Attribute tracking on current product page
    selected_attributes: Dict[str, str] = field(default_factory=dict)
    attribute_options: Dict[str, List[str]] = field(default_factory=dict)
    current_price: Optional[float] = None
    # Stall tracking
    _stall_asin: Optional[str] = None
    _stall_turns: int = 0


def detect_page_type(observation: str, has_search_bar: bool, clickables: List[str]) -> str:
    """Determine page type from environment signals (API-stable across train/test)."""
    if has_search_bar:
        return PAGE_HOME
    if "Total results:" in observation:
        return PAGE_SEARCH_RESULTS
    if "buy now" in clickables or (
        "Description" in observation and "Features" in observation and "Reviews" in observation
    ):
        return PAGE_PRODUCT_DETAIL
    return PAGE_UNKNOWN


def extract_asin(clickables: List[str]) -> Optional[str]:
    """ASIN is a 10-char uppercase alphanumeric clickable (e.g. B08YNK1X66)."""
    for c in clickables:
        if re.fullmatch(r"[A-Z0-9]{10}", c.upper()) and len(c) == 10:
            return c.upper()
    return None


def extract_price(observation: str) -> Optional[float]:
    """Parse 'Price: $XX.XX' from product page observation."""
    m = re.search(r"[Pp]rice[:\s]+\$?([\d,]+(?:\.\d{1,2})?)", observation)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


# Attribute section headers as they appear in WebShop observations.
# Options under these headers are assigned to the mapped group.
_ATTR_COLOR_HDRS = frozenset({"color", "colours", "color name", "colour"})
_ATTR_SIZE_HDRS  = frozenset({"size", "sizes"})
_ATTR_OTHER_HDRS = frozenset({
    "flavor", "flavors", "flavour", "flavours", "flavor name",
    "scent", "scents", "fragrance",
    "style", "styles",
    "material", "materials",
    "pattern", "patterns",
    "configuration",
    "option", "options",
    "type", "format",
})


def _extract_attr_with_obs(
    observation: str,
    clickables: List[str],
) -> Dict[str, List[str]]:
    """Primary attribute extractor using observation section headers.

    Parses '[SEP]'-delimited observation to detect section headers (e.g.
    'color', 'size', 'flavor name') and assigns subsequent clickable items to
    the correct group. This correctly handles compound names ('metallic gun
    metal'), alphanumeric codes ('c08'), and slug-style names ('greenwithoutlogo')
    that token-based classification misses.

    Falls back to extract_attribute_options for any clickables not placed by
    the observation parser (e.g. when the observation is truncated).
    """
    clickable_lower_map: Dict[str, str] = {c.lower(): c for c in clickables}
    parts = [p.strip() for p in re.split(r'\[SEP\]', observation)]

    result: Dict[str, List[str]] = {}
    current_group: Optional[str] = None
    assigned: set = set()  # lowercase clickable values placed by obs parsing

    for part in parts:
        pl = part.lower()
        if not pl:
            continue
        # Navigation stops and page-level tokens end any open attribute section
        if pl in _NAV_CLICKABLES or pl in ("webshop",) or pl.startswith("instruction:"):
            current_group = None
            continue
        # Section header detection
        if pl in _ATTR_COLOR_HDRS:
            current_group = "color"
            continue
        if pl in _ATTR_SIZE_HDRS:
            current_group = "size"
            continue
        if pl in _ATTR_OTHER_HDRS:
            current_group = "other"
            continue
        # Assign this part to the current section if it is a clickable option
        if current_group is not None and pl in clickable_lower_map:
            result.setdefault(current_group, []).append(clickable_lower_map[pl])
            assigned.add(pl)

    # Fallback: classify any clickables not covered by observation parsing
    unassigned = [
        c for c in clickables
        if c.lower() not in assigned
        and c.lower() not in _NAV_CLICKABLES
        and not re.fullmatch(r"[a-z0-9]{10}", c.lower())
    ]
    if unassigned:
        fallback = extract_attribute_options(unassigned)
        for group, opts in fallback.items():
            for opt in opts:
                if opt.lower() not in assigned:
                    result.setdefault(group, []).append(opt)

    return {k: v for k, v in result.items() if v}


def extract_attribute_options(clickables: List[str]) -> Dict[str, List[str]]:
    """
    Classify non-navigation clickables into attribute categories.
    Returns e.g. {"color": ["black", "pink"], "size": ["large", "x-large"],
                  "spec": ["8 ounce", "6.6 feet"], "other": [...]}.

    Handles compound options like "sky blue", "light brown", "rose gold",
    "turquoise | ivory", "a1-green" by checking whether any known color/size
    token appears as a word inside the option string.
    """
    options: Dict[str, List[str]] = {"color": [], "size": [], "spec": [], "other": []}
    for c in clickables:
        cl = c.lower().strip()
        if cl in _NAV_CLICKABLES:
            continue
        if re.fullmatch(r"[a-z0-9]{10}", cl):
            continue  # ASIN-like, skip

        # Exact match first
        if cl in _COLOR_TOKENS:
            options["color"].append(c)
        elif cl in _SIZE_TOKENS:
            options["size"].append(c)
        elif _MEASUREMENT_RE.search(cl):
            options["spec"].append(c)
        else:
            # Compound option: check if a color or size token appears as a word inside
            # e.g. "sky blue", "light brown", "rose gold", "a1-green", "4x-large"
            found_color = any(
                re.search(r'\b' + re.escape(ct) + r'\b', cl)
                for ct in _COLOR_TOKENS
            )
            if found_color:
                options["color"].append(c)
                continue
            found_size = any(
                re.search(r'\b' + re.escape(st) + r'\b', cl)
                for st in _SIZE_TOKENS
            )
            if found_size:
                options["size"].append(c)
                continue
            options["other"].append(c)
    # Drop empty categories
    return {k: v for k, v in options.items() if v}


def _asin_from_action(action: Optional[str]) -> Optional[str]:
    """Extract ASIN from a click[ASIN] action string.

    Product pages don't include the ASIN in their own clickables list, so we
    infer it from the navigation action that brought us here.
    """
    if action and action.startswith("click[") and action.endswith("]"):
        val = action[6:-1].strip()
        if re.fullmatch(r"[A-Za-z0-9]{10}", val):
            return val.upper()
    return None


def update_page_state(
    state: WebShopPageState,
    action: Optional[str],
    observation: str,
    has_search_bar: bool,
    clickables: List[str],
) -> None:
    """H1: Update WebShopPageState in-place after each step."""
    new_page_type = detect_page_type(observation, has_search_bar, clickables)

    # Track search queries
    if action and action.startswith("search[") and action.endswith("]"):
        query = action[7:-1].strip()
        state.search_queries.append(query)

    # Track back-to-search clicks
    if action and action == "click[back to search]":
        state.back_to_search_count += 1

    # Track ASIN and attribute state on product pages
    if new_page_type == PAGE_PRODUCT_DETAIL:
        # Product pages do NOT contain the ASIN in their own clickables (only search
        # result pages do).  Derive the ASIN from the navigation action instead.
        asin = _asin_from_action(action)

        if asin and asin != state.current_asin:
            # Navigated to a new product — reset attribute state
            state.current_asin = asin
            if asin not in state.asins_visited:
                state.asins_visited.append(asin)
            state.selected_attributes = {}
            state.attribute_options = _extract_attr_with_obs(observation, clickables)
            state.current_price = extract_price(observation)
            state._stall_asin = asin
            state._stall_turns = 0
        else:
            # Same product page: track the click BEFORE refreshing options, because
            # WebShop often removes the selected option from the clickables list after
            # selection (e.g. "sky blue" disappears once chosen). If we refresh first,
            # the selected option is gone and we can't match it.
            pre_click_opts = {k: list(v) for k, v in state.attribute_options.items()}

            new_opts = _extract_attr_with_obs(observation, clickables)
            if new_opts:
                state.attribute_options = new_opts
            # Track attribute selection using PRE-click options.
            # WebShop removes the selected option from the clickables list after
            # the click (e.g. "sky blue" disappears after selection), so we must
            # look up the clicked value in the options that existed BEFORE the
            # page re-rendered.
            if action and action.startswith("click[") and action.endswith("]"):
                clicked = action[6:-1].strip().lower()
                # First try pre-click options (handles removed-after-click case)
                tracked = False
                for category, opts in pre_click_opts.items():
                    if clicked in [o.lower() for o in opts]:
                        state.selected_attributes[category] = clicked
                        tracked = True
                        break
                # Fallback: try current (post-click) options
                if not tracked:
                    for category, opts in state.attribute_options.items():
                        if clicked in [o.lower() for o in opts]:
                            state.selected_attributes[category] = clicked
                            break
            # Update price in case page re-rendered
            price = extract_price(observation)
            if price is not None:
                state.current_price = price

        # Stall counter
        if state._stall_asin == state.current_asin:
            state._stall_turns += 1
        else:
            state._stall_asin = state.current_asin
            state._stall_turns = 1
    else:
        state._stall_turns = 0
        state._stall_asin = None

    state.page_type = new_page_type


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Action Gate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_match_click(value: str, clickables: List[str], threshold: float) -> Optional[str]:
    """Return the best-matching clickable for value, or None if below threshold."""
    value_l = value.strip().lower()
    best, best_score = None, -1.0
    for c in clickables:
        score = SequenceMatcher(None, value_l, c.lower()).ratio()
        if score > best_score:
            best_score = score
            best = c
    if best_score >= threshold:
        return best
    return None


# ─────────────────────────────────────────────────────────────────────────────
# H3 — Tool description patching
# ─────────────────────────────────────────────────────────────────────────────

_H3_SEARCH_HINT = (
    "Search for the product using specific keywords matching all required attributes "
    "(item type, color, size, material, etc.). Do NOT include price or budget words — "
    "WebShop search ignores price; use the product name and key features only."
)
_H3_CLICK_HINT = (
    "On a product page, select ALL required attributes (color, size, variant, etc.) "
    "that match the task description BEFORE clicking 'buy now'. "
    "Check each option group on the page; leave nothing unselected if the task specifies it."
)


def patch_webshop_tool_descriptions(
    tools: Optional[List[Dict[str, Any]]],
    extra_hint: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """H3: Append shopping strategy hints to tool descriptions."""
    if not tools:
        return tools
    patched = copy.deepcopy(tools)
    extra = str(extra_hint or "").strip()
    for tool in patched:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if name == "search_action":
            fn["description"] = fn.get("description", "") + " " + _H3_SEARCH_HINT
        elif name == "click_action":
            fn["description"] = fn.get("description", "") + " " + _H3_CLICK_HINT
        if extra:
            description = fn.get("description", "").strip()
            if extra not in description:
                fn["description"] = f"{description}\n\n{extra}".strip()
        tool["function"] = fn
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# H5 — Attribute checklist builder
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_ALIASES = {
    "oz": "ounce", "ounces": "ounce", "ozs": "ounce",
    "lbs": "pound", "lb": "pound", "pounds": "pound",
    "inches": "inch", "in": "inch", "\"": "inch",
    "ft": "feet", "foot": "feet",
    "grams": "gram", "kgs": "kg",
    "milliliter": "ml", "milliliters": "ml",
    "liters": "liter", "litres": "liter", "litre": "liter",
    "gallons": "gallon", "quarts": "quart",
    "packs": "pack", "counts": "count", "pieces": "piece", "pcs": "piece", "pairs": "pair",
}


def _normalize_spec(s: str) -> str:
    """Normalize a measurement value for matching: lowercase, remove hyphens,
    expand unit aliases, collapse spaces."""
    s = s.lower().strip()
    s = re.sub(r'(\d)\s*-\s*([a-z])', r'\1 \2', s)  # "13-ounce" → "13 ounce"
    s = re.sub(r'\s+', ' ', s)
    # Replace unit aliases
    for alias, canonical in _UNIT_ALIASES.items():
        s = re.sub(r'\b' + re.escape(alias) + r'\b', canonical, s)
    return s


def _match_option_in_list(required_value: str, opts: List[str]) -> Optional[str]:
    """Find the best matching option for required_value in opts list.

    Matching priority:
    1. Exact (case-insensitive, with spec normalization)
    2. Required value is a substring of an option (e.g. "32gb" in "32gb ram | 1 tb ssd")
    3. An option is a substring of required value
    Returns the matched option string, or None.
    """
    req_l = required_value.lower().strip()
    req_norm = _normalize_spec(required_value)
    for o in opts:
        if o.lower().strip() == req_l or _normalize_spec(o) == req_norm:
            return o
    for o in opts:
        if req_l in o.lower() or req_norm in _normalize_spec(o):
            return o
    for o in opts:
        if o.lower() in req_l or _normalize_spec(o) in req_norm:
            return o
    return None


def _instruction_grounded_matches(
    req: WebShopTaskRequirements,
    other_opts: List[str],
) -> List[str]:
    """H5 helper: find "other" page options that overlap with task keywords.

    Returns a list of option strings that are semantically grounded in the
    task instruction (e.g. "huckleberry" option matches "huckleberry syrup" task,
    "counter height" matches "counter height barstools" task).

    Uses word-overlap: at least one meaningful keyword from the instruction
    must appear as a whole word inside the option string.
    Excludes options that are entirely composed of stopwords or navigation tokens.

    IG is intentionally not used for color/style variants that H0 already handles:
    if ALL overlap words between the option and task_keywords are in _COLOR_TOKENS
    or _AMBIGUOUS_COLOR_TOKENS, skip — the color check already covers this, and IG
    matching on color-like words (e.g. "stonewash") creates false "All Checked".
    """
    if not req.task_keywords or not other_opts:
        return []
    keyword_set = set(req.task_keywords)
    # Words already covered by explicit H0 color/size detection — IG shouldn't
    # shadow them or create a separate requirement for the same attribute.
    color_covered: set = set()
    if req.color:
        color_covered.update(req.color.lower().split())
    if req.color_alt:
        color_covered.update(req.color_alt.lower().split())
    if req.size:
        color_covered.update(req.size.lower().split())

    # Generic adjective/descriptor words that alone don't constitute a meaningful
    # IG match — they appear in many unrelated option strings.
    _IG_WEAK_WORDS = {
        "dark", "light", "long", "short", "soft", "hard", "deep", "flat",
        "wide", "thin", "free", "open", "pure", "slim", "tall", "warm",
        "cool", "bold", "full", "fine", "mini", "maxi", "high", "plus",
    }

    matched: List[str] = []
    for opt in other_opts:
        opt_l = opt.lower()

        # Skip long options (>4 words): these are almost always product category
        # descriptions offered as bundle/variant options, not true attribute selectors.
        # e.g. "neon gel nail polish set", "v-neck pocket maxi - feather ruby".
        if len(opt_l.split()) > 4:
            continue

        # Apostrophe normalization: "men's" → "mens" so it can match task_keywords
        # (which were extracted the same way — see parse_task_requirements).
        opt_normalized = re.sub(r"'([a-z])", r'\1', opt_l)
        opt_words = set(re.findall(r'\b[a-z]{4,}\b', opt_normalized))
        overlap = opt_words & keyword_set
        if not overlap:
            continue
        # Skip if the entire overlap is color/ambiguous-color tokens.
        # These are either already tracked by H0 (color check) or are
        # style variants that shouldn't generate an extra IG requirement.
        all_color_like = overlap.issubset(_AMBIGUOUS_COLOR_TOKENS | _CORE_COLOR_TOKENS | color_covered)
        if all_color_like:
            continue
        # Skip if ALL overlapping words are generic adjectives/descriptors.
        # A single weak word like "dark" matching "dark blonde mix" isn't meaningful.
        meaningful_overlap = overlap - _IG_WEAK_WORDS
        if not meaningful_overlap:
            continue
        matched.append(opt)
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Search Results Product Ranking
# Parses the search results observation, scores each product title against
# task keywords, and returns the best-matching ASIN to help agent selection.
# ─────────────────────────────────────────────────────────────────────────────

_ASIN_RE = re.compile(r'\b(b[0-9a-z]{9})\b', re.IGNORECASE)

def _rank_search_results(
    observation: str,
    req: WebShopTaskRequirements,
    clickables: List[str],
) -> Optional[str]:
    """Score products on the search results page and suggest the best match.

    Parses [SEP]-delimited observation to extract ASIN → title pairs.
    Scores each title by keyword overlap with task_keywords.
    Returns a short suggestion string, or None if no useful ranking.
    """
    # Parse search results: pattern is ASIN [SEP] Title [SEP] Price [SEP] ASIN ...
    parts = re.split(r'\[SEP\]', observation)
    # Build list of (asin, title, price) tuples
    products: List[dict] = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        # Check if this part is an ASIN (clickable)
        asin_match = _ASIN_RE.fullmatch(part)
        if asin_match and asin_match.group(1).lower() in [c.lower() for c in clickables]:
            asin = asin_match.group(1).lower()
            title = parts[i + 1].strip() if i + 1 < len(parts) else ""
            price_str = parts[i + 2].strip() if i + 2 < len(parts) else ""
            products.append({"asin": asin, "title": title, "price_str": price_str})
            i += 3
        else:
            i += 1

    if not products:
        return None

    # Score each product by task keyword overlap
    # Normalize apostrophes for matching
    kw_set = set(req.task_keywords)
    # Also add color/size as scoring signals
    if req.color:
        kw_set.update(req.color.lower().split())
    if req.material:
        kw_set.add(req.material.lower())

    scored = []
    for p in products:
        title_normalized = re.sub(r"'([a-z])", r'\1', p["title"].lower())
        title_words = set(re.findall(r'\b[a-z]{3,}\b', title_normalized))
        overlap = title_words & kw_set
        score = len(overlap)

        # Price check: penalize if over budget
        price_ok = True
        if req.price_max is not None:
            # Extract highest price from range like "$14.77 to $21.37"
            prices = re.findall(r'\$?([\d,]+(?:\.\d+)?)', p["price_str"])
            if prices:
                max_price = max(float(px.replace(",", "")) for px in prices)
                if max_price > req.price_max * 1.05:
                    price_ok = False

        scored.append({
            **p,
            "score": score,
            "overlap": overlap,
            "price_ok": price_ok,
        })

    # Sort: price-ok products first, then by score descending
    scored.sort(key=lambda x: (x["price_ok"], x["score"]), reverse=True)

    best = scored[0]
    # Require at least 2 distinct task keywords to match before suggesting.
    # A single-keyword match (e.g. only "moisturizer" for "buttercream scent
    # moisturizer") is too weak to reliably identify the right product.
    if best["score"] < 2:
        return None

    # Include the product title so the agent can verify the suggestion is correct.
    title_preview = best["title"][:60].strip() if best["title"] else best["asin"]
    suggestion = f"Best match on this page: '{title_preview}' — click '{best['asin']}'"
    if not best["price_ok"]:
        suggestion += " (WARNING: may exceed budget)"

    return suggestion


# ─────────────────────────────────────────────────────────────────────────────
# H4 sub-check — Product Title / Category Verification
# Fires once per new ASIN (stall_turns == 1) inside post_step_monitor.
# Controlled by h4_enabled; no separate config flag needed.
# ─────────────────────────────────────────────────────────────────────────────

# Words that are too generic to be useful for category matching
_CATEGORY_GENERIC_WORDS = _TASK_STOPWORDS | {
    "price", "lower", "looking", "dollars", "please", "cheap", "great",
    "nice", "perfect", "best", "want", "need", "find", "look",
    # Verb/gerund forms that appear in task instructions but aren't product nouns
    "interested", "buying", "wanting", "searching", "finding", "lasting",
    "adding", "getting", "using", "featuring", "offering", "providing",
    "including", "looking", "matching", "fitting", "going", "coming",
    # Common adjectives/descriptors (not product type nouns)
    "classic", "modern", "simple", "basic", "standard", "typical",
    "various", "multiple", "different", "similar", "compatible",
    "general", "special", "unique", "original", "genuine", "authentic",
    "durable", "strong", "sturdy", "solid", "elegant", "stylish",
    "affordable", "budget", "premium", "everyday", "daily", "regular",
    # Size/color/material meta-words that aren't product categories
    "colored", "shade", "style", "finish", "design", "pattern",
}

def _product_title_check(
    req: WebShopTaskRequirements,
    observation: str,
) -> Optional[str]:
    """H4 sub-check: whether the current product page is the right item category.

    Extracts meaningful nouns from item_keywords and checks how many appear
    in the product page observation. When overlap is very low, warns the agent
    to reconsider and go back to search.

    Designed to fire only on first landing on a new product page (caller
    should only call when ASIN changed).

    Returns a warning string or None.
    """
    if req is None:
        return None

    # Build a candidate set of item-type words from the task instruction.
    # Use item_keywords as primary source; also pull longer words from raw_instruction
    # that aren't stopwords/color/size/price tokens.
    _color_words = _COLOR_TOKENS
    _size_words = _SIZE_TOKENS
    item_text = req.item_keywords.lower()
    raw_text = req.raw_instruction.lower()

    candidate_words: List[str] = []
    for w in re.findall(r'\b[a-z]{4,}\b', item_text):
        if (w not in _CATEGORY_GENERIC_WORDS
                and w not in _color_words
                and w not in _size_words):
            candidate_words.append(w)

    # Deduplicate, keep top 6 most discriminative (longer words first)
    seen: set = set()
    item_nouns: List[str] = []
    for w in sorted(set(candidate_words), key=len, reverse=True):
        if w not in seen:
            seen.add(w)
            item_nouns.append(w)
        if len(item_nouns) >= 6:
            break

    if not item_nouns:
        return None

    # Flatten the observation into a single searchable string.
    # WebShop observation format: "Instruction: [SEP] task [SEP] Back to Search [SEP] ..."
    # We want to search in the product portion, not the instruction portion.
    # Heuristic: take text after the second "[SEP]" (skip task instruction).
    obs_lower = observation.lower()
    sep_positions = [m.start() for m in re.finditer(r'\[sep\]', obs_lower)]
    if len(sep_positions) >= 2:
        product_portion = obs_lower[sep_positions[1]:]
    else:
        product_portion = obs_lower

    overlap_count = sum(1 for w in item_nouns if w in product_portion)
    overlap_ratio = overlap_count / len(item_nouns)

    # item_nouns is sorted longest-first, so item_nouns[0] is the most specific noun.
    # Warn only when BOTH conditions hold:
    #   - primary noun (longest, most discriminative) is missing, AND
    #   - overall overlap is below 50% (more than half the item nouns are missing).
    # Using AND prevents false positives where the task phrase has multiple content
    # words but the correct product page is missing one (e.g. "batteries remote control
    # for tv" → a TV remote may lack "batteries" but has "remote" + "control" → 0.67).
    primary_noun_found = item_nouns[0] in product_portion
    if not primary_noun_found and overlap_ratio < 0.50:
        missing = [w for w in item_nouns if w not in product_portion][:3]
        return (
            f"Harness: this product may not match the task category. "
            f"Expected item keywords not found in product: {', '.join(missing)}. "
            f"Consider going back to search for a better match."
        )
    return None


def _build_attribute_checklist(
    req: WebShopTaskRequirements,
    state: WebShopPageState,
    clickables: List[str],
) -> str:
    """
    Build a human-readable attribute checklist for the current product page.
    Covers: color, size, measurement specs, and style/other attributes.
    Shows: selected ✓ | click 'X' | not available on this page.
    """
    lines = []
    attr_statuses: Dict[str, str] = {}  # category → 'ok' | 'missing' | 'unavailable'

    def _check_attr(category: str, required_value: str) -> Tuple[str, str]:
        """Returns (line, status) where status ∈ 'ok' | 'missing' | 'unavailable'."""
        req_l = required_value.lower()
        selected = state.selected_attributes.get(category, "").lower()
        # Selected check: exact or substring.
        #   req_l in selected  — requirement is a substring of the selected value
        #                        (e.g. req="cotton", sel="100% cotton blend")
        #   sel_words ⊇ req_words — selected words are a superset of required words
        #                        (replaces the old `selected in req_l` string-substring
        #                         check, which falsely accepted "small" as satisfying
        #                         "small petite" because "small" ⊂ "small petite" str)
        req_words = set(req_l.split())
        sel_words = set(selected.split()) if selected else set()
        if selected and (
            selected == req_l
            or req_l in selected
            or sel_words.issuperset(req_words)
        ):
            return f"  - {category}={required_value}: [selected ✓]", "ok"
        opts = state.attribute_options.get(category, [])
        match = _match_option_in_list(required_value, opts)
        if match:
            # If agent already selected the best available match, treat as ok
            if selected and selected == match.lower():
                return f"  - {category}={required_value}: [selected ✓]", "ok"
            return f"  - {category}={required_value}: [click '{match}']", "missing"
        if opts:
            return (
                f"  - {category}={required_value}: "
                f"[not available — page has: {', '.join(opts[:4])}]",
                "unavailable",
            )
        return (
            f"  - {category}={required_value}: "
            f"[no {category} options — consider going back to search]",
            "unavailable",
        )

    def _check_attr_or(category: str, primary: str, alt: str) -> Tuple[str, str]:
        """Like _check_attr but accepts either of two values (OR alternatives).
        Returns 'ok' if EITHER is selected, 'missing' if EITHER can be clicked,
        'unavailable' only if NEITHER is available.
        """
        req_l1, req_l2 = primary.lower(), alt.lower()
        selected = state.selected_attributes.get(category, "").lower()
        if selected and (selected in (req_l1, req_l2) or req_l1 in selected or req_l2 in selected
                        or selected in req_l1 or selected in req_l2):
            return f"  - {category}={primary} or {alt}: [selected ✓]", "ok"
        opts = state.attribute_options.get(category, [])
        match1 = _match_option_in_list(primary, opts)
        match2 = _match_option_in_list(alt, opts)
        if match1 or match2:
            clickable = match1 or match2
            # If agent already selected the best available match, treat as ok
            if selected and selected == clickable.lower():
                return f"  - {category}={primary} or {alt}: [selected ✓]", "ok"
            return f"  - {category}={primary} or {alt}: [click '{clickable}']", "missing"
        if opts:
            return (
                f"  - {category}={primary} or {alt}: "
                f"[not available — page has: {', '.join(opts[:4])}]",
                "unavailable",
            )
        return (
            f"  - {category}={primary} or {alt}: "
            f"[no {category} options — consider going back to search]",
            "unavailable",
        )

    # ── Color ──────────────────────────────────────────────────────────────────
    if req.color:
        if req.color_alt:
            line, status = _check_attr_or("color", req.color, req.color_alt)
        else:
            line, status = _check_attr("color", req.color)
        lines.append(line)
        attr_statuses["color"] = status

    # ── Size ───────────────────────────────────────────────────────────────────
    if req.size:
        line, status = _check_attr("size", req.size)
        lines.append(line)
        attr_statuses["size"] = status

    # ── Specs (measurements, tech specs) ───────────────────────────────────────
    # Try to match each required spec against the page's "spec" and "other" options.
    # This handles "8 ounce", "6.6 feet", "32gb ram | 1 tb ssd" etc.
    # Always shows the requirement even when page has no matching options (unlike before).
    spec_opts = (
        state.attribute_options.get("spec", [])
        + state.attribute_options.get("other", [])
    )
    for spec_req in req.specs:
        # Skip specs already well-covered by color/size matching
        if req.color and spec_req.lower() in req.color.lower():
            continue
        match = _match_option_in_list(spec_req, spec_opts)
        sel_spec = state.selected_attributes.get("spec", "")
        if sel_spec and spec_req.lower() in sel_spec.lower():
            lines.append(f"  - spec={spec_req}: [selected ✓]")
            attr_statuses[f"spec:{spec_req}"] = "ok"
        elif match:
            lines.append(f"  - spec={spec_req}: [click '{match}']")
            attr_statuses[f"spec:{spec_req}"] = "missing"
        elif spec_opts:
            lines.append(f"  - spec={spec_req}: [not found — page has: {', '.join(spec_opts[:3])}]")
            attr_statuses[f"spec:{spec_req}"] = "unavailable"
        else:
            # Always show spec requirement even when no page options available
            lines.append(f"  - spec={spec_req}: [no options on this page — verify manually]")
            attr_statuses[f"spec:{spec_req}"] = "unavailable"

    # ── Style keywords (catch-all, hardcoded patterns) ────────────────────────
    other_opts = state.attribute_options.get("other", [])
    if other_opts and req.style_keywords:
        matched_style = [o for o in other_opts for kw in req.style_keywords if kw[:4].lower() in o.lower()]
        if matched_style:
            sel = state.selected_attributes.get("other", "")
            if sel:
                lines.append(f"  - style: [selected ✓: {sel}]")
                attr_statuses["style"] = "ok"
            else:
                lines.append(f"  - style: [consider: {', '.join(matched_style[:3])}]")
                attr_statuses["style"] = "missing"

    # ── Instruction-grounded option matching ───────────────────────────────────
    # Dynamically match page "other" options to task keywords not covered by
    # color/size/spec/style checks (e.g. "huckleberry", "counter height", "beard balm").
    # This runs regardless of whether explicit requirements were found.
    ig_matches = _instruction_grounded_matches(req, other_opts)
    # Remove options already covered by style keyword matching above
    already_covered_opts = {o.lower() for o in (matched_style if other_opts and req.style_keywords else [])}
    ig_matches = [o for o in ig_matches if o.lower() not in already_covered_opts]
    # IG matches from the same "other" group are MUTUALLY EXCLUSIVE (radio buttons):
    # only ONE can be selected at a time. Treat them as an OR group — satisfying
    # ANY ONE is enough. This prevents the ping-pong loop where the agent keeps
    # toggling between two alternatives forever.
    if ig_matches:
        sel_other = state.selected_attributes.get("other", "").lower()
        ig_any_selected = sel_other and any(
            ig_opt.lower() in sel_other for ig_opt in ig_matches[:2]
        )
        if ig_any_selected:
            # Show which one is currently selected
            matched_ig = next(
                (o for o in ig_matches[:2] if o.lower() in sel_other), ig_matches[0]
            )
            lines.append(f"  - feature: [selected ✓: '{matched_ig}']")
            attr_statuses["feature"] = "ok"
        else:
            # None selected yet — show the best candidate(s) as alternatives
            if len(ig_matches) >= 2:
                opts_str = "' or '".join(ig_matches[:2])
                lines.append(f"  - feature: [click one of: '{opts_str}']")
            else:
                lines.append(f"  - feature: [click '{ig_matches[0]}']")
            attr_statuses["feature"] = "missing"

    # ── Price ──────────────────────────────────────────────────────────────────
    price_ok = (
        req.price_max is None
        or state.current_price is None
        or state.current_price <= req.price_max * 1.05
    )
    if req.price_max is not None and state.current_price is not None:
        mark = "✓" if price_ok else "✗ OVER BUDGET"
        lines.append(f"  - price=${state.current_price:.2f} (budget: ${req.price_max:.0f}): [{mark}]")

    # ── Defensive: unselected attribute groups ──────────────────────────────
    # When H0 didn't extract a requirement for color/size but the page has
    # selectable options in those groups, warn the agent to select something.
    for group in ("color", "size"):
        if group in attr_statuses:
            continue  # already covered by H0-parsed requirement check above
        opts = state.attribute_options.get(group, [])
        sel = state.selected_attributes.get(group, "")
        if opts and not sel:
            lines.append(f"  - {group}: [unselected — choose one: {', '.join(opts[:4])}]")
            attr_statuses[group] = "missing"
        elif opts and sel:
            lines.append(f"  - {group}: [selected ✓: {sel}]")
            attr_statuses[group] = "ok"

    if not lines:
        return ""

    checklist = "Attribute checklist:\n" + "\n".join(lines)

    # ── Verdict (only when buy now is available) ───────────────────────────────
    if "buy now" in [c.lower() for c in clickables]:
        has_unavailable = any(s == "unavailable" for s in attr_statuses.values())
        has_missing = any(s == "missing" for s in attr_statuses.values())

        # "all_ok" only fires when we actually tracked and verified requirements.
        has_tracked_attrs = bool(attr_statuses)
        all_ok = has_tracked_attrs and all(s == "ok" for s in attr_statuses.values())

        if not price_ok:
            checklist += "\n→ Price over budget. Go back to search for a cheaper product."
        elif has_missing:
            # Selectable attrs still exist on page but aren't clicked yet — prioritize these.
            miss_keys = [
                k.replace("spec:", "").replace("feature:", "")
                for k, v in attr_statuses.items() if v == "missing"
            ]
            checklist += f"\n→ Still need to select: {', '.join(miss_keys)}."
        elif has_unavailable:
            # No selector on page for these attrs — product is likely a single-variant item.
            # Build human-readable attribute value descriptions.
            ua_value_strs = []
            for k, v in attr_statuses.items():
                if v != "unavailable":
                    continue
                if k == "color":
                    v_str = req.color + (f" or {req.color_alt}" if req.color_alt else "")
                    ua_value_strs.append(f"color={v_str}")
                elif k == "size":
                    ua_value_strs.append(f"size={req.size}")
                elif k.startswith("spec:"):
                    ua_value_strs.append(k.replace("spec:", ""))
                else:
                    ua_value_strs.append(k)
            val_str = ", ".join(ua_value_strs[:2])
            # Soft verdict: don't force "go back" — single-variant products have
            # no selector for their fixed attribute. Let the agent decide based on
            # whether the product title/description matches the task.
            checklist += (
                f"\n→ No {val_str} selector on this page (may be a single-variant product). "
                f"If the product title/description confirms it has {val_str}, click 'buy now'. "
                f"Otherwise go back to search."
            )
        elif all_ok:
            # Check whether important task descriptors aren't covered by any tracked attr.
            # If so, append a verification nudge so the agent confirms the right product.
            tracked_words: set = set()
            if req.color:
                tracked_words.update(req.color.lower().split())
            if req.size:
                tracked_words.update(req.size.lower().split())
            if req.material:
                tracked_words.update(req.material.lower().split())
            for sp in req.specs:
                tracked_words.update(sp.lower().split())
            # Also include words matched by IG/style so we don't double-nudge
            for k in attr_statuses:
                tracked_words.update(re.findall(r'[a-z]{4,}', k.lower()))

            unchecked_kws = [
                w for w in req.task_keywords
                if w not in tracked_words
                and w not in _CATEGORY_GENERIC_WORDS
                and w not in _COLOR_TOKENS   # covered by color check
                and w not in _SIZE_TOKENS    # covered by size check
                and len(w) >= 4
            ][:3]

            if unchecked_kws:
                kw_str = ", ".join(f"'{w}'" for w in unchecked_kws)
                checklist += (
                    f"\n→ All attributes selected. Also verify the product mentions "
                    f"{kw_str} before clicking 'buy now'."
                )
            else:
                checklist += "\n→ All checked. Click 'buy now' to complete the purchase."
        elif not has_tracked_attrs:
            # No attribute selectors on this page. Build explicit requirement list
            # so the agent knows exactly what to verify in the title/description.
            req_parts = []
            if req.color:
                c_str = req.color + (f" or {req.color_alt}" if req.color_alt else "")
                req_parts.append(f"color={c_str}")
            if req.size:
                req_parts.append(f"size={req.size}")
            if req.material:
                req_parts.append(f"material={req.material}")
            for sp in req.specs[:2]:
                req_parts.append(f"spec={sp}")
            if req_parts:
                req_str = ", ".join(req_parts)
                checklist += (
                    f"\n→ No attribute selectors on this page. "
                    f"Required: {req_str}. "
                    f"Verify the title/description confirms BOTH product type AND these requirements "
                    f"before clicking 'buy now'. If unsure, go back to search."
                )
            else:
                top_kws = req.task_keywords[:4]
                if top_kws:
                    kw_str = ", ".join(f"'{k}'" for k in top_kws)
                    checklist += (
                        f"\n→ Verify this product matches the task — look for {kw_str} "
                        f"in the title/description, then click 'buy now'."
                    )
                else:
                    checklist += "\n→ Verify this product matches the task description, then click 'buy now'."

    return checklist


# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WebShopHarnessRuntime:
    config: WebShopHarnessConfig
    requirements: Optional[WebShopTaskRequirements] = field(default=None)
    page_state: WebShopPageState = field(default_factory=WebShopPageState)
    force_next_action: Optional[str] = field(default=None)

    # H2 state
    _last_click: Optional[str] = field(default=None)
    _repeat_click_count: int = field(default=0)
    _total_buy_now_blocks: int = field(default=0)  # safety cap across all block reasons

    # H5 dedup
    _last_hint: Optional[str] = field(default=None)
    _last_hint_page: str = field(default="")

    # H4 search loop escalation: counts how many times the loop recovery has
    # already fired this episode. After enough resets, switch to "just buy" mode.
    _search_loop_resets: int = field(default=0)

    # H4 stall escalation: per-ASIN count of stall interventions. When an agent
    # keeps returning to the same product page and stalling, escalate to a hard
    # "decide now" message instead of repeating the attribute checklist.
    _asin_stall_count: Dict[str, int] = field(default_factory=dict)

    # ── H0 ───────────────────────────────────────────────────────────────────

    def init_task(self, instruction: str) -> None:
        """H0: Parse task requirements and reset episode state."""
        self.requirements = parse_task_requirements(instruction)
        self.page_state = WebShopPageState()
        self.force_next_action = None
        self._last_click = None
        self._repeat_click_count = 0
        self._total_buy_now_blocks = 0  # safety cap across all block reasons
        self._defensive_buy_blocks = 0  # count defensive attr guard blocks
        self._product_mismatch_warned = False  # H4 flagged product mismatch
        self._mismatch_buy_blocks = 0  # times buy-now blocked after mismatch
        self._last_hint = None
        self._last_hint_page = ""
        self._search_loop_resets = 0
        self._asin_stall_count = {}
        self._price_warned_asins: set = set()  # ASINs already warned about price, to avoid repeat H4 loops
        self._h4_intervened = False  # set by post_step_monitor; suppresses H4-E on same turn

    # ── H5 cold-start ────────────────────────────────────────────────────────

    def cold_start_skill_hints(self) -> List[Dict[str, str]]:
        """
        H5 cold-start: two-layer skill retrieval.
          1. Filter WEBSHOP_SKILLS by task_type tag.
          2. BM25-rank filtered candidates against the raw instruction.
          3. Return top-k (config.h5_top_k) skills.
        """
        if not self.config.h5_enabled or self.requirements is None:
            return []
        skills = retrieve_webshop_skills(
            task_type=self.requirements.task_type,
            query=self.requirements.raw_instruction,
            top_k=self.config.h5_top_k,
            score_threshold=self.config.h5_score_threshold,
        )
        result = []
        for skill in skills:
            text = skill["text"]
            result.append({
                "id": skill["id"],
                "text": text,
                "trigger": "cold_start",
                "token_cost": str(len(text.split())),
            })
        return result

    # ── H2 ───────────────────────────────────────────────────────────────────

    def pre_validate_action(
        self,
        tool_name: str,
        raw_value: str,
        has_search_bar: bool,
        clickables: List[str],
    ) -> Dict[str, Any]:
        """
        H2: Validate and optionally canonicalize the agent's tool call.

        Returns:
          action: str | None — final action string to execute (None = block)
          blocked: bool
          reason: str
          canonicalized: bool
        """
        response: Dict[str, Any] = {
            "action": None,
            "blocked": False,
            "reason": "",
            "canonicalized": False,
        }

        # Apply force_next_action if set (from H5 enforcement or H4 budget)
        if self.force_next_action:
            forced = self.force_next_action
            # Only yield to agent if it's doing something critical (buy now)
            if tool_name == "click_action" and raw_value.lower().strip() == "buy now":
                self.force_next_action = None
                pass  # let agent's buy now through (will hit precheck below)
            elif forced == "click[buy now]":
                # Before force-buying, verify required attributes are selected.
                # If something is still missing, cancel the force and let H5
                # continue guiding — a premature force with missing attrs just
                # locks in a partial reward.
                block_msg = self._buy_now_precheck(clickables)
                if block_msg:
                    # Attributes still missing: don't force, clear the queued action.
                    self.force_next_action = None
                    response["blocked"] = True
                    response["reason"] = "force_buy_now_attrs_missing"
                    response["block_message"] = block_msg
                    return response
                self.force_next_action = None
                response["action"] = forced
                response["reason"] = "force_next_action"
                return response
            else:
                self.force_next_action = None
                response["action"] = forced
                response["reason"] = "force_next_action"
                return response

        if tool_name == "search_action":
            if not has_search_bar:
                response["blocked"] = True
                response["reason"] = "search_not_available"
                return response
            # H2: Strip price/budget text from search queries — WebShop search
            # doesn't filter by price, so price words waste keyword slots.
            cleaned = re.sub(
                r'\b(price\s+(less|lower|under|below)\s+than\s+\$?\d+[\d.,]*'
                r'|under\s+\$\d+[\d.,]*'
                r'|less\s+than\s+\$?\d+[\d.,]*\s*(dollars?)?'
                r'|lower\s+than\s+\$?\d+[\d.,]*\s*(dollars?)?'
                r'|price\s+lower\s+than\s+\$?\d+[\d.,]*\s*(dollars?)?'
                r'|\$\d+[\d.,]*\s*(dollars?)?'
                r'|price\s+\S*\s*\d+[\d.,]*\s*(dollars?)?'
                r'|\d+[\d.,]*\s*dollars?)\b',
                '', raw_value, flags=re.IGNORECASE
            ).strip()
            # Also strip trailing price filler
            cleaned = re.sub(r'\s+(price|budget|cost|dollars?)\s*$', '', cleaned, flags=re.IGNORECASE).strip()
            if cleaned:
                raw_value = cleaned
            response["action"] = f"search[{raw_value}]"
            return response

        if tool_name == "click_action":
            clickables_lower = [c.lower() for c in clickables]
            value_l = raw_value.strip().lower()

            # Exact match (case-insensitive)
            if value_l in clickables_lower:
                idx = clickables_lower.index(value_l)
                final_value = clickables[idx]
            else:
                # Fuzzy match
                match = _fuzzy_match_click(raw_value, clickables, self.config.h2_click_similarity_threshold)
                if match:
                    final_value = match
                    response["canonicalized"] = True
                else:
                    response["blocked"] = True
                    response["reason"] = f"click_value_not_in_admissible: {raw_value!r}"
                    return response

            # H2 buy-now pre-check: block premature buy if required attributes are
            # available on this page but not yet selected.
            if final_value.lower() == "buy now" and self.config.h2_enabled:
                block_msg = self._buy_now_precheck(clickables)
                if block_msg:
                    response["blocked"] = True
                    response["reason"] = "buy_now_missing_attrs"
                    response["block_message"] = block_msg
                    return response

            # Repeat-click detection
            if final_value.lower() == (self._last_click or "").lower():
                self._repeat_click_count += 1
                if self._repeat_click_count >= self.config.h2_repeat_click_block_after:
                    # Never block navigation clicks
                    if final_value.lower() not in {"back to search", "next >", "< prev", "buy now"}:
                        response["blocked"] = True
                        response["reason"] = f"repeat_click_blocked: {final_value!r}"
                        return response
            else:
                self._last_click = final_value
                self._repeat_click_count = 1

            response["action"] = f"click[{final_value}]"
            return response

        # Unknown tool
        response["blocked"] = True
        response["reason"] = f"unknown_tool: {tool_name}"
        return response

    def _buy_now_precheck(
        self,
        clickables: List[str],
        *,
        record_block: bool = True,
        respect_safety_valve: bool = True,
    ) -> Optional[str]:
        """H2 buy-now guard: return a block message if required attributes are
        available on this page but not yet selected, otherwise return None.

        Only fires when the attribute IS in the page's clickables (admissible).
        Never blocks when attribute is unavailable — in that case the agent
        should buy or go back; we let H5 guide that decision.
        """
        if self.requirements is None:
            return None
        req = self.requirements
        state = self.page_state
        if state.page_type != PAGE_PRODUCT_DETAIL:
            return None

        # Safety valve: after too many blocks this episode, let buy now through
        # to prevent the agent from getting stuck in an infinite retry loop.
        if respect_safety_valve and self._total_buy_now_blocks >= 4:
            return None

        to_click: List[str] = []

        # Color check (handles OR alternatives: e.g. "lavender or ivory")
        if req.color:
            sel = state.selected_attributes.get("color", "").lower()
            colors_needed = [req.color]
            if req.color_alt:
                colors_needed.append(req.color_alt)
            # Selected if ANY of the allowed colors is selected
            is_sel = sel and any(
                sel == c.lower() or c.lower() in sel or sel in c.lower()
                for c in colors_needed
            )
            if not is_sel:
                opts = state.attribute_options.get("color", [])
                # Try to find a match for any accepted color
                match = None
                for c in colors_needed:
                    match = _match_option_in_list(c, opts)
                    if match:
                        break
                if match:
                    # If the agent already selected the best available match, don't block
                    if sel and sel == match.lower():
                        pass  # already selected the best option
                    else:
                        color_desc = f"{req.color} or {req.color_alt}" if req.color_alt else req.color
                        to_click.append(f"click '{match}' for color ({color_desc})")

        # Size check
        if req.size:
            sel = state.selected_attributes.get("size", "").lower()
            is_sel = sel and (sel == req.size.lower() or req.size.lower() in sel or sel in req.size.lower())
            if not is_sel:
                opts = state.attribute_options.get("size", [])
                match = _match_option_in_list(req.size, opts)
                if match:
                    # If the agent already selected the best available match, don't block
                    if sel and sel == match.lower():
                        pass
                    else:
                        to_click.append(f"click '{match}' for size")

        # Spec check
        spec_opts = (
            state.attribute_options.get("spec", [])
            + state.attribute_options.get("other", [])
        )
        for spec_req in req.specs:
            sel_spec = state.selected_attributes.get("spec", "").lower()
            if sel_spec and spec_req.lower() in sel_spec:
                continue
            match = _match_option_in_list(spec_req, spec_opts)
            if match:
                # If the agent already selected the best available match, don't block
                if sel_spec and sel_spec == match.lower():
                    continue
                to_click.append(f"click '{match}' for {spec_req}")

        # Defensive attribute guard: if any attribute group has options but
        # nothing is selected, block buy-now even if H0 didn't parse a
        # requirement for that group. This catches cases where the instruction
        # doesn't mention color/size but the product requires a selection.
        # Extends to "other" group (catches alphanumeric codes like "c08" and
        # slug-style names like "greenwithoutlogo" when observation-based parsing
        # places them correctly under "color", and also "other" variants).
        # Allow through after 2 defensive blocks to prevent infinite loops.
        if not to_click and (
            not respect_safety_valve or self._defensive_buy_blocks < 2
        ):
            any_selected = any(v for v in state.selected_attributes.values())
            # Only fire defensive guard when agent hasn't selected ANY attribute yet.
            # If any selection was made, trust the agent is on the right track.
            if not any_selected:
                unselected_groups = []
                for group in ("color", "size"):
                    opts = state.attribute_options.get(group, [])
                    sel = state.selected_attributes.get(group, "")
                    if opts and not sel:
                        unselected_groups.append((group, opts))
                other_def_opts = state.attribute_options.get("other", [])
                if other_def_opts and not state.selected_attributes.get("other", ""):
                    unselected_groups.append(("attribute", other_def_opts))
                if unselected_groups:
                    if record_block:
                        self._defensive_buy_blocks += 1
                    for group, opts in unselected_groups:
                        to_click.append(f"select a {group} (options: {', '.join(opts[:4])})")

        if not to_click:
            return None

        if record_block:
            self._total_buy_now_blocks += 1
        instructions = "; ".join(to_click[:3])
        return (
            f"You must select required attributes before buying. "
            f"Please: {instructions}. Then click 'buy now'."
        )

    # ── H1 (called after env.step) ───────────────────────────────────────────

    def update_state(
        self,
        action: Optional[str],
        observation: str,
        has_search_bar: bool,
        clickables: List[str],
    ) -> None:
        """H1: Update page state after each environment step."""
        update_page_state(self.page_state, action, observation, has_search_bar, clickables)

    # ── H4 ───────────────────────────────────────────────────────────────────

    def post_step_monitor(
        self,
        action: Optional[str],
        observation: str,
        has_search_bar: bool,
        clickables: List[str],
    ) -> Dict[str, Any]:
        """
        H4: Shopping state monitor. Returns a recovery_prompt if intervention needed,
        otherwise returns empty dict.
        """
        response: Dict[str, Any] = {
            "audit_reason": "",
            "intervention_level": "none",
            "recovery_prompt": None,
        }
        self._h4_intervened = False  # reset each step
        if not self.config.h4_enabled or self.requirements is None:
            return response

        state = self.page_state
        req = self.requirements

        # ⓪ H4 product-category check — fires once per new ASIN (stall_turns==1).
        # Warns agent when product title doesn't match the task's item type,
        # before it wastes turns selecting attributes on a completely wrong product.
        if (
            state.page_type == PAGE_PRODUCT_DETAIL
            and state.current_asin is not None
            and state._stall_turns == 1  # first turn on this product page
        ):
            title_warn = _product_title_check(req, observation)
            if title_warn:
                self._product_mismatch_warned = True
                self._mismatch_buy_blocks = 0
                self._h4_intervened = True
                response["audit_reason"] = "product_category_mismatch"
                response["intervention_level"] = "soft"
                response["recovery_prompt"] = title_warn
                return response
            else:
                # Product title matches — clear any previous mismatch flag
                self._product_mismatch_warned = False

        # ① Search loop detection
        if (
            state.back_to_search_count >= self.config.h4_search_loop_threshold
            or self._duplicate_search_count() >= self.config.h4_duplicate_search_threshold
        ):
            self._search_loop_resets += 1
            # Build a simplified search suggestion using task_keywords (already filtered,
            # no stopwords or sentence fragments) for better query quality.
            simplified = list(req.task_keywords[:5])
            if req.color and req.color not in simplified:
                simplified.insert(0, req.color)
            if req.size and req.size not in simplified:
                simplified.append(req.size)
            if req.price_max:
                simplified.append(f"under ${req.price_max:.0f}")
            hint_query = " ".join(simplified[:6])
            response["audit_reason"] = "search_loop"
            response["intervention_level"] = "soft"
            if self._search_loop_resets >= 2:
                # Second+ loop reset: hard escape — the perfect product may not exist.
                # Instruct agent to pick the closest match from current results.
                response["recovery_prompt"] = (
                    f"Harness: still searching after many attempts. "
                    f"Stop searching — pick the best matching product from the current results "
                    f"and click buy now. A close match is acceptable."
                )
            else:
                response["recovery_prompt"] = (
                    f"Harness: you have searched many times without success. "
                    f"Try a simpler query, e.g. '{hint_query}', or pick any close match and buy it."
                )
            # Reset both counters so this doesn't fire on every subsequent step.
            # Keep only the last query so _duplicate_search_count() starts from 1.
            state.back_to_search_count = 0
            if state.search_queries:
                state.search_queries = state.search_queries[-1:]
            self._h4_intervened = True
            return response

        # ② Price over budget (on product page) — fires at most once per ASIN to
        # prevent repeat-fire loops (agent gets stuck between price-warn and
        # search-block when search isn't available on product pages).
        asin_for_price = state.current_asin or ""
        if (
            state.page_type == PAGE_PRODUCT_DETAIL
            and state.current_price is not None
            and req.price_max is not None
            and state.current_price > req.price_max * (1 + self.config.price_tolerance)
            and asin_for_price not in self._price_warned_asins
        ):
            self._price_warned_asins.add(asin_for_price)
            response["audit_reason"] = "price_over_budget"
            response["intervention_level"] = "soft"
            response["recovery_prompt"] = (
                f"Harness: this product costs ${state.current_price:.2f} but your budget is "
                f"${req.price_max:.0f}. Click 'back to search' and find a cheaper option."
            )
            self._h4_intervened = True
            return response

        # ③ Product page stall
        if (
            state.page_type == PAGE_PRODUCT_DETAIL
            and state._stall_turns >= self.config.h4_product_stall_turns
        ):
            asin_key = state.current_asin or "unknown"
            self._asin_stall_count[asin_key] = self._asin_stall_count.get(asin_key, 0) + 1
            stall_count = self._asin_stall_count[asin_key]
            response["audit_reason"] = "product_stall"
            response["intervention_level"] = "soft"
            if stall_count >= 2:
                # Second stall on same product: escalate to a hard "decide now" message.
                # The agent clearly can't make progress — force a binary choice.
                response["recovery_prompt"] = (
                    "Harness: you have been on this product page too long. "
                    "Make a decision NOW: click 'buy now' if it matches the task, "
                    "or click 'back to search' to try a different product."
                )
            else:
                checklist = _build_attribute_checklist(req, state, clickables)
                response["recovery_prompt"] = (
                    f"Harness: you seem stuck on this product page.\n{checklist}"
                    if checklist else
                    "Harness: you seem stuck. Click 'buy now' to purchase or 'back to search' to try another product."
                )
            state._stall_turns = 0  # reset so we don't fire every turn
            self._h4_intervened = True
            return response

        return response

    def _duplicate_search_count(self) -> int:
        """Count max occurrences of any single search query."""
        if not self.page_state.search_queries:
            return 0
        from collections import Counter
        counts = Counter(self.page_state.search_queries)
        return max(counts.values())

    # ── H5 ───────────────────────────────────────────────────────────────────

    def step_guidance(
        self,
        step_num: int,
        max_steps: int,
        observation: str,
        has_search_bar: bool,
        clickables: List[str],
    ) -> Optional[str]:
        """H4-E: State-driven per-step guidance using H1 page state and parsed requirements."""
        if not self.config.h4_enabled or self.requirements is None:
            return None
        # Skip if post_step_monitor already injected a recovery prompt this turn
        if self._h4_intervened:
            return None

        state = self.page_state
        req = self.requirements
        hint: Optional[str] = None

        if state.page_type == PAGE_HOME:
            if step_num == 0:
                # Suggest initial search query
                parts = [req.item_keywords[:30]]
                if req.color:
                    parts.append(req.color)
                if req.size:
                    parts.append(req.size)
                if req.material:
                    parts.append(req.material)
                query_suggestion = " ".join(parts)
                hint = f"Hint: search for '{query_suggestion}' to find a matching product."

        elif state.page_type == PAGE_SEARCH_RESULTS:
            summary = requirements_summary(req)
            next_page_available = "next >" in [c.lower() for c in clickables]

            # Product ranking: suggest best match from search results
            product_suggestion = _rank_search_results(observation, req, clickables)

            if state.back_to_search_count >= 2:
                if next_page_available:
                    hint = (
                        f"Hint: you've searched several times. "
                        f"Try clicking 'next >' to see more results before searching again. "
                        f"Requirements: {summary}."
                    )
                else:
                    hint = (
                        f"Hint: you've searched several times. Try different keywords or select "
                        f"a close match. Requirements: {summary}."
                    )
            elif req.price_max is not None:
                hint = (
                    f"Hint: check prices before clicking (budget: ${req.price_max:.0f}). "
                    f"Look for a product matching: {summary}."
                )
            else:
                hint = f"Hint: select a product matching: {summary}."

            # Append product suggestion if available
            if product_suggestion:
                hint += f"\n{product_suggestion}"

            if next_page_available and state.back_to_search_count >= 1 and not product_suggestion:
                hint += "\nNo good match on this page? Click 'next >' to browse more results."

        elif state.page_type == PAGE_PRODUCT_DETAIL:
            checklist = _build_attribute_checklist(req, state, clickables)
            if checklist:
                hint = f"Hint: {checklist}"
                # Auto-force buy now: when checklist says all-ok and buy now is
                # available, queue force so the NEXT turn auto-buys if agent
                # produces text instead of clicking buy now.
                buy_now_in = "buy now" in [c.lower() for c in clickables]
                if buy_now_in and ("All checked" in checklist or "All attributes selected" in checklist):
                    self.force_next_action = "click[buy now]"

        if hint is None:
            return None

        # Word budget cap
        words = hint.split()
        if len(words) > self.config.h4_hint_max_words:
            hint = " ".join(words[:self.config.h4_hint_max_words])

        # Dedup: suppress if page type and content unchanged
        if hint == self._last_hint and state.page_type == self._last_hint_page:
            return None

        self._last_hint = hint
        self._last_hint_page = state.page_type
        return hint

    # ── H4-D ─────────────────────────────────────────────────────────────────

    def budget_check(
        self,
        remaining_steps: int,
        clickables: List[str],
    ) -> Dict[str, Any]:
        """H4-D: Step-budget management — urgency hint and forced Buy Now."""
        response: Dict[str, Any] = {"hint": None, "force_action": None}
        if not self.config.h4_enabled or self.requirements is None:
            return response

        state = self.page_state
        clickables_lower = [c.lower() for c in clickables]
        buy_now_available = "buy now" in clickables_lower

        # Hard force: buy now in admissible and steps critical
        if remaining_steps < self.config.h4_force_threshold and buy_now_available:
            response["force_action"] = "click[buy now]"
            response["hint"] = (
                f"[{remaining_steps} steps left] Harness: forcing 'buy now' to complete purchase."
            )
            return response

        # Soft warn: on search results page with few steps left
        if (
            remaining_steps < self.config.h4_warn_threshold
            and state.page_type in (PAGE_SEARCH_RESULTS, PAGE_HOME)
        ):
            response["hint"] = (
                f"[{remaining_steps} steps left] Urgently: click a product and buy it now."
            )

        # Soft warn: on product page, buy now available, few steps
        if (
            remaining_steps < self.config.h4_warn_threshold
            and state.page_type == PAGE_PRODUCT_DETAIL
            and buy_now_available
        ):
            response["hint"] = (
                f"[{remaining_steps} steps left] Buy now is available — click 'buy now'."
            )

        return response
