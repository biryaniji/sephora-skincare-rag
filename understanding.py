"""Query understanding: turn whatever the shopper typed (plus the conversation so far) into a
structured search request.

Two layers:
  1. Rules (always on, free, instant): prices in many phrasings and currencies, skin type,
     greetings, routines, "cheapest", follow-up detection, skincare vocabulary.
  2. LLM analyzer (when a model is connected): reads the chat history and returns JSON with a
     standalone question and constraints. It handles phrasings the rules miss
     ("won't break the bank", "something for my pimples", "same but for night").
Rules fill in anything the LLM leaves out, and the newest message always wins on conflicts.
"""
import json
import re

INR_PER_USD = 83.0          # rough rate, only to translate "under ₹2000" into the catalogue's USD prices
PRODUCT_TYPES = ["cleanser", "toner", "serum", "moisturizer", "sunscreen", "eye", "mask", "exfoliant",
                 "oil", "mist", "lip", "self_tanner"]
SKIN_TYPES = ["dry", "oily", "combination", "normal", "sensitive"]

_CUR_BEFORE = r"(?:\$|usd\s*|rs\.?\s*|inr\s*|₹\s*)"
_CUR_AFTER = r"(?:\s*(?:\$|usd|dollars?|bucks|rs\.?|rupees?|inr|₹))"
_NUM = r"(\d+(?:[.,]\d+)?)\s*(k)?"
_AMOUNT = rf"({_CUR_BEFORE})?\s*{_NUM}({_CUR_AFTER})?"

_MAX_WORDS = (r"under|below|less than|cheaper than|within|max(?:imum)?|up ?to|no more than|not more than|"
              r"at most|lower than|<=?|budget(?::| is| of)?(?: only| just)?(?: around| about| roughly)?|"
              r"(?:i )?(?:only |just )?have(?: only| just)?|can (?:only )?(?:spend|afford)|"
              r"spend(?: up to| at most| around)?|afford")
_MIN_WORDS = r"over|above|more than|at least|min(?:imum)?|>=?|starting (?:at|from)"
_AROUND_WORDS = r"around|about|roughly|approximately|~"

GREETING = re.compile(r"(?i)^\s*(hi+|hello+|hey+|hiya|yo|namaste|good (morning|afternoon|evening)|thanks?( a lot| so much)?"
                      r"|thank (you|u)( so much| a lot| very much)?|thx|ty|ok(ay)?|cool|great|nice|awesome|perfect|bye|goodbye|see you)"
                      r"[\s,!.?]*(there|claude|bot|advisor|buddy)?[\s!.?]*$")
SKINCARE_VOCAB = re.compile(
    r"(?i)\b(skin\w*|acne|pimples?|zits?|breakouts?|blemish\w*|blackheads?|whiteheads?|pores?|wrinkles?|fine lines|"
    r"aging|anti-?aging|dark spots?|hyperpigment\w*|pigmentation|redness|dull\w*|glow\w*|hydrat\w*|moistu?ri\w*|"
    r"oily|dry|sensitive|combination|spf|sunscreens?|sunblock|serums?|toners?|cleansers?|face ?wash|retinol|"
    r"retinoid|niacinamide|hyaluronic|vitamin c|salicylic|glycolic|lactic|azelaic|aha|bha|peptides?|ceramides?|"
    r"squalane|exfoliat\w*|eye creams?|dark circles|puff\w*|routine|regimen|face|facial|lips?|masks?|creams?|"
    r"lotions?|balms?|ingredients?|tan\w*|sephora|products?)\b")
ROUTINE = re.compile(r"(?i)\b(routine|regimen|skin ?care steps|am (and|&) pm|morning (and|&) night|step[- ]by[- ]step|"
                     r"full set of products|starter kit for)\b")
NIGHT_ONLY = re.compile(r"(?i)\b(night|pm|evening|bedtime)\b")
MORNING_ONLY = re.compile(r"(?i)\b(morning|am)\b")
CHEAPEST = re.compile(r"(?i)\b(cheap(est|er)?|lowest[- ]priced?|least expensive|most affordable|inexpensive|"
                      r"budget[- ]friendly|affordable|won'?t break the bank)\b")
REFERS_BACK = re.compile(r"(?i)\b(those|these|them|they|it|its|it's|this one|that one|the (first|second|third|"
                         r"last|other|same|cheaper|cheapest|previous) (one|options?|products?)?|which one|"
                         r"of (them|those|these)|either|both)\b")
FOLLOW_UP_LEAD = re.compile(r"(?i)^\s*(and|also|what about|how about|same (but|for|thing)|instead|now|ok(ay)?,? (and|now|what)|"
                            r"then|but|any (other|cheaper|more)|anything (else|cheaper|similar)|show me (more|others|cheaper))\b")
SKIN_TYPE_RE = re.compile(r"(?i)\b(dry|oily|combination|combo|normal|sensitive)(?:[,/ ]+(?:and |& )?"
                          r"(?:acne[- ]prone|dehydrated|mature))?\s+skin\b|\bi have (dry|oily|combination|combo|"
                          r"normal|sensitive)\b|\bmy skin is (dry|oily|combination|combo|normal|sensitive)\b")


# ---------------------------------------------------------------- rules
def _to_usd(value: str, k: str | None, cur_before: str | None, cur_after: str | None) -> float:
    amount = float(value.replace(",", "")) * (1000 if k else 1)
    cur = f" {cur_before or ''} {cur_after or ''} ".lower()
    if re.search(r"\b(rs\.?|inr|rupees?)(?=[\s.]|$)|₹", cur):
        amount = amount / INR_PER_USD
    return round(amount, 2)


def parse_price(text: str) -> dict:
    """Price constraints in USD from many phrasings:
    'under $20', 'budget of 20$', 'i only have 20 bucks', '20 dollars max', '$20 or less',
    'between 15 and 30', '15-30$', 'around $25', 'over $50', 'under ₹2000'."""
    q = text.lower().replace("’", "'")
    out: dict = {}
    m = re.search(rf"(?:between|from)\s*{_AMOUNT}\s*(?:and|to|-|–)\s*{_AMOUNT}", q) or \
        re.search(rf"{_AMOUNT}\s*(?:-|–|to)\s*{_AMOUNT}", q)
    if m:
        g = m.groups()
        lo, hi = _to_usd(g[1], g[2], g[0], g[3]), _to_usd(g[5], g[6], g[4] or g[0], g[7] or g[3])
        # only treat "15-30" as a price range when there's a currency or a price word nearby
        if g[0] or g[3] or g[4] or g[7] or re.search(r"\b(budget|price|cost|spend|dollars?|bucks|between)\b", q):
            out["min_price"], out["max_price"] = sorted([lo, hi])
            return out
    m = re.search(rf"(?:{_MAX_WORDS})\s*{_AMOUNT}", q)
    if m and (m.group(1) or m.group(4) or not re.match(r"\s*(?:spf|%|years?|yo\b|y/o|reviews?)", q[m.end():])):
        value = _to_usd(m.group(2), m.group(3), m.group(1), m.group(4))
        if not re.search(r"(?:have|spend|afford)\s*$", q[:m.start(1) if m.group(1) else m.start(2)]) or \
                m.group(1) or m.group(4):
            out["max_price"] = value
    if "max_price" not in out:
        m = re.search(rf"{_AMOUNT}\s*(?:or less|or under|and under|max(?:imum)?|tops|budget|at most|limit)\b", q)
        if m and (m.group(1) or m.group(4) or "budget" in q[m.end() - 7:m.end()]):
            out["max_price"] = _to_usd(m.group(2), m.group(3), m.group(1), m.group(4))
    if "max_price" not in out:
        m = re.search(rf"(?:{_AROUND_WORDS})\s*{_AMOUNT}", q)
        if m and (m.group(1) or m.group(4)):
            out["max_price"] = round(_to_usd(m.group(2), m.group(3), m.group(1), m.group(4)) * 1.15, 2)
    if "max_price" not in out and re.search(r"\bbudget\b", q):
        m = re.search(rf"({_CUR_BEFORE})\s*{_NUM}|{_NUM}({_CUR_AFTER})", q)
        if m:
            g = m.groups()
            out["max_price"] = _to_usd(g[1], g[2], g[0], None) if g[1] else _to_usd(g[3], g[4], None, g[5])
    m = None
    for cand in re.finditer(rf"(?:{_MIN_WORDS})\s*{_AMOUNT}", q):
        if not re.search(r"\b(no|not)\s*$", q[:cand.start()]):
            m = cand
            break
    m = m or re.search(rf"{_AMOUNT}\s*(?:or more|and up|\+)", q)
    price_words = re.search(r"\b(budget|price[ds]?|cost|spend|dollars?|bucks|expensive|splurge|luxury)\b|[$₹]", q)
    if m and (m.group(1) or m.group(4) or price_words):
        if not re.match(r"\s*(?:spf|%|reviews?|stars?|years?|yo\b|y/o)", q[m.end():]):
            out["min_price"] = _to_usd(m.group(2), m.group(3), m.group(1), m.group(4))
    if out.get("min_price") and out.get("max_price") and out["min_price"] > out["max_price"]:
        out.pop("min_price")
    return out


def skin_type_in(text: str) -> str | None:
    m = SKIN_TYPE_RE.search(text)
    if not m:
        return None
    word = next(g for g in m.groups() if g).lower()
    return "combination" if word == "combo" else word


def is_small_talk(text: str) -> bool:
    return bool(GREETING.match(text))


def small_talk_reply(text: str) -> str:
    low = text.lower()
    if re.search(r"thank|thx|\bty\b", low):
        return ("You're welcome! Ask me anything else about Sephora skincare: another product type, "
                "a different budget, or what reviewers with your skin type say.")
    if re.search(r"bye|see you", low):
        return "Bye! Come back any time you need help choosing skincare."
    if re.search(r"\b(ok|okay|cool|great|nice|awesome|perfect)\b", low):
        return "Great! Want me to narrow it down further, by budget, skin type or ingredients?"
    return ("Hi! I'm your Sephora skincare advisor. Tell me your **skin type**, **concern** and **budget**, "
            "for example: *a gentle cleanser for oily, acne-prone skin under $25*, or ask about any product's "
            "ingredients and reviews.")


def routine_steps(text: str, types: list[str] | None = None) -> list[str] | None:
    if not ROUTINE.search(text):
        return None
    if types and len(types) >= 2:
        return types
    if NIGHT_ONLY.search(text) and not MORNING_ONLY.search(text):
        return ["cleanser", "serum", "moisturizer"]
    return ["cleanser", "serum", "moisturizer", "sunscreen"]


def looks_constraint_only(text: str, has_type_or_brand: bool) -> bool:
    """'i just have budget of 20$', 'for oily skin', 'vegan please' -> refines the previous request."""
    if has_type_or_brand:
        return False
    words = re.findall(r"[a-z0-9$₹]+", text.lower())
    has_constraint = bool(parse_price(text) or skin_type_in(text) or
                          re.search(r"(?i)\b(vegan|clean|fragrance|unscented|cheap\w*|affordable|"
                                    r"budget|spf|price)\b", text))
    return has_constraint or (len(words) <= 5 and bool(SKINCARE_VOCAB.search(text)))


# ---------------------------------------------------------------- LLM analyzer
ANALYZER_PROMPT = """You turn a shopper's latest chat message into a search request for Sephora's skincare catalogue.
The message may be in English, Hindi (Devanagari) or Hinglish (Hindi in Roman letters). Whatever the language, write "standalone_question" in plain English, translating Hindi skincare words (e.g. तैलीय = oily, शुष्क/ड्राई = dry, मुंहासे = acne, झुर्रियाँ = wrinkles, फेस वॉश = face wash, सनस्क्रीन = sunscreen) and keep product and brand names as written.
Use the conversation to resolve follow-ups: keep the product type, skin type, budget and other constraints from earlier turns unless the shopper changes them. The newest message wins on conflicts.
Return ONLY a JSON object with these keys:
{"standalone_question": "the full request as one self-contained sentence, including carried-over constraints",
 "is_skincare": true or false (false only when clearly unrelated to skincare or beauty shopping),
 "is_small_talk": true or false (greetings, thanks, chit-chat with no request),
 "product_types": list using only: cleanser, toner, serum, moisturizer, sunscreen, eye, mask, exfoliant, oil, mist, lip, self_tanner,
 "skin_type": one of dry, oily, combination, normal, sensitive, or "",
 "concerns": list of short phrases such as "acne", "dark spots", "redness", "dryness", "wrinkles",
 "max_price_usd": number or null, "min_price_usd": number or null,
 "vegan": true or false, "fragrance_free": true or false, "clean": true or false,
 "routine": true or false (true when they want several products that work together),
 "sort": "relevance", "price_low" or "rating"}
Prices are in US dollars. Convert rupees at about 83 INR per USD. "Cheap" or "budget-friendly" without a number means sort "price_low", not a price limit."""


def parse_analysis(text: str) -> dict | None:
    """Validate the analyzer's JSON; anything malformed is dropped rather than trusted."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    out: dict = {}
    sq = raw.get("standalone_question")
    if isinstance(sq, str) and 3 <= len(sq.strip()) <= 400:
        out["standalone_question"] = sq.strip()
    for key in ("is_skincare", "is_small_talk", "vegan", "fragrance_free", "clean", "routine"):
        if isinstance(raw.get(key), bool):
            out[key] = raw[key]
    types = raw.get("product_types") or []
    if isinstance(types, list):
        valid = [t for t in (str(x).lower().strip() for x in types) if t in PRODUCT_TYPES]
        if valid:
            out["product_types"] = valid
    st = str(raw.get("skin_type") or "").lower().strip()
    if st in SKIN_TYPES:
        out["skin_type"] = st
    concerns = raw.get("concerns") or []
    if isinstance(concerns, list) and any(str(c).strip() for c in concerns):
        out["concerns"] = [str(c)[:40] for c in concerns if str(c).strip()][:5]
    for src, dst in (("max_price_usd", "max_price"), ("min_price_usd", "min_price")):
        v = raw.get(src)
        if isinstance(v, (int, float)) and 0 < v < 5000:
            out[dst] = float(v)
    if raw.get("sort") in ("price_low", "rating"):
        out["sort"] = raw["sort"]
    return out


# ---------------------------------------------------------------- spelling
SHORT_FIXES = {"skn": "skin", "fac": "face", "fce": "face", "spf30": "spf 30", "moist": "moisturizer",
               "moisturiser": "moisturizer", "moisturisers": "moisturizers", "facewash": "face wash",
               "sunscrn": "sunscreen", "vit": "vitamin", "c-serum": "vitamin c serum", "hyaluronic": "hyaluronic",
               "recomend": "recommend", "reccomend": "recommend", "reccommend": "recommend", "pls": "please",
               "plz": "please", "u": "you", "ur": "your", "wanna": "want to", "gimme": "give me"}
DOMAIN_WORDS = ["cleanser", "cleansers", "toner", "toners", "serum", "serums", "moisturizer", "moisturizers",
                "sunscreen", "sunscreens", "exfoliator", "exfoliant", "mask", "masks", "face", "wash", "cream",
                "creams", "lotion", "gel", "balm", "essence", "mist", "eye", "lip", "retinol", "niacinamide",
                "hyaluronic", "salicylic", "glycolic", "azelaic", "lactic", "vitamin", "peptide", "peptides",
                "ceramide", "ceramides", "squalane", "bakuchiol", "acne", "pimples", "blackheads", "wrinkles",
                "hydrating", "brightening", "sensitive", "combination", "oily", "dry", "normal", "skin", "skincare",
                "fragrance", "vegan", "budget", "cheapest", "affordable", "recommend", "suggest", "routine",
                "pigmentation", "hyperpigmentation", "redness", "pores", "dullness", "sunburn", "dark", "spots",
                "circles", "puffiness", "anti-aging", "aging", "tinted", "mineral", "chemical", "gentle", "foaming",
                "exfoliating", "blackhead", "whiteheads", "sunblock", "moisturizing", "cleansing"]
# Real ingredient / product words that look like typos of other words: never "correct" them.
PROTECTED = {"retinal", "retinoid", "retinoids", "retinyl", "retinaldehyde", "creme", "crème", "lotion", "balm",
             "toning", "tinted", "parabens", "sulfates", "silicones", "sulfur", "zinc", "clay", "oils", "acids"}


class SpellFixer:
    """Fixes typos in the shopper's message before it's parsed and searched.
    Candidates are skincare terms, product-name words and brand words. A word is changed when it
    is close to a candidate (difflib similarity >= 0.8) and the candidate is used far more often in
    the catalogue than the word itself (so real but rare words, and brand names, are left alone).
    Core skincare terms win ties ("cleanzer" -> "cleanser", not "cleaner")."""

    def __init__(self, word_counts: dict[str, int], domain_words: set[str]):
        import difflib
        self._difflib = difflib
        self.counts = word_counts
        self.core = [w for w in DOMAIN_WORDS if len(w) >= 4]
        self.by_len: dict[int, list[str]] = {}
        for w in set(domain_words) | set(DOMAIN_WORDS):
            if len(w) >= 4 and self.counts.get(w, 0) >= 3 or w in DOMAIN_WORDS:
                self.by_len.setdefault(len(w), []).append(w)
        self.cache: dict[str, str] = {}

    def _best(self, low: str, pool: list[str], cutoff: float, ratio: int) -> str | None:
        m = self._difflib.get_close_matches(low, pool, n=3, cutoff=cutoff)
        own = self.counts.get(low, 0)
        for cand in m:
            if own == 0 or self.counts.get(cand, 0) >= ratio * own:
                return cand
        return None

    def fix_word(self, word: str) -> str:
        low = word.lower()
        if low in SHORT_FIXES:
            return SHORT_FIXES[low]
        if len(low) < 4 or not low.isalpha() or low in DOMAIN_WORDS or low in PROTECTED \
                or self.counts.get(low, 0) >= 200:
            return word
        if low not in self.cache:
            fixed = self._best(low, self.core, 0.8, ratio=5)
            if not fixed:
                pool = [w for n in range(len(low) - 2, len(low) + 3) for w in self.by_len.get(n, [])]
                fixed = self._best(low, pool, 0.84, ratio=20)
            self.cache[low] = fixed or word
        return self.cache[low]

    def fix(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Returns (corrected text, [(typo, fix), ...])."""
        changes = []

        def sub(m):
            w = m.group(0)
            fixed = self.fix_word(w)
            if fixed.lower() != w.lower():
                changes.append((w, fixed))
            return fixed
        return re.sub(r"[A-Za-z][A-Za-z'-]*", sub, text), changes
