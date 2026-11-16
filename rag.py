"""Step 3: hybrid retrieval (vector + BM25), guards and grounded answer generation.

Flow for one question
  0. Understand the message with the conversation (understanding.py): prices in any phrasing,
     skin type, routines, "cheapest", greetings. Follow-ups ("i only have $20", "which of those
     is cheapest?") inherit the previous request's constraints; a connected LLM turns the message
     into a standalone request (JSON). "Those / them" reuses the products from the last answer.
  1. Scope guards (no LLM call):
       * makeup / hair / fragrance questions -> "outside the skincare catalogue"
  2. Parse hard filters from the question -> metadata filter:
       price range, vegan, clean, fragrance-free, brand, product type
       (serum, moisturizer, sunscreen/SPF, eye, mask ...). Types are ANDed, so
       "moisturizer with SPF" = moisturizer AND sunscreen. Eye products and devices are
       excluded unless asked for.
  3. Vector search in Chroma (bge-small, cosine) -> top 20 chunks.
     BM25 keyword search over the same chunks   -> top 20 chunks.
     Reciprocal Rank Fusion (+ a boost when a distinctive product-name word such as
     "Protini" is in the question), max 2 chunks per product, top 8.
     2+ brands named -> one sub-search per brand (query decomposition).
     "Best rated" questions -> re-rank by a review-count-weighted rating.
     No match with the filters -> relax product type, then brand; never silently drop
     price / vegan constraints.
  4. Notes for the LLM when needed:
       * named brands that Sephora doesn't sell,
       * medical topics (pregnancy, prescriptions, skin conditions...).

Fallback (two layers)
  * Retrieval layer: if the best vector similarity is below SIMILARITY_THRESHOLD the
    question is off-topic for this catalogue, so we answer without calling the LLM.
  * Prompt layer: the LLM answers only from the numbered context, cites it, and says
    which part of the question the data doesn't cover.
"""
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

from llm import pick_backend
from understanding import (ANALYZER_PROMPT, CHEAPEST, FOLLOW_UP_LEAD, REFERS_BACK, SKINCARE_VOCAB,
                           is_small_talk, looks_constraint_only, parse_analysis, parse_price,
                           routine_steps, skin_type_in, small_talk_reply, SpellFixer)

CHROMA_DIR = "data/chroma"
CHUNKS_FILE = "data/processed/chunks.jsonl"
COLLECTION = "sephora_skincare"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction

CANDIDATES = 20        # per retriever
FINAL_CHUNKS = 8       # sent to the LLM
MAX_PER_PRODUCT = 2
RRF_K = 60
SIMILARITY_THRESHOLD = 0.65  # evaluate.py --calibrate (16 Nov 2026): off-topic max 0.614 (biryani),
                             # on-topic min 0.726; set below the midpoint so vague skincare questions pass
RATING_PRIOR_MEAN = 4.2      # Bayesian rating: pulls ratings with few reviews toward the average
RATING_PRIOR_WEIGHT = 50

DATA_NOTE = "Prices are USD from a March 2023 snapshot of sephora.com and may have changed."

FALLBACK_MESSAGE = ("I couldn't find that in the Sephora skincare catalogue data I have. "
                    "I can help with Sephora skincare products: ingredients, skin types, "
                    "concerns, prices, ratings and what reviewers say.")
SCOPE_MESSAGE = ("That sounds like {topic}, which isn't in my data: I only cover Sephora's **skincare** "
                 "catalogue (cleansers, serums, moisturizers, sunscreen, masks, eye care and similar). "
                 "Ask me about skincare and I'll help.")
NO_MATCH_MESSAGE = ("I couldn't find any skincare products in the catalogue that match {constraints}. "
                    "Try a higher budget or fewer requirements.")

SYSTEM_PROMPT = f"""You are the Sephora Skincare Advisor, a shopping assistant for Sephora's skincare catalogue.

Rules:
- Answer ONLY from the numbered product context in the user's message. Do not use outside knowledge about products, brands, prices, ingredients or what an ingredient does.
- Cite the context for every product fact with its number, like [2]. Quote product names, brands and prices exactly as they appear.
- If the context doesn't contain what is needed, say clearly that the catalogue data doesn't cover it. Don't guess. If only part of the question is covered, answer that part and say which part isn't covered.
- When recommending, give 2-4 options, each with one line on why it fits (skin type, concern, price, rating, what reviewers say).
- Ratings: always give the review count with a rating. Never call a product "best rated" or "top rated" when its rating comes from very few reviews; say so instead.
- Reviews are individual customers' experiences. Present them as "reviewers say", never as facts about the product. Reviews marked "received product free" were incentivized; mention that if you quote one.
- Medical topics (pregnancy, breastfeeding, prescriptions or medication, skin conditions such as eczema, rosacea or dermatitis, allergies, reactions): never say or imply a product is safe, unsafe or effective for the condition, even if a review says so. Say the catalogue can't answer that and suggest asking a dermatologist or doctor. You may still give factual catalogue details such as the ingredient list.
- If a note says a brand isn't sold in this catalogue, say that first, then offer comparable products from the context.
- {DATA_NOTE} Mention this when you quote prices.
- Respect the shopper's constraints (budget, skin type, vegan...). The context has already been filtered to them; if the notes say a filter was relaxed, say the match is approximate.
- For follow-up questions, answer about the products discussed before when they're in the context.
- If the shopper's request is vague, give your best picks from the context and ask one short question that would narrow it down (skin type, concern or budget).
- Keep answers short and scannable. Use bullet points for lists of products."""

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation, so 'Supergoop!' matches 'supergoop',
    'Kiehl's' matches 'kiehls' and 'Avène' matches 'avene'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("'", "")
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip() + " "


# ---------------------------------------------------------------- query understanding
# Product-type words -> type flags set in build_index.py. All matched types are ANDed.
QUERY_TYPES = [
    (r"\b(eyes?|under-?eye)\b", "eye"),
    (r"\bsunscreens?\b|\bspf\b|\bsun ?(block|screen|protection|cream)\b", "sunscreen"),
    (r"\bmoisturi[sz]ers?\b|\b(face|day|night) creams?\b|\bface lotions?\b", "moisturizer"),
    (r"\bserums?\b|\bampoules?\b", "serum"),
    (r"\bcleansers?\b|\bfa?ce? ?wash(es)?\b|\bfacewash(es)?\b|\bcleansing (balm|oil|gel|foam|milk)s?\b|\bmakeup removers?\b", "cleanser"),
    (r"\btoners?\b", "toner"),
    (r"\bexfoliat(or|ors|ant|ants)\b|\bpeels?\b|\bscrubs?\b", "exfoliant"),
    (r"\bmasks?\b", "mask"),
    (r"\b(face|facial) oils?\b", "oil"),
    (r"\b(face )?mists?\b|\bessences?\b", "mist"),
    (r"\blip (balm|mask|oil|treatment|scrub)s?\b", "lip"),
    (r"\bself[- ]?tan\w*|\bsunless\b", "self_tanner"),
]
FORMULA_TYPES = {"sunscreen", "moisturizer", "serum", "cleanser", "toner", "exfoliant", "mask", "oil", "mist", "lip"}
DEVICE_WORDS = re.compile(r"(?i)\b(device|tool|roller|gua sha|led|wand|brush|microcurrent)\b")

OUT_OF_SCOPE = [
    (r"\b(foundation|concealer|mascara|eyeliner|eye ?shadow|lipstick|lip gloss|blush|bronzer|highlighter|"
     r"setting spray|brow|false lashes|makeup palette)s?\b", "makeup"),
    (r"\b(shampoo|conditioner|hair (care|mask|oil|serum|spray|dye|colou?r|loss|growth|products?)|dry shampoo)\b",
     "hair care"),
    (r"\b(perfumes?|colognes?|eau de (parfum|toilette)|(signature|men'?s|women'?s) (scent|fragrance))\b", "fragrance"),
    (r"\bnail (polish|care|art)\b|\bnail polishes\b", "nail care"),
]

MEDICAL = re.compile(r"(?i)\b(pregnan\w*|breast-?feed\w*|nursing|trying to conceive|ttc|prescri\w*|accutane|"
                     r"isotretinoin|tretinoin|medicat\w*|antibiotic\w*|eczema|psoriasis|rosacea|dermatitis|"
                     r"allerg\w*|reaction|rash|infection|burns?|wound|melasma|cancer|melanoma|moles?|"
                     r"chemo\w*|safe (for|to use|with|during)|side effects?|interact\w*|diagnos\w*)\b")

BEST_RATED = re.compile(r"(?i)\b(best|highest|top)[- ]?(rated|reviewed|rating)\b|\bbest rating\b|\bmost (popular|loved)\b")

# Popular skincare brands that Sephora (in this 2023 snapshot) doesn't sell. Checked against
# the catalogue at startup, so a brand that IS in the data is never flagged.
UNSOLD_BRANDS = ["CeraVe", "La Roche-Posay", "Cetaphil", "Neutrogena", "Olay", "Bioderma", "Avène", "Avene",
                 "Eucerin", "Vichy", "Aveeno", "Garnier", "Nivea", "Simple", "Minimalist", "The Derma Co",
                 "Mamaearth", "Plum", "Dot & Key", "Cosrx", "COSRX", "Some By Mi", "Beauty of Joseon",
                 "Skin1004", "Anua", "Paula's Choice Korea", "Vaseline", "Clean & Clear", "Differin",
                 "Pond's", "Lakme", "Lakmé", "Biotique", "Himalaya", "Kama Ayurveda", "Forest Essentials"]
AMBIGUOUS_BRANDS = {"fresh", "simple", "plum", "glow", "clean"}   # also everyday words


@dataclass
class Retrieved:
    chunks: list[dict] = field(default_factory=list)
    best_similarity: float = 0.0
    filters: dict = field(default_factory=dict)


def parse_filters(question: str) -> dict:
    q = question.lower()
    filters: dict = dict(parse_price(question))
    if CHEAPEST.search(q):
        filters["sort"] = "price_low"
    elif BEST_RATED.search(q):
        filters["sort"] = "rating"
    if re.search(r"\bvegan\b", q):
        filters["vegan"] = True
    if re.search(r"\bclean (beauty|products?|skincare|formula)|\bclean at sephora\b", q):
        filters["clean"] = True
    if re.search(r"\b(fragrance|scent|perfume)[- ]?free\b|\bunscented\b|\bwithout (fragrance|scent)\b", q):
        filters["fragrance_free"] = True

    types = [t for pattern, t in QUERY_TYPES if re.search(pattern, q)]
    if types:
        filters["types"] = types
        if "eye" not in types and set(types) & FORMULA_TYPES:
            filters["exclude_eye"] = True     # "moisturizer" shouldn't return eye creams
    if (not types or set(types) & FORMULA_TYPES) and not DEVICE_WORDS.search(q):
        filters["exclude_devices"] = True
    return filters


def out_of_scope_topic(question: str) -> str | None:
    q = question.lower()
    if any(re.search(p, q) for p, _ in QUERY_TYPES):
        return None          # "SPF foundation" or "lip mask" still has a skincare angle
    for pattern, topic in OUT_OF_SCOPE:
        if re.search(pattern, q):
            return topic
    return None


def medical_terms(question: str) -> list[str]:
    return sorted({m.group(0).lower() for m in MEDICAL.finditer(question)})


def strip_medical(question: str) -> str:
    """Search for the product, not for reviews that happen to mention pregnancy etc.
    The medical part of the question is handled by the guard note instead."""
    q = MEDICAL.sub(" ", question)
    q = re.sub(r"(?i)\b(is it|is (this|that|the)|safe(ly)?|okay|ok|to use|use|using|during|while|when|"
               r"i'?m|am|my|if)\b", " ", q)
    q = re.sub(r"\s+", " ", re.sub(r"[?!.]", " ", q)).strip()
    return q if len(tokenize(q)) >= 2 else question


def matches(meta: dict, filters: dict) -> bool:
    if "max_price" in filters and meta["price"] > filters["max_price"]:
        return False
    if "min_price" in filters and meta["price"] < filters["min_price"]:
        return False
    for key, flag in (("vegan", "is_vegan"), ("clean", "is_clean"), ("fragrance_free", "is_fragrance_free")):
        if filters.get(key) and not meta.get(flag, False):
            return False
    if filters.get("brand") and meta["brand"] != filters["brand"]:
        return False
    for t in filters.get("types", []):
        if not meta.get(f"type_{t}", False):
            return False
    if filters.get("exclude_eye") and meta.get("type_eye"):
        return False
    if filters.get("exclude_devices") and meta.get("is_device"):
        return False
    return True


def chroma_where(filters: dict):
    conds = []
    if "max_price" in filters:
        conds.append({"price": {"$lte": filters["max_price"]}})
    if "min_price" in filters:
        conds.append({"price": {"$gte": filters["min_price"]}})
    for key, flag in (("vegan", "is_vegan"), ("clean", "is_clean"), ("fragrance_free", "is_fragrance_free")):
        if filters.get(key):
            conds.append({flag: True})
    if filters.get("brand"):
        conds.append({"brand": filters["brand"]})
    for t in filters.get("types", []):
        conds.append({f"type_{t}": True})
    if filters.get("exclude_eye"):
        conds.append({"type_eye": False})
    if filters.get("exclude_devices"):
        conds.append({"is_device": False})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def hard_constraints(filters: dict) -> str:
    parts = []
    if "min_price" in filters and "max_price" in filters:
        parts.append(f"a price between ${filters['min_price']:g} and ${filters['max_price']:g}")
    elif "max_price" in filters:
        parts.append(f"a price under ${filters['max_price']:g}")
    elif "min_price" in filters:
        parts.append(f"a price over ${filters['min_price']:g}")
    for key, label in (("vegan", "vegan"), ("clean", "Clean at Sephora"), ("fragrance_free", "fragrance-free")):
        if filters.get(key):
            parts.append(label)
    return " and ".join(parts) or "your requirements"


def bayes_rating(meta: dict) -> float:
    if not meta.get("has_rating", meta.get("rating", 0) > 0):
        return RATING_PRIOR_MEAN - 1
    v = meta.get("reviews", 0)
    return (v * meta["rating"] + RATING_PRIOR_WEIGHT * RATING_PRIOR_MEAN) / (v + RATING_PRIOR_WEIGHT)


def merge_preferences(filters: dict, preferences: dict | None) -> dict:
    """Sidebar shopping preferences act like words in the question (the question wins on price)."""
    if not preferences:
        return filters
    f = dict(filters)
    if preferences.get("max_price") and "max_price" not in f:
        f["max_price"] = float(preferences["max_price"])
    for key in ("vegan", "clean", "fragrance_free"):
        if preferences.get(key):
            f[key] = True
    return f


def preference_text(preferences: dict | None) -> str:
    if not preferences:
        return ""
    bits = []
    if preferences.get("skin_type"):
        bits.append(f"skin type {preferences['skin_type'].lower()}")
    if preferences.get("max_price"):
        bits.append(f"budget up to ${preferences['max_price']:g}")
    bits += [label for key, label in (("vegan", "vegan only"), ("clean", "Clean at Sephora only"),
                                       ("fragrance_free", "fragrance-free only")) if preferences.get(key)]
    return f"Shopper preferences: {', '.join(bits)}.\n" if bits else ""


COMMON_WORDS = {"night", "nights", "daily", "every", "full", "light", "super", "ultra", "power", "magic", "first",
                "night", "please", "something", "anything", "which", "would", "could", "about", "their", "there",
                "build", "budget", "cheap", "cheapest", "best", "better", "really", "great", "little", "after"}


# ---------------------------------------------------------------- the RAG system
class SephoraRAG:
    def __init__(self, collection=None, embedder=None, chunks: list[dict] | None = None):
        """collection / embedder / chunks can be injected for tests; by default they're loaded from disk."""
        from rank_bm25 import BM25Okapi

        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(EMBED_MODEL)
        if collection is None:
            import chromadb
            collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
        if chunks is None:
            with open(CHUNKS_FILE, encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f]
        self.embedder, self.collection, self.chunks = embedder, collection, chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.brands = sorted({c["meta"]["brand"] for c in chunks}, key=len, reverse=True)
        sold = {normalize(b) for b in self.brands}
        self.unsold = [b for b in UNSOLD_BRANDS if normalize(b) not in sold]
        self.product_count = len({c["meta"]["product_id"] for c in chunks})
        self.review_count = sum(c["meta"].get("reviews", 0) for c in chunks if c["meta"]["chunk_type"] == "overview"
                                and c["id"].endswith("-ov0"))
        self.backend = pick_backend()
        self.use_analyzer = True      # LLM query understanding (one extra short model call per message)
        # Rare words in product names ("protini", "niacinamide") identify a named product.
        names = {c["meta"]["product_id"]: set(tokenize(c["meta"]["product_name"])) for c in chunks}
        self.name_tokens = names
        df: dict[str, int] = {}
        for toks in names.values():
            for t in toks:
                df[t] = df.get(t, 0) + 1
        brand_words = {t for b in self.brands for t in tokenize(b)}
        self.rare_name_words = {t for t, n in df.items() if n <= 5 and len(t) >= 4 and t not in brand_words}
        # Stricter set for "does this message name a specific product?": no skincare vocabulary.
        self.product_name_words = {t for t in self.rare_name_words
                                   if df[t] <= 3 and len(t) >= 5 and not SKINCARE_VOCAB.fullmatch(t)
                                   and t not in COMMON_WORDS}
        # Spelling: typos are matched to skincare terms, product-name words and brand words.
        counts: dict[str, int] = {}
        for c in chunks:
            for t in tokenize(c["text"]):
                counts[t] = counts.get(t, 0) + 1
        domain = {t for toks in names.values() for t in toks if t.isalpha()} | brand_words
        self.speller = SpellFixer(counts, domain)

    # ---------- entity detection
    def _find_brands(self, question: str, names: list[str]) -> list[str]:
        q, found = normalize(question), []
        for b in names:
            nb = normalize(b)
            if len(nb.strip()) < 3 or nb not in q:
                continue
            if nb.strip() in AMBIGUOUS_BRANDS and not self._clearly_brand(question, nb.strip()):
                continue   # "fresh" as an adjective, not the brand fresh
            if not any(nb.strip() in normalize(f).strip() for f in found):
                found.append(b)
        return found

    @staticmethod
    def _clearly_brand(question: str, word: str) -> bool:
        """Everyday-word brands count only when used as a brand: 'by fresh', 'fresh's',
        'Fresh brand', or capitalised mid-sentence ('compare Fresh and Tatcha')."""
        low = question.lower().replace("’", "'")
        if re.search(rf"\b(by|from|brand)\s+{word}\b|\b{word}('s| brand)\b", low):
            return True
        return bool(re.search(rf"(?<=[a-z,] ){word.capitalize()}\b", question))

    def brands_in(self, question: str) -> list[str]:
        """Catalogue brands named in the question (longest first, so 'The INKEY List' beats 'INKEY')."""
        return self._find_brands(question, self.brands)

    def unsold_brands_in(self, question: str) -> list[str]:
        return self._find_brands(question, self.unsold)

    # ---------- retrieval
    def vector_search(self, question: str, filters: dict) -> tuple[list[tuple[str, float]], float]:
        qv = self.embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
        # Best similarity is measured without filters: it answers "is this on-topic at all?"
        unfiltered = self.collection.query(query_embeddings=[qv], n_results=1)
        best = 1 - unfiltered["distances"][0][0] if unfiltered["ids"][0] else 0.0
        res = self.collection.query(query_embeddings=[qv], n_results=CANDIDATES, where=chroma_where(filters))
        hits = [(cid, 1 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]
        return hits, best

    def keyword_search(self, question: str, filters: dict) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= CANDIDATES:
                break
            if matches(self.chunks[i]["meta"], filters):
                hits.append((self.chunks[i]["id"], float(scores[i])))
        return hits

    def retrieve(self, question: str, preferences: dict | None = None, base_filters: dict | None = None,
                 routine: list[str] | None = None) -> Retrieved:
        """Query decomposition:
          * routine ("AM routine for oily skin") -> one sub-search per step (cleanser, serum, ...),
          * comparison of 2+ brands -> one sub-search per brand, interleaved,
        so one part can't crowd out the others. Otherwise a single hybrid search."""
        filters = merge_preferences(dict(base_filters) if base_filters is not None else parse_filters(question),
                                    preferences)
        if preferences and preferences.get("skin_type") and preferences["skin_type"].lower() not in question.lower():
            question = f"{question} for {preferences['skin_type'].lower()} skin"
        if routine:
            return self._retrieve_routine(question, filters, routine)
        brands = self.brands_in(question)
        if len(brands) == 1:
            filters["brand"] = brands[0]
        if len(brands) < 2:
            return self._retrieve_one(question, filters, FINAL_CHUNKS)
        per_brand = max(2, FINAL_CHUNKS // len(brands))
        parts = [self._retrieve_one(question, {**filters, "brand": b}, per_brand) for b in brands]
        interleaved = [c for group in zip(*[p.chunks for p in parts]) for c in group]
        seen = {c["id"] for c in interleaved}
        leftovers = [c for p in parts for c in p.chunks if c["id"] not in seen]
        relaxed = sorted({r for p in parts for r in p.filters.get("relaxed", [])})
        merged = {**filters, "sub_queries": brands, **({"relaxed": relaxed} if relaxed else {})}
        return Retrieved(chunks=(interleaved + leftovers)[:FINAL_CHUNKS],
                         best_similarity=max(p.best_similarity for p in parts), filters=merged)

    def _retrieve_routine(self, question: str, filters: dict, steps: list[str]) -> Retrieved:
        per_step = max(2, FINAL_CHUNKS // len(steps))
        base = {k: v for k, v in filters.items() if k not in ("types", "exclude_eye")}
        if "max_price" in base:
            base.setdefault("sort", "price_low")    # a routine has to fit a budget: cheaper options first
        parts = []
        for step in steps:
            f = {**base, "types": [step], "exclude_eye": step != "eye", "overview_only": True}
            parts.append(self._retrieve_one(f"{step} {question}", f, per_step))
        chunks = [c for p in parts for c in p.chunks][:max(FINAL_CHUNKS, per_step * len(steps))]
        return Retrieved(chunks=chunks, best_similarity=max(p.best_similarity for p in parts),
                         filters={**base, "routine": steps})

    def _search(self, question: str, filters: dict):
        vec_hits, best = self.vector_search(question, filters)
        return vec_hits, self.keyword_search(question, filters), best

    def _retrieve_one(self, question: str, filters: dict, limit: int) -> Retrieved:
        filters = dict(filters)
        vec_hits, kw_hits, best = self._search(question, filters)
        # Too strict? Relax the soft filters (product type, then brand). Price, vegan, clean
        # and fragrance-free are hard constraints the shopper asked for: never relaxed.
        for soft in (("types", "exclude_eye"), ("brand",)):
            if vec_hits or kw_hits or not any(k in filters for k in soft):
                continue
            filters = {k: v for k, v in filters.items() if k not in soft}
            filters.setdefault("relaxed", []).append(soft[0])
            vec_hits, kw_hits, best = self._search(question, filters)

        fused: dict[str, float] = {}
        sources: dict[str, set] = {}
        for name, hits in (("vector", vec_hits), ("keyword", kw_hits)):
            for rank, (cid, _) in enumerate(hits):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                sources.setdefault(cid, set()).add(name)

        # Named-product boost: a distinctive word from a product name in the question.
        named = set(tokenize(question)) & self.rare_name_words
        if named:
            for cid in fused:
                if named & self.name_tokens[self.by_id[cid]["meta"]["product_id"]]:
                    fused[cid] += 1.0 / (RRF_K + 1)
                    sources[cid].add("name")
        order = sorted(fused, key=fused.get, reverse=True)
        if filters.get("overview_only"):
            order = [cid for cid in order if self.by_id[cid]["meta"]["chunk_type"] == "overview"]
        sort = filters.get("sort") or ("rating" if BEST_RATED.search(question) else None)
        pool, rest = order[:CANDIDATES], order[CANDIDATES:]
        if sort == "rating":
            # Among the relevant candidates, prefer high ratings backed by many reviews.
            order = sorted(pool, key=lambda cid: bayes_rating(self.by_id[cid]["meta"]), reverse=True) + rest
            filters["sort"] = "rating"
        elif sort == "price_low":
            # Cheapest first, but only among relevant candidates (top half of the fused list).
            relevant = pool[:max(8, len(pool) // 2)]
            order = sorted(relevant, key=lambda cid: self.by_id[cid]["meta"]["price"]) + \
                [c for c in order if c not in relevant]

        per_product: dict[str, int] = {}
        selected = []
        for cid in order:
            chunk = self.by_id[cid]
            pid = chunk["meta"]["product_id"]
            if per_product.get(pid, 0) >= MAX_PER_PRODUCT:
                continue
            per_product[pid] = per_product.get(pid, 0) + 1
            selected.append({**chunk, "score": round(fused[cid], 4), "found_by": " + ".join(sorted(sources[cid]))})
            if len(selected) >= limit:
                break
        return Retrieved(chunks=selected, best_similarity=round(best, 3), filters=filters)

    # ---------- generation
    def set_backend(self, preference: str | None = None, ollama_model: str | None = None):
        """Pick the LLM: 'auto' (Claude if ANTHROPIC_API_KEY, else local Ollama), 'claude', 'ollama' or 'none'."""
        self.backend = pick_backend(preference, ollama_model)
        return self.backend

    def llm_configured(self) -> bool:
        return self.backend is not None

    # ---------- understanding the request
    def analyze(self, question: str, history: list[dict]) -> dict | None:
        """LLM query analyzer: JSON with a standalone question and constraints (see understanding.py)."""
        if self.backend is None:
            return None
        convo = "\n".join(f"{t['role']}: {t['content'][:500]}" for t in history[-6:])
        msg = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Latest message: {question}"
        try:
            text = self.backend.complete(ANALYZER_PROMPT, [{"role": "user", "content": msg}],
                                         max_tokens=400, fast=True, json_mode=True)
        except Exception:
            return None
        return parse_analysis(text)

    def understand(self, question: str, history: list[dict], previous: dict | None = None) -> dict:
        """Work out what the shopper wants, using the conversation.
        Precedence: constraints in the latest message > the LLM analyzer > constraints carried over
        from the previous request."""
        prev_q = (previous or {}).get("search_question") or \
            next((t["content"] for t in reversed(history) if t["role"] == "user"), None)
        own = parse_filters(question)
        explicit = {k: v for k, v in own.items() if k not in ("exclude_devices", "exclude_eye")}
        has_anchor = bool(own.get("types") or self.brands_in(question) or
                          set(tokenize(question)) & self.product_name_words)
        refers_back = bool(prev_q and REFERS_BACK.search(question))
        follow_up = bool(prev_q) and (refers_back or bool(FOLLOW_UP_LEAD.search(question)) or
                                      looks_constraint_only(question, has_anchor))

        # Layer 3: rules. A follow-up inherits the previous request's constraints.
        if follow_up:
            filters, text = parse_filters(prev_q), f"{prev_q} {question}"
        else:
            filters, text = dict(own), question
        source = "rules + previous request" if follow_up else "rules"

        # Layer 2: LLM analyzer, when a model is connected and the message needs interpreting.
        analysis = None
        needs_llm = bool(history) or not (has_anchor or explicit.get("max_price") or explicit.get("types"))
        if self.backend is not None and needs_llm and self.use_analyzer:
            analysis = self.analyze(question, history)
        if analysis:
            source = f"{self.backend.label} + rules"
            if analysis.get("standalone_question"):
                text = analysis["standalone_question"]
                filters = {**filters, **{k: v for k, v in parse_filters(text).items()
                                         if k in ("min_price", "max_price", "types", "vegan", "clean", "fragrance_free")}}
            if analysis.get("product_types"):
                filters["types"] = analysis["product_types"]
            for k in ("max_price", "min_price", "sort"):
                if analysis.get(k):
                    filters[k] = analysis[k]
            for k in ("vegan", "clean", "fragrance_free"):
                if analysis.get(k):
                    filters[k] = True

        # Layer 1: the latest message always wins.
        filters.update(explicit)
        if "max_price" in explicit and "min_price" not in explicit and filters.get("min_price", 0) > explicit["max_price"]:
            filters.pop("min_price")
        types = filters.get("types") or []
        filters.pop("exclude_eye", None)
        if types and "eye" not in types and set(types) & FORMULA_TYPES:
            filters["exclude_eye"] = True
        if (not types or set(types) & FORMULA_TYPES) and not DEVICE_WORDS.search(text):
            filters["exclude_devices"] = True
        else:
            filters.pop("exclude_devices", None)

        prev_routine = ((previous or {}).get("retrieved") or Retrieved()).filters.get("routine") if follow_up else None
        routine = routine_steps(question, own.get("types")) or \
            (routine_steps(text, types) if analysis is None or analysis.get("routine") is not False else None) or \
            (types if analysis and analysis.get("routine") and len(types) >= 2 else None) or \
            (["cleanser", "serum", "moisturizer", "sunscreen"] if analysis and analysis.get("routine") else None) or \
            prev_routine
        # Skin type describes the shopper, not one request: remember it for the whole conversation.
        skin = skin_type_in(question) or (analysis or {}).get("skin_type") or skin_type_in(text) or \
            next((st for t in reversed(history) if t["role"] == "user" for st in [skin_type_in(t["content"])] if st), None)
        skincare = bool(SKINCARE_VOCAB.search(text) or (analysis and analysis.get("is_skincare")) or types
                        or has_anchor)
        if not skincare:
            skin = None                       # don't decorate an off-topic message with a remembered skin type
        small_talk = bool(analysis and analysis.get("is_small_talk") and not types and not explicit
                          and not SKINCARE_VOCAB.search(question))
        return {"standalone": text, "filters": filters, "routine": routine, "skin_type": skin,
                "follow_up": follow_up, "refers_back": refers_back, "analysis": analysis, "source": source,
                "small_talk": small_talk, "skincare": skincare}

    @staticmethod
    def describe(u: dict) -> str:
        f, bits = u["filters"], []
        if u.get("routine"):
            bits.append("routine: " + " → ".join(u["routine"]))
        elif f.get("types"):
            bits.append(" + ".join(f["types"]))
        if u.get("skin_type"):
            bits.append(f"{u['skin_type']} skin")
        if "min_price" in f and "max_price" in f:
            bits.append(f"${f['min_price']:g}–${f['max_price']:g}")
        elif "max_price" in f:
            bits.append(f"≤ ${f['max_price']:g}")
        elif "min_price" in f:
            bits.append(f"≥ ${f['min_price']:g}")
        bits += [label for k, label in (("vegan", "vegan"), ("clean", "clean"), ("fragrance_free", "fragrance-free"))
                 if f.get(k)]
        if f.get("brand"):
            bits.append(f["brand"])
        if f.get("sort") == "price_low":
            bits.append("cheapest first")
        elif f.get("sort") == "rating":
            bits.append("best rated first")
        return " · ".join(bits) or "no specific constraints"

    def prepare(self, question: str, history: list[dict] | None = None, preferences: dict | None = None,
                previous: dict | None = None) -> dict:
        """Everything before generation: understanding, guards, retrieval, notes.
        `previous` is the last assistant result (its products are reused for "which of those...").
        Returns a result dict; if result["fallback"] is set, result["answer"] is final."""
        t0 = time.perf_counter()
        history = [{"role": t["role"], "content": t["content"]} for t in (history or []) if t.get("content")]
        timings: dict[str, float] = {}
        result = {"question": question, "search_question": question, "notes": [], "timings": timings,
                  "retrieved": Retrieved(), "fallback": None, "answer": "", "model": None,
                  "steps": [], "history": history, "understood": ""}
        steps = result["steps"]

        original = question
        question, typos = self.speller.fix(question)
        if typos:
            steps.append(("Spelling", ", ".join(f"{a} → {b}" for a, b in typos)))
            result["corrected"] = question
        if is_small_talk(question):
            steps.append(("Small talk", "greeting or thanks, so no search needed"))
            result.update(answer=small_talk_reply(question), fallback="small_talk")
            return result

        u = self.understand(question, history, previous)
        timings["understanding_ms"] = (time.perf_counter() - t0) * 1000
        search_q = u["standalone"]
        result["search_question"], result["understood"] = search_q, self.describe(u)
        f = u["filters"]
        result["profile"] = {
            "skin": u["skin_type"], "products": u["routine"] or f.get("types") or [],
            "routine": bool(u["routine"]), "max_price": f.get("max_price"), "min_price": f.get("min_price"),
            "musts": [label for k, label in (("vegan", "vegan"), ("clean", "Clean at Sephora"),
                                              ("fragrance_free", "fragrance-free")) if f.get(k)],
            "sort": f.get("sort"), "brand": f.get("brand")}
        steps.append(("Understood as", f"{search_q}  [{result['understood']}]  · via {u['source']}"))
        if u["small_talk"]:
            result.update(answer=small_talk_reply(question), fallback="small_talk")
            return result

        topic = out_of_scope_topic(search_q)
        if topic:
            steps.append(("Scope guard", f"{topic} is outside the skincare catalogue, so no search and no LLM"))
            result.update(answer=SCOPE_MESSAGE.format(topic=topic), fallback="out_of_scope")
            return result

        med = medical_terms(search_q)
        retrieval_q = strip_medical(search_q) if med else search_q
        if u["skin_type"] and u["skincare"] and u["skin_type"] not in retrieval_q.lower():
            retrieval_q += f" for {u['skin_type']} skin"
        t1 = time.perf_counter()
        retrieved = self.retrieve(retrieval_q, preferences, base_filters=u["filters"], routine=u["routine"])
        if u["refers_back"] and previous and previous.get("retrieved") and previous["retrieved"].chunks:
            retrieved = self._with_previous(retrieved, previous["retrieved"])
            steps.append(("Follow-up", "reused the products from the previous answer"))
        timings["retrieval_ms"] = (time.perf_counter() - t1) * 1000
        result["retrieved"] = retrieved
        if retrieval_q != search_q:
            steps.append(("Search text", retrieval_q))
        shown = {k: v for k, v in retrieved.filters.items() if k not in ("exclude_devices", "overview_only")}
        steps.append(("Filters", ", ".join(f"{k}={v}" for k, v in shown.items()) or "none"))
        n_vec = sum("vector" in c["found_by"] for c in retrieved.chunks)
        n_kw = sum("keyword" in c["found_by"] for c in retrieved.chunks)
        steps.append(("Hybrid search", f"{len(retrieved.chunks)} chunks: {n_vec} found by vector, {n_kw} by keyword "
                                       f"(BM25), fused with RRF"))
        ok = retrieved.best_similarity >= SIMILARITY_THRESHOLD
        why = "" if ok or not u["skincare"] else " (but the question is clearly about skincare, so we answer)"
        steps.append(("Relevance check", f"best similarity {retrieved.best_similarity} "
                                         f"{'≥' if ok else '<'} threshold {SIMILARITY_THRESHOLD}{why}"))
        if not ok and not u["skincare"]:
            result.update(answer=FALLBACK_MESSAGE, fallback="low_similarity")
            return result
        if not retrieved.chunks:
            result.update(answer=NO_MATCH_MESSAGE.format(constraints=hard_constraints(retrieved.filters)),
                          fallback="no_match")
            return result

        notes = result["notes"]
        f = retrieved.filters
        unsold = self.unsold_brands_in(search_q)
        if unsold:
            notes.append(f"The customer named {', '.join(unsold)}, which this Sephora catalogue doesn't sell. "
                         "Say that first, then suggest comparable products from the context.")
        if med:
            notes.append(f"Medical topic detected ({', '.join(med)}). Follow the medical rule: no safety or "
                         "efficacy claims, reviews are not evidence, suggest a dermatologist or doctor.")
        if "relaxed" in f:
            notes.append(f"No product matched every filter, so these filters were relaxed: "
                         f"{', '.join(f['relaxed'])}. Say the match is approximate.")
        if f.get("routine"):
            budget = f" and keep the total within ${f['max_price']:g} if possible (say if it can't be done)" \
                if "max_price" in f else ""
            notes.append(f"The shopper wants a routine: recommend one product per step in this order: "
                         f"{', '.join(f['routine'])}{budget}. Give the total price.")
        elif "max_price" in f:
            notes.append(f"The shopper's budget is up to ${f['max_price']:g} per product. Every product in the context "
                         "fits it; only recommend those.")
        if f.get("sort") == "price_low":
            notes.append("The shopper wants affordable options: lead with the lowest prices.")
        if u["follow_up"]:
            notes.append(f"This is a follow-up. The shopper's full request is: {search_q}")
        if u["skin_type"]:
            notes.append(f"The shopper's skin type is {u['skin_type']}: say why each pick suits it, using the "
                         "Sephora skin-type tags and the recommend rate by skin type where available.")
        for n in notes:
            steps.append(("Guard" if n.startswith(("Medical", "The customer named")) else "Note", n))

        context = "\n\n---\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(retrieved.chunks))
        note_block = ("Notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n\n") if notes else ""
        prefs = preference_text(preferences)
        user_turn = (f"Product context from the Sephora catalogue:\n\n{context}\n\n{note_block}"
                     f"{prefs}Customer message: {original}" + (f" (read as: {question})" if typos else ""))
        messages = list(history[-4:])
        while messages and messages[0]["role"] != "user":
            messages.pop(0)                      # the API expects the first turn to be the user's
        messages.append({"role": "user", "content": user_turn})
        result["messages"] = messages
        if self.backend is None:
            result.update(answer=self._product_list(retrieved, med), fallback="no_llm")
        return result

    def _with_previous(self, new: Retrieved, prev: Retrieved) -> Retrieved:
        """'Which of those is cheapest?': put the earlier products first (if they still satisfy the
        hard constraints), then fill with new results."""
        hard = {k: v for k, v in new.filters.items() if k in ("min_price", "max_price", "vegan", "clean", "fragrance_free")}
        carry = [{**c, "found_by": "previous answer"} for c in prev.chunks
                 if c["meta"]["chunk_type"] == "overview" and matches(c["meta"], hard)]
        if new.filters.get("sort") == "price_low":
            carry.sort(key=lambda c: c["meta"]["price"])
        ids = {c["id"] for c in carry}
        chunks = (carry + [c for c in new.chunks if c["id"] not in ids])[:FINAL_CHUNKS]
        return Retrieved(chunks=chunks, best_similarity=max(new.best_similarity, prev.best_similarity),
                         filters={**new.filters, "reused_previous": len(carry)})

    @staticmethod
    def _product_list(retrieved: Retrieved, medical: list[str]) -> str:
        lines = []
        for i, c in enumerate(retrieved.chunks):
            m = c["meta"]
            rating = (f"{m['rating']:.1f}/5 from {m['reviews']} reviews" if m.get("has_rating", True)
                      else "no ratings yet")
            lines.append(f"- [{i + 1}] **{m['product_name']}** by {m['brand']} (${m['price']:.2f}, {rating})")
        prefix = "**Please ask a dermatologist or doctor about medical questions.**\n\n" if medical else ""
        return (prefix + "_No language model connected, so here are the matching products from the catalogue "
                "(start Ollama or add an API key to get written answers):_\n\n" + "\n".join(lines))

    def generate(self, result: dict) -> Iterator[str]:
        """Stream the answer text. Fills result["answer"], result["model"] and the timing when done."""
        if result["fallback"] or "messages" not in result:
            yield result["answer"]
            return
        t0 = time.perf_counter()
        parts = []
        for piece in self.backend.stream(SYSTEM_PROMPT, result["messages"], max_tokens=1500):
            parts.append(piece)
            yield piece
        result["answer"] = "".join(parts).strip()
        result["model"] = self.backend.label
        result["timings"]["generation_ms"] = (time.perf_counter() - t0) * 1000
        result["steps"].append(("Answer written", f"{self.backend.label}, grounded in {len(result['retrieved'].chunks)} "
                                                  "numbered sources"))

    def answer(self, question: str, history: list[dict] | None = None, preferences: dict | None = None,
               previous: dict | None = None) -> dict:
        result = self.prepare(question, history, preferences, previous)
        result["answer"] = "".join(self.generate(result)).strip()
        return result
