"""Step 2: chunk the products, embed them and store them in Chroma.

Chunking strategy (record-based, not fixed-size):
  * "overview" chunk  - name, brand, category, price, rating, skin types, concerns,
                        highlights, recommend rate by skin type, review snippets.
                        This is what most shopper questions match.
  * "ingredients" chunk - the product header repeated + the full INCI list.
                        Kept separate because a 700-char chemical list would dilute the
                        overview embedding, and ingredient questions are usually keyword-like.
  Every chunk starts with the product header (name, brand, category, price), so no chunk is
  ever orphaned. The repeated header is the overlap between parts of one product.

Size limit: bge-small reads at most 512 tokens and silently drops the rest. Chunks are
split so each one fits in MAX_TOKENS, measured with the embedder's own tokenizer
(a character estimate is used when the tokenizer isn't loaded, e.g. in tests).
  * long overviews are split on review boundaries,
  * long ingredient lists are split between ingredients ("part 1 of 2").

Metadata on every chunk: price, rating, has_rating, brand, category, skin types, vegan,
clean, device, and product-type flags (type_serum, type_moisturizer, ...) that retrieval
uses as hard filters.

Run:  python build_index.py
"""
import json
import os
import re
from typing import Callable

import pandas as pd

DATA = "data/processed/skincare_products.csv"
CHROMA_DIR = "data/chroma"
CHUNKS_FILE = "data/processed/chunks.jsonl"
COLLECTION = "sephora_skincare"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 500          # bge-small limit is 512 incl. special tokens
MAX_CHARS = 2000          # hard cap as well, keeps chunks readable in the UI


def estimate_tokens(text: str) -> int:
    """WordPiece estimate used only when the real tokenizer isn't loaded (tests).
    Short words are ~1 piece; long chemical names split into several; punctuation is 1."""
    words = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text)
    return 2 + sum(max(1, -(-len(w) // 5)) if w.isalpha() else len(w) if w.isdigit() else 1 for w in words)


# ---------------------------------------------------------------- product types
TYPE_RULES: dict[str, Callable[[str, str, str], bool]] = {
    # name, category, subcategory -> bool
    "eye": lambda n, c, s: c == "Eye Care" or bool(re.search(r"\beyes?\b|\bunder-eye\b", n)),
    "sunscreen": lambda n, c, s: c == "Sunscreen" or bool(re.search(r"\bspf\b|sunscreen|\bsun (cream|lotion|fluid)\b", n)),
    "moisturizer": lambda n, c, s: s in {"Moisturizers", "Night Creams", "Decollete & Neck Creams", "BB & CC Creams"}
                                   or bool(re.search(r"moisturi[sz]|\bcream\b|\blotion\b|gel-cream|\bemulsion\b", n)),
    "serum": lambda n, c, s: s == "Face Serums" or bool(re.search(r"\bserum\b|\bampoules?\b|\bconcentrate\b", n)),
    "cleanser": lambda n, c, s: s in {"Face Wash & Cleansers", "Makeup Removers", "Face Wipes"}
                                or bool(re.search(r"cleanser|face wash|cleansing (balm|oil|gel|foam|milk)", n)),
    "toner": lambda n, c, s: s == "Toners" or bool(re.search(r"\btoner\b", n)),
    "exfoliant": lambda n, c, s: s in {"Exfoliators", "Facial Peels"} or bool(re.search(r"exfoliat|\bpeel\b|\bscrub\b", n)),
    "mask": lambda n, c, s: c == "Masks" or s == "Eye Masks" or bool(re.search(r"\bmasks?\b", n)),
    "oil": lambda n, c, s: s == "Face Oils" or bool(re.search(r"\bface oil\b|\bfacial oil\b", n)),
    "mist": lambda n, c, s: s == "Mists & Essences" or bool(re.search(r"\bmist\b|\bessence\b", n)),
    "lip": lambda n, c, s: c == "Lip Balms & Treatments" or bool(re.search(r"\blips?\b", n)),
    "self_tanner": lambda n, c, s: c == "Self Tanners" or bool(re.search(r"self[- ]tan|sunless", n)),
}


def product_types(row) -> dict[str, bool]:
    name = str(row["product_name"]).lower()
    cat = row["category"] if isinstance(row["category"], str) else ""
    sub = row["subcategory"] if isinstance(row["subcategory"], str) else ""
    return {f"type_{t}": bool(rule(name, cat, sub)) for t, rule in TYPE_RULES.items()}


# ---------------------------------------------------------------- chunking
def header(row) -> str:
    return f"Product: {row['product_name']} by {row['brand_name']} ({row['category']}, ${row['price_usd']:.2f})"


def fits(text: str, token_len: Callable[[str], int]) -> bool:
    return len(text) <= MAX_CHARS and token_len(text) <= MAX_TOKENS


def split_overview(desc: str, head: str, token_len: Callable[[str], int]) -> list[str]:
    """Keep the overview whole if it fits, otherwise move reviews into extra chunks
    (each one starting with the product header)."""
    if fits(desc, token_len):
        return [desc]
    main, sep, reviews = desc.partition("What reviewers say:\n")
    chunks, current = [main.strip()], []
    for review in [r for r in reviews.split("\n") if r.strip()]:
        candidate = f"{head}\nWhat reviewers say:\n" + "\n".join(current + [review])
        if current and not fits(candidate, token_len):
            chunks.append(f"{head}\nWhat reviewers say:\n" + "\n".join(current))
            current = []
        current.append(review)
    if current:
        tail = "\n".join(current)
        folded = chunks[0] + "\nWhat reviewers say:\n" + tail
        if len(chunks) == 1 and fits(folded, token_len):
            chunks[0] = folded
        else:
            chunks.append(f"{head}\nWhat reviewers say:\n{tail}")
    return chunks


def split_ingredients(ingredients: str, head: str, token_len: Callable[[str], int]) -> list[str]:
    """One ingredients chunk if it fits, otherwise split between ingredients."""
    whole = f"{head}\nIngredients: {ingredients}"
    if fits(whole, token_len):
        return [whole]
    items = [i.strip() for i in re.split(r",\s*(?![^()]*\))", ingredients) if i.strip()]
    parts, current = [], []
    for item in items:
        if current and not fits(f"{head}\nIngredients (part 9 of 9): " + ", ".join(current + [item]), token_len):
            parts.append(current)
            current = []
        current.append(item)
    if current:
        parts.append(current)
    n = len(parts)
    return [f"{head}\nIngredients (part {i} of {n}): " + ", ".join(p) for i, p in enumerate(parts, 1)]


def chunk_metadata(row) -> dict:
    has_rating = pd.notna(row["rating"])
    return {
        "product_id": str(row["product_id"]),
        "product_name": row["product_name"],
        "brand": row["brand_name"],
        "category": row["category"] if isinstance(row["category"], str) else "",
        "subcategory": row["subcategory"] if isinstance(row["subcategory"], str) else "",
        "price": float(row["price_usd"]),
        "rating": float(row["rating"]) if has_rating else 0.0,   # Chroma can't store None
        "has_rating": bool(has_rating),
        "reviews": int(row["reviews"]) if pd.notna(row["reviews"]) else 0,
        "skin_types": row["skin_types"] if isinstance(row["skin_types"], str) else "",
        "is_vegan": bool(row["is_vegan"]),
        "is_clean": bool(row["is_clean"]),
        "is_device": bool(row.get("is_device", False)),
        "is_fragrance_free": "Fragrance Free" in str(row["description"]),
        **product_types(row),
    }


def make_chunks(df: pd.DataFrame, token_len: Callable[[str], int] = estimate_tokens) -> list[dict]:
    chunks = []
    for _, row in df.iterrows():
        meta = chunk_metadata(row)
        head = header(row)
        for i, text in enumerate(split_overview(row["description"], head, token_len)):
            chunks.append({"id": f"{row['product_id']}-ov{i}", "text": text,
                           "meta": {**meta, "chunk_type": "overview"}})
        ing = row["ingredients_clean"]
        if isinstance(ing, str) and ing.strip():
            for i, text in enumerate(split_ingredients(ing, head, token_len)):
                chunks.append({"id": f"{row['product_id']}-ing{i}", "text": text,
                               "meta": {**meta, "chunk_type": "ingredients"}})
    return chunks


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(DATA, dtype={"product_id": str})
    model = SentenceTransformer(EMBED_MODEL)
    tok = model.tokenizer

    def token_len(text: str) -> int:
        return len(tok(text, add_special_tokens=True)["input_ids"])

    chunks = make_chunks(df, token_len)
    by_type = pd.Series([c["meta"]["chunk_type"] for c in chunks]).value_counts().to_dict()
    longest = max(token_len(c["text"]) for c in chunks)
    print(f"{len(df)} products -> {len(chunks)} chunks {by_type}; longest chunk {longest} tokens")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    vectors = model.encode([c["text"] for c in chunks], batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), 1000):
        batch = chunks[start:start + 1000]
        col.add(ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["meta"] for c in batch],
                embeddings=vectors[start:start + 1000].tolist())
    print(f"stored {col.count()} chunks in Chroma at {CHROMA_DIR}/")
    print("next: python check_data.py   (checks the chunks too)")


if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)
    main()
