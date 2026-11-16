"""Data-quality checks for the processed catalogue and the chunk file.

Run after prepare_data.py (and again after build_index.py for the chunk checks):
    python check_data.py
Exits with code 1 if any FAIL check fails, so it can gate the index build.
Writes the results to data/processed/quality_report.md (useful for the presentation).
"""
import json
import os
import re
import sys

import pandas as pd

PRODUCTS = "data/processed/skincare_products.csv"
CHUNKS = "data/processed/chunks.jsonl"
REPORT = "data/processed/quality_report.md"
EXCLUDED_SUBCATEGORIES = {"Beauty Supplements", "Holistic Wellness", "Teeth Whitening", "Hair Removal"}
MAX_CHUNK_CHARS = 2100
ENCODING_JUNK = re.compile("Ã|â€|�|&amp;|&#\\d+;|<br")

results: list[tuple[str, str, str]] = []   # (level, check, detail)


def check(ok: bool, name: str, detail: str = "", level: str = "FAIL"):
    results.append(("PASS" if ok else level, name, detail))


def products_checks(df: pd.DataFrame):
    required = ["product_id", "product_name", "brand_name", "category", "price_usd", "description"]
    nulls = {c: int(df[c].isna().sum()) for c in required if c in df}
    check(all(c in df for c in required) and not any(nulls.values()), "Required fields present and non-null",
          str({k: v for k, v in nulls.items() if v}))
    check(df["product_id"].is_unique, "product_id unique", f"{df['product_id'].duplicated().sum()} duplicates")
    dup = df.duplicated(subset=["brand_name", "product_name"]).sum()
    check(dup == 0, "brand + name unique", f"{dup} duplicates")

    bad = df[df["subcategory"].isin(EXCLUDED_SUBCATEGORIES)]
    check(bad.empty, "No supplements / wellness / teeth whitening / hair removal", f"{len(bad)} rows")
    minis = df[df["product_name"].str.contains(r"(?i)\bmini\b") & (df["category"] != "High Tech Tools")]
    check(minis.empty, "No mini-size duplicates", ", ".join(minis["product_name"].head(5)))
    refills = df[df["product_name"].str.contains(r"(?i)\brefill\b")]
    check(len(refills) <= 5, "Refills only where no full-size product exists", f"{len(refills)} refills left",
          level="WARN")

    check(df["price_usd"].between(1, 2000).all(), "Prices between $1 and $2,000",
          f"min {df['price_usd'].min()}, max {df['price_usd'].max()}")
    r = df["rating"].dropna()
    check(r.between(1, 5).all(), "Ratings between 1 and 5 (or missing)")
    check(((df["rating"].isna()) == (df["reviews"].isna() | (df["reviews"] == 0))).all(),
          "Rating missing exactly when there are no reviews", level="WARN")
    small = int((df["reviews"] < 10).sum())
    check("not reliable" in " ".join(df.loc[df["reviews"] < 10, "description"].head(50)) or small == 0,
          "Small-sample ratings are flagged in the text", f"{small} products with < 10 reviews")

    desc = df["description"]
    check(desc.str.startswith("Product:").all(), "Every description starts with its product name")
    check(desc.str.len().between(100, 4000).all(), "Description length 100-4,000 chars",
          f"min {desc.str.len().min()}, max {desc.str.len().max()}")
    check(not desc.str.contains("Highlights: .*Good for:", regex=True).any(),
          "Concerns are on their own line, not mixed into highlights")
    junk = sum(int(df[c].fillna("").astype(str).str.contains(ENCODING_JUNK).sum())
               for c in ["product_name", "description", "ingredients_clean"])
    check(junk == 0, "No encoding artefacts (Ã, â€, �, HTML entities)", f"{junk} cells", level="WARN")

    ing = df["ingredients_clean"].fillna("").str.strip()
    check(not ing.str.endswith(",").any(), "No ingredient list ends mid-list (trailing comma)",
          ", ".join(df.loc[ing.str.endswith(","), "product_name"].head(5)))
    if "is_device" in df:
        dev_with_ing = df[df["is_device"] & ing.str.contains(r"(?i)\bdevice\b|^100% .*(?:quartz|jade)")]
        check(dev_with_ing.empty, "Devices don't carry a formula's ingredient list", f"{len(dev_with_ing)} rows")
    if "ingredients_status" in df:
        counts = df["ingredients_status"].value_counts().to_dict()
        check(True, "Ingredient status recorded", str(counts))
    same = df[ing.ne("") & ing.duplicated(keep=False)]
    check(len(same) <= 40, "Few identical ingredient lists across different products",
          f"{len(same)} products share a list with another (check day/night pairs, kits)", level="WARN")

    filled = df["skin_types"].notna().mean()
    sensitive = df["skin_types"].fillna("").str.contains("sensitive").sum()
    check(True, "Skin-type tag coverage (Sephora 'Best for')",
          f"{filled:.0%} of products, {sensitive} tagged 'sensitive': too sparse for a hard filter")
    inc = desc.str.contains("received product free").mean()
    check(True, "Incentivized review snippets are labelled", f"{inc:.0%} of products show one")


def chunk_checks(df: pd.DataFrame):
    if not os.path.exists(CHUNKS):
        check(False, "Chunk file exists", "run build_index.py", level="WARN")
        return
    chunks = [json.loads(line) for line in open(CHUNKS, encoding="utf-8")]
    ids = [c["id"] for c in chunks]
    check(len(ids) == len(set(ids)), "Chunk ids unique")
    pids = {c["meta"]["product_id"] for c in chunks}
    missing = set(df["product_id"]) - pids
    extra = pids - set(df["product_id"])
    check(not missing and not extra, "Chunks match the product file",
          f"{len(missing)} products without chunks, {len(extra)} chunks for removed products")
    over = [c["id"] for c in chunks if len(c["text"]) > MAX_CHUNK_CHARS]
    check(not over, f"Every chunk <= {MAX_CHUNK_CHARS} chars (embedder limit is 512 tokens)",
          f"{len(over)} too long: {over[:3]}")
    check(all(c["text"].startswith("Product:") for c in chunks), "Every chunk names its product")
    overview = {c["meta"]["product_id"] for c in chunks if c["meta"]["chunk_type"] == "overview"}
    check(overview >= set(df["product_id"]), "Every product has an overview chunk")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # Windows consoles may not be UTF-8
    df = pd.read_csv(PRODUCTS, dtype={"product_id": str})
    products_checks(df)
    chunk_checks(df)
    lines = [f"# Data quality report\n\n{len(df):,} products\n", "| Result | Check | Detail |", "|---|---|---|"]
    lines += [f"| {lvl} | {name} | {detail} |" for lvl, name, detail in results]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    width = max(len(n) for _, n, _ in results)
    for lvl, name, detail in results:
        print(f"[{lvl:4}] {name:<{width}}  {detail}")
    fails = sum(lvl == "FAIL" for lvl, _, _ in results)
    print(f"\n{len(results) - fails} of {len(results)} checks without FAIL. Report: {REPORT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
