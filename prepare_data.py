"""Step 1: clean the Sephora Kaggle data and build one description per product.

Input : Kaggle dataset nadyinky/sephora-products-and-skincare-reviews
        (product_info.csv + reviews_*.csv), downloaded with kagglehub
        or copied into data/raw/.
Output: data/processed/skincare_products.csv
        data/processed/cleaning_report.md   (what was dropped and why)

Run:  python prepare_data.py
Then: python check_data.py      (automated data-quality checks)
"""
import ast
import glob
import os
import re
import warnings

import pandas as pd

# Some scraped ingredient strings contain backslashes; literal_eval warns but parses fine.
warnings.filterwarnings("ignore", category=SyntaxWarning)

KAGGLE_ID = "nadyinky/sephora-products-and-skincare-reviews"
OUT_DIR = "data/processed"
OUT_FILE = os.path.join(OUT_DIR, "skincare_products.csv")
REPORT_FILE = os.path.join(OUT_DIR, "cleaning_report.md")

POSITIVE_REVIEWS = 2           # most helpful 4-5 star reviews per product
CRITICAL_REVIEWS = 1           # most helpful 1-2 star review per product
MIN_REVIEWS_FOR_RATING = 10    # below this, the rating is flagged as a small sample
MIN_SKIN_TYPE_VOTES = 15       # min reviews per skin type before we report its recommend rate

# Not skincare, and several make health claims (prenatal, menopause, detox, sleep) that the
# bot must not advise on. Matched against Sephora's tertiary category.
EXCLUDED_SUBCATEGORIES = {"Beauty Supplements", "Holistic Wellness", "Teeth Whitening", "Hair Removal"}

DEVICE_NAME = re.compile(r"(?i)\b(device|roller|gua sha|wand|massager|brush|razor|blades?|pillowcase|"
                         r"humidifier|attachment|tools?|holder|light mask|toning device)\b")
KIT_NAME = re.compile(r"(?i)\b(kit|duo|trio|collection|bundle|starter set|gift set)\b|\bset$")
INCENTIVIZED = re.compile(r"(?i)complimentary|received (?:this|it|the product)[^.]{0,30}free|"
                          r"free (?:product|sample)[^.]{0,20}(?:from|in exchange)|in exchange for (?:my |an )?"
                          r"(?:honest )?(?:review|opinion)|influenster|sampling purposes|gifted|\bPR package")
INGREDIENT_BOILERPLATE = re.compile(r"(?i)please be aware|may change or vary|refer to the (?:ingredient list|"
                                    r"product packag)|most up-to-date list|clean at sephora|formulated without|"
                                    r"is free (?:of|from)|this product (?:is|does)|we are committed|"
                                    r"ingredient list may")


# ---------------------------------------------------------------- parsing helpers
def find_raw_dir() -> str:
    """Use data/raw if the CSVs were copied there, otherwise download via kagglehub."""
    if os.path.exists("data/raw/product_info.csv"):
        return "data/raw"
    import kagglehub

    return kagglehub.dataset_download(KAGGLE_ID)


def parse_list(value) -> list[str]:
    """highlights / ingredients are stored as the string form of a Python list."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, str):
            parsed = [parsed]
        return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        return [value.strip()]


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,+", ",", s)
    return s


def _is_header(s: str) -> bool:
    """Component label inside a kit, e.g. 'Ultra Facial Cream:'."""
    return s.endswith(":") and "," not in s and len(s) <= 80 and not s.startswith("-")


def _is_marketing(s: str) -> bool:
    """'-Niacinamide: reduces pores' style notes, and disclaimers."""
    if s.startswith("-") or s.startswith("•"):
        return True
    if INGREDIENT_BOILERPLATE.search(s) and s.count(",") < 4:
        return True
    # "Hyaluronic Acid: plumps skin." -> a short label followed by prose, few commas
    if re.match(r"^[^,:]{2,45}:\s+\S", s) and s.count(",") < 3 and not re.match(r"(?i)^\+?/?-?\s*may contain", s):
        return True
    return False


def clean_ingredients(items: list[str]) -> tuple[str, int]:
    """Rebuild the INCI list from Sephora's scraped list.

    The old version kept only the element with the most commas, which dropped the start or
    end of lists that the scraper had split over several elements (e.g. a list that began at
    'EDTA, ...'). Now every INCI-looking element is kept, in order:
      * marketing notes ('-Niacinamide: ...') and disclaimers are dropped,
      * kit component labels ('Ultra Facial Cream:') start a new labelled block,
      * consecutive fragments are joined, exact repeats are removed.
    Returns (text, number_of_labelled_components).
    """
    blocks: list[list] = []          # [label, [fragments]]
    label, new_block = None, True
    candidates = [_tidy(s) for s in items if _tidy(s)]
    for s in candidates:
        if _is_header(s):
            label, new_block = s.rstrip(":").strip(), True
            continue
        if _is_marketing(s):
            continue
        prev = blocks[-1][1][-1] if blocks and blocks[-1][1] else ""
        continuation = prev.endswith(",") or (s[:1].islower() or s[:1] == "(")
        looks_inci = s.count(",") >= 1 or continuation or len(candidates) == 1
        if not looks_inci:
            continue
        if new_block or not blocks:
            blocks.append([label, []])
            new_block = False
        if s not in blocks[-1][1]:
            blocks[-1][1].append(s)

    parts = []
    for lab, frags in blocks:
        text = _tidy(", ".join(f.rstrip(",. ") for f in frags)).strip(" ,")
        if text:
            parts.append(f"[{lab}] {text}" if lab else text)
    labelled = sum(1 for lab, _ in blocks if lab)
    # Same list repeated for two sizes/shades: keep one copy.
    parts = list(dict.fromkeys(parts))
    return " | ".join(parts), labelled


def split_highlights(highlights: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Separate Sephora's highlight tags into (skin types, concerns, other tags)."""
    skin, concerns, other = set(), [], []
    for h in highlights:
        low = h.lower()
        if low.startswith("best for"):
            for t in ["dry", "oily", "combo", "normal", "sensitive"]:
                if re.search(rf"\b{t}\b", low):
                    skin.add("combination" if t == "combo" else t)
        elif low.startswith("good for:"):
            concerns.append(h.split(":", 1)[1].strip())
        else:
            other.append(h)
    return sorted(skin), concerns, other


def is_device(row) -> bool:
    return bool(DEVICE_NAME.search(row["product_name"])) or row.get("tertiary_category") in {
        "Facial Rollers", "Facial Cleansing Brushes", "Blotting Papers"}


def is_kit(name: str) -> bool:
    return bool(KIT_NAME.search(name))


def _norm_name(name: str) -> list[str]:
    name = re.sub(r"(?i)\b(refill(?:able)?|pack|mini|jumbo|travel size|limited edition)\b", " ", name)
    return re.findall(r"[a-z0-9]+", name.lower())


def find_refill_duplicates(p: pd.DataFrame) -> pd.Index:
    """Refills whose full-size product (same brand, same first three name words) is also listed."""
    drop = []
    for idx, row in p[p["product_name"].str.contains(r"(?i)\brefill\b")].iterrows():
        key = _norm_name(row["product_name"])[:3]
        siblings = p[(p["brand_name"] == row["brand_name"]) & (p.index != idx)
                     & ~p["product_name"].str.contains(r"(?i)\brefill\b")]
        if any(_norm_name(n)[:3] == key for n in siblings["product_name"]):
            drop.append(idx)
    return pd.Index(drop)


# ---------------------------------------------------------------- products
def load_products(raw: str, log: list[str]) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(raw, "product_info.csv"), dtype={"product_id": str})
    log.append(f"| Start: all Sephora products | {len(p):,} |")

    def step(mask, why):
        nonlocal p
        before = len(p)
        p = p[mask].copy()
        log.append(f"| {why} | {len(p):,} (-{before - len(p):,}) |")

    step(p["primary_category"] == "Skincare", "Keep primary_category == Skincare")
    step(~p["secondary_category"].isin(["Value & Gift Sets", "Mini Size"]),
         "Drop gift sets and the Mini Size category (duplicates of full-size items)")
    step(~p["tertiary_category"].isin(EXCLUDED_SUBCATEGORIES),
         "Drop non-skincare / health-claim items: " + ", ".join(sorted(EXCLUDED_SUBCATEGORIES)))
    mini = p["product_name"].str.contains(r"(?i)\bmini\b") & (p["secondary_category"] != "High Tech Tools")
    step(~mini, "Drop 'Mini' size variants anywhere in the name (devices like 'LUNA Mini 3' are kept: different models)")
    step(~p["product_id"].duplicated(), "Deduplicate by product_id")
    step(~p.duplicated(subset=["brand_name", "product_name"]), "Deduplicate by brand + name")
    refills = find_refill_duplicates(p)
    step(~p.index.isin(refills), "Drop refills whose full-size product is listed")

    p["highlights_list"] = p["highlights"].apply(parse_list)
    parsed = p["ingredients"].apply(parse_list).apply(clean_ingredients)
    p["ingredients_clean"] = parsed.str[0]
    p["kit_components"] = parsed.str[1]
    p["is_device"] = p.apply(is_device, axis=1)
    p["is_kit"] = p["product_name"].apply(is_kit)
    # Devices have no formula; a scraped "ingredient list" on a device belongs to something else.
    p.loc[p["is_device"] & (p["ingredients_clean"].str.count(",") < 5), "ingredients_clean"] = ""
    p.loc[p["is_device"] & p["product_name"].str.contains(r"(?i)\bdevice\b"), "ingredients_clean"] = ""
    p["ingredients_status"] = "ok"
    p.loc[p["ingredients_clean"] == "", "ingredients_status"] = "missing"
    p.loc[p["is_device"] & (p["ingredients_clean"] == ""), "ingredients_status"] = "not_applicable"
    p.loc[p["is_kit"] & (p["kit_components"] < 2) & (p["ingredients_clean"] != ""), "ingredients_status"] = "partial_kit"

    has_content = (p["highlights_list"].str.len() > 0) | (p["ingredients_clean"].str.len() > 0)
    step(has_content, "Drop products with neither highlights nor ingredients")
    return p


# ---------------------------------------------------------------- reviews
def load_reviews(raw: str, product_ids: set[str], log: list[str]) -> pd.DataFrame:
    cols = ["product_id", "rating", "is_recommended", "total_pos_feedback_count",
            "review_text", "skin_type"]
    frames = []
    for f in sorted(glob.glob(os.path.join(raw, "reviews_*.csv"))):
        df = pd.read_csv(f, usecols=cols, dtype={"product_id": str}, low_memory=False)
        frames.append(df[df["product_id"].isin(product_ids)])
    r = pd.concat(frames, ignore_index=True)
    n0 = len(r)
    r["review_text"] = r["review_text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    r = r[r["review_text"].str.len().between(80, 600)]
    r = r.drop_duplicates(subset=["product_id", "review_text"])
    r["incentivized"] = r["review_text"].str.contains(INCENTIVIZED)
    log.append(f"\nReviews for kept products: {n0:,}; after length filter (80-600 chars) and dedup: {len(r):,}; "
               f"incentivized (free product): {r['incentivized'].mean():.1%}")
    return r


def skin_type_feedback(g: pd.DataFrame) -> str:
    """Recommend rate per reviewer skin type, e.g. 'dry 93% (n=210), oily 71% (n=58)'.
    Rates, not raw counts: counts mostly reflect how many reviewers have each skin type."""
    g = g.dropna(subset=["skin_type", "is_recommended"])
    stats = g.groupby("skin_type")["is_recommended"].agg(["mean", "size"])
    stats = stats[stats["size"] >= MIN_SKIN_TYPE_VOTES].sort_values("size", ascending=False)
    return ", ".join(f"{st} {m:.0%} (n={n})" for st, (m, n) in stats.iterrows())


def _snippet(row) -> str:
    tags = [f"{int(row['rating'])}/5"]
    if isinstance(row["skin_type"], str):
        tags.append(f"{row['skin_type']} skin")
    if row["incentivized"]:
        tags.append("received product free")
    return f'"{row["review_text"]}" ({", ".join(tags)})'


def summarise_reviews(r: pd.DataFrame) -> pd.DataFrame:
    """Per product: most helpful positive and critical reviews (organic ones first) and
    the recommend rate per skin type."""
    rows = []
    for pid, g in r.groupby("product_id"):
        g = g.sort_values(["incentivized", "total_pos_feedback_count"], ascending=[True, False])
        picks = pd.concat([g[g["rating"] >= 4].head(POSITIVE_REVIEWS),
                           g[g["rating"] <= 2].head(CRITICAL_REVIEWS)])
        rows.append({"product_id": pid,
                     "review_snippets": "\n".join(picks.apply(_snippet, axis=1)) if len(picks) else "",
                     "skin_type_feedback": skin_type_feedback(g)})
    return pd.DataFrame(rows, columns=["product_id", "review_snippets", "skin_type_feedback"])


# ---------------------------------------------------------------- description
def rating_line(row) -> str:
    if pd.isna(row["rating"]) or pd.isna(row["reviews"]) or not row["reviews"]:
        return "Rating: no ratings yet"
    n = int(row["reviews"])
    line = f"Rating: {row['rating']:.1f}/5 from {n:,} reviews"
    if n < MIN_REVIEWS_FOR_RATING:
        line += " (very few reviews, so this rating is not reliable)"
    return line


def build_description(row) -> str:
    cat = " > ".join(str(c) for c in [row["primary_category"], row["secondary_category"],
                                      row["tertiary_category"]] if isinstance(c, str))
    lines = [
        f"Product: {row['product_name']} by {row['brand_name']}",
        f"Category: {cat}",
        f"Price: ${row['price_usd']:.2f}" + (f" | Size: {row['size']}" if isinstance(row["size"], str) else ""),
        rating_line(row),
    ]
    if row["skin_types"]:
        lines.append(f"Best for skin types (Sephora): {row['skin_types']}")
    if row["concerns"]:
        lines.append(f"Good for concerns (Sephora): {row['concerns']}")
    if row["other_highlights"]:
        lines.append("Highlights: " + "; ".join(row["other_highlights"]))
    flags = [label for col, label in [("sephora_exclusive", "Sephora exclusive"), ("new", "New"),
                                      ("limited_edition", "Limited edition"), ("online_only", "Online only")]
             if row.get(col) == 1]
    if flags:
        lines.append("Availability: " + ", ".join(flags))
    if row["is_kit"]:
        lines.append("Note: this is a set or kit with more than one item.")
    if row.get("skin_type_feedback"):
        lines.append(f"Share of reviewers who recommend it, by their skin type: {row['skin_type_feedback']}")
    if row.get("review_snippets"):
        lines.append("What reviewers say:\n" + row["review_snippets"])
    return "\n".join(lines)


def main():
    log = ["# Cleaning report", "", "| Step | Products |", "|---|---|"]
    raw = find_raw_dir()
    print("reading from", raw)
    p = load_products(raw, log)
    reviews = load_reviews(raw, set(p["product_id"]), log)
    p = p.merge(summarise_reviews(reviews), on="product_id", how="left")
    p[["review_snippets", "skin_type_feedback"]] = p[["review_snippets", "skin_type_feedback"]].fillna("")

    split = p["highlights_list"].apply(split_highlights)
    p["skin_types"] = split.str[0].apply(", ".join)
    p["concerns"] = split.str[1].apply(", ".join)
    p["other_highlights"] = split.str[2]
    p["is_vegan"] = p["highlights_list"].apply(lambda h: "Vegan" in h)
    p["is_clean"] = p["highlights_list"].apply(lambda h: "Clean at Sephora" in h)
    p["description"] = p.apply(build_description, axis=1)

    keep = ["product_id", "product_name", "brand_name", "secondary_category", "tertiary_category",
            "price_usd", "rating", "reviews", "size", "skin_types", "concerns", "is_vegan",
            "is_clean", "is_device", "is_kit", "ingredients_status", "ingredients_clean", "description"]
    out = p[keep].rename(columns={"secondary_category": "category", "tertiary_category": "subcategory"})
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    log += ["", f"**Final: {len(out):,} products.** Average description length "
                f"{out['description'].str.len().mean():.0f} characters.", "",
            "Ingredient status: " + ", ".join(f"{k} {v}" for k, v in out["ingredients_status"].value_counts().items()),
            "", "Assumptions: prices are USD from the March 2023 scrape; reviewer skin type is self-reported; "
            "ratings from fewer than 10 reviews are marked as unreliable in the text."]
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\nwrote {OUT_FILE} and {REPORT_FILE}")


if __name__ == "__main__":
    main()
