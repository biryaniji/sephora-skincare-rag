"""Tests for the cleaning rules, on a tiny raw-format fixture that reproduces each problem
found in the data audit.  Run:  python -m pytest tests -q"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare_data as pdx  # noqa: E402


# ------------------------------------------------------------- ingredient parsing
def test_fragments_are_joined_not_dropped():
    # The old "most commas wins" rule kept only the second element and lost Water/Glycerin.
    items = ["Water, Glycerin, Niacinamide,", "EDTA, Sodium Citrate, Caprylyl Glycol, Citric Acid, Caramel,"]
    text, _ = pdx.clean_ingredients(items)
    assert text.startswith("Water, Glycerin, Niacinamide, EDTA")
    assert not text.endswith(",")


def test_marketing_notes_and_disclaimers_removed():
    items = ["-Niacinamide: Reduces the look of pores.", "-Zinc: Balances oil.",
             "Aqua (Water), Niacinamide, Pentylene Glycol, Zinc PCA, Dimethyl Isosorbide",
             "Please be aware that ingredient lists may change or vary from time to time."]
    text, _ = pdx.clean_ingredients(items)
    assert text == "Aqua (Water), Niacinamide, Pentylene Glycol, Zinc PCA, Dimethyl Isosorbide"


def test_kit_components_are_labelled():
    items = ["Ultra Facial Cream:", "Water, Glycerin, Squalane", "Ultra Facial Cleanser:", "Water, Sodium Laureth Sulfate, Glycerin"]
    text, labelled = pdx.clean_ingredients(items)
    assert labelled == 2
    assert "[Ultra Facial Cream] Water, Glycerin, Squalane" in text
    assert "[Ultra Facial Cleanser]" in text


def test_single_ingredient_product_kept():
    assert pdx.clean_ingredients(["Squalane Oil."])[0] == "Squalane Oil"


def test_duplicate_lists_kept_once():
    text, _ = pdx.clean_ingredients(["Water, Glycerin, Squalane", "Water, Glycerin, Squalane"])
    assert text == "Water, Glycerin, Squalane"


def test_may_contain_shades_kept():
    text, _ = pdx.clean_ingredients(["Ricinus Communis Seed Oil, Cera Alba, Tocopherol",
                                     "+/- May Contain: Iron Oxides, Titanium Dioxide, Mica"])
    assert "May Contain" in text


# ------------------------------------------------------------- highlights
def test_split_highlights():
    skin, concerns, other = pdx.split_highlights(
        ["Vegan", "Good for: Dryness", "Best for Dry, Combo, Normal Skin", "Without Parabens"])
    assert skin == ["combination", "dry", "normal"]
    assert concerns == ["Dryness"]
    assert other == ["Vegan", "Without Parabens"]


# ------------------------------------------------------------- end to end on a raw fixture
def _product(pid, name, brand="BrandA", cat2="Moisturizers", cat3="Moisturizers", price=30.0,
             rating=4.5, reviews=200, ingredients="['Water, Glycerin, Squalane']",
             highlights="['Vegan', 'Good for: Dryness', 'Best for Dry, Combo, Normal Skin']"):
    return dict(product_id=pid, product_name=name, brand_id=1, brand_name=brand, loves_count=1,
                rating=rating, reviews=reviews, size="1 oz/ 30 mL", variation_type=None, variation_value=None,
                variation_desc=None, ingredients=ingredients, price_usd=price, value_price_usd=None,
                sale_price_usd=None, limited_edition=0, new=0, online_only=0, out_of_stock=0,
                sephora_exclusive=0, highlights=highlights, primary_category="Skincare",
                secondary_category=cat2, tertiary_category=cat3, child_count=0, child_max_price=None,
                child_min_price=None)


@pytest.fixture()
def raw_dir(tmp_path):
    products = [
        _product("P1", "Daily Cream"),
        _product("P2", "Daily Cream Refill", price=24.0),                      # refill of P1 -> drop
        _product("P3", "Rescue Mask Mini", cat2="Masks", cat3="Face Masks"),  # mini at end -> drop
        _product("P4", "Prenatal Support Supplement", cat2="Wellness", cat3="Beauty Supplements"),  # drop
        _product("P5", "Whitening Pen", cat2="Teeth Whitening", cat3="Teeth Whitening"),
        _product("P6", "BEAR Facial Toning Device", cat2="High Tech Tools", cat3="Anti-Aging",
                 ingredients="['Water, Glycerin, Butylene Glycol, Niacinamide, Peptides, Phenoxyethanol']"),
        _product("P7", "LUNA Mini 3", cat2="High Tech Tools", cat3="Facial Cleansing Brushes", ingredients=None),
        _product("P8", "Glow Starter Kit", cat2="Treatments", cat3="Face Serums"),
        _product("P9", "New Serum", cat2="Treatments", cat3="Face Serums", rating=5.0, reviews=2),
        _product("P10", "Unrated Toner", cat2="Treatments", cat3="Toners", rating=None, reviews=None),
        _product("P11", "Lipstick", ingredients=None, highlights=None),          # nothing to retrieve -> drop
    ]
    pd.DataFrame(products).to_csv(tmp_path / "product_info.csv", index=False)
    long = "This cream is lovely and hydrating, my skin feels soft all day and it layers well under makeup."
    reviews = [
        dict(product_id="P1", rating=5, is_recommended=1, total_pos_feedback_count=50,
             review_text=long + " I received this product complimentary for testing.", skin_type="dry"),
        dict(product_id="P1", rating=5, is_recommended=1, total_pos_feedback_count=10, review_text=long, skin_type="dry"),
        dict(product_id="P1", rating=4, is_recommended=1, total_pos_feedback_count=5, review_text=long + " Good.", skin_type="oily"),
        dict(product_id="P1", rating=1, is_recommended=0, total_pos_feedback_count=8,
             review_text="Broke me out badly within two days and the scent is overwhelming. Would not buy again.", skin_type="oily"),
    ]
    # enough votes for a dry-skin recommend rate (needs >= 15)
    reviews += [dict(product_id="P1", rating=5, is_recommended=1, total_pos_feedback_count=0,
                     review_text=long + f" Review number {i}.", skin_type="dry") for i in range(20)]
    pd.DataFrame(reviews).to_csv(tmp_path / "reviews_0-250.csv", index=False)
    return tmp_path


def test_pipeline_on_fixture(raw_dir):
    log = []
    p = pdx.load_products(str(raw_dir), log)
    names = set(p["product_name"])
    assert "Daily Cream" in names and "Daily Cream Refill" not in names
    assert "Rescue Mask Mini" not in names
    assert "Prenatal Support Supplement" not in names and "Whitening Pen" not in names
    assert "LUNA Mini 3" in names                                   # device model, not a size
    assert "Lipstick" not in names
    dev = p[p["product_name"] == "BEAR Facial Toning Device"].iloc[0]
    assert dev["ingredients_clean"] == "" and dev["ingredients_status"] == "not_applicable"
    assert p[p["product_name"] == "Glow Starter Kit"].iloc[0]["ingredients_status"] == "partial_kit"

    r = pdx.load_reviews(str(raw_dir), set(p["product_id"]), log)
    s = pdx.summarise_reviews(r).set_index("product_id").loc["P1"]
    first = s["review_snippets"].split("\n")[0]
    assert "received product free" not in first                     # organic review preferred
    assert "dry 100% (n=" in s["skin_type_feedback"]

    row = p[p["product_name"] == "New Serum"].iloc[0]
    assert "not reliable" in pdx.rating_line(row)
    row = p[p["product_name"] == "Unrated Toner"].iloc[0]
    assert pdx.rating_line(row) == "Rating: no ratings yet"


def test_main_writes_outputs(raw_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/raw", exist_ok=True)
    for f in os.listdir(raw_dir):
        if f.endswith(".csv"):
            os.replace(raw_dir / f, tmp_path / "data/raw" / f)
    pdx.main()
    out = pd.read_csv("data/processed/skincare_products.csv")
    d = out.set_index("product_name").loc["Daily Cream", "description"]
    assert "Best for skin types (Sephora): combination, dry, normal" in d
    assert "Good for concerns (Sephora): Dryness" in d
    assert "Highlights: Vegan" in d
    assert os.path.exists("data/processed/cleaning_report.md")
