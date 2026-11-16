import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_index as bi  # noqa: E402

HEAD = "Product: Test Cream by BrandA (Moisturizers, $30.00)"


def row(name, cat="Moisturizers", sub="Moisturizers"):
    return pd.Series({"product_name": name, "category": cat, "subcategory": sub})


def test_long_ingredient_list_is_split_and_every_part_named():
    ingredients = ", ".join(f"Hydroxyethylpiperazine Ethane Sulfonic Acid {i}" for i in range(120))
    parts = bi.split_ingredients(ingredients, HEAD, bi.estimate_tokens)
    assert len(parts) > 1
    assert all(p.startswith(HEAD) and bi.fits(p, bi.estimate_tokens) for p in parts)
    assert parts[0].count("part 1 of")
    # nothing lost
    joined = ", ".join(p.split(": ", 1)[1] for p in parts)
    assert joined.count("Sulfonic") == 120


def test_parentheses_not_split():
    parts = bi.split_ingredients("Water (Aqua, Eau), Glycerin", HEAD, bi.estimate_tokens)
    assert parts == [f"{HEAD}\nIngredients: Water (Aqua, Eau), Glycerin"]


def test_long_overview_split_on_reviews_with_header():
    reviews = "\n".join(f'"{"Lovely cream that hydrates well. " * 12}" (5/5, dry skin)' for _ in range(6))
    desc = f"{HEAD}\nRating: 4.5/5 from 100 reviews\nWhat reviewers say:\n{reviews}"
    chunks = bi.split_overview(desc, HEAD, bi.estimate_tokens)
    assert len(chunks) > 1
    assert all(c.startswith("Product:") and bi.fits(c, bi.estimate_tokens) for c in chunks)
    assert sum(c.count("(5/5, dry skin)") for c in chunks) == 6


def test_short_overview_untouched():
    desc = f"{HEAD}\nRating: 4.5/5\nWhat reviewers say:\n\"Nice.\" (5/5)"
    assert bi.split_overview(desc, HEAD, bi.estimate_tokens) == [desc]


def test_product_types():
    t = bi.product_types(row("Hydra Vizor Invisible Moisturizer Broad Spectrum SPF 30"))
    assert t["type_moisturizer"] and t["type_sunscreen"] and not t["type_eye"]
    t = bi.product_types(row("Retinol Eye Serum", "Eye Care", "Eye Creams & Treatments"))
    assert t["type_eye"] and t["type_serum"]
    t = bi.product_types(row("Hydrating Eye Masks", "Eye Care", "Eye Masks"))
    assert t["type_eye"] and t["type_mask"]
    t = bi.product_types(row("C-Firma Fresh Day Serum", "Treatments", "Face Serums"))
    assert t["type_serum"] and not t["type_moisturizer"]
