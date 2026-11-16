# Data quality report

1,918 products

| Result | Check | Detail |
|---|---|---|
| PASS | Required fields present and non-null | {} |
| PASS | product_id unique | 0 duplicates |
| PASS | brand + name unique | 0 duplicates |
| PASS | No supplements / wellness / teeth whitening / hair removal | 0 rows |
| PASS | No mini-size duplicates |  |
| PASS | Refills only where no full-size product exists | 1 refills left |
| PASS | Prices between $1 and $2,000 | min 3.0, max 1900.0 |
| PASS | Ratings between 1 and 5 (or missing) |  |
| PASS | Rating missing exactly when there are no reviews |  |
| PASS | Small-sample ratings are flagged in the text | 133 products with < 10 reviews |
| PASS | Every description starts with its product name |  |
| PASS | Description length 100-4,000 chars | min 173, max 2244 |
| PASS | Concerns are on their own line, not mixed into highlights |  |
| WARN | No encoding artefacts (Ã, â€, �, HTML entities) | 1 cells |
| PASS | No ingredient list ends mid-list (trailing comma) |  |
| PASS | Devices don't carry a formula's ingredient list | 0 rows |
| PASS | Ingredient status recorded | {'ok': 1839, 'not_applicable': 44, 'missing': 30, 'partial_kit': 5} |
| WARN | Few identical ingredient lists across different products | 50 products share a list with another (check day/night pairs, kits) |
| PASS | Skin-type tag coverage (Sephora 'Best for') | 32% of products, 0 tagged 'sensitive': too sparse for a hard filter |
| PASS | Incentivized review snippets are labelled | 1% of products show one |
| PASS | Chunk ids unique |  |
| PASS | Chunks match the product file | 0 products without chunks, 0 chunks for removed products |
| PASS | Every chunk <= 2100 chars (embedder limit is 512 tokens) | 0 too long: [] |
| PASS | Every chunk names its product |  |
| PASS | Every product has an overview chunk |  |
