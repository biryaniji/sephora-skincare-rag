# Cleaning report

| Step | Products |
|---|---|
| Start: all Sephora products | 8,494 |
| Keep primary_category == Skincare | 2,420 (-6,074) |
| Drop gift sets and the Mini Size category (duplicates of full-size items) | 2,112 (-308) |
| Drop non-skincare / health-claim items: Beauty Supplements, Hair Removal, Holistic Wellness, Teeth Whitening | 2,043 (-69) |
| Drop 'Mini' size variants anywhere in the name (devices like 'LUNA Mini 3' are kept: different models) | 1,976 (-67) |
| Deduplicate by product_id | 1,976 (-0) |
| Deduplicate by brand + name | 1,965 (-11) |
| Drop refills whose full-size product is listed | 1,960 (-5) |
| Drop products with neither highlights nor ingredients | 1,918 (-42) |

Reviews for kept products: 945,641; after length filter (80-600 chars) and dedup: 799,768; incentivized (free product): 12.9%

**Final: 1,918 products.** Average description length 1427 characters.

Ingredient status: ok 1839, not_applicable 44, missing 30, partial_kit 5

Assumptions: prices are USD from the March 2023 scrape; reviewer skin type is self-reported; ratings from fewer than 10 reviews are marked as unreliable in the text.
