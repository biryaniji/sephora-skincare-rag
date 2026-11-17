# Sephora Skincare Advisor: a RAG chatbot

A Streamlit chatbot that answers shopper questions about **Sephora's skincare catalogue**
using retrieved product data and customer reviews, cites its sources, and says so when the
data has no answer.

## Use case
- **Company / industry:** Sephora, Beauty & Personal Care
- **User:** an online shopper with a skin type, concern and budget, facing hundreds of similar products
- **Purpose:** product discovery and product Q&A (ingredients, skin type, concerns, price, ratings, reviews)
- **Out of scope:** medical advice, makeup / hair / fragrance, orders, live stock, other retailers

## Dataset
[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle, scraped from sephora.com in March 2023). Real data, not synthetic.

| File | Rows | Used for |
|---|---|---|
| `product_info.csv` | 8,494 products, 27 columns | name, brand, category, price, rating, highlights, ingredients |
| `reviews_*.csv` (5 files) | about 1.1M reviews | review text, rating, recommended, skin type of reviewer |

**Cleaning (`prepare_data.py`)**. Every step and its row count is written to `data/processed/cleaning_report.md`.
- Kept `primary_category == "Skincare"`. Dropped gift sets and the Mini Size category.
- **Dropped non-skincare and health-claim items** (Beauty Supplements, Holistic Wellness, Teeth Whitening,
  Hair Removal: 53 products, e.g. prenatal and menopause supplements). The bot must not advise on them.
- Dropped "Mini" variants anywhere in the name (device models such as "LUNA Mini 3" are kept) and refills
  whose full-size product is listed. Deduplicated by `product_id` and by brand + name.
- **Ingredients:** rebuilt the INCI list from *all* list fragments (the scraper sometimes splits one list
  over several elements), removed marketing notes and disclaimers, labelled each component of a kit,
  and removed "ingredients" from devices. Each product gets an `ingredients_status`
  (`ok`, `partial_kit`, `missing`, `not_applicable`).
- **Highlights** are split into *skin types* (Sephora "Best for"), *concerns* ("Good for") and other tags,
  each on its own line, so the LLM can't mix them up.
- **Ratings** from fewer than 10 reviews are marked as unreliable in the text. Missing ratings stay missing.
- **Reviews:** kept 80–600 characters, deduplicated. Per product: the 2 most helpful positive reviews and the
  most helpful critical one, **preferring organic reviews**. Incentivized ones ("received this product free")
  are labelled. Instead of "most reviewers are combination skin" (a base rate), we store the
  **recommend rate per skin type**, e.g. `dry 93% (n=210)`.
- The dataset has no long description column, so we build one per product from all of the above.
- Assumptions: prices are USD from March 2023; reviewer skin type is self-reported.

**Data-quality checks (`check_data.py`)**: 20+ automated PASS/WARN/FAIL checks: unique ids, no excluded
categories, no minis, price and rating ranges, ingredient lists not cut mid-list, devices without formulas,
encoding, and for the chunks: every product indexed, every chunk under the size limit and naming its product.
Results go to `data/processed/quality_report.md`. On the original processed data, 6 checks failed. Each one is
fixed by the new cleaning.

## Pipeline
```
Kaggle CSVs → clean → build description → chunk (token-aware) → embed (bge-small) → Chroma
                                                                         ↘ BM25 index
question → follow-up? rewrite with chat history → scope guard (makeup/hair/fragrance)
         → filters (price range, vegan, clean, fragrance-free, brand, product type)
         → [2+ brands? one sub-search per brand]
         → vector top-20 + BM25 top-20 → RRF fusion (+ named-product boost, rating re-rank if "best rated")
         → similarity check (fallback) → guard notes (unsold brand, medical) → Claude → cited answer + sources
```

| Stage | Choice | Why |
|---|---|---|
| Chunking | **Record-based.** Each product gets an *overview* chunk (facts, skin types, concerns, highlights, reviews) and an *ingredients* chunk. Chunks are limited to **500 tokens measured with bge's own tokenizer**. Long overviews split on review boundaries and long ingredient lists split between ingredients ("part 1 of 2"), with the product header repeated in every part | Fixed-size chunks would cut products in half and mix products. A separate ingredients chunk stops a 700-character chemical list from diluting the overview embedding. bge-small reads 512 tokens and silently drops the rest, so the cap is in tokens, not characters. The repeated header plays the role of overlap: every chunk names its product. |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d, free, local, 512-token limit) | Strong retrieval quality for its size |
| Vector store | Chroma (persistent, cosine) with metadata: price, rating, review count, brand, category, vegan, clean, fragrance-free, device, and **product-type flags** (`type_serum`, `type_moisturizer`, `type_sunscreen`, `type_eye`, …) | Metadata filters handle "under $30", "vegan", "moisturizer with SPF" (moisturizer AND sunscreen), which embeddings can't do |
| Retrieval | **Hybrid RAG**: vector + BM25 merged with Reciprocal Rank Fusion (k=60), max 2 chunks per product. **Query decomposition** for comparisons of 2+ brands | BM25 matches exact ingredient and product names ("azelaic acid", "Protini"); vectors match meaning ("for flaky skin") |
| Ranking extras | "Best/top rated" → re-rank candidates by a Bayesian rating (pulls ratings with few reviews toward the average) | A 5.0★ from 2 reviews shouldn't beat a 4.6★ from 2,000 |
| Generation | Claude (`claude-opus-5-5`, change with `CLAUDE_MODEL`), grounding system prompt, numbered context, **citations [n]**. Follow-ups are rewritten with `claude-haiku-5-5` | Answers only from context, quotes names and prices exactly, no medical advice |
| Fallback | (1) Off-topic: best similarity < 0.65 → fallback without calling the LLM. (2) Makeup / hair / fragrance → scope message, no LLM. (3) Price / vegan / clean constraints are **never relaxed**: no match → say so. (4) The prompt tells the model to say what the context doesn't cover. | Cheap, predictable refusals, and the LLM never sees a context it would have to invent around |

## Understanding what the shopper means (`understanding.py`)
Shoppers don't type clean queries, so each message goes through two layers before retrieval:
1. **Rules (always on):** prices in many phrasings and currencies ("budget of 20$", "i only have 20 bucks",
   "$20 or less", "15-30$", "under ₹2000" ≈ $24), skin type ("my skin is combo"), routines, "cheapest",
   greetings. The rules don't mistake "SPF 30", "Zinc 1%" or "women over 40" for prices.
   **Typos are fixed first** ("fac wash", "mosturizer", "serrum", "cleanzer", "niacinimide", "skn"). Words are matched to
   skincare terms, product-name words and brand words, and a word is only changed if the fix is used far more often
   in the catalogue. Brand names and real ingredients such as "retinal" are never touched.
2. **LLM analyzer (when a model is connected):** reads the conversation and returns JSON: a standalone question,
   product types, skin type, budget, vegan / fragrance-free, routine, sort order. Ollama runs it in JSON mode.
   Invalid output is discarded and the rules take over.

**Conversation memory.** A follow-up such as "i just have budget of 20$" or "only fragrance-free ones" inherits the
previous request (product type, skin type, constraints) and the newest message wins on conflicts. Skin type is
remembered for the whole chat. "Which of those is cheapest?" reuses the products from the last answer, sorted by
price. "Build me a routine" runs one search per step (cleanser → serum → moisturizer → SPF) within the budget.
Clear skincare questions are never refused by the similarity threshold. The threshold only blocks messages with no
skincare vocabulary.

## Answer engine: Claude or a free local model
`llm.py` has two backends, chosen in the app sidebar (or with `LLM_BACKEND=auto|claude|ollama|none`):
- **Claude** (`claude-opus-5-5`) when `ANTHROPIC_API_KEY` is set. It gives the best grounding and refusals.
- **Ollama**, a free local model on your laptop. Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b`
  (or `llama3.2:3b` on a slower laptop). The app finds the models you've pulled and picks the best one.
- **Search only** when neither is available: the app lists the matching products with prices and ratings.

## The interface (`app.py`)
- Starter questions grouped by type (discover, ingredients, compare, reviews, guardrails)
- Streaming answers with **citation badges** that match the **product cards** (price, stars, review count,
  vegan / clean / fragrance-free tags, "few reviews" warning)
- Status badges: grounded in N sources, medical guard, brand not sold here, off-topic (no LLM call)
- **"How this answer was found"**: the pipeline trace for each answer (follow-up rewrite, filters, hybrid
  search counts, similarity vs threshold, guards, model, timings) and the exact source text sent to the model
- Sidebar: answer engine and model picker, shopping preferences (skin type, budget, vegan, clean, fragrance-free)
  that act as filters, follow-up suggestions, thumbs feedback, export of the chat as Markdown

## Voice and languages (`voice.py`, `languages.py`)
- **Voice input**: the chat box has a mic button (`st.chat_input(accept_audio=True)`). The recording is transcribed
  on the laptop by Whisper (no audio leaves the machine) and then goes through the same search as a typed question.
  English uses `openai/whisper-base.en` (~290 MB); Hindi and mixed modes use the multilingual `openai/whisper-small`
  (~1 GB). Both download from Hugging Face the first time they're needed. `python check_voice.py` is a quick check on
  Windows: it speaks a sentence with Windows text-to-speech and prints what Whisper heard.
- **Answer language** (sidebar): English, हिन्दी (Hindi), English + हिन्दी (English answer, then the same in Hindi)
  or Hinglish (Hindi in Roman letters). The catalogue, index and filter rules stay English: a Hindi question is turned
  into an English search request by the LLM query analyzer, and only the answer is written in the chosen language.
  Product names, brands, prices, ingredient names and `[n]` citations are kept exactly as in the data. Fixed
  messages (no match, off-topic, greetings) have Hindi versions.
- Limits: Hindi questions need an LLM (Claude works best; small local models write weak Hindi). Without one, the app
  says so in both languages. Buttons and headings are still English.

## Run it
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py      # downloads the Kaggle data via kagglehub, cleans it -> cleaning_report.md
python check_data.py        # data-quality checks on the products
python build_index.py       # chunks + embeds + stores in data/chroma (a few minutes)
python check_data.py        # again: now also checks the chunks
python evaluate.py --calibrate   # check the off-topic threshold after a rebuild
export ANTHROPIC_API_KEY=...     # optional: without it the app uses Ollama, or lists products
streamlit run app.py
python evaluate.py          # retrieval + routing checks (free)
python evaluate.py --llm    # also generates and grades answers (Claude or Ollama) -> eval_results.md
python -m pytest tests -q   # unit tests (no models or API needed)
```

## Test questions
| Type | Question | Expected behaviour |
|---|---|---|
| Factual | What are the ingredients in The Ordinary Niacinamide 10% + Zinc 1%? | Ingredients chunk for that product is retrieved |
| Filtered | Suggest a vegan moisturizer under $30 for dry skin | Only vegan moisturizers ≤ $30, no eye creams |
| Filtered (SPF) | I need a moisturizer with SPF under $40 | Moisturizers that are also SPF, ≤ $40 |
| Comparison | Compare The Ordinary and The INKEY List serums for acne | Evidence from both brands |
| Multi-part | Which retinol serum has the best rating, and what do oily-skin reviewers say? | Well-reviewed retinol serums, rating with review count, reviewer quotes |
| Keyword | Do you have anything with azelaic acid? | BM25 finds the exact ingredient |
| Unsold brand | Compare CeraVe and La Roche-Posay cleansers | Says Sephora doesn't sell them, offers comparable cleansers |
| Medical (in scope) | Is the Drunk Elephant Protini cream safe during pregnancy? | Retrieves Protini; no safety claim; suggests a doctor |
| Follow-up | "which of those is cheapest?" after the vegan question | Rewritten into a standalone question; filters carried over |
| No match | Show me a toner under $1 | Says nothing matches; price not relaxed |
| Out of scope (beauty) | Best foundation for oily skin? | Scope message, no LLM call |
| Off-topic | What is the best laptop for students? | Fallback without calling the LLM |

`evaluate.py --llm` grades each answer two ways. **Rule checks:** citations point to real sources, every
quoted price is in the context, and required behaviour such as "suggests a doctor". **LLM judge:**
faithfulness, relevance and correctness on a 1–5 scale, with any unsupported claims listed. Results are in `eval_results.md`.

## Evaluation results
Rerun after rebuilding the index (`python evaluate.py` and `python evaluate.py --llm`). The numbers are in
`eval_results.md`.

*Baseline before the data and retrieval fixes (7 Nov 2026):* 8/8 retrieval checks passed. Off-topic
questions scored 0.49 / 0.45 best similarity and on-topic ones 0.70+, so the 0.55 threshold sat in the gap.
**After the fixes (16 Nov 2026, `--calibrate`):** on-topic questions scored 0.726–0.848, off-topic 0.488–0.614.
The old 0.55 threshold would have let "how do I fix a flat bicycle tyre" (0.574) and "how do I cook biryani" (0.614)
through, so it was raised to **0.65**: above every off-topic score, and below the 0.67 midpoint so vaguer
skincare questions still pass. Cleaning: 8,494 → 1,918 products, 3,991 chunks (longest 500 tokens), all 25 data checks pass.

## Failure cases found and what we changed
1. **One brand crowded out the other in comparisons.** *Fixed:* query decomposition, with one sub-search per brand.
2. **Wrong product type for "moisturizer" queries** (eye creams ranked high). *Fixed:* product-type flags;
   eye products are excluded unless the question mentions eyes.
3. **"Moisturizer with SPF" only searched the Sunscreen category** and missed the 42 SPF moisturizers.
   *Fixed:* types are ANDed (moisturizer AND sunscreen) instead of mapping to one category.
4. **Supplements and wellness items in a skincare bot** (prenatal, menopause, detox). *Fixed:* dropped in cleaning.
5. **Wrong or truncated ingredient lists** (a list starting at "EDTA", a device carrying a serum's list).
   *Fixed:* ingredients rebuilt from all fragments; devices have none; kits are labelled.
6. **Ratings from a handful of reviews looked "best".** *Fixed:* flagged in the text; review-weighted re-rank.
7. **Brands Sephora doesn't sell** (CeraVe, La Roche-Posay) retrieved other brands silently. *Fixed:* a list of
   popular unsold brands, checked against the catalogue; the LLM is told to say so first.
8. **Medical questions about in-catalogue products.** Retrieval matched reviews that mention pregnancy.
   *Fixed:* medical terms are removed from the search query (we search for the product), a guard note forbids
   safety claims, and the prompt treats reviews as anecdotes.
9. **Follow-up questions** ("which of those is cheapest?") retrieved on the follow-up text alone.
   *Fixed:* rewritten into a standalone question with the chat history.
10. **Budget follow-ups were treated as off-topic.** "i just have budget of 20$" after a cleanser question was
    searched on its own, scored below the threshold and got the fallback. *Fixed:* conversation-aware query
    understanding (rules + LLM analyzer) and constraint carry-over; added to the evaluation.
11. **Still open: stale data.** Prices and stock are a March 2023 snapshot. The UI and prompt say so.
12. **Still open: sensitive skin.** Sephora's "Best for" tags never include "sensitive", so those questions rely
    on review text and highlights, not a filter.
