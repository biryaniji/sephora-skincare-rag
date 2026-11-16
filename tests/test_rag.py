"""Retrieval-logic tests on the real catalogue, with fake vector store / embedder.
Run:  python -m pytest tests -q"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fakes import FakeCollection, FakeEmbedder, install_bm25_stub  # noqa: E402

install_bm25_stub()
import build_index  # noqa: E402
import rag  # noqa: E402

PRODUCTS = os.path.join(os.path.dirname(HERE), "data/processed/skincare_products.csv")


@pytest.fixture(scope="module")
def bot():
    if not os.path.exists(PRODUCTS):
        pytest.skip("run prepare_data.py first (needs data/processed/skincare_products.csv)")
    df = pd.read_csv(PRODUCTS, dtype={"product_id": str})
    chunks = build_index.make_chunks(df)
    emb = FakeEmbedder()
    b = rag.SephoraRAG(collection=FakeCollection(chunks, emb), embedder=emb, chunks=chunks)
    b.backend = None
    return b


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(rag, "SIMILARITY_THRESHOLD", 0.0)   # fake embeddings aren't calibrated


# ------------------------------------------------------------- query parsing
def test_spf_moisturizer_is_both_types():
    f = rag.parse_filters("Moisturizer with SPF 30 under $40")
    assert set(f["types"]) == {"sunscreen", "moisturizer"} and f["max_price"] == 40 and f["exclude_eye"]


def test_eye_mask():
    f = rag.parse_filters("a cooling eye mask")
    assert set(f["types"]) == {"eye", "mask"} and "exclude_eye" not in f


def test_price_range_and_flags():
    f = rag.parse_filters("fragrance-free vegan cleanser between $20 and $50")
    assert (f["min_price"], f["max_price"]) == (20, 50)
    assert f["vegan"] and f["fragrance_free"] and f["types"] == ["cleanser"]


def test_moisturizing_adjective_is_not_a_type():
    assert "types" not in rag.parse_filters("a moisturizing cleanser") or \
        rag.parse_filters("a moisturizing cleanser")["types"] == ["cleanser"]


@pytest.mark.parametrize("q,topic", [
    ("best foundation for oily skin", "makeup"),
    ("which shampoo stops hair loss", "hair care"),
    ("a perfume for my mom", "fragrance"),
    ("Does Protini contain fragrance?", None),
    ("tinted moisturizer with SPF", None),
    ("lip mask for chapped lips", None),
])
def test_out_of_scope(q, topic):
    assert rag.out_of_scope_topic(q) == topic


def test_medical_terms():
    assert "pregnancy" in rag.medical_terms("Is Protini safe during pregnancy?")
    assert rag.medical_terms("vegan moisturizer for dry skin") == []


# ------------------------------------------------------------- brands
def test_two_brands_found(bot):
    assert set(bot.brands_in("Compare The Ordinary and The INKEY List serums")) == {"The Ordinary", "The INKEY List"}


def test_everyday_word_brand(bot):
    assert bot.brands_in("a fresh smelling gel moisturizer") == []
    assert bot.brands_in("what's good by fresh?") == ["fresh"]


def test_unsold_brands(bot):
    assert set(bot.unsold_brands_in("Compare CeraVe and La Roche-Posay cleansers")) == {"CeraVe", "La Roche-Posay"}
    assert bot.unsold_brands_in("Avene thermal water") == ["Avène"]


# ------------------------------------------------------------- retrieval
def test_filters_are_respected(bot):
    r = bot.retrieve("Suggest a vegan moisturizer under $30 for dry skin")
    assert r.chunks
    for c in r.chunks:
        m = c["meta"]
        assert m["is_vegan"] and m["price"] <= 30 and m["type_moisturizer"] and not m["type_eye"]


def test_spf_moisturizers_are_found(bot):
    r = bot.retrieve("moisturizer with SPF for oily skin")
    assert r.chunks and all(c["meta"]["type_moisturizer"] and c["meta"]["type_sunscreen"] for c in r.chunks)


def test_comparison_returns_both_brands(bot):
    r = bot.retrieve("Compare The Ordinary and The INKEY List serums for acne")
    assert {c["meta"]["brand"] for c in r.chunks} >= {"The Ordinary", "The INKEY List"}


def test_single_brand_filter(bot):
    r = bot.retrieve("What are the ingredients in The Ordinary Niacinamide 10% + Zinc 1%?")
    assert r.filters["brand"] == "The Ordinary"
    assert any("Niacinamide 10% + Zinc 1%" in c["text"] and c["meta"]["chunk_type"] == "ingredients" for c in r.chunks)


def test_keyword_ingredient(bot):
    r = bot.retrieve("Do you have anything with azelaic acid?")
    assert any("azelaic" in c["text"].lower() for c in r.chunks)


def test_best_rated_prefers_well_reviewed(bot):
    r = bot.retrieve("Which retinol serum has the best rating?")
    assert r.filters.get("sort")
    assert r.chunks[0]["meta"]["reviews"] >= 50


def test_price_is_never_relaxed(bot):
    out = bot.answer("a toner under $1")
    assert out["fallback"] == "no_match" and "under $1" in out["answer"]


# ------------------------------------------------------------- answer() routing (no LLM key)
def test_answer_out_of_scope(bot):
    assert bot.answer("best foundation for oily skin")["fallback"] == "out_of_scope"


def test_answer_notes(bot):
    out = bot.answer("Compare CeraVe and La Roche-Posay cleansers for sensitive skin")
    assert any("doesn't sell" in n for n in out["notes"])
    out = bot.answer("Is the Drunk Elephant Protini cream safe to use during pregnancy?")
    assert any("Medical topic" in n for n in out["notes"]) and "dermatologist" in out["answer"]


def test_follow_up_uses_history(bot):
    history = [{"role": "user", "content": "vegan moisturizer under $30 for dry skin"},
               {"role": "assistant", "content": "Here are three options..."}]
    out = bot.answer("which of those is cheapest?", history)
    assert "vegan moisturizer" in out["search_question"]
    assert all(c["meta"]["price"] <= 30 for c in out["retrieved"].chunks)


def test_medical_question_retrieves_the_product(bot):
    assert rag.strip_medical("Is the Drunk Elephant Protini cream safe to use during pregnancy?") == \
        "Drunk Elephant Protini cream"
    out = bot.answer("Is the Drunk Elephant Protini cream safe to use during pregnancy?")
    assert "Protini" in out["retrieved"].chunks[0]["meta"]["product_name"]


# ------------------------------------------------------------- preferences and generation
def test_preferences_act_as_filters(bot):
    r = bot.retrieve("a gentle cleanser", {"max_price": 25, "vegan": True, "skin_type": "Oily"})
    assert r.chunks and all(c["meta"]["price"] <= 25 and c["meta"]["is_vegan"] for c in r.chunks)
    assert rag.merge_preferences({"max_price": 10.0}, {"max_price": 50})["max_price"] == 10.0  # question wins


class EchoBackend:
    label = "fake · echo"

    def __init__(self, analysis=None):
        self.seen = None
        self.analysis = analysis

    def stream(self, system, messages, max_tokens=1500):
        self.seen = messages
        for piece in ["Try ", "this ", "one [1]."]:
            yield piece

    def complete(self, system, messages, max_tokens=1500, fast=False, json_mode=False):
        return self.analysis or "not json"


def test_streaming_generation(bot):
    bot.backend = EchoBackend()
    try:
        prep = bot.prepare("Suggest a vegan moisturizer under $30 for dry skin")
        assert prep["fallback"] is None and prep["messages"][-1]["role"] == "user"
        pieces = list(bot.generate(prep))
        assert pieces == ["Try ", "this ", "one [1]."]
        assert prep["answer"] == "Try this one [1]." and prep["model"] == "fake · echo"
        assert "generation_ms" in prep["timings"] and prep["steps"][-1][0] == "Answer written"
    finally:
        bot.backend = None


# ------------------------------------------------------------- conversations
def _turn(bot, question, prev=None, history=None):
    history = list(history or [])
    if prev:
        history += [{"role": "user", "content": prev["question"]}, {"role": "assistant", "content": "Here are some options."}]
    return bot.answer(question, history, previous=prev), history


def test_screenshot_conversation_budget_follow_up(bot):
    first, _ = _turn(bot, "Suggest a hydrating cleanser for dry skin")
    second, _ = _turn(bot, "i just have budget of 20$", prev=first)
    assert second["fallback"] in (None, "no_llm")                     # not "off-topic" any more
    f = second["retrieved"].filters
    assert f["max_price"] == 20 and f["types"] == ["cleanser"]
    assert second["retrieved"].chunks
    assert all(c["meta"]["price"] <= 20 and c["meta"]["type_cleanser"] for c in second["retrieved"].chunks)
    assert "cleanser" in second["understood"] and "≤ $20" in second["understood"]


def test_follow_up_without_previous_result_uses_history(bot):
    history = [{"role": "user", "content": "vegan moisturizer under $30"}, {"role": "assistant", "content": "..."}]
    out = bot.answer("my budget is only 15 dollars", history)
    f = out["retrieved"].filters
    assert f["max_price"] == 15 and f["vegan"] and f["types"] == ["moisturizer"]   # newest budget wins


def test_which_of_those_reuses_previous_products(bot):
    first, _ = _turn(bot, "Suggest a vegan moisturizer under $30 for dry skin")
    prev_ids = [c["meta"]["product_id"] for c in first["retrieved"].chunks]
    second, _ = _turn(bot, "which of those is the cheapest?", prev=first)
    chunks = second["retrieved"].chunks
    assert chunks[0]["found_by"] == "previous answer" and chunks[0]["meta"]["product_id"] in prev_ids
    reused = [c for c in chunks if c["found_by"] == "previous answer"]
    assert [c["meta"]["price"] for c in reused] == sorted(c["meta"]["price"] for c in reused)


def test_small_talk(bot):
    out = bot.answer("hi")
    assert out["fallback"] == "small_talk" and "skin type" in out["answer"]
    assert bot.answer("thanks!", [{"role": "user", "content": "x"}])["fallback"] == "small_talk"


def test_routine_searches_each_step(bot):
    out = bot.answer("build me a morning skincare routine for oily skin, budget $80")
    f = out["retrieved"].filters
    assert f["routine"] == ["cleanser", "serum", "moisturizer", "sunscreen"] and f["max_price"] == 80
    found = {t for c in out["retrieved"].chunks for t in ("cleanser", "serum", "moisturizer", "sunscreen")
             if c["meta"][f"type_{t}"]}
    assert found == {"cleanser", "serum", "moisturizer", "sunscreen"}


def test_cheapest_sorts_by_price(bot):
    out = bot.answer("cheapest vitamin c serum")
    prices = [c["meta"]["price"] for c in out["retrieved"].chunks]
    assert out["retrieved"].filters["sort"] == "price_low" and prices[0] == min(prices)


def test_skincare_question_not_blocked_by_threshold(bot, monkeypatch):
    monkeypatch.setattr(rag, "SIMILARITY_THRESHOLD", 0.99)
    assert bot.answer("anything for dark spots?")["fallback"] != "low_similarity"
    assert bot.answer("What is the best laptop for students?")["fallback"] == "low_similarity"


def test_llm_analyzer_understands_vague_requests(bot):
    analysis = ('{"standalone_question": "affordable treatment for acne and pimples", "is_skincare": true, '
                '"is_small_talk": false, "product_types": ["serum"], "skin_type": "oily", "concerns": ["acne"], '
                '"max_price_usd": null, "min_price_usd": null, "vegan": false, "fragrance_free": false, '
                '"clean": false, "routine": false, "sort": "price_low"}')
    bot.backend = EchoBackend(analysis)
    try:
        prep = bot.prepare("got something for my pimples that won't break the bank? my face gets greasy")
        assert prep["search_question"] == "affordable treatment for acne and pimples"
        f = prep["retrieved"].filters
        assert f["types"] == ["serum"] and f["sort"] == "price_low"
        assert "oily skin" in prep["understood"] and "fake · echo" in prep["steps"][0][1]
        # malformed analyzer output falls back to the rules
        bot.backend = EchoBackend("sorry, I can't do JSON")
        prep = bot.prepare("vegan cleanser under $20")
        assert prep["retrieved"].filters["max_price"] == 20 and "rules" in prep["steps"][0][1]
    finally:
        bot.backend = None


def test_skin_type_remembered_across_requests(bot):
    history = [{"role": "user", "content": "I have dry skin, need a cleanser"}, {"role": "assistant", "content": "..."}]
    out = bot.answer("build me a night routine under $70", history)
    assert "dry skin" in out["understood"] and out["retrieved"].filters["routine"] == ["cleanser", "serum", "moisturizer"]


def test_off_topic_after_skincare_is_not_a_follow_up(bot, monkeypatch):
    monkeypatch.setattr(rag, "SIMILARITY_THRESHOLD", 0.45)     # fake-embedder scale
    first, _ = _turn(bot, "Suggest a hydrating cleanser for dry skin")
    u = bot.understand("best laptop for gaming", [{"role": "user", "content": first["question"]}], first)
    assert not u["follow_up"] and not u["skincare"] and u["skin_type"] is None
    out, _ = _turn(bot, "best laptop for gaming", prev=first)
    assert out["fallback"] == "low_similarity"


def test_what_about_switches_type_but_keeps_budget(bot):
    first, _ = _turn(bot, "vegan cleanser under $20 for oily skin")
    out, _ = _turn(bot, "what about toners?", prev=first)
    f = out["retrieved"].filters
    assert f["types"] == ["toner"] and f["max_price"] == 20 and f["vegan"]


# ------------------------------------------------------------- typos
@pytest.mark.parametrize("typo,fixed", [
    ("i have a dry skin reccomend ms a fac wash", "face"), ("mosturizer for oily skn", "moisturizer"),
    ("best serrum with niacinimide", "niacinamide"), ("gentle cleanzer for sensative skin", "cleanser"),
    ("sunscren that doesnt leave white cast", "sunscreen"), ("exfoliater for blackheds", "blackheads"),
])
def test_typos_are_fixed(bot, typo, fixed):
    text, changes = bot.speller.fix(typo)
    assert fixed in text.lower() and changes


@pytest.mark.parametrize("text", ["retinal serum", "Drunk Elephant Protini", "Sunday Riley Good Genes",
                                  "Glow Recipe watermelon toner", "Kiehl's ultra facial cream", "eye creme"])
def test_real_words_and_names_untouched(bot, text):
    assert bot.speller.fix(text)[1] == []


def test_screenshot_misspelled_face_wash(bot):
    out = bot.answer("i have a dry skin reccomend ms a fac wash")
    assert out["fallback"] in (None, "no_llm")
    assert out["retrieved"].filters["types"] == ["cleanser"] and "dry skin" in out["understood"]
    assert all(c["meta"]["type_cleanser"] for c in out["retrieved"].chunks)
    assert out["steps"][0][0] == "Spelling"
