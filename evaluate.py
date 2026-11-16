"""Step 5: evaluate retrieval, guards and (optionally) the generated answers.

    python evaluate.py               retrieval + routing checks (free, no API key)
    python evaluate.py --llm         also generates answers and grades them:
                                       * rule checks: citations valid, every quoted price is
                                         in the context, required behaviour (e.g. suggests a doctor)
                                       * LLM judge: faithfulness, relevance, correctness (1-5)
                                     uses Claude if ANTHROPIC_API_KEY is set, else local Ollama
                                     (a local grader is less reliable than Claude)
    python evaluate.py --calibrate   prints the best similarity of on/off-topic questions
                                     and suggests SIMILARITY_THRESHOLD (rerun after rebuilding)

Results are written to eval_results.md for the README / presentation.
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field

import rag
from llm import ClaudeBackend
from rag import SephoraRAG

JUDGE_MODEL = os.getenv("CLAUDE_JUDGE_MODEL", "claude-sonnet-5-5")
RESULTS_FILE = "eval_results.md"


@dataclass
class Case:
    kind: str
    question: str
    retrieved_text: list[str] = field(default_factory=list)   # must appear in some retrieved chunk
    fallback: str | None = None                                # expected routing (None = answer normally)
    meta_check: str | None = None                              # name of a check on every chunk's metadata
    answer_must: list[str] = field(default_factory=list)       # any of these words must be in the answer (--llm)
    history: list[dict] = field(default_factory=list)


CASES = [
    Case("factual", "What are the ingredients in The Ordinary Niacinamide 10% + Zinc 1%?",
         retrieved_text=["Niacinamide 10% + Zinc 1%"]),
    Case("filtered", "Suggest a vegan moisturizer under $30 for dry skin", meta_check="vegan_moisturizer_30"),
    Case("filtered (SPF)", "I need a moisturizer with SPF under $40", meta_check="spf_moisturizer_40"),
    Case("comparison", "Compare The Ordinary and The INKEY List serums for acne",
         retrieved_text=["The Ordinary", "The INKEY List"]),
    Case("multi-part", "Which retinol serum has the best rating, and what do oily-skin reviewers say about it?",
         retrieved_text=["etinol"], answer_must=["review"]),
    Case("keyword", "Do you have anything with azelaic acid?", retrieved_text=["Azelaic"]),
    Case("unsold brand", "Compare CeraVe and La Roche-Posay cleansers for sensitive skin",
         answer_must=["not sold", "doesn't sell", "don't sell", "does not sell", "not carried", "isn't sold",
                      "aren't sold", "not available", "doesn't carry", "don't carry", "not in"]),
    Case("medical (in scope)", "Is the Drunk Elephant Protini cream safe to use during pregnancy?",
         retrieved_text=["Protini"], answer_must=["doctor", "dermatologist", "healthcare", "physician", "OB"]),
    Case("follow-up", "which of those is the cheapest?",
         history=[{"role": "user", "content": "Suggest a vegan moisturizer under $30 for dry skin"},
                  {"role": "assistant", "content": "Here are some vegan moisturizers under $30 ..."}],
         meta_check="vegan_moisturizer_30"),
    Case("follow-up (budget)", "i just have budget of 20$",
         history=[{"role": "user", "content": "Suggest a hydrating cleanser for dry skin"},
                  {"role": "assistant", "content": "Here are some hydrating cleansers for dry skin ..."}],
         meta_check="cleanser_20"),
    Case("routine", "Build me a morning skincare routine for oily skin under $80", meta_check="under_80"),
    Case("greeting", "hi", fallback="small_talk"),
    Case("no match (hard filter)", "Show me a toner under $1", fallback="no_match"),
    Case("out of scope (beauty)", "What's the best foundation for oily skin?", fallback="out_of_scope"),
    Case("out of scope (beauty)", "Which shampoo is good for hair loss?", fallback="out_of_scope"),
    Case("off-topic", "What is the best laptop for students?", fallback="low_similarity"),
    Case("off-topic", "Who won the cricket world cup?", fallback="low_similarity"),
]

META_CHECKS = {
    "vegan_moisturizer_30": lambda m: m["is_vegan"] and m["price"] <= 30 and m["type_moisturizer"] and not m["type_eye"],
    "spf_moisturizer_40": lambda m: m["price"] <= 40 and m["type_moisturizer"] and m["type_sunscreen"],
    "cleanser_20": lambda m: m["price"] <= 20 and m["type_cleanser"],
    "under_80": lambda m: m["price"] <= 80,
}

CALIBRATION_ON = ["vegan moisturizer for dry skin", "does this serum have niacinamide", "gentle cleanser for acne",
                  "best eye cream for dark circles", "sunscreen that doesn't leave a white cast",
                  "what do reviewers say about Drunk Elephant Protini", "toner with glycolic acid",
                  "hydrating sheet mask under $10"]
CALIBRATION_OFF = ["best laptop for students", "who won the cricket world cup", "how do I cook biryani",
                   "what's the weather in Delhi", "explain quantum computing", "cheap flights to Goa",
                   "how do I fix a flat bicycle tyre", "recommend a good thriller novel"]


# ---------------------------------------------------------------- answer grading
def rule_checks(case: Case, out: dict) -> list[str]:
    """Deterministic faithfulness checks. Returns a list of problems (empty = ok)."""
    problems = []
    answer, chunks = out["answer"], out["retrieved"].chunks
    context = "\n".join(c["text"] for c in chunks)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if chunks and out["fallback"] is None:
        if not cited:
            problems.append("no citations")
        if bad := sorted(n for n in cited if not 1 <= n <= len(chunks)):
            problems.append(f"citations out of range: {bad}")
        asked = {float(p) for p in re.findall(r"\$(\d+(?:\.\d+)?)", case.question)}   # the shopper's budget
        for price in set(re.findall(r"\$\d+(?:\.\d{2})?", answer)):
            value = float(price[1:])
            if value not in asked and not re.search(rf"\${value:.2f}\b|\${value:g}\b", context):
                problems.append(f"price {price} not in context")
    if case.answer_must and not any(w.lower() in answer.lower() for w in case.answer_must):
        problems.append(f"expected one of {case.answer_must[:3]}...")
    return problems


JUDGE_PROMPT = """You grade answers from a RAG shopping assistant for Sephora's skincare catalogue.
Grade ONLY against the numbered context the assistant was given, not your own knowledge.
Return JSON only: {"faithfulness": 1-5, "relevance": 1-5, "correctness": 1-5, "unsupported_claims": ["..."], "comment": "..."}
- faithfulness: every claim is supported by the context (5 = fully, 1 = mostly invented)
- relevance: answers what was asked (5 = directly and completely)
- correctness: the facts quoted (names, prices, ratings, ingredients) match the context exactly; for medical questions, gives no safety claims and suggests a doctor; for unknown brands, says they aren't in the catalogue"""


def judge(bot: SephoraRAG, question: str, out: dict) -> dict:
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(out["retrieved"].chunks))
    msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{out['answer']}"
    grader = bot.backend
    if isinstance(grader, ClaudeBackend):
        grader = ClaudeBackend(model=JUDGE_MODEL)      # a different model grades than answers
    text = grader.complete(JUDGE_PROMPT, [{"role": "user", "content": msg}], max_tokens=1500)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"comment": "judge returned no JSON"}
    except json.JSONDecodeError:
        return {"comment": "judge returned invalid JSON"}


# ---------------------------------------------------------------- runs
def run(use_llm: bool):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # Windows consoles may not be UTF-8
    bot = SephoraRAG()
    if not use_llm:
        bot.backend = None                           # keep the default run free and fast
    elif bot.backend is None:
        sys.exit("--llm needs a model: set ANTHROPIC_API_KEY, or start Ollama with a model pulled "
                 "(e.g. ollama pull qwen2.5:7b)")
    else:
        print(f"answering with {bot.backend.label}")
    rows, passed = [], 0
    for case in CASES:
        # Free mode runs the full pipeline (understanding, guards, retrieval) with no model attached.
        out = bot.answer(case.question, case.history)
        r = out["retrieved"]
        actual_fb = out["fallback"] if out["fallback"] != "no_llm" else None
        ok = actual_fb == case.fallback
        if ok and case.fallback is None:
            ok = all(any(t.lower() in c["text"].lower() for c in r.chunks) for t in case.retrieved_text)
            if case.meta_check:
                ok = ok and bool(r.chunks) and all(META_CHECKS[case.meta_check](c["meta"]) for c in r.chunks)
        problems, grades = [], {}
        if use_llm and out["fallback"] is None and ok:
            problems = rule_checks(case, out)
            grades = judge(bot, case.question, out)
            ok = ok and not problems
        passed += ok
        rows.append((case, out, ok, problems, grades))

        print(f"\n[{'PASS' if ok else 'FAIL'}] ({case.kind}) {case.question}")
        if out.get("search_question", case.question) != case.question:
            print(f"   rewritten: {out['search_question']}")
        print(f"   best similarity {r.best_similarity}  filters {r.filters}  routing={out['fallback']}")
        for c in r.chunks[:5]:
            m = c["meta"]
            print(f"   - {m['product_name'][:60]} | {m['brand']} | ${m['price']:.0f} | "
                  f"{m['rating']:.1f}* ({m['reviews']}) | {m['chunk_type']} | {c['found_by']}")
        if use_llm:
            print("   ANSWER:", out["answer"].replace("\n", "\n   "))
            if problems:
                print("   PROBLEMS:", problems)
            if grades:
                print("   JUDGE:", grades)
    print(f"\nchecks passed: {passed}/{len(CASES)}")
    write_results(rows, use_llm, passed)


def write_results(rows, use_llm: bool, passed: int):
    lines = [f"# Evaluation results\n\n**{passed}/{len(rows)} checks pass** "
             f"({'retrieval + answers' if use_llm else 'retrieval and routing only'}).\n",
             "| Type | Question | Routing | Best sim. | Result |" + (" Faithful | Relevant | Correct | Problems |" if use_llm else ""),
             "|---|---|---|---|---|" + ("---|---|---|---|" if use_llm else "")]
    for case, out, ok, problems, g in rows:
        row = (f"| {case.kind} | {case.question} | {out['fallback'] or 'answered'} | "
               f"{out['retrieved'].best_similarity} | {'PASS' if ok else 'FAIL'} |")
        if use_llm:
            row += (f" {g.get('faithfulness', '-')} | {g.get('relevance', '-')} | {g.get('correctness', '-')} | "
                    f"{'; '.join(problems + g.get('unsupported_claims', [])) or '-'} |")
        lines.append(row)
    if use_llm:
        scored = [g for *_, g in rows if "faithfulness" in g]
        if scored:
            avg = {k: sum(float(g[k]) for g in scored) / len(scored) for k in ("faithfulness", "relevance", "correctness")}
            lines.append("\nAverage judge scores (1-5): " + ", ".join(f"{k} {v:.2f}" for k, v in avg.items()))
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {RESULTS_FILE}")


def calibrate():
    bot = SephoraRAG()
    on = [bot.retrieve(q).best_similarity for q in CALIBRATION_ON]
    off = [bot.retrieve(q).best_similarity for q in CALIBRATION_OFF]
    for label, qs, sims in (("ON-TOPIC", CALIBRATION_ON, on), ("OFF-TOPIC", CALIBRATION_OFF, off)):
        print(label)
        for q, s in sorted(zip(qs, sims), key=lambda x: x[1]):
            print(f"   {s:.3f}  {q}")
    lo, hi = min(on), max(off)
    print(f"\nlowest on-topic {lo:.3f}, highest off-topic {hi:.3f}, current threshold {rag.SIMILARITY_THRESHOLD}")
    if hi < lo:
        print(f"suggested SIMILARITY_THRESHOLD = {(lo + hi) / 2:.2f} (midpoint of the gap)")
    else:
        print("the two groups overlap: keep the threshold and rely on the prompt-level fallback for the overlap")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        run("--llm" in sys.argv)
