import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from understanding import (is_small_talk, looks_constraint_only, parse_analysis, parse_price,  # noqa: E402
                           routine_steps, skin_type_in)


@pytest.mark.parametrize("text,expected", [
    ("i just have budget of 20$", {"max_price": 20}), ("under $20", {"max_price": 20}),
    ("budget 20", {"max_price": 20}), ("i only have 20 bucks", {"max_price": 20}),
    ("20 dollars max", {"max_price": 20}), ("$20 or less", {"max_price": 20}),
    ("between 15 and 30 dollars", {"min_price": 15, "max_price": 30}), ("15-30$", {"min_price": 15, "max_price": 30}),
    ("over $50", {"min_price": 50}), ("I can spend up to $35", {"max_price": 35}),
    ("budget: $60 for the whole routine", {"max_price": 60}), ("no more than $30", {"max_price": 30}),
    ("under ₹2000", {"max_price": 24.1}), ("my budget is 2000 rs", {"max_price": 24.1}),
    ("skincare for women over 40", {}), ("Niacinamide 10% + Zinc 1%", {}), ("SPF 30 moisturizer", {}),
    ("more than 1000 reviews", {}), ("i have 2 kids", {}), ("2-3 times a day", {}),
])
def test_parse_price(text, expected):
    assert parse_price(text) == pytest.approx(expected)


def test_skin_type_and_small_talk_and_routine():
    assert skin_type_in("I have oily skin") == "oily" and skin_type_in("my skin is combo") == "combination"
    assert skin_type_in("vitamin c serum") is None
    assert all(is_small_talk(t) for t in ["hi", "Hello there!", "thanks", "thank you so much", "ok"])
    assert not is_small_talk("hi, I need a cleanser")
    assert routine_steps("night routine please") == ["cleanser", "serum", "moisturizer"]
    assert routine_steps("vitamin c serum") is None


def test_constraint_only_messages():
    assert looks_constraint_only("i just have budget of 20$", False)
    assert looks_constraint_only("for oily skin", False)
    assert not looks_constraint_only("what's a good retinol serum for beginners", False)
    assert not looks_constraint_only("under $20", True)
    assert not looks_constraint_only("best laptop for gaming", False)


def test_parse_analysis_validates():
    a = parse_analysis('noise {"standalone_question": "cheap cleanser", "product_types": ["cleanser", "shoes"], '
                       '"skin_type": "Oily", "max_price_usd": 20, "sort": "price_low", "vegan": "yes"} noise')
    assert a == {"standalone_question": "cheap cleanser", "product_types": ["cleanser"], "skin_type": "oily",
                 "max_price": 20.0, "sort": "price_low"}
    assert parse_analysis("no json here") is None
    assert parse_analysis('{"max_price_usd": -5}') == {}
