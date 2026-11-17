import numpy as np

from languages import (BOTH, ENGLISH, HI, HINDI, HINGLISH, LANGUAGES, has_devanagari, localize, max_tokens,
                       prompt_rule, small_talk_key)
from voice import wav_to_array


def test_detects_devanagari():
    assert has_devanagari("मुझे फेस वॉश चाहिए")
    assert has_devanagari("I need a सीरम")
    assert not has_devanagari("mujhe face wash chahiye")
    assert not has_devanagari("")


def test_localize_modes():
    en = "No match."
    assert localize(en, "no_match", ENGLISH, constraints="x") == en
    assert localize(en, "no_match", HINGLISH, constraints="x") == en
    assert localize(en, "no_match", HINDI, constraints="x") == HI["no_match"].format(constraints="x")
    both = localize(en, "no_match", BOTH, constraints="x")
    assert both.startswith(en) and both.endswith(HI["no_match"].format(constraints="x")) and "---" in both


def test_every_language_has_a_prompt_rule_and_budget():
    for lang in LANGUAGES:
        assert prompt_rule(lang).startswith("LANGUAGE:")
        assert max_tokens(lang) >= 1500
    assert "Devanagari" in prompt_rule(HINDI)
    assert "---" in prompt_rule(BOTH)
    assert "Roman" in prompt_rule(HINGLISH)
    assert max_tokens(BOTH) > max_tokens(HINDI) > max_tokens(ENGLISH)


def test_hindi_messages_keep_their_placeholders():
    HI["scope"].format(topic="makeup")
    HI["no_match"].format(constraints="under $10")


def test_small_talk_key_in_both_scripts():
    assert small_talk_key("thanks a lot") == "thanks"
    assert small_talk_key("धन्यवाद") == "thanks"
    assert small_talk_key("bye") == "bye"
    assert small_talk_key("नमस्ते") == "hello"


def test_wav_decoding_downmixes_and_resamples():
    import io
    import wave
    rate = 8000
    stereo = (np.sin(np.linspace(0, 100, rate)) * 12000).astype("<i2")
    stereo = np.repeat(stereo[:, None], 2, axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(stereo.tobytes())
    audio = wav_to_array(buf.getvalue())
    assert audio.ndim == 1 and abs(len(audio) - 16000) <= 1 and audio.dtype == np.float32
    assert float(np.abs(audio).max()) <= 1.0
