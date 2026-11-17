"""Answer languages: English, Hindi, English + Hindi, Hinglish.

The catalogue, the search index and the filter rules are all English. So a Hindi question is first
translated to an English search request by the LLM (see ANALYZER_PROMPT in understanding.py), the search runs
in English, and only the written answer (and the app's fixed messages) are produced in the chosen language.
"""
import re

ENGLISH, HINDI, BOTH, HINGLISH = "English", "हिन्दी (Hindi)", "English + हिन्दी", "Hinglish"
LANGUAGES = [ENGLISH, HINDI, BOTH, HINGLISH]

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text or ""))


# Added to the system prompt. Product names, brands, prices, ingredient names and [n] citations stay as in the data.
KEEP = ("Keep product names, brand names, prices (like $24.00), ingredient names and the citation numbers like [2] "
        "exactly as they appear in the context, in English letters. Never translate or transliterate them.")
PROMPT_RULES = {
    ENGLISH: "LANGUAGE: Answer in English, even if the customer wrote in Hindi or Hinglish.",
    HINDI: ("LANGUAGE: Write the whole answer in simple, friendly Hindi in Devanagari script, even if the customer wrote "
            "in English. " + KEEP + " Use a gender-neutral tone (avoid first-person verb forms that mark gender)."),
    BOTH: ("LANGUAGE: Write the answer twice. First write it fully in English. Then write a line containing only --- "
           "and then the same answer in simple Hindi in Devanagari script. " + KEEP + " Add nothing else, and keep the "
           "Hindi part just as complete as the English part."),
    HINGLISH: ("LANGUAGE: Write in Hinglish: Hindi written in Roman (English) letters, mixed naturally with English "
               "skincare words, the way friends chat, for example 'Yeh serum oily skin ke liye accha hai [2]'. " + KEEP),
}


def prompt_rule(language: str) -> str:
    return PROMPT_RULES.get(language, PROMPT_RULES[ENGLISH])


def max_tokens(language: str) -> int:
    """Hindi text takes many more tokens than English, and the bilingual answer is written twice."""
    return {ENGLISH: 1500, HINGLISH: 2000, HINDI: 3500, BOTH: 4500}.get(language, 1500)


# Fixed messages the app shows without calling the LLM.
HI = {
    "fallback": ("मुझे Sephora स्किनकेयर कैटलॉग के डेटा में यह नहीं मिला। Sephora स्किनकेयर प्रोडक्ट्स के बारे में आप मुझसे "
                 "पूछ सकते हैं: इंग्रीडिएंट्स, स्किन टाइप, समस्याएँ, कीमत, रेटिंग और रिव्यू में लोग क्या कहते हैं।"),
    "scope": ("यह {topic} जैसा लगता है, जो मेरे डेटा में नहीं है। यहाँ सिर्फ़ Sephora के **स्किनकेयर** कैटलॉग की जानकारी है "
              "(क्लींज़र, सीरम, मॉइस्चराइज़र, सनस्क्रीन, मास्क, आई केयर आदि)। कृपया स्किनकेयर से जुड़ा सवाल पूछें।"),
    "no_match": "कैटलॉग में {constraints} से मेल खाने वाला कोई स्किनकेयर प्रोडक्ट नहीं मिला। बजट बढ़ाकर या शर्तें कम करके देखें।",
    "no_llm_list": ("_कोई भाषा मॉडल जुड़ा नहीं है, इसलिए कैटलॉग से मेल खाने वाले प्रोडक्ट यहाँ दिए हैं "
                    "(लिखित जवाब के लिए Ollama चलाएँ या API key जोड़ें):_"),
    "medical": "**चिकित्सा से जुड़े सवालों के लिए कृपया त्वचा विशेषज्ञ या डॉक्टर से पूछें।**",
    "needs_llm": ("हिंदी में सवाल समझने के लिए भाषा मॉडल चाहिए। साइडबार में **Claude** (API key) या **Ollama** चुनें, "
                  "या अपना सवाल अंग्रेज़ी में लिखें।"),
    "hello": ("नमस्ते! अपनी **त्वचा का प्रकार**, **समस्या** और **बजट** बताइए, जैसे: *तैलीय, मुंहासों वाली त्वचा के लिए "
              "$25 से कम का जेंटल क्लींज़र*। किसी भी प्रोडक्ट के इंग्रीडिएंट्स और रिव्यू भी पूछ सकते हैं।"),
    "thanks": "कोई बात नहीं! Sephora स्किनकेयर के बारे में कुछ और पूछना हो तो बताइए: दूसरा प्रोडक्ट, अलग बजट या रिव्यू।",
    "bye": "अलविदा! स्किनकेयर चुनने में मदद चाहिए तो कभी भी आइए।",
}
HI_NEEDS_LLM_EN = ("To understand questions written in Hindi I need a language model. Choose **Claude** (API key) or "
                   "**Ollama** in the sidebar, or write your question in English.")


def localize(en: str, hi_key: str, language: str, **fmt) -> str:
    """The English message, the Hindi one, or both, depending on the chosen language."""
    hi = HI[hi_key].format(**fmt) if fmt else HI[hi_key]
    if language == HINDI:
        return hi
    if language == BOTH:
        return f"{en}\n\n---\n\n{hi}"
    return en


def small_talk_key(text: str) -> str:
    low = text.lower()
    if re.search(r"thank|thx|\bty\b|धन्यवाद|शुक्रिया|थैंक", low):
        return "thanks"
    if re.search(r"bye|see you|अलविदा|बाय|फिर मिलेंगे", low):
        return "bye"
    return "hello"
