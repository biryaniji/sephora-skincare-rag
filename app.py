"""Step 4: Streamlit chat UI.   Run:  streamlit run app.py

Design: a beauty-shopping app with two looks, switchable in the sidebar.
  * Light & luxurious (default): pearl white, champagne gold, espresso text, Bodoni Moda headlines
    with Jost for reading, thin hairlines, nude / rose / champagne bottles with gold caps.
  * Bold boutique: white and black with hot-pink accents and bold sans-serif type.
Products are shown as shop tiles. The dataset has no product photos, so each tile gets a simple bottle
illustration (dropper for serums, tube for face washes, jar for creams, spray for toners) on a soft
colour for its category. The opening screen works like a store homepage: shop by concern, shop by
category, or ask in your own words. Remembered preferences show as filter chips.

Answer engine (sidebar): Claude if ANTHROPIC_API_KEY is set, otherwise a free local model through
Ollama (e.g. `ollama pull qwen2.5:7b`), otherwise search-only mode that lists matching products.
"""
import html
import os
import re
import time

import streamlit as st

from llm import OllamaBackend, pick_backend
from rag import DATA_NOTE, SephoraRAG

st.set_page_config(page_title="Skincare Advisor", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

THEMES = {
    "Light & luxurious": {
        "fonts": "family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..700;1,6..96,400&family=Jost:wght@300;400;500;600",
        "body": "'Jost', 'Inter', system-ui, sans-serif", "display": "'Bodoni Moda', 'Didot', Georgia, serif",
        "display_weight": "500", "page": "#FBF9F6", "surface": "#FFFFFF", "ink": "#2A2420", "muted": "#8A7F76",
        "line": "#ECE4D9", "accent": "#B08D57", "accent_dark": "#8C6B3A", "accent_soft": "#F6EFE4",
        "header_bg": "#FBF9F6", "header_fg": "#2A2420", "header_line": "1px solid #D9C7A7",
        "hero_bg": "radial-gradient(90% 120% at 85% 50%, #F3E6DA 0%, #F8F1EA 45%, #FBF9F6 100%)",
        "user_bg": "#2A2420", "user_fg": "#FBF9F6", "chip_radius": "999px", "chip_border": "1px solid #CDB58C",
        "chip_hover_bg": "#B08D57", "chip_hover_fg": "#FFFFFF", "tile_radius": "6px", "card_radius": "8px",
        "label_case": "uppercase", "label_spacing": ".18em", "label_weight": "500", "wordmark": "Skincare Advisor",
        "wordmark_font": "'Bodoni Moda', Georgia, serif", "wordmark_style": "italic", "wordmark_spacing": ".04em",
        "wordmark_size": "1.35rem", "icon_invert": "0", "cap": "#C8A46A", "hero_title": "Your skin, beautifully understood.",
    },
    "Bold boutique": {
        "fonts": "family=Plus+Jakarta+Sans:wght@400;500;600;700;800",
        "body": "'Plus Jakarta Sans', 'Inter', system-ui, sans-serif", "display": "'Plus Jakarta Sans', 'Inter', sans-serif",
        "display_weight": "800", "page": "#FFFFFF", "surface": "#FFFFFF", "ink": "#111111", "muted": "#6E6E73",
        "line": "#EAEAEA", "accent": "#E8336D", "accent_dark": "#C21D55", "accent_soft": "#FFF0F5",
        "header_bg": "#111111", "header_fg": "#FFFFFF", "header_line": "0",
        "hero_bg": "linear-gradient(120deg, #FFE3EC 0%, #FFF4F7 55%, #FFFFFF 100%)",
        "user_bg": "#111111", "user_fg": "#FFFFFF", "chip_radius": "999px", "chip_border": "1.5px solid #111111",
        "chip_hover_bg": "#111111", "chip_hover_fg": "#FFFFFF", "tile_radius": "16px", "card_radius": "14px",
        "label_case": "uppercase", "label_spacing": ".08em", "label_weight": "800", "wordmark": "SKINCARE ADVISOR",
        "wordmark_font": "'Plus Jakarta Sans', sans-serif", "wordmark_style": "normal", "wordmark_spacing": ".32em",
        "wordmark_size": ".95rem", "icon_invert": "1", "cap": "#2B2B2E", "hero_title": "Find your perfect skincare match.",
    },
}
LOOK = st.session_state.get("look", "Light & luxurious")
T = THEMES.get(LOOK, THEMES["Light & luxurious"])

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?{T["fonts"]}&display=swap');
:root {{ --page:{T["page"]}; --surface:{T["surface"]}; --ink:{T["ink"]}; --muted:{T["muted"]}; --line:{T["line"]};
  --accent:{T["accent"]}; --accentd:{T["accent_dark"]}; --accents:{T["accent_soft"]}; --display:{T["display"]}; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea,
[data-baseweb="select"] * {{ font-family: {T["body"]}; }}
[data-testid="stAppViewContainer"] {{ background: var(--page); }}
.block-container {{ max-width: 1100px; padding-top: 4.8rem; padding-bottom: 7rem; }}
.stAppDeployButton, footer {{ display: none !important; }}
*:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}

/* top bar */
[data-testid="stHeader"] {{ background: {T["header_bg"]}; height: 3.4rem; border-bottom: {T["header_line"]}; }}
[data-testid="stHeader"]::before {{ content: "{T["wordmark"]}"; position: absolute; left: 50%; top: 50%;
  transform: translate(-50%, -50%); color: {T["header_fg"]}; font-family: {T["wordmark_font"]}; font-style: {T["wordmark_style"]};
  font-weight: 600; letter-spacing: {T["wordmark_spacing"]}; font-size: {T["wordmark_size"]}; white-space: nowrap; }}
[data-testid="stHeader"] button, [data-testid="stHeader"] svg {{ color: {T["header_fg"]} !important; fill: {T["header_fg"]} !important; }}
[data-testid="stToolbar"] {{ filter: invert({T["icon_invert"]}); }}

/* sidebar */
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--line); }}
.side-title {{ font-family: var(--display); font-weight: {T["display_weight"]}; font-size: 1.3rem; color: var(--ink); }}
.side-sub {{ color: var(--muted); font-size: .85rem; margin: .15rem 0 1rem; }}
.side-h {{ font-weight: 600; font-size: .78rem; letter-spacing: {T["label_spacing"]}; text-transform: {T["label_case"]};
  color: var(--ink); margin: 1.3rem 0 .4rem; }}
.engine {{ background: var(--accents); color: var(--accentd); padding: .6rem .8rem; border-radius: {T["card_radius"]};
  font-size: .85rem; margin-top: .4rem; }}
.engine.off {{ background: #F4F2EF; color: var(--muted); }}

/* hero */
.hero {{ display: grid; grid-template-columns: 1.25fr 1fr; align-items: center; gap: 1rem; border-radius: {T["card_radius"]};
  background: {T["hero_bg"]}; padding: 3rem 2.8rem; overflow: hidden; border: 1px solid var(--line); }}
.hero h1 {{ font-family: var(--display); font-weight: {T["display_weight"]}; font-size: clamp(2.1rem, 4.4vw, 3.4rem);
  line-height: 1.04; letter-spacing: -.01em; color: var(--ink); margin: 0 0 .9rem; }}
.hero p {{ color: var(--muted); font-size: 1.06rem; line-height: 1.6; margin: 0 0 1.3rem; max-width: 46ch; }}
.hero .count {{ display: inline-block; border: 1px solid var(--accent); color: var(--accentd); font-weight: 500; font-size: .8rem;
  letter-spacing: {T["label_spacing"]}; text-transform: {T["label_case"]}; border-radius: 999px; padding: .45rem 1rem; }}
.shelf {{ display: flex; justify-content: center; align-items: flex-end; gap: .5rem; }}
.shelf svg {{ filter: drop-shadow(0 12px 16px rgba(120, 90, 50, .16)); }}
.sec-h {{ font-family: var(--display); font-weight: {T["display_weight"]}; font-size: 1.55rem; color: var(--ink); margin: 2.6rem 0 .9rem; }}

/* chips and follow-ups */
.st-key-chips button, .st-key-next button {{ border: {T["chip_border"]}; background: var(--surface); color: var(--ink);
  border-radius: {T["chip_radius"]}; font-weight: 500; font-size: .88rem; padding: .4rem 1.05rem; }}
.st-key-chips button:hover, .st-key-next button:hover {{ background: {T["chip_hover_bg"]}; color: {T["chip_hover_fg"]};
  border-color: {T["chip_hover_bg"]}; }}

/* categories */
.cat {{ display: flex; justify-content: center; }}
.cat-circle {{ width: 98px; height: 98px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--line); }}
.st-key-cats button {{ border: 0; background: transparent; font-weight: 500; color: var(--ink); font-size: .92rem; padding: .25rem; }}
.st-key-cats button:hover {{ color: var(--accent); background: transparent; }}

/* example questions */
.st-key-examples button {{ background: var(--surface); border: 1px solid var(--line); border-radius: {T["card_radius"]}; text-align: left;
  justify-content: flex-start; padding: .85rem 1.05rem; color: var(--ink); min-height: 64px; }}
.st-key-examples button:hover {{ border-color: var(--accent); background: var(--accents); color: var(--ink); }}
.st-key-examples button p {{ text-align: left; font-size: .94rem; }}

/* shopping-for-you chips */
.st-key-memory {{ position: sticky; top: 3.4rem; z-index: 50; background: var(--page); padding: .75rem 0; border-bottom: 1px solid var(--line); }}
.filters {{ display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; }}
.filters .lbl {{ font-family: var(--display); font-weight: {T["display_weight"]}; font-size: 1rem; margin-right: .35rem; color: var(--ink); }}
.chip-on {{ background: var(--accents); color: var(--accentd); border: 1px solid var(--line); border-radius: 999px; padding: .25rem .85rem;
  font-size: .84rem; font-weight: 500; }}
.chip-off {{ color: var(--muted); font-size: .86rem; }}

/* chat */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"] {{ display: none; }}
[data-testid="stChatMessage"] {{ background: transparent; padding: .35rem 0; gap: 0; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ flex-direction: row-reverse; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
  background: {T["user_bg"]}; color: {T["user_fg"]}; border-radius: 18px 18px 4px 18px; padding: .6rem 1.05rem; max-width: 70%;
  flex: 0 1 auto; margin-left: auto; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {{ color: {T["user_fg"]}; }}
[data-testid="stChatMessageContent"] {{ font-size: 1rem; line-height: 1.68; color: var(--ink); }}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {{ max-width: 72ch; }}
[data-testid="stChatMessageContent"] strong {{ font-weight: 600; }}
.who {{ display: flex; align-items: center; gap: .55rem; font-family: var(--display); font-weight: {T["display_weight"]};
  font-size: 1.02rem; margin-bottom: .25rem; color: var(--ink); }}
.who i {{ width: 26px; height: 26px; border-radius: 50%; background: var(--accent); display: inline-flex; align-items: center;
  justify-content: center; color: #fff; font-style: normal; font-size: .8rem; font-family: {T["body"]}; }}
.cite {{ display: inline-flex; align-items: center; justify-content: center; min-width: 1.2rem; height: 1.2rem; padding: 0 .3rem;
  border-radius: 999px; background: var(--accent); color: #fff; font-size: .7rem; font-weight: 600; vertical-align: .12em; margin: 0 .1rem; }}
.basis {{ font-size: .82rem; color: var(--muted); margin: .7rem 0 .2rem; }}
.notice {{ background: var(--accents); color: var(--accentd); border-radius: {T["card_radius"]}; padding: .65rem .95rem; font-size: .9rem;
  margin: .5rem 0; max-width: 72ch; }}
.notice.quiet {{ background: #F4F2EF; color: var(--muted); }}

/* product tiles */
.tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 1.1rem; margin: 1.1rem 0 .4rem; }}
.tile {{ border: 1px solid var(--line); border-radius: {T["tile_radius"]}; overflow: hidden; background: var(--surface);
  transition: border-color .2s, box-shadow .2s; }}
.tile:hover {{ border-color: var(--accent); box-shadow: 0 10px 28px rgba(60, 40, 20, .08); }}
.tile-img {{ position: relative; height: 158px; display: flex; align-items: center; justify-content: center; }}
.tile-photo {{ position: absolute; inset: 0; background-size: cover; background-position: center; }}
.tile-note {{ position: absolute; left: .6rem; bottom: .55rem; font-size: .62rem; color: #fff; background: rgba(0,0,0,.32);
  border-radius: 999px; padding: .1rem .5rem; letter-spacing: .02em; }}
.hero-photo {{ position: relative; height: 300px; border-radius: {T["card_radius"]}; overflow: hidden; }}
.hero-photo .shelf {{ position: absolute; inset: 0; align-items: center; }}
.hero-photo .bg {{ position: absolute; inset: 0; background-size: cover; background-position: center; }}
.cat-circle {{ position: relative; overflow: hidden; }}
.cat-circle .bg {{ position: absolute; inset: 0; background-size: cover; background-position: center; }}
.credits {{ font-size: .72rem; color: var(--muted); margin-top: .6rem; }}
.tile-num {{ z-index: 2; position: absolute; top: .65rem; left: .65rem; display: flex; gap: .2rem; }}
.tile-flag {{ z-index: 2; position: absolute; top: .65rem; right: .65rem; background: rgba(255,255,255,.85); color: var(--ink); font-size: .66rem;
  font-weight: 500; letter-spacing: {T["label_spacing"]}; text-transform: {T["label_case"]}; border-radius: 999px; padding: .18rem .55rem; }}
.tile-body {{ padding: .85rem .9rem 1rem; }}
.tile-brand {{ font-size: .68rem; font-weight: {T["label_weight"]}; letter-spacing: {T["label_spacing"]}; text-transform: {T["label_case"]};
  color: var(--muted); }}
.tile-name {{ font-family: var(--display); font-size: 1rem; line-height: 1.3; color: var(--ink); margin: .25rem 0 .5rem; min-height: 2.6em;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
.stars {{ color: var(--accent); font-size: .82rem; letter-spacing: .05em; }}
.stars .off {{ color: #DDD5CB; }}
.rev {{ color: var(--muted); font-size: .78rem; margin-left: .3rem; }}
.tile-price {{ font-weight: 600; font-size: 1.05rem; margin-top: .4rem; color: var(--ink); }}
.tags {{ display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .55rem; }}
.tag {{ font-size: .68rem; font-weight: 500; border-radius: 8px; padding: .14rem .55rem; background: var(--accents); color: var(--accentd); }}
.tag.warn {{ background: #FFF4E0; color: #8A5A12; }}

/* how it was found */
.steps {{ counter-reset: step; list-style: none; padding: 0; margin: .2rem 0 0; }}
.steps li {{ counter-increment: step; display: grid; grid-template-columns: 1.9rem 1fr; padding: .45rem 0; border-bottom: 1px solid var(--line);
  font-size: .88rem; color: var(--ink); }}
.steps li:before {{ content: counter(step); width: 1.4rem; height: 1.4rem; border-radius: 50%; border: 1px solid var(--accent); color: var(--accentd);
  font-size: .72rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.timing {{ font-size: .82rem; color: var(--muted); margin-top: .5rem; }}
.next-h {{ font-family: var(--display); font-weight: {T["display_weight"]}; font-size: 1rem; margin: .7rem 0 .15rem; color: var(--ink); }}
[data-testid="stChatInput"] {{ border-radius: 999px; }}
@media (max-width: 720px) {{
  .hero {{ grid-template-columns: 1fr; padding: 1.7rem; }} .shelf, .hero-photo {{ display: none; }}
  .tiles {{ grid-template-columns: repeat(2, 1fr); gap: .6rem; }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{ max-width: 90%; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- product illustrations
# (pastel background, bottle colour, shape) per product type
TYPE_STYLE_BOUTIQUE = {
    "serum": ("#EFE4FB", "#B993E6", "dropper"), "oil": ("#FDF0D9", "#E8B25C", "dropper"),
    "cleanser": ("#DFF5EA", "#6FC39C", "tube"), "exfoliant": ("#EAF2DC", "#9CBF6A", "tube"),
    "sunscreen": ("#FFF4CF", "#F0C23B", "tube"), "lip": ("#FFE1E7", "#EE7F95", "tube"),
    "self_tanner": ("#F9E7D9", "#D99A6C", "tube"), "moisturizer": ("#FFE6DA", "#F4A07F", "jar"),
    "mask": ("#FFE0EB", "#F287AE", "jar"), "eye": ("#F4E5F8", "#C792D9", "jar"),
    "toner": ("#E0F0FF", "#7FB2E8", "spray"), "mist": ("#E2F5F8", "#7FCBDA", "spray"),
    "other": ("#F2F2F4", "#B9B9C0", "pump"),
}
# Light & luxurious: nude, rose, champagne, pearl and sage tones on soft pearl backgrounds.
TYPE_STYLE_LUXE = {
    "serum": ("#F5EDE6", "#D8B6A4", "dropper"), "oil": ("#F7EFE2", "#D6B27C", "dropper"),
    "cleanser": ("#EEF0EA", "#B7C2AE", "tube"), "exfoliant": ("#F1EEE6", "#C9BC9C", "tube"),
    "sunscreen": ("#F8F1E3", "#E2C89A", "tube"), "lip": ("#F7E9E8", "#D9A3A0", "tube"),
    "self_tanner": ("#F4EAE1", "#C99C7E", "tube"), "moisturizer": ("#F7EEE9", "#E6C9BC", "jar"),
    "mask": ("#F6EAEC", "#D9AEB6", "jar"), "eye": ("#F1ECF1", "#C9B6C8", "jar"),
    "toner": ("#EEF1F2", "#B9C7CC", "spray"), "mist": ("#EEF2F0", "#BCCDC5", "spray"),
    "other": ("#F4F1EC", "#CFC4B6", "pump"),
}
TYPE_STYLE = TYPE_STYLE_LUXE if LOOK == "Light & luxurious" else TYPE_STYLE_BOUTIQUE
TYPE_ORDER = ["eye", "sunscreen", "serum", "cleanser", "toner", "moisturizer", "mask", "exfoliant", "oil", "mist", "lip",
              "self_tanner"]


def bottle_svg(shape: str, colour: str, size: int = 92) -> str:
    dark = T["cap"]
    shapes = {
        "dropper": f'<rect x="25" y="2" width="10" height="18" rx="5" fill="{dark}"/><rect x="21" y="18" width="18" height="10" rx="2" fill="{dark}"/>'
                   f'<rect x="11" y="27" width="38" height="66" rx="12" fill="{colour}"/>',
        "tube": f'<path d="M12 4 H48 L43 74 H17 Z" fill="{colour}"/><rect x="20" y="73" width="20" height="20" rx="4" fill="{dark}"/>',
        "jar": f'<rect x="6" y="30" width="48" height="13" rx="5" fill="{dark}"/><rect x="8" y="42" width="44" height="46" rx="12" fill="{colour}"/>',
        "spray": f'<rect x="24" y="4" width="12" height="9" rx="2" fill="{dark}"/><rect x="34" y="6" width="9" height="4" rx="2" fill="{dark}"/>'
                 f'<rect x="21" y="13" width="18" height="12" rx="3" fill="{dark}"/><rect x="13" y="24" width="34" height="69" rx="10" fill="{colour}"/>',
        "pump": f'<rect x="24" y="2" width="20" height="6" rx="3" fill="{dark}"/><rect x="27" y="7" width="6" height="12" fill="{dark}"/>'
                f'<rect x="22" y="17" width="16" height="10" rx="3" fill="{dark}"/><rect x="12" y="26" width="36" height="67" rx="10" fill="{colour}"/>',
    }
    label_y = {"dropper": 48, "tube": 30, "jar": 56, "spray": 48, "pump": 48}[shape]
    return (f'<svg width="{size * 0.66:.0f}" height="{size}" viewBox="0 0 60 96" aria-hidden="true">{shapes[shape]}'
            f'<rect x="19" y="{label_y}" width="22" height="18" rx="3" fill="#fff" opacity=".85"/></svg>')


SHAPE_VARIANTS = {"tube": ["tube", "pump"], "dropper": ["dropper", "pump"], "jar": ["jar", "jar", "pump"],
                  "spray": ["spray", "pump"], "pump": ["pump", "tube"]}


def shade(hex_colour: str, amount: float) -> str:
    """Lighten (amount > 0) or darken (amount < 0) a colour."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    f = (lambda c: c + (255 - c) * amount) if amount > 0 else (lambda c: c * (1 + amount))
    return "#" + "".join(f"{int(max(0, min(255, f(c)))):02X}" for c in (r, g, b))


def product_look(meta: dict) -> tuple[str, str, str]:
    """Background, bottle colour and shape for a product: its category's palette, varied per product
    so a grid of the same category doesn't look identical."""
    t = product_type(meta)
    bg, colour, shape = TYPE_STYLE[t]
    seed = sum(ord(ch) for ch in str(meta.get("product_id", "")))
    colour = shade(colour, [0, -0.12, 0.18, -0.22, 0.08][seed % 5])
    shape = SHAPE_VARIANTS[shape][seed // 5 % len(SHAPE_VARIANTS[shape])]
    return bg, colour, shape


def product_type(meta: dict) -> str:
    return next((t for t in TYPE_ORDER if meta.get(f"type_{t}")), "other")


# ---------------------------------------------------------------- photos
# The Sephora dataset has no product images, so tiles show free-licence photography chosen per
# category (Pexels and Unsplash licences: free to use, no permission needed). These are illustrative,
# not the actual product, and are labelled as such. Put your own picture in images/<product_id>.jpg
# (or .png / .webp) and it replaces the illustrative photo for that product.
def _pexels(i: int, w: int = 640) -> str:
    return f"https://images.pexels.com/photos/{i}/pexels-photo-{i}.jpeg?auto=compress&cs=tinysrgb&w={w}"


def _unsplash(i: str, w: int = 640) -> str:
    return f"https://images.unsplash.com/photo-{i}?fm=jpg&q=70&w={w}&auto=format&fit=crop"


SPRAY = [_pexels(8167156), _pexels(8167167), _pexels(8167288), _pexels(8167290)]
DROPPER = [_pexels(6167869), _pexels(8167170), _unsplash("1741896136777-deb999cab264")]
BOTTLE = [_pexels(8167117), _pexels(8166777), _pexels(8167163), _pexels(8166824)]
JAR = [_unsplash("1642683496509-121deffa9cf0"), _unsplash("1667242003558-e42942d2b911"), _pexels(8166568)]
TUBE = [_unsplash("1598662972299-5408ddb8a3dc"), _pexels(8166611), _pexels(8167163)]
PHOTOS = {"toner": SPRAY, "mist": SPRAY, "serum": DROPPER, "oil": DROPPER, "cleanser": BOTTLE, "exfoliant": BOTTLE,
          "moisturizer": JAR, "mask": JAR, "eye": JAR, "sunscreen": TUBE, "lip": TUBE, "self_tanner": TUBE,
          "other": BOTTLE}
HERO_PHOTO = _unsplash("1741896136777-deb999cab264", 1400)
PHOTO_CREDITS = ("Illustrative photos: Daria Liudnaya and Vie Studio (Pexels); Maria Lupan, everdrop and "
                 "Lina Verovaya (Unsplash). Free licences.")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


@st.cache_data(show_spinner=False)
def own_photo(product_id: str) -> str | None:
    """Data URI for images/<product_id>.<ext>, if the user added one."""
    import base64
    for ext, mime in (("jpg", "jpeg"), ("jpeg", "jpeg"), ("png", "png"), ("webp", "webp")):
        path = os.path.join(IMAGES_DIR, f"{product_id}.{ext}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()
    return None


def photo_for(meta: dict) -> tuple[str, bool]:
    """(image URL, is_illustrative) for a product."""
    own = own_photo(str(meta.get("product_id", "")))
    if own:
        return own, False
    options = PHOTOS[product_type(meta)]
    seed = sum(ord(ch) for ch in str(meta.get("product_id", "")))
    return options[seed % len(options)], True


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the catalogue and search index...")
def load_rag(code_version: float) -> SephoraRAG:
    """Cached across reruns; code_version changes when rag.py / understanding.py / llm.py change,
    so an edited engine is reloaded instead of mixing old and new code."""
    return SephoraRAG()


def code_version() -> float:
    here = os.path.dirname(os.path.abspath(__file__))
    return max(os.path.getmtime(os.path.join(here, f)) for f in ("rag.py", "understanding.py", "llm.py"))


@st.cache_data(ttl=20, show_spinner=False)
def ollama_models() -> list[str]:
    return OllamaBackend.list_models()


rag = load_rag(code_version())
st.session_state.setdefault("messages", [])
st.session_state.setdefault("feedback", {})


# ---------------------------------------------------------------- content
CONCERNS = {"Dry skin": "I have dry skin. What moisturizer should I use?",
            "Acne & breakouts": "What's good for acne-prone skin?",
            "Dark spots": "What helps with dark spots and uneven tone?",
            "Oily skin": "I have oily skin. Recommend a lightweight moisturizer.",
            "Sensitive skin": "Recommend a gentle, fragrance-free cleanser for sensitive skin.",
            "Fine lines": "What's a good serum for fine lines?",
            "Under $25": "What are the best rated skincare products under $25?"}
CATEGORIES = [("cleanser", "Face wash", "Show me the best rated face washes"),
              ("toner", "Toners", "Show me the best rated toners"),
              ("serum", "Serums", "Show me the best rated serums"),
              ("moisturizer", "Moisturizers", "Show me the best rated moisturizers"),
              ("sunscreen", "Sunscreen", "Show me the best rated sunscreens"),
              ("eye", "Eye care", "Show me the best rated eye creams")]
EXAMPLES = ["I have dry skin, recommend a face wash under $20",
            "What's in The Ordinary Niacinamide 10% + Zinc 1%?",
            "Compare The Ordinary and The INKEY List serums for acne",
            "Which retinol serum is best rated, and what do oily-skin reviewers say?",
            "Build me a morning routine for oily skin under $80",
            "Is the Drunk Elephant Protini cream safe during pregnancy?"]
TYPE_NAMES = {"cleanser": "Face wash", "toner": "Toner", "serum": "Serum", "moisturizer": "Moisturizer",
              "sunscreen": "Sunscreen", "eye": "Eye care", "mask": "Mask", "exfoliant": "Exfoliator", "oil": "Face oil",
              "mist": "Mist", "lip": "Lip care", "self_tanner": "Self-tanner"}
ROUTING_NOTE = {
    "low_similarity": "That's outside what this skincare catalogue covers. Ask me about skincare products, ingredients, skin types or budgets.",
    "out_of_scope": "Makeup, hair and fragrance aren't in this catalogue. Skincare only, for now.",
    "no_match": "Nothing matches every requirement. Try a higher budget or fewer must-haves. Your limits are never loosened silently.",
    "no_llm": "No answer engine is connected, so here are the matching products. Pick one in the sidebar to get written answers.",
    "error": "The answer couldn't be completed.",
}


# ---------------------------------------------------------------- helpers
def esc(text) -> str:
    return html.escape(str(text))


WHO = '<div class="who"><i>✿</i>Skincare Advisor</div>'


def styled_answer(text: str) -> str:
    """Markdown answer with [n] turned into pink badges. Kept as markdown (not wrapped in an HTML
    block) so bold text and bullet lists render."""
    return re.sub(r"\[(\d{1,2})\]", r'<span class="cite">\1</span>', text)


def money(v: float) -> str:
    return f"${v:,.0f}" if float(v).is_integer() else f"${v:,.2f}"


def stars(rating: float) -> str:
    full = max(0, min(5, int(round(rating))))
    return f'<span class="stars">{"★" * full}<span class="off">{"★" * (5 - full)}</span></span>'


def product_tiles(chunks: list[dict], only: set[int] | None = None) -> str:
    products: dict[str, dict] = {}
    for i, c in enumerate(chunks, 1):
        if only and i not in only:
            continue
        products.setdefault(c["meta"]["product_id"], {"meta": c["meta"], "sources": []})["sources"].append(i)
    tiles = []
    for p in products.values():
        m = p["meta"]
        t = product_type(m)
        bg, colour, shape = product_look(m)
        nums = "".join(f'<span class="cite">{n}</span>' for n in p["sources"])
        if m.get("has_rating", True):
            rating = f'{stars(m["rating"])}<span class="rev">{m["rating"]:.1f} ({m["reviews"]:,})</span>'
        else:
            rating = '<span class="rev">No ratings yet</span>'
        tags = [f'<span class="tag">{label}</span>' for flag, label in
                (("is_vegan", "Vegan"), ("is_clean", "Clean"), ("is_fragrance_free", "Fragrance-free")) if m.get(flag)]
        if m.get("skin_types"):
            tags.append(f'<span class="tag">For {esc(m["skin_types"])} skin</span>')
        if m.get("has_rating", True) and m.get("reviews", 0) < 10:
            tags.append('<span class="tag warn">Few reviews</span>')
        flag = f'<span class="tile-flag">{TYPE_NAMES.get(t, "Skincare")}</span>'
        url, illustrative = photo_for(m)
        photo = (f'<div class="tile-photo" role="img" aria-label="" style="background-image:url(\'{url}\')"></div>'
                 + ('<span class="tile-note">Illustrative photo</span>' if illustrative else ""))
        tiles.append(f'<div class="tile"><div class="tile-img" style="background:{bg}">{bottle_svg(shape, colour)}{photo}'
                     f'<div class="tile-num">{nums}</div>{flag}</div><div class="tile-body"><div class="tile-brand">{esc(m["brand"])}</div>'
                     f'<div class="tile-name">{esc(m["product_name"])}</div><div>{rating}</div>'
                     f'<div class="tile-price">{money(m["price"])}</div><div class="tags">{"".join(tags)}</div></div></div>')
    return f'<div class="tiles">{"".join(tiles)}</div>'


def notices(result: dict) -> str:
    out = []
    for note in result.get("notes", []):
        if note.startswith("Medical"):
            out.append("This touches on a medical question. I can share what the catalogue says, but please check with a "
                       "dermatologist or doctor before using anything.")
        elif "doesn't sell" in note:
            out.append("That brand isn't sold in this Sephora catalogue, so here are the closest matches.")
        elif "relaxed" in note:
            out.append("Nothing matched every detail, so these are the closest matches.")
    html_out = "".join(f'<div class="notice">{esc(n)}</div>' for n in out)
    if result.get("fallback") in ROUTING_NOTE:
        html_out += f'<div class="notice quiet">{esc(ROUTING_NOTE[result["fallback"]])}</div>'
    return html_out


def basis_line(result: dict) -> str:
    chunks = result["retrieved"].chunks
    if not chunks or result.get("fallback") not in (None, "no_llm"):
        return ""
    t = result.get("timings", {})
    secs = sum(t.get(k, 0) for k in ("understanding_ms", "retrieval_ms", "generation_ms")) / 1000
    who = f" Written by {esc(result['model'])}." if result.get("model") else ""
    return f'<div class="basis">Based on {len(chunks)} catalogue entries (the numbers in the answer).{who} {secs:.1f} s.</div>'


def steps_html(result: dict) -> str:
    items = "".join(f"<li><span><b>{esc(name)}.</b> {esc(detail)}</span></li>" for name, detail in result.get("steps", []))
    t = result.get("timings", {})
    parts = [f"{label} {t[k] / 1000:.1f} s" for k, label in
             (("understanding_ms", "understanding"), ("retrieval_ms", "search"), ("generation_ms", "writing")) if k in t]
    return f'<ol class="steps">{items}</ol>' + (f'<div class="timing">Time: {", ".join(parts)}.</div>' if parts else "")


def render_details(result: dict, idx: int):
    if result.get("fallback") == "small_talk":
        return
    st.markdown(notices(result) + basis_line(result), unsafe_allow_html=True)
    chunks = result["retrieved"].chunks
    if chunks and result.get("fallback") in (None, "no_llm") and st.session_state.get("show_products", True):
        cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", result.get("answer", "")) if 1 <= int(n) <= len(chunks)}
        st.markdown(product_tiles(chunks, cited or None), unsafe_allow_html=True)
        others = [c for i, c in enumerate(chunks, 1) if cited and i not in cited]
        n_other = len({c["meta"]["product_id"] for c in others} - {chunks[i - 1]["meta"]["product_id"] for i in cited})
        if n_other:
            with st.expander(f"See {n_other} more product{'s' if n_other > 1 else ''} I considered"):
                st.markdown(product_tiles(others), unsafe_allow_html=True)
    if st.session_state.get("show_trace", True) and result.get("steps"):
        with st.expander("How I found this"):
            st.markdown(steps_html(result), unsafe_allow_html=True)
            if chunks:
                st.markdown("**The exact catalogue text I used**")
                for i, c in enumerate(chunks, 1):
                    st.caption(f"[{i}] {c['meta']['product_name']} ({c['meta']['chunk_type']} entry, found by {c['found_by']})")
                    st.text(c["text"][:900] + ("..." if len(c["text"]) > 900 else ""))
    if hasattr(st, "feedback"):
        st.session_state.feedback[idx] = st.feedback("thumbs", key=f"fb_{idx}")


def filter_chips(profile: dict, preferences: dict):
    chips = []
    skin = profile.get("skin") or (preferences.get("skin_type") or "").lower()
    if skin:
        chips.append(f"{skin.capitalize()} skin")
    prods = profile.get("products") or []
    if prods:
        chips.append(("Routine: " if profile.get("routine") else "") + ", ".join(TYPE_NAMES.get(t, t) for t in prods))
    hi, lo = profile.get("max_price") or preferences.get("max_price"), profile.get("min_price")
    if lo and hi:
        chips.append(f"{money(lo)} to {money(hi)}")
    elif hi:
        chips.append(f"Under {money(hi)}")
    elif lo:
        chips.append(f"Over {money(lo)}")
    musts = profile.get("musts", []) + [l for k, l in (("vegan", "vegan"), ("clean", "Clean at Sephora"),
                                                        ("fragrance_free", "fragrance-free")) if preferences.get(k)]
    chips += [m.capitalize() for m in dict.fromkeys(musts)]
    if profile.get("brand"):
        chips.append(profile["brand"])
    if profile.get("sort") == "price_low":
        chips.append("Lowest price first")
    elif profile.get("sort") == "rating":
        chips.append("Best rated first")
    body = "".join(f'<span class="chip-on">{esc(c)}</span>' for c in chips) or \
        '<span class="chip-off">Tell me your skin type, what you need and your budget</span>'
    with st.container(key="memory"):
        st.markdown(f'<div class="filters"><span class="lbl">Shopping for you</span>{body}</div>', unsafe_allow_html=True)


def follow_ups(result: dict) -> list[str]:
    p, out = result.get("profile") or {}, ["Which of those is the cheapest?"]
    if p.get("routine"):
        return ["Make it cheaper", "What order do I apply them in?", "Swap the serum for one with niacinamide"]
    if not p.get("max_price"):
        out.append("I only have $25")
    if not p.get("skin"):
        out.append("Which suits oily skin best?")
    if "fragrance-free" not in p.get("musts", []):
        out.append("Only fragrance-free ones")
    return out[:3]


def chat_markdown() -> str:
    lines = ["# Skincare Advisor conversation", ""]
    for m in st.session_state.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'Advisor'}:** {m['content']}\n")
        r = m.get("result")
        if r and r["retrieved"].chunks and r.get("fallback") in (None, "no_llm"):
            lines.append("Sources: " + "; ".join(f"[{i}] {c['meta']['product_name']} ({c['meta']['brand']})"
                                                  for i, c in enumerate(r["retrieved"].chunks, 1)) + "\n")
    return "\n".join(lines)


def ask(question: str):
    st.session_state.pending = question
    st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-title">Skincare Advisor</div>'
                '<div class="side-sub">Answers from Sephora\'s skincare catalogue and its customer reviews.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-h">Look</div>', unsafe_allow_html=True)
    st.radio("Look", list(THEMES), key="look", horizontal=True, label_visibility="collapsed")

    st.markdown('<div class="side-h">Who writes the answers</div>', unsafe_allow_html=True)
    models = ollama_models()
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    engine = st.radio("Who writes the answers", ["Best available", "Claude", "A model on this laptop", "Just list products"],
                      label_visibility="collapsed", key="engine",
                      help="Best available uses Claude if an API key is set, otherwise a local Ollama model.")
    chosen_model = None
    if engine == "Claude" and not has_key:
        key_in = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
        if key_in:
            os.environ["ANTHROPIC_API_KEY"] = key_in.strip()
            has_key = True
    if engine in ("Best available", "A model on this laptop") and models:
        best = OllamaBackend.best_model(models)
        chosen_model = st.selectbox("Local model", models, index=models.index(best), key="ollama_model")
    pref = {"Best available": "auto", "Claude": "claude", "A model on this laptop": "ollama", "Just list products": "none"}[engine]
    sig = (pref, chosen_model, has_key)
    if st.session_state.get("backend_sig") != sig:
        try:
            st.session_state.backend = pick_backend(pref, chosen_model)
        except Exception as e:
            st.session_state.backend = None
            st.error(f"That engine didn't start: {e}")
        st.session_state.backend_sig = sig
    rag.backend = st.session_state.backend
    if rag.backend is not None:
        st.markdown(f'<div class="engine">Answers are written by <b>{esc(rag.backend.model)}</b> {rag.backend.where}.</div>',
                    unsafe_allow_html=True)
    elif pref == "none":
        st.markdown('<div class="engine off">You\'ll see matching products without a written answer.</div>', unsafe_allow_html=True)
    elif pref == "claude":
        st.markdown('<div class="engine off">Paste an Anthropic API key above to use Claude.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="engine off">No local model found. Install Ollama, run <code>ollama pull qwen2.5:7b</code>, '
                    'then refresh.</div>', unsafe_allow_html=True)
    rag.use_analyzer = st.toggle("Read each message in context", value=True, key="smart",
                                 help="Uses the model to interpret follow-ups and vague requests. Turn off on a slow laptop.")

    st.markdown('<div class="side-h">Always apply</div>', unsafe_allow_html=True)
    skin = st.selectbox("My skin type", ["Not set", "Dry", "Oily", "Combination", "Normal", "Sensitive"], key="skin")
    budget = st.slider("Most I'll pay per product", 0, 200, 0, step=5, key="budget", format="$%d",
                       help="Leave at $0 for no limit. A price in your message takes priority.")
    c1, c2 = st.columns(2)
    vegan = c1.toggle("Vegan", key="vegan")
    clean = c2.toggle("Clean", key="clean")
    ff = st.toggle("Fragrance-free", key="ff")
    preferences = {"skin_type": None if skin == "Not set" else skin, "max_price": budget or None,
                   "vegan": vegan, "clean": clean, "fragrance_free": ff}

    st.markdown('<div class="side-h">Show</div>', unsafe_allow_html=True)
    st.toggle("Product tiles", value=True, key="show_products")
    st.toggle("How I found each answer", value=True, key="show_trace")
    c1, c2 = st.columns(2)
    if c1.button("Start over", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    c2.download_button("Save chat", chat_markdown(), file_name="skincare-advisor-chat.md",
                       use_container_width=True, disabled=not st.session_state.messages)
    st.caption(f"{DATA_NOTE} Not medical advice.")
    st.markdown(f'<div class="credits">{PHOTO_CREDITS} Add your own product photos as images/&lt;product_id&gt;.jpg.</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- store homepage
if not st.session_state.messages and not st.session_state.get("pending"):
    shelf = "".join(bottle_svg(TYPE_STYLE[t][2], TYPE_STYLE[t][1], size) for t, size in
                    (("toner", 120), ("serum", 150), ("moisturizer", 110), ("cleanser", 135), ("sunscreen", 118)))
    st.markdown(f'<div class="hero"><div><h1>{T["hero_title"]}</h1>'
                f'<p>Tell me about your skin, your concerns and your budget. I\'ll pick from Sephora\'s skincare range '
                f'and show you what real customers say about each product.</p>'
                f'<span class="count">{rag.product_count:,} products to choose from</span></div>'
                f'<div class="hero-photo"><div class="shelf">{shelf}</div>'
                f'<div class="bg" style="background-image:url(\'{HERO_PHOTO}\')"></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-h">Shop by concern</div>', unsafe_allow_html=True)
    with st.container(key="chips"):
        cols = st.columns(len(CONCERNS))
        for i, (label, q) in enumerate(CONCERNS.items()):
            if cols[i].button(label, key=f"concern_{i}", use_container_width=True):
                ask(q)

    st.markdown('<div class="sec-h">Shop by category</div>', unsafe_allow_html=True)
    with st.container(key="cats"):
        cols = st.columns(len(CATEGORIES))
        for i, (t, label, q) in enumerate(CATEGORIES):
            bg, colour, shape = TYPE_STYLE[t]
            cols[i].markdown(f'<div class="cat"><div class="cat-circle" style="background:{bg}">'
                             f'{bottle_svg(shape, colour, 64)}<div class="bg" style="background-image:url(\'{PHOTOS[t][i % len(PHOTOS[t])].replace("w=640", "w=240")}\')"></div>'
                             f'</div></div>', unsafe_allow_html=True)
            if cols[i].button(label, key=f"cat_{i}", use_container_width=True):
                ask(q)

    st.markdown('<div class="sec-h">Or just ask</div>', unsafe_allow_html=True)
    with st.container(key="examples"):
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                ask(q)

# ---------------------------------------------------------------- conversation
last_profile = next((m["result"].get("profile") for m in reversed(st.session_state.messages)
                     if m.get("result") and m["result"].get("profile")), None)
if st.session_state.messages or st.session_state.get("pending"):
    filter_chips(last_profile or {}, preferences)

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(WHO, unsafe_allow_html=True)
            st.markdown(styled_answer(msg["content"]), unsafe_allow_html=True)
            if msg.get("result"):
                render_details(msg["result"], idx)
        else:
            st.markdown(msg["content"])

last = st.session_state.messages[-1] if st.session_state.messages else None
if last and last["role"] == "assistant" and last.get("result") and last["result"].get("fallback") in (None, "no_llm") \
        and last["result"]["retrieved"].chunks and not st.session_state.get("pending"):
    st.markdown('<div class="next-h">You could also ask</div>', unsafe_allow_html=True)
    with st.container(key="next"):
        suggestions = follow_ups(last["result"])
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if cols[i].button(q, key=f"fu_{len(st.session_state.messages)}_{i}", use_container_width=True):
                ask(q)

question = st.chat_input("Ask about your skin, a product, a budget or an ingredient")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(WHO, unsafe_allow_html=True)
        result = None
        try:
            with st.status("Finding your matches...", expanded=False) as status:
                previous = next((m.get("result") for m in reversed(st.session_state.messages[:-1])
                                 if m["role"] == "assistant" and m.get("result")
                                 and m["result"].get("fallback") in (None, "no_llm")), None)
                result = rag.prepare(question, history, preferences, previous)
                n = len(result["retrieved"].chunks)
                status.update(label=f"Found {n} matching catalogue entries" if n and result.get("fallback") in
                              (None, "no_llm") else "Done", state="complete")
            box, text = st.empty(), ""
            for piece in rag.generate(result):
                text += piece
                box.markdown(styled_answer(text + " ▍"), unsafe_allow_html=True)
                time.sleep(0.004)
            answer = result["answer"] or text
            box.markdown(styled_answer(answer), unsafe_allow_html=True)
        except Exception as e:  # show model / connection problems instead of crashing the demo
            if "Connection refused" in str(e) or "urlopen" in str(e) or "timed out" in str(e):
                answer = ("The local model isn't responding. Start Ollama (`ollama serve`) or choose another option "
                          "under *Who writes the answers*.")
            else:
                answer = f"The answer couldn't be completed ({type(e).__name__}: {e}). Send the message again."
            st.markdown(answer)
            if result is not None:
                result["fallback"] = result.get("fallback") or "error"
        if result is not None:
            render_details(result, len(st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": answer, "result": result})
    st.rerun()
