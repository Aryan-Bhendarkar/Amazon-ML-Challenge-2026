"""Baseline text normalization (v0). Country-agnostic by default; tiny country-aware rules
only where an abbreviation is ambiguous (e.g. 'r' = 'rue' only in France). Unknown countries
fall back to generic rules — never assume the country set is {US, India}.

Everything here is hand-written general knowledge (abbreviations, legal forms) — allowed.
Do NOT add downloaded gazetteers / postal DBs (fair-play rule, see docs/rules.md).

Improve iteratively: every change must be measured (blocking recall + val F0.5) and logged.
Known TODOs (docs/ideas_backlog.md): data-driven native-script token map (state names,
'praivet'->'pvt'), learned abbreviation table from train pairs, French dept->region map.
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

from anyascii import anyascii

# ----------------------------------------------------------------------------- generic
_PRE_ASCII = [
    (re.compile(r"[Nn]\s?[°º]"), " no "),     # N° 21 -> no 21 (before anyascii turns ° into 'deg')
    (re.compile(r"[°º]"), " "),
    (re.compile(r"[​-‍﻿]"), ""),  # zero-width chars (seen in Odia names)
]


def to_ascii(s: str) -> str:
    """NFKC -> transliterate any script to ASCII (anyascii, ISC license) -> lowercase."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for pat, rep in _PRE_ASCII:
        s = pat.sub(rep, s)
    return anyascii(s).lower()


def script_of(s: str) -> str:
    """Dominant non-Latin script name ('latin' if none). Useful feature: native-script records."""
    for ch in s or "":
        if ord(ch) > 0x24F and ch.isalpha():
            try:
                return unicodedata.name(ch).split(" ")[0].lower()
            except ValueError:
                return "other"
    return "latin"


_WS = re.compile(r"\s+")
_DOTTED_ABBR = re.compile(r"\b(?:[a-z]\.){2,}")          # l.l.c. s.a.r.l. -> llc sarl
_NONALNUM = re.compile(r"[^a-z0-9]+")


def _squash(s: str) -> str:
    return _WS.sub(" ", s).strip()


# ----------------------------------------------------------------------------- names
LEGAL_CANON = {
    # US / generic
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co", "ltd": "ltd", "limited": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "lllp": "lllp", "gmbh": "gmbh",
    # India (incl. anyascii transliterations seen in data; extend from train pairs)
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd", "praivet": "pvt", "praivett": "pvt",
    "pra": "pvt", "li": "ltd", "limitet": "ltd", "limitedd": "ltd", "elelpi": "llp",
    "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "earl": "earl", "gie": "gie", "scop": "scop",
}
NAME_STOP = {"the", "and", "of", "m/s", "ms", "sri", "shri", "shree", "a", "an", "de", "la", "le", "les", "du", "des"}

_ID_TAG = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|\bid\s*:\s*\d+|#\s*\d+\s*$")
_ALIAS_SPLIT = re.compile(r"\b(?:formerly known as|formerly|f/k/a|fka|a/k/a|aka|d/b/a|dba|doing business as|trading as|t/a)\b")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|fr|biz|info|co|us|io)$")
_HANDLE = re.compile(r"^@([a-z0-9_\.]+)$")


class NameNorm(NamedTuple):
    full: str          # all tokens, legal forms canonicalized
    core: str          # tokens minus legal forms and stopwords (the business identity)
    legal: str         # sorted canonical legal forms, space-joined ('' if none)
    compact: str       # core with spaces removed (matches domains/handles: 'wilfordhancock')
    alias: str         # other name in 'X formerly known as Y' / 'X dba Y' ('' if none)
    kind: str          # 'name' | 'domain' | 'handle' | 'empty'


def normalize_name(raw: str) -> NameNorm:
    s = to_ascii(raw or "")
    s = _squash(s)
    if s in ("", "na", "n/a", "null", "<null>", "none", "-"):
        return NameNorm("", "", "", "", "", "empty")
    kind = "name"
    m = _DOMAIN.match(s)
    if m:
        s, kind = m.group(1).replace("-", " "), "domain"
    else:
        m = _HANDLE.match(s)
        if m:
            s, kind = m.group(1).replace("_", " ").replace(".", " "), "handle"
    s = _ID_TAG.sub(" ", s)
    alias = ""
    parts = _ALIAS_SPLIT.split(s, maxsplit=1)
    if len(parts) == 2:
        s, alias = parts[0], parts[1]
    s = s.replace("&", " and ").replace("+", " ")
    s = _DOTTED_ABBR.sub(lambda mm: mm.group(0).replace(".", ""), s)
    full_toks, core_toks, legal = [], [], []
    for t in _NONALNUM.sub(" ", s).split():
        c = LEGAL_CANON.get(t)
        if c is not None:
            full_toks.append(c)
            legal.extend(c.split())
        else:
            full_toks.append(t)
            if t not in NAME_STOP:
                core_toks.append(t)
    alias_n = normalize_name(alias).core if alias else ""
    core = " ".join(core_toks)
    return NameNorm(" ".join(full_toks), core, " ".join(sorted(set(legal))),
                    core.replace(" ", ""), alias_n, kind)


# ----------------------------------------------------------------------------- addresses
STREET_CANON = {
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "road": "rd",
    "drive": "dr", "lane": "ln", "court": "ct", "boulevard": "blvd", "bd": "blvd", "bld": "blvd",
    "place": "pl", "trail": "trl", "parkway": "pkwy", "highway": "hwy", "circle": "cir",
    "terrace": "ter", "square": "sq", "suite": "ste", "sainte": "ste", "apartment": "apt",
    "building": "bldg", "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "route": "rte", "chemin": "ch", "impasse": "imp", "allee": "all", "faubourg": "fg",
    "near": "nr", "opposite": "opp", "opp": "opp", "sector": "sec", "extension": "extn",
    "ext": "extn", "phase": "ph", "nagar": "ngr", "colony": "col",
}
COUNTRY_TOKEN_RULES = {  # ambiguous single letters etc. — applied only for that country
    "france": {"r": "rue"},
}
ADDR_NOISE = {"null", "no", "h", "hno", "door", "plot", "flat", "house", "number", "num", "bis"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
IN_STATE_CODE_ALIASES = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk"}

_NUM = re.compile(r"\d+[a-z]?(?:[-/]\d+[a-z]?)*")


class AddrNorm(NamedTuple):
    full: str        # normalized tokens, street words canonicalized, noise tokens dropped
    street: str      # alphabetic tokens only (no numbers), minus state tokens
    numbers: str     # all number-like tokens in order, space-joined ('12', '2-7-136/1')
    house: str       # first number-like token ('' if none)
    postcode: str    # 5-digit or 6-digit standalone number ('' if none)
    state: str       # canonical state code if recognized at component level ('' if not)
    n_parts: int     # number of comma-separated components
    empty: bool


def _state_of(component: str, country: str) -> str:
    c = component.strip()
    if country == "us":
        if c in US_STATES:
            return US_STATES[c]
        if len(c) == 2 and c in US_STATES.values():
            return c
    elif country == "india":
        if c in IN_STATES:
            return IN_STATES[c]
        c2 = IN_STATE_CODE_ALIASES.get(c, c)
        if len(c2) == 2 and c2 in IN_STATES.values():
            return c2
    return ""


def normalize_address(raw: str, country: str = "") -> AddrNorm:
    ctry = (country or "").strip().lower()
    s = to_ascii(raw or "").replace("<null>", " ")
    s = _squash(s)
    if not s:
        return AddrNorm("", "", "", "", "", "", 0, True)
    comps = [c.strip() for c in s.split(",") if c.strip()]
    extra = COUNTRY_TOKEN_RULES.get(ctry, {})
    nums = _NUM.findall(s)
    state, toks, street_toks = "", [], []
    for c in comps:
        alpha = _squash(re.sub(r"[^a-z]+", " ", c))
        st = _state_of(alpha, ctry)           # state is usually a whole component ('up 201301' ok)
        if st:
            state = st
            toks.append(st)
            toks.extend(n for n in _NUM.findall(c))
            continue
        for t in _NONALNUM.sub(" ", re.sub(r"(?<=\d)[-/](?=\d)", "_", c)).split():
            t = t.replace("_", "-")
            t = extra.get(t, t)
            t = STREET_CANON.get(t, t)
            if t in ADDR_NOISE:
                continue
            toks.append(t)
            if t.isalpha():
                street_toks.append(t)
    street = " ".join(street_toks)
    # postcode: a standalone 5/6-digit number that is NOT the leading house number
    postcode = next((n for n in reversed(nums[1:]) if n.isdigit() and len(n) in (5, 6)), "")
    return AddrNorm(" ".join(toks), street, " ".join(nums), nums[0] if nums else "",
                    postcode, state, len(comps), False)
