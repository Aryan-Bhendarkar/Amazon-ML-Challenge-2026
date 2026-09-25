"""Administrative-area aliases learned ONLY from the provided (unlabeled) address records of one country.

Why: some countries' addresses name the admin area inconsistently across sources (S1 'Lille, Hauts-de-France'
vs S2 'LILLE, Nord'). Hand-written state tables exist for some countries; for any other country this module
learns an alias map from the records themselves, no external data and no labels:

  1. admin vocabulary A = frequent components that are the LAST comma-component of S1 addresses in at least
     `last_rate` of their S1 occurrences (S1 is the clean reference; its addresses end with the admin area)
  2. every frequent component gets its S1 admin distribution through the components it co-occurs with
     (e.g. 'calais' appears with 'hauts de france' in S1 -> P(admin | calais))
  3. a frequent non-admin component Y is an ALIAS of admin X when Y substitutes X (Y and X almost never share
     an address) and the components Y co-occurs with belong to X with probability >= `purity`
     ('pas de calais' appears with 'calais', whose S1 admin is 'hauts de france' -> alias)

Validated on TRAIN US/India (hand-written state tables as the answer key) before use; see
scripts/learn_admin_map.py. Applied in normalization only for countries whose S1 addresses get no state from
the hand-written parser, so countries with a parser (and hence all training data) are unchanged.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .normalize import _squash, to_ascii

_ALPHA = re.compile(r"[^a-z]+")


def component_key(c: str) -> str:
    """same component form as normalize._state_of sees: ascii, letters only, squashed."""
    return _squash(_ALPHA.sub(" ", to_ascii(c)))


def _comps(addr: str) -> list[str]:
    out = []
    for c in (addr or "").split(","):
        if any(ch.isdigit() for ch in c):          # street lines / postcodes are never admin areas
            continue
        k = component_key(c)
        if k:
            out.append(k)
    return out


def learn(s1_addrs, other_addrs, min_count: int = 200, min_frac: float = 1e-3, last_rate: float = 0.5,
          purity: float = 0.9, max_co: float = 0.02) -> dict:
    """Returns {'admins': sorted list, 'alias': {component: admin}} for one country."""
    s1_addrs, other_addrs = list(s1_addrs), list(other_addrs)
    n = len(s1_addrs) + len(other_addrs)
    thr = max(min_count, min_frac * n)
    cnt, s1_cnt, s1_last = Counter(), Counter(), Counter()
    recs = []
    for i, a in enumerate(s1_addrs + other_addrs):
        cs = _comps(a)
        if i < len(s1_addrs) and cs:
            last = (a or "").split(",")[-1]
            if not any(ch.isdigit() for ch in last):
                s1_last[component_key(last)] += 1
            s1_cnt.update(set(cs))
        u = set(cs)
        cnt.update(u)
        recs.append(u)
    F = {c for c, k in cnt.items() if k >= thr}
    admins = {c for c in F if s1_cnt[c] >= thr / 4 and s1_last[c] >= last_rate * s1_cnt[c]}
    # co-occurrence among frequent components
    co = defaultdict(Counter)
    for u in recs:
        f = u & F
        if len(f) < 2:
            continue
        for x in f:
            for y in f:
                if x != y:
                    co[x][y] += 1
    # P(admin | component) for non-admin components, from their direct co-occurrence with admins
    padm = {}
    for c in F - admins:
        d = {x: co[c][x] for x in admins if co[c][x]}
        tot = sum(d.values())
        if tot:
            padm[c] = {x: v / tot for x, v in d.items()}
    alias = {}
    for y in F - admins:
        # only components that never sit next to an admin themselves (substitutes, not cities)
        if sum(co[y][x] for x in admins) > max_co * cnt[y]:
            continue
        votes, tot = Counter(), 0
        for z, v in co[y].items():
            if z in padm:
                for x, p in padm[z].items():
                    votes[x] += v * p
                tot += v
        if tot < thr / 4:
            continue
        x, v = votes.most_common(1)[0] if votes else (None, 0)
        if x is not None and v / tot >= purity:
            alias[y] = x
    return {"admins": sorted(admins), "alias": dict(sorted(alias.items()))}
