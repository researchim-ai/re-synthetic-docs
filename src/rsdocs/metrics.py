import re
from typing import Tuple


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = text.replace("  \n", "\n").replace("**", "").strip()
    s = "\n".join(re.sub(r"\s+", " ", line).strip() for line in s.splitlines())
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def compute_em_and_cer(pred: str, gt: str) -> Tuple[int, float]:
    p = normalize_text(pred)
    g = normalize_text(gt)
    em = int(p == g)
    cer = levenshtein_distance(p, g) / max(len(g), 1)
    return em, cer


