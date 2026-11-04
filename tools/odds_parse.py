"""BOAT RACE 公式サイトのオッズページの読み取り（締切前のオッズを取るため）

  odds3t  3連単      odds3f  3連複      odds2tf 2連単・2連複
  oddsk   拡連複     oddstf  単勝・複勝
返す形は data/odds の保存形式と同じ（t, f, e, q, k, w, p）。
"""
from __future__ import annotations

import itertools
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # pragma: no cover
    pass

B = range(1, 7)
TRI = list(itertools.permutations(B, 3))
EXA = list(itertools.permutations(B, 2))
QUI = list(itertools.combinations(B, 2))
TRIO = list(itertools.combinations(B, 3))


def _t(td) -> str:
    return unicodedata.normalize("NFKC", td.get_text(" ", strip=True)) if td is not None else ""


def _num(s: str):
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else None


def _range(s: str):
    v = re.findall(r"\d+(?:\.\d+)?", s or "")
    return [float(v[0]), float(v[1])] if len(v) >= 2 else ([float(v[0]), float(v[0])] if v else [None, None])


def updated_at(html: str) -> str | None:
    m = re.search(r"オッズ更新時間\s*([0-9]{1,2}:[0-9]{2})", html)
    return m.group(1) if m else None


def _tables(soup):
    return [t for t in soup.select("div.table1 table") if t.select("td.oddsPoint")]


def _three(table) -> dict:
    """3連単・3連複の表: 列のまとまり（1着の艇）ごとに [2着(行をまたぐ), 3着, オッズ]"""
    out = {}
    left = [0] * 6
    second = [None] * 6
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td", recursive=False)
        i = 0
        for g in range(6):
            if left[g] > 0:
                left[g] -= 1
            else:
                if i >= len(tds):
                    break
                td = tds[i]; i += 1
                second[g] = _num(_t(td))
                left[g] = int(td.get("rowspan", 1)) - 1
            if i + 1 >= len(tds) + 1:
                break
            third_td, odds_td = tds[i], tds[i + 1] if i + 1 < len(tds) else None
            i += 2
            if odds_td is None or "oddsPoint" not in (odds_td.get("class") or []):
                continue
            s, t = second[g], _num(_t(third_td))
            if s and t:
                out[(g + 1, int(s), int(t))] = _num(_t(odds_td))
    return out


def _two(table, ranged=False) -> dict:
    """2連単・2連複・拡連複の表: 列のまとまりごとに [2着, オッズ]"""
    out = {}
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td", recursive=False)
        for g in range(min(6, len(tds) // 2)):
            a, o = tds[2 * g], tds[2 * g + 1]
            if "oddsPoint" not in (o.get("class") or []):
                continue
            s = _num(_t(a))
            if s:
                out[(g + 1, int(s))] = _range(_t(o)) if ranged else _num(_t(o))
    return out


def parse(page: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    tabs = _tables(soup)
    res: dict = {"at": updated_at(html)}
    if page == "odds3t" and tabs:
        d = _three(tabs[0])
        res["t"] = [d.get(c) for c in TRI]
    elif page == "odds3f" and tabs:
        d = {tuple(sorted(k)): v for k, v in _three(tabs[0]).items()}
        res["f"] = [d.get(c) for c in TRIO]
    elif page == "odds2tf" and len(tabs) >= 2:
        e = _two(tabs[0]); q = {tuple(sorted(k)): v for k, v in _two(tabs[1]).items()}
        res["e"] = [e.get(c) for c in EXA]
        res["q"] = [q.get(c) for c in QUI]
    elif page == "oddsk" and tabs:
        k = {tuple(sorted(kk)): v for kk, v in _two(tabs[0], ranged=True).items()}
        res["k"] = [k.get(c, [None, None]) for c in QUI]
    elif page == "oddstf" and len(tabs) >= 2:
        def single(table, ranged):
            out = {}
            for tr in table.select("tbody tr"):
                tds = tr.find_all("td", recursive=False)
                if len(tds) >= 3 and "oddsPoint" in (tds[-1].get("class") or []):
                    w = _num(_t(tds[0]))
                    if w:
                        out[int(w)] = _range(_t(tds[-1])) if ranged else _num(_t(tds[-1]))
            return out
        w = single(tabs[0], False); p = single(tabs[1], True)
        res["w"] = [w.get(b) for b in B]
        res["p"] = [p.get(b, [None, None]) for b in B]
    return res
