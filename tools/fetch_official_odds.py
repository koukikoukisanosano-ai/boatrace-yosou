"""公式サイト（boatrace.jp）の「締切時オッズ」から、2連単の確定オッズを過去の日の分だけ取る。

Turnmark API の確定オッズは 2026-01-01 からしか無いので、それより前の検証のために使う。
レースの一覧は data/openapi の出走表から作る。保存は data/odds と同じ形（2連単 "e" 30通りだけ）。
すでにある日のファイルは上書きしない。公式サイトに負担をかけないよう、1件ごとに間をあける。

  python tools/fetch_official_odds.py --from 2025-01-01 --to 2025-01-31
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
import os
OUT = Path(os.environ.get("ODDS_OUT", ROOT / "data" / "odds"))
URL = "https://www.boatrace.jp/owpc/pc/race/odds2tf"
UA = "boatrace-yosou/1.0 (+https://github.com/koukikoukisanosano-ai/boatrace-yosou)"
EXA = list(itertools.permutations(range(1, 7), 2))
CELL = re.compile(r'<td class="is-fs14 is-boatColor\d[^"]*">(\d)</td>\s*<td class="oddsPoint[^"]*">([^<]*)</td>')


def parse(html: str):
    """2連単の表（最初の表）→ 30通り。列は1着の艇（1〜6）、各行に [2着の艇, オッズ] が並ぶ"""
    i = html.find("2連単オッズ")
    if i < 0:
        return None
    j = html.find("title7_mainLabel", i + 10)
    body = html[i:j if j > 0 else None]
    cells = CELL.findall(body)
    if len(cells) != 30:
        return None
    got = {}
    for n, (second, v) in enumerate(cells):
        first = n % 6 + 1
        try:
            got[(first, int(second))] = float(v)
        except ValueError:
            got[(first, int(second))] = None   # 欠場など
    if set(got) != set(EXA):
        return None
    return [got[k] for k in EXA]


def races_of(d: date):
    p = ROOT / "data" / "openapi" / str(d.year) / f"{d:%Y%m%d}.json.gz"
    if not p.exists():
        return []
    st = (json.loads(gzip.decompress(p.read_bytes())).get("programs") or {}).get("stadiums") or {}
    return sorted((int(j), int(r)) for j, v in st.items() for r in (v.get("races") or {}))


def get(s, params):
    for attempt in range(5):
        try:
            r = s.get(URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.text
            print(f"  {params} → {r.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  {params} → {e}", file=sys.stderr)
        time.sleep(10 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--limit", type=int, default=0, help="1日あたりのレース数の上限（確かめる用）")
    ap.add_argument("--wait", type=float, default=1.0, help="1件ごとの間隔（秒）")
    a = ap.parse_args()
    s = requests.Session()
    s.headers["User-Agent"] = UA
    d, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    while d <= end:
        p = OUT / str(d.year) / f"{d:%Y%m%d}.json.gz"
        rs = races_of(d)
        if p.exists() or not rs:
            print(f"{d}: {'既にあります' if p.exists() else '出走表がありません'}")
            d += timedelta(days=1)
            continue
        out, bad = {}, 0
        for jcd, rno in rs[: a.limit or None]:
            html = get(s, {"rno": rno, "jcd": f"{jcd:02d}", "hd": f"{d:%Y%m%d}"})
            e = parse(html) if html else None
            if e and any(x is not None for x in e):
                out[f"{jcd}-{rno}"] = {"e": e}
            else:
                bad += 1
            time.sleep(a.wait)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":"), sort_keys=True).encode(), 9, mtime=0))
        print(f"{d}: {len(out)} レース（取れなかった {bad}）", flush=True)
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
