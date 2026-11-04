"""過去の確定オッズ（Turnmark API・非公式）を data/odds/YYYY/YYYYMMDD.json.gz に小さくまとめて保存する。

Turnmark API は前日分を翌日の朝〜昼に公開する（2026-01-01以降）。
保存形式（レースごと "場-R"）:
  t: 3連単 120通り（1着→2着→3着 の辞書順）  e: 2連単 30通り  q: 2連複 15通り  f: 3連複 20通り
  w: 単勝 6  p: 複勝 [下限,上限]×6  k: 拡連複 [下限,上限]×15     値が無いところは null

  python tools/fetch_odds.py                 # 未取得の日（＋直近3日）
  python tools/fetch_odds.py --from 2026-01-01 --to 2026-01-31
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

JST = timezone(timedelta(hours=9))
BASE = "https://turnmark.github.io/api/v1"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "odds"
FIRST_DAY = date(2026, 1, 1)
UA = "boatrace-yosou/1.0 (+https://github.com/koukikoukisanosano-ai/boatrace-yosou)"

B = range(1, 7)
TRI = list(itertools.permutations(B, 3))
EXA = list(itertools.permutations(B, 2))
QUI = list(itertools.combinations(B, 2))
TRIO = list(itertools.combinations(B, 3))


def num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def dig(o, *keys):
    for k in keys:
        if not isinstance(o, dict):
            return None
        o = o.get(str(k))
    return o


def rng(v):
    if not isinstance(v, dict):
        return [None, None]
    return [num(v.get("lower_limit")), num(v.get("upper_limit"))]


def compact(odds: dict) -> dict:
    return {
        "t": [num(dig(odds.get("trifecta"), a, b, c)) for a, b, c in TRI],
        "e": [num(dig(odds.get("exacta"), a, b)) for a, b in EXA],
        "q": [num(dig(odds.get("quinella"), a, b)) for a, b in QUI],
        "f": [num(dig(odds.get("trio"), a, b, c)) for a, b, c in TRIO],
        "w": [num(dig(odds.get("win"), a)) for a in B],
        "p": [rng(dig(odds.get("place"), a)) for a in B],
        "k": [rng(dig(odds.get("quinella_place"), a, b)) for a, b in QUI],
    }


def path_for(d: date) -> Path:
    return OUT / f"{d:%Y}" / f"{d:%Y%m%d}.json.gz"


def fetch_day(s: requests.Session, d: date) -> dict | None:
    url = f"{BASE}/{d:%Y}/{d:%Y%m%d}.json"
    for attempt in range(4):
        try:
            r = s.get(url, timeout=90)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  再試行 {d} ({e})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"取得できませんでした: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--refresh", type=int, default=3)
    a = ap.parse_args()
    today = datetime.now(JST).date()
    start = date.fromisoformat(a.start) if a.start else FIRST_DAY
    end = date.fromisoformat(a.end) if a.end else today - timedelta(days=1)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    d, saved, skipped, missing = start, 0, 0, 0
    while d <= end:
        recent = (today - d).days <= a.refresh
        if path_for(d).exists() and not recent and not (a.start or a.end):
            skipped += 1
        else:
            j = fetch_day(s, d)
            if j is None:
                missing += 1
                print(f"{d}: まだ公開されていません")
            else:
                races = {}
                for jcd, st in ((j.get("programs") or {}).get("stadiums") or {}).items():
                    for rno, rc in (st.get("races") or {}).items():
                        if rc.get("odds"):
                            races[f"{int(jcd)}-{int(rno)}"] = compact(rc["odds"])
                p = path_for(d)
                p.parent.mkdir(parents=True, exist_ok=True)
                data = gzip.compress(json.dumps(races, separators=(",", ":"), sort_keys=True).encode(), 9, mtime=0)
                if not (p.exists() and p.read_bytes() == data):
                    p.write_bytes(data)
                    saved += 1
                print(f"{d}: {len(races)} レース")
            time.sleep(0.3)
        d += timedelta(days=1)
    print(f"完了: 保存 {saved} 日 / 既存 {skipped} 日 / 未公開 {missing} 日")


if __name__ == "__main__":
    main()
