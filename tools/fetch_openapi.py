"""Boatrace Open API（非公式）の日別JSONを data/openapi/YYYY/YYYYMMDD.json.gz に保存する。

使い方:
  python tools/fetch_openapi.py                      # 2026-01-01〜昨日のうち、まだ無い日＋直近3日を取得
  python tools/fetch_openapi.py --from 2026-10-01 --to 2026-10-31
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

JST = timezone(timedelta(hours=9))
BASE = "https://boatraceopenapi.github.io/api/v1"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "openapi"
FIRST_DAY = date(2026, 1, 1)
UA = "boatrace-yosou/1.0 (+https://github.com/koukikoukisanosano-ai/boatrace-yosou)"


def path_for(d: date) -> Path:
    return OUT / f"{d:%Y}" / f"{d:%Y%m%d}.json.gz"


def fetch_day(s: requests.Session, d: date) -> dict | None:
    url = f"{BASE}/{d:%Y}/{d:%Y%m%d}.json"
    for attempt in range(4):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  再試行 {d} ({e})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"取得できませんでした: {url}")


def save(d: date, j: dict) -> int:
    p = path_for(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(j, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    data = gzip.compress(raw, compresslevel=9, mtime=0)
    if p.exists() and p.read_bytes() == data:
        return 0
    p.write_bytes(data)
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--refresh", type=int, default=3, help="直近N日は既に有っても取り直す")
    a = ap.parse_args()
    today = datetime.now(JST).date()
    start = date.fromisoformat(a.start) if a.start else FIRST_DAY
    end = date.fromisoformat(a.end) if a.end else today - timedelta(days=1)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    d, n_new, n_skip = start, 0, 0
    while d <= end:
        recent = (today - d).days <= a.refresh
        if path_for(d).exists() and not recent and not (a.start or a.end):
            n_skip += 1
        else:
            j = fetch_day(s, d)
            if j is None:
                print(f"{d}: データなし(404)")
            else:
                size = save(d, j)
                n_races = sum(len(st.get("races", {})) for st in j.get("programs", {}).get("stadiums", {}).values())
                print(f"{d}: {n_races}レース {'更新' if size else '変更なし'}")
                n_new += bool(size)
            time.sleep(0.3)
        d += timedelta(days=1)
    print(f"完了: 保存/更新 {n_new} 日, 既存のため省略 {n_skip} 日")


if __name__ == "__main__":
    main()
