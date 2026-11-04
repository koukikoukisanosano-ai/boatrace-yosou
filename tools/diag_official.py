"""GitHub Actions から BOAT RACE 公式サイトの直前情報・オッズが取得できるかを確認する。
取得したHTMLの見本を data/debug/ に保存する（構造確認用）。"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

JST = timezone(timedelta(hours=9))
BASE = "https://www.boatrace.jp/owpc/pc/race"
OUT = Path(__file__).resolve().parent.parent / "data" / "debug"
UA = "Mozilla/5.0 (compatible; boatrace-yosou/1.0; +https://github.com/koukikoukisanosano-ai/boatrace-yosou)"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
    now = datetime.now(JST)
    hd = now.strftime("%Y%m%d")

    def get(path: str, **params) -> requests.Response:
        t = time.time()
        r = s.get(f"{BASE}/{path}", params=params, timeout=30)
        print(f"GET {path} {params} -> {r.status_code} {len(r.content)}B {time.time() - t:.2f}s")
        time.sleep(1.5)
        return r

    r = get("index", hd=hd)
    if r.status_code != 200:
        print("公式サイトに接続できませんでした")
        return 1
    (OUT / "index.html").write_bytes(r.content)
    jcds = sorted(set(re.findall(r"raceindex\?jcd=(\d{2})", r.text)))
    print("開催場:", jcds)
    best = None
    for jcd in jcds:
        ri = get("raceindex", jcd=jcd, hd=hd)
        if jcd == jcds[0]:
            (OUT / "raceindex.html").write_bytes(ri.content)
        for m in re.finditer(r"rno=(\d+)&amp;jcd=\d{2}&amp;hd=\d{8}[^<]*?</a>.*?(\d{1,2}:\d{2})", ri.text, re.S):
            pass
        times = re.findall(r">(\d{1,2}:\d{2})<", ri.text)
        for i, t in enumerate(times[:12], start=1):
            hh, mm = map(int, t.split(":"))
            dl = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            mins = (dl - now).total_seconds() / 60
            if -5 <= mins <= 25 and (best is None or abs(mins - 8) < abs(best[2] - 8)):
                best = (jcd, i, mins)
        if len(jcds) > 6 and best:
            break
    print("対象レース:", best)
    if not best:
        jcd, rno = jcds[0], 12
    else:
        jcd, rno = best[0], best[1]
    for page in ("beforeinfo", "racelist", "odds3t", "raceresult"):
        r = get(page, rno=rno, jcd=jcd, hd=hd)
        (OUT / f"{page}.html").write_bytes(r.content)
        if page == "beforeinfo":
            print("  展示タイム表あり:", "is-w748" in r.text, " スタート展示あり:", "table1_boatImage1Time" in r.text)
        if page == "odds3t":
            print("  オッズ数:", len(re.findall(r"oddsPoint", r.text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
