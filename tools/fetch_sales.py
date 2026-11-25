"""券種ごとの売上（レースごと）を集める: いまは常滑だけが、翌日の出走表PDF（その2「記録表」）に
レースごとの券種別「売上枚数・的中枚数」を載せている。data/sales/YYYYMMDD.json に保存
  {"8": {"1": {"total": 105756500, "q": [枚, 的中枚, 配当], "e": [...], "t": [...], "f": [...]}}}
  q=2連複 e=2連単 t=3連単 f=3連複（1枚=100円）
  python tools/fetch_sales.py [YYYYMMDD ...]   （省略時は昨日と今日。PDFは翌日付けで出る）
"""
from __future__ import annotations

import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sales"
UA = "Mozilla/5.0 (compatible; boatrace-yosou/1.0; +https://github.com/koukikoukisanosano-ai/boatrace-yosou)"
JST = timezone(timedelta(hours=9))
Z2H = str.maketrans("０１２３４５６７８９", "0123456789")
SOURCES = {8: "https://www.boatrace-tokoname.jp/uploads/cdn/pdf/syussou/08_{D}_2.pdf"}   # {D}=翌日 YYYYMMDD
NUM = r"([\d,]+)"
PAT = {
    "e": re.compile(r"２連単\s*([１-６])A([１-６])\s*" + NUM + r"円\s*\d+\s*" + NUM + r"枚\s*" + NUM + "枚"),
    "q": re.compile(r"２連複\s*([１-６])B([１-６])\s*" + NUM + r"円\s*\d+\s*" + NUM + r"枚\s*" + NUM + "枚"),
    "t": re.compile(r"３連単\s*([１-６])A([１-６])A([１-６])\s*" + NUM + r"円\s*\d+\s*" + NUM + r"枚\s*" + NUM + "枚"),
    "f": re.compile(r"３連複\s*([１-６])B([１-６])B([１-６])\s*" + NUM + r"円\s*\d+\s*" + NUM + r"枚\s*" + NUM + "枚"),
}
TOTAL = re.compile(r"売上合計\s+" + NUM + r"円\s*返還合計")
DAYHEAD = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日.{0,6}記録表")
n = lambda s: int(s.replace(",", ""))


def parse(pdf_bytes: bytes):
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(p.extract_text(layout=True) or "" for p in pdf.pages)
    m = DAYHEAD.search(text.replace(" ", ""))
    day = f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}" if m else None
    found = {k: [x for x in p.finditer(text)] for k, p in PAT.items()}
    totals = [n(x.group(1)) for x in TOTAL.finditer(text)]
    races = {}
    nr = len(totals)
    if not nr or any(len(v) != nr for v in found.values()):
        return day, None, {k: len(v) for k, v in found.items()} | {"total": nr}
    for i in range(nr):
        rc = {"total": totals[i]}
        for k, v in found.items():
            g = v[i].groups()
            combo = "".join(g[:-3]).translate(Z2H)
            rc[k] = [n(g[-2]), n(g[-1]), n(g[-3]), combo]   # 売上枚数, 的中枚数, 配当(円), 組
        races[str(i + 1)] = rc
    return day, races, None


def fetch(day: str, s: requests.Session) -> int:
    nxt = (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    got = 0
    path = OUT / f"{day}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for jcd, url in SOURCES.items():
        if str(jcd) in data:
            continue
        try:
            r = s.get(url.format(D=nxt), timeout=60)
        except Exception as e:  # noqa: BLE001
            print(day, jcd, "取得失敗", e); continue
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            print(day, jcd, "PDFなし", r.status_code); continue
        d, races, err = parse(r.content)
        if d != day:
            print(day, jcd, "別の日の記録表", d); continue
        if not races:
            print(day, jcd, "読めず", err); continue
        data[str(jcd)] = races; got += len(races)
        q = [v["q"][0] * 100 for v in races.values()]
        print(day, jcd, f"{len(races)}レース 2連複の売上 {min(q):,}〜{max(q):,}円")
    if data:
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return got


def main() -> int:
    days = sys.argv[1:]
    if not days:
        t = datetime.now(JST)
        days = [(t - timedelta(days=1)).strftime("%Y%m%d"), t.strftime("%Y%m%d")]
    s = requests.Session(); s.headers.update({"User-Agent": UA})
    for d in days:
        fetch(d, s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
