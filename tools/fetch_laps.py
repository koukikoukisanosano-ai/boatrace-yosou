"""各競艇場のサイトから「オリジナル展示データ」（一周・まわり足・直線・半周など）を、その日の分だけ集める。

公式サイト（boatrace.jp）には載っておらず、24場それぞれのサイトが独自に計測して載せている。
過去の日付の分は残っていないので、毎日（昼の部の後と夜の部の後）その日の全レースを取りに行く。

  python tools/fetch_laps.py [--date YYYY-MM-DD]

- 場ごとに「載っていそうなページ」の候補を順に試し、表が読めたページを data/laps/sources.json に覚える
- 結果は data/laps/YYYYMMDD.json: {"場-R": {"cols": [...], "boats": {"1": {"一周": 37.27, ...}}, "url": ...}}
- 1件ごとに間をあける（各場のサイトに負担をかけないように）
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "laps"
JST = timezone(timedelta(hours=9))
UA = "Mozilla/5.0 (compatible; boatrace-yosou/1.0; +https://github.com/koukikoukisanosano-ai/boatrace-yosou)"

# 場ごとのサイト（多くの場は同じ作りの「直前情報」ページ: /sp/index.php?page=yosou-cyokuzen&race=R&day=YYYYMMDD）
SITES = {
    1: "https://www.boatrace-kiryu.com", 2: "https://www.boatrace-toda.jp", 3: "https://www.boatrace-edogawa.com",
    4: "https://www.heiwajima.gr.jp", 5: "https://www.boatrace-tamagawa.com", 6: "https://www.boatrace-hamanako.jp",
    7: "https://www.gamagori-kyotei.com", 8: "https://www.boatrace-tokoname.jp", 9: "https://www.boatrace-tsu.com",
    10: "https://www.boatrace-mikuni.jp", 11: "https://www.boatrace-biwako.jp", 12: "https://www.boatrace-suminoe.jp",
    13: "https://www.boatrace-amagasaki.jp", 14: "https://www.n14.jp", 15: "https://www.marugameboat.jp",
    16: "https://www.kojimaboat.jp", 17: "https://www.boatrace-miyajima.com", 18: "https://www.boatrace-tokuyama.jp",
    19: "https://www.boatrace-shimonoseki.jp", 20: "https://www.wmb.jp", 21: "https://www.boatrace-ashiya.com",
    22: "https://www.boatrace-fukuoka.com", 23: "https://www.boatrace-karatsu.jp", 24: "https://omurakyotei.jp",
}
# 場ごとの特別なページ（わかっているもの）。{R}=レース番号, {RR}=2桁, {D}=YYYYMMDD
SPECIAL = {
    8: ["/raceguide/kyogi15/{R}/"],
    12: ["/asp/kyogi/12/pc/st02{RR}.htm", "/asp/kyogi/12/sp/yoso05{RR}.htm"],
    16: ["/asp/kyogi/16/sp/yoso05{RR}.htm", "/asp/kyogi/16/pc/st02{RR}.htm"],
}
COMMON = ["/sp/index.php?page=yosou-cyokuzen&race={R}&day={D}", "/sp/index.php?page=yosou-cyokuzen&race={R}"]
LABELS = [("半周", "半周"), ("一周", "一周"), ("1周", "一周"), ("まわり足", "まわり足"), ("回り足", "まわり足"), ("直線", "直線"), ("展示", "展示")]


def label_of(h: str):
    h = h.replace(" ", "").replace("　", "")
    for k, v in LABELS:
        if k in h:
            return v
    return None


def num(s: str):
    s = (s or "").strip().replace("−", "-")
    m = re.fullmatch(r"-?\d+(\.\d+)?", s)
    return float(s) if m else None


def parse_tables(html: str):
    """一周・まわり足などの見出しがある表を読む → {枠: {項目: 値}}"""
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        if not rows:
            continue
        head = None
        for i, tr in enumerate(rows[:3]):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            labs = [label_of(c) for c in cells]
            if any(l in ("一周", "まわり足", "半周") for l in labs):
                head, start = labs, i + 1
                break
        if not head:
            continue
        boats = {}
        for tr in rows[start:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if not cells:
                continue
            w = re.match(r"^([1-6])\b", cells[0])
            if not w or w.group(1) in boats:
                continue
            # 見出しと列の数が合わない（枠の列が結合など）ときは右寄せで合わせる
            off = len(cells) - len(head)
            vals = {}
            for j, lab in enumerate(head):
                k = j + off if off > 0 else j
                if lab and 0 <= k < len(cells):
                    v = num(cells[k].split(" ")[0])
                    if v is not None:
                        vals[lab] = v
            if vals:
                boats[w.group(1)] = vals
        if len(boats) >= 5 and any(any(k in v for k in ("一周", "まわり足", "半周", "直線")) for v in boats.values()):
            return boats
    return None


def parse_text(html: str):
    """表になっていないページ: 「オリジナル展示 枠 体重 … 一周 まわり足 直線 …」の後の数字を読む"""
    t = re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" "))
    m = re.search(r"オリジナル展示\s*枠((?:\s*[^\d\s]+){3,9}?)\s+1\s", t)
    if not m:
        return None
    heads = m.group(1).split()
    labs = [label_of(h) for h in heads]
    if not any(l in ("一周", "まわり足", "半周") for l in labs):
        return None
    body = t[m.end() - 2: m.end() + 600]
    boats = {}
    for w in range(1, 7):
        mm = re.search(rf"(?:^|\s){w}((?:\s+(?:-?\d+\.\d+|-+|--?\.-+))+)", body)
        if not mm:
            continue
        toks = mm.group(1).split()
        vals = {}
        for lab, tok in zip(labs, toks):
            v = num(tok)
            if lab and v is not None:
                vals[lab] = v
        if vals:
            boats[str(w)] = vals
    return boats if len(boats) >= 5 else None


def races_today(d: date):
    """その日に開催している場とレース（自動判定の記録 → 公開データの順に探す）"""
    ymd = d.strftime("%Y%m%d")
    out = {}
    p = ROOT / "docs" / "data" / "programs" / f"{ymd}.json"
    if p.exists():
        for k in json.loads(p.read_text()):
            j, r = map(int, k.split("-"))
            out.setdefault(j, set()).add(r)
    q = ROOT / "data" / "openapi" / str(d.year) / f"{ymd}.json.gz"
    if not out and q.exists():
        st = json.loads(gzip.decompress(q.read_bytes())).get("programs", {}).get("stadiums") or {}
        for j, v in st.items():
            out[int(j)] = {int(r) for r in (v.get("races") or {})}
    return {j: sorted(rs) for j, rs in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(JST).strftime("%Y-%m-%d"))
    ap.add_argument("--wait", type=float, default=1.5)
    ap.add_argument("--venues", default="", help="この場だけ（例 11,12）")
    a = ap.parse_args()
    d = date.fromisoformat(a.date)
    ymd = d.strftime("%Y%m%d")
    OUT.mkdir(parents=True, exist_ok=True)
    src_p = OUT / "sources.json"
    sources = json.loads(src_p.read_text()) if src_p.exists() else {}
    out_p = OUT / f"{ymd}.json"
    got = json.loads(out_p.read_text()) if out_p.exists() else {}
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
    venues = races_today(d)
    if a.venues:
        venues = {j: rs for j, rs in venues.items() if str(j) in a.venues.split(",")}
    print(f"{a.date}: {len(venues)} 場", file=sys.stderr)
    summary = {}
    for jcd, rs in sorted(venues.items()):
        base = SITES.get(jcd)
        if not base:
            continue
        known = sources.get(str(jcd), {}).get("path")
        cands = ([known] if known else []) + [x for x in SPECIAL.get(jcd, []) + COMMON if x != known]
        ok = 0
        for rno in rs:
            key = f"{jcd}-{rno}"
            if key in got:
                ok += 1
                continue
            for path in cands:
                url = base + path.format(R=rno, RR=f"{rno:02d}", D=ymd)
                try:
                    r = s.get(url, timeout=20)
                    time.sleep(a.wait)
                except requests.RequestException:
                    continue
                if r.status_code != 200:
                    continue
                r.encoding = r.apparent_encoding if not r.encoding or r.encoding.lower() == "iso-8859-1" else r.encoding
                boats = parse_tables(r.text) or parse_text(r.text)
                if boats:
                    cols = sorted({k for v in boats.values() for k in v})
                    got[key] = {"cols": cols, "boats": boats, "url": url, "at": datetime.now(JST).isoformat(timespec="seconds")}
                    ok += 1
                    if sources.get(str(jcd), {}).get("path") != path:
                        sources[str(jcd)] = {"path": path, "cols": cols, "since": a.date}
                        cands = [path] + [x for x in cands if x != path]
                    break
            else:
                if rno == rs[0] and not known:
                    break   # 1R でどの候補も読めない場は、今日はあきらめる（無駄に取りに行かない）
        summary[jcd] = f"{ok}/{len(rs)}"
        print(f"  場{jcd}: {ok}/{len(rs)} レース", file=sys.stderr)
    out_p.write_text(json.dumps(got, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    src_p.write_text(json.dumps(sources, ensure_ascii=False, indent=1, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
