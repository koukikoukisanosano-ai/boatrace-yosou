"""公式の過去データ（番組表 B・競走成績 K の LZH ファイル）を、Open API v1 と同じ形の日別 JSON に直す

  python tools/mbrace.py --from 2016-01-01 --to 2024-08-31 --out work/mbrace
    https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh（番組表）
    https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh（競走成績: 展示タイム・進入・ST・決まり手・払戻）
  展開には lha（lhasa）が要る:  sudo apt-get install lhasa
  無いもの（null にする）: スタート展示のST・チルト・部品交換・F/L数・平均ST・3連率・グレード
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://www1.mbrace.or.jp/od2"
RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4}
TECH = {"逃げ": 1, "差し": 2, "まくり": 3, "まくり差し": 4, "抜き": 5, "恵まれ": 6}
WIND_DIR = {"北": 1, "北北東": 2, "北東": 3, "東北東": 4, "東": 5, "東南東": 6, "南東": 7, "南南東": 8, "南": 9, "南南西": 10, "南西": 11,
            "西南西": 12, "西": 13, "西北西": 14, "北西": 15, "北北西": 16, "無風": 17}
WEATHER = {"晴": 1, "曇り": 2, "曇": 2, "雨": 3, "雪": 4, "霧": 6}
PAYKEY = {"単勝": "win", "複勝": "place", "2連単": "exacta", "2連複": "quinella", "拡連複": "quinella_place", "3連単": "trifecta", "3連複": "trio"}


def z2h(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def num(s):
    try:
        return float(s) if "." in s else int(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 競走成績（K）
RE_K_HEAD = re.compile(r"^(\d\d)KBGN")
RE_K_DATE = re.compile(r"(\d{4})/\s*(\d+)/\s*(\d+)")
RE_K_RACE = re.compile(r"^\s+(\d+)R\s+(.+?)\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm")
RE_K_BOAT = re.compile(r"^\s+(\d\d|[FLKS]\d?|欠|転|落|沈|妨|エ|不)\s+([1-6])\s+(\d{4})\s(.{8})(.*)$")
RE_K_PAY = re.compile(r"^\s+(単勝|複勝|2連単|2連複|拡連複|3連単|3連複)?\s+([\d\-=]+)\s+(\d+)(.*)$")   # 全体を半角にしてから読む


def tech_of(s: str):
    s = re.sub(r"\s", "", z2h(s))
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)   # カタカナ → ひらがな（半角ｶﾅで書かれている年がある）
    return TECH.get(s) or {"にげ": 1, "さし": 2, "まくりさし": 4, "まくりざし": 4, "ぬき": 5, "めぐまれ": 6}.get(s)


def parse_k(text: str) -> dict:
    """{jcd: {date, races: {rno: {...}}}}"""
    text = z2h(text)   # 半角カナ・全角英数をそろえる（全角空白も1文字の空白になるので桁の位置は変わらない）
    out, jcd, ymd, race, rno, payk = {}, None, None, None, None, None
    for ln in text.splitlines():
        m = RE_K_HEAD.match(ln)
        if m:
            jcd = int(m.group(1)); ymd = None; race = None
            continue
        if jcd is None:
            continue
        if ymd is None:
            m = RE_K_DATE.search(ln)
            if m and "ボートレース" in ln:
                ymd = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                out[jcd] = {"date": ymd, "races": {}}
            continue
        h = z2h(ln)
        m = RE_K_RACE.match(h)
        if m and "R" in h[:10]:
            rno = int(m.group(1))
            race = out[jcd]["races"][rno] = {
                "title": m.group(2).strip(), "distance": int(m.group(3)), "weather": WEATHER.get(m.group(4).strip()),
                "wind_dir": WIND_DIR.get(m.group(5).strip()), "wind": int(m.group(6)), "wave": int(m.group(7)),
                "boats": {}, "payouts": {}, "technique": None, "refunds": []}
            payk = None
            continue
        if race is None:
            continue
        if "レースタイム" in ln:
            race["technique"] = tech_of(ln.split("レースタイム")[-1])
            continue
        m = RE_K_BOAT.match(ln)
        if m:
            pl, w = m.group(1), int(m.group(2))
            tok = m.group(5).split()
            motor = int(tok[0]) if len(tok) > 0 and tok[0].isdigit() else None
            boat = int(tok[1]) if len(tok) > 1 and tok[1].isdigit() else None
            rest = tok[2:]
            ex = course = st = None
            if rest and re.fullmatch(r"\d\.\d\d", rest[0]):
                ex = float(rest.pop(0))
            if rest and re.fullmatch(r"[1-6]", rest[0]):
                course = int(rest.pop(0))
            if rest:
                t = rest[0]
                if re.fullmatch(r"F?\d?\.\d\d", t):
                    v = float(t.lstrip("F") if not t.lstrip("F").startswith(".") else "0" + t.lstrip("F"))
                    st = -v if t.startswith("F") else v
            place = int(pl) if pl.isdigit() and 1 <= int(pl) <= 6 else None
            src = str(place) if place else ("欠" if pl[0] in "K欠" else pl[0])   # Open API の書き方に合わせる（欠場=欠, F, L, S=失格）
            race["boats"][w] = {"place": place, "place_src": src, "number": int(m.group(3)), "name": z2h(m.group(4)).replace(" ", ""),
                                "motor": motor, "boat": boat, "ex_time": ex, "course": course, "st": st}
            if src in ("欠", "F", "L"):   # 返還になる艇
                race["refunds"].append(w)
            continue
        m = RE_K_PAY.match(ln)
        if m:
            if m.group(1):
                payk = PAYKEY[m.group(1)]
            if payk:
                combo = m.group(2)
                if payk in ("quinella", "quinella_place", "trio"):
                    combo = "=".join(sorted(re.split("[-=]", combo)))
                race["payouts"].setdefault(payk, []).append({"combination": combo, "amount": int(m.group(3))})
                if payk == "place":
                    for c2, a2 in re.findall(r"([1-6])\s+(\d+)", m.group(4)):
                        race["payouts"]["place"].append({"combination": c2, "amount": int(a2)})
    return out


# ---------------------------------------------------------------- 番組表（B）
RE_B_HEAD = re.compile(r"^(\d\d)BBGN")
RE_B_DATE = re.compile(r"(\d{4})年\s*(\d+)月\s*(\d+)日")
RE_B_RACE = re.compile(r"^\s*(\d+)R\s+(.*?)\s+H(\d+)m.*?(\d+):(\d+)")
RATE = r"\s*(\d{1,3}\.\d\d)"   # 桁があふれると前とくっつく（例 25.00110 = 2率25.00・ボート110）
RE_B_BOAT = re.compile(r"^([1-6]) (\d{4})(.{4})(\d{2})(.{2})(\d{2})([AB][12])" + RATE * 4 + r"\s*(\d{1,3})" + RATE + r"\s*(\d{1,3})" + RATE)


def parse_b(text: str) -> dict:
    out, jcd, ymd, rno = {}, None, None, None
    for ln in text.splitlines():
        m = RE_B_HEAD.match(ln)
        if m:
            jcd = int(m.group(1)); ymd = None
            continue
        if jcd is None:
            continue
        h = z2h(ln)
        if ymd is None:
            m = RE_B_DATE.search(h)
            if m and "ボートレース" in h:
                ymd = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                out[jcd] = {"date": ymd, "races": {}, "title": "", "day": None}
            m2 = re.search(r"第\s*(\d+)日", h)
            if m2 and jcd in out:
                out[jcd]["day"] = int(m2.group(1))
            continue
        m = RE_B_RACE.match(h)
        if m and "締切" in h:
            rno = int(m.group(1))
            out[jcd]["races"][rno] = {"subtitle": m.group(2).strip(), "distance": int(m.group(3)),
                                      "closed": f"{int(m.group(4)):02d}:{int(m.group(5)):02d}", "fixed": "進入固定" in h, "boats": {}}
            continue
        m = RE_B_BOAT.match(ln)
        if m and rno is not None:
            g = m.groups()
            out[jcd]["races"][rno]["boats"][int(g[0])] = {
                "number": int(g[1]), "name": g[2].replace("　", ""), "age": int(g[3]), "branch": g[4].replace("　", ""), "weight": int(g[5]),
                "rank": RANK.get(g[6]), "nat_win": num(g[7]), "nat2": num(g[8]), "loc_win": num(g[9]), "loc2": num(g[10]),
                "motor": int(g[11]), "motor2": num(g[12]), "boat": int(g[13]), "boat2": num(g[14])}
    return out


# ---------------------------------------------------------------- Open API の形に
def to_openapi(b: dict, k: dict) -> dict:
    st = {}
    for jcd in sorted(set(b) | set(k)):
        bj, kj = b.get(jcd, {}), k.get(jcd, {})
        ymd = bj.get("date") or kj.get("date")
        races = {}
        for rno in sorted(set(bj.get("races", {})) | set(kj.get("races", {}))):
            br, kr = bj.get("races", {}).get(rno, {}), kj.get("races", {}).get(rno, {})
            racers, pv, rs = {}, {}, {}
            for w in range(1, 7):
                bb, kb = br.get("boats", {}).get(w), kr.get("boats", {}).get(w)
                if not bb and not kb:
                    continue
                bb = bb or {}; kb = kb or {}
                racers[str(w)] = {"number": bb.get("number") or kb.get("number"), "name": bb.get("name") or kb.get("name"),
                                  "rank_number": bb.get("rank"), "branch_number_source": bb.get("branch", ""), "age": bb.get("age"),
                                  "weight": bb.get("weight"), "flying_count": None, "late_count": None, "average_start_timing": None,
                                  "national_win_rate": bb.get("nat_win"), "national_top_2_percent": bb.get("nat2"), "national_top_3_percent": None,
                                  "local_win_rate": bb.get("loc_win"), "local_top_2_percent": bb.get("loc2"), "local_top_3_percent": None,
                                  "motor_number": bb.get("motor") or kb.get("motor"), "motor_top_2_percent": bb.get("motor2"), "motor_top_3_percent": None,
                                  "boat_number": bb.get("boat") or kb.get("boat"), "boat_top_2_percent": bb.get("boat2"), "boat_top_3_percent": None}
                if kb:
                    # 競走成績の「進入」は本番の進入（スタート展示の進入とほぼ同じ）。展示STは無い
                    pv[str(w)] = {"course_number": kb.get("course"), "exhibition_time": kb.get("ex_time"), "start_timing": None, "tilt_adjustment": None,
                                  "parts": [], "propeller": None}
                    rs[str(w)] = {"course_number": kb.get("course"), "start_timing": kb.get("st"), "place_number": kb.get("place"),
                                  "place_number_source": kb.get("place_src")}
            rc = {"date": ymd, "race_number": rno, "closed_at": f"{ymd} {br.get('closed', '00:00')}:00", "title": "", "subtitle": br.get("subtitle", ""),
                  "distance": br.get("distance") or kr.get("distance"), "day_number": bj.get("day"), "grade_number": None, "racers": racers}
            if kr:
                rc["preview"] = {"racers": pv, "wind_speed": kr.get("wind"), "wind_direction_number": kr.get("wind_dir"), "wave_height": kr.get("wave"),
                                 "weather_number": kr.get("weather"), "source": "mbrace"}
                rc["result"] = {"racers": rs, "technique_number": kr.get("technique"), "payouts": kr.get("payouts", {}), "refunds": kr.get("refunds", [])}
            races[str(rno)] = rc
        st[str(jcd)] = {"races": races}
    return {"programs": {"stadiums": st}, "source": "mbrace"}


def fetch_lzh(s: requests.Session, kind: str, d: date) -> str | None:
    url = f"{BASE}/{kind}/{d:%Y%m}/{kind.lower()}{d:%y%m%d}.lzh"
    for i in range(3):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.ok and r.content[2:7] == b"-lh5-":
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp, "a.lzh"); p.write_bytes(r.content)
                    subprocess.run(["lha", "xq", "a.lzh"], cwd=tmp, check=True, capture_output=True)
                    txt = [f for f in Path(tmp).iterdir() if f.suffix.lower() == ".txt"]
                    return txt[0].read_bytes().decode("cp932", errors="replace") if txt else None
        except Exception as e:  # noqa: BLE001
            print("  やり直し", url, e, file=sys.stderr)
        time.sleep(2 * (i + 1))
    return None


def debug(bt: str, kt: str):
    print("== K の見出し行")
    for ln in sorted({ln for ln in kt.splitlines() if "レースタイム" in ln})[:8]:
        print(repr(ln))
    print("== K で読めなかった、艇らしい行")
    for ln in [ln for ln in kt.splitlines() if re.match(r"^\s+\S+\s+[1-6]\s+\d{4}", ln) and not RE_K_BOAT.match(ln)][:10]:
        print(repr(ln))
    print("== B の場ごとの見出し・読めなかった行")
    cur = None
    for ln in bt.splitlines():
        if RE_B_HEAD.match(ln):
            cur = ln
        if "ボートレース" in ln and ("年" in ln or "日" in ln):
            print(repr(ln))
    for ln in [ln for ln in bt.splitlines() if re.match(r"^[1-6] \d{4}", ln) and not RE_B_BOAT.match(ln)][:15]:
        print("x", repr(ln))
    for ln in [ln for ln in bt.splitlines() if "締切" in ln and not RE_B_RACE.match(z2h(ln))][:5]:
        print("r", repr(ln))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true", help="読めなかった行を表示する")
    ap.add_argument("--parse-only", nargs=2, metavar=("B.txt", "K.txt"), help="展開済みのテキストを読む（確認用）")
    a = ap.parse_args()
    if a.parse_only:
        bt, kt = (Path(x).read_text(encoding="utf-8") for x in a.parse_only)
        print(json.dumps(to_openapi(parse_b(bt), parse_k(kt)), ensure_ascii=False)[:3000])
        return
    s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (boatrace-yosou research)"
    d, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    n = 0
    from collections import Counter
    cnt = Counter()
    while d <= end:
        out = Path(a.out, f"{d:%Y}", f"{d:%Y%m%d}.json.gz")
        if not out.exists():
            bt, kt = fetch_lzh(s, "B", d), fetch_lzh(s, "K", d)
            if a.debug and d == date.fromisoformat(a.start):
                debug(bt or "", kt or "")
            if bt or kt:
                j = to_openapi(parse_b(bt or ""), parse_k(kt or ""))
                for sv in j["programs"]["stadiums"].values():
                    for rc in sv["races"].values():
                        cnt["レース"] += 1
                        res = rc.get("result")
                        if not res:
                            cnt["結果なし"] += 1; continue
                        cnt["決まり手なし"] += res["technique_number"] is None
                        cnt["2連単なし"] += not res["payouts"].get("exacta")
                        cnt["選手なし(B)"] += not rc["racers"] or rc["racers"].get("1", {}).get("national_win_rate") is None
                        for w, x in res["racers"].items():
                            cnt["着:" + str(x["place_number_source"])] += 1
                            cnt["進入なし"] += x["course_number"] is None
                            cnt["展示なし"] += rc["preview"]["racers"][w]["exhibition_time"] is None
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(gzip.compress(json.dumps(j, ensure_ascii=False, separators=(",", ":")).encode()))
                n += 1
            time.sleep(0.3)
        d += timedelta(days=1)
    print(f"{n} 日分を書き出しました → {a.out}")
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
