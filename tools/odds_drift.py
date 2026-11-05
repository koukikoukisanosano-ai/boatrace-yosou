"""締切前オッズの動きと、締切前オッズで選んだときの成績（毎日たまっていく実際のデータで検証）

  python tools/odds_drift.py
入力（毎日自動でたまる）:
  data/preds/YYYY/YYYYMMDD.json.gz  その日のモデルで出した2連単の確率（report.mjs。オッズは混ぜない）
  docs/live/YYYYMMDD.json           締切前に取った2連単オッズ（e0 = 買い目を決めた約5分前、e = 締切1分前ごろ）
  data/odds/YYYY/YYYYMMDD.json.gz   確定オッズ（翌日の昼に取り込み）
  data/openapi/YYYY/YYYYMMDD.json.gz 結果と払戻
出力: docs/data/drift.json
  - オッズの動き（締切1分前 → 確定、約5分前 → 確定）
  - 今の買い方を「約5分前（買い目を決めて通知する時点）」「締切1分前」「確定」のオッズで選んだときの回収率（払戻は実際の払戻）
  - 約5分前のオッズで選んだ場合の買い方の総当たり（十分たまったら、学習し直しのときにこの買い方を使う）
"""
from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXA = [f"{a}-{b}" for a in range(1, 7) for b in range(1, 7) if a != b]
MIN_DAYS_ADOPT = 30        # これだけ日数がたまったら、約5分前のオッズで選んだ買い方を使ってよい
MIN_BETS_ADOPT = 1500


def jload(p: Path):
    if not p.exists():
        return None
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_text())


def payouts(day_json):
    out = {}
    for jcd, st in (day_json.get("programs", {}).get("stadiums") or {}).items():
        for rno, rc in (st.get("races") or {}).items():
            res = rc.get("result") or {}
            pays = (res.get("payouts") or {}).get("exacta") or []
            refunds = [int(x) for x in (res.get("refunds") or res.get("refund_numbers") or []) if str(x).isdigit()]
            got = {}
            for p in pays:
                combo = p.get("combination") or p.get("combo")
                amt = p.get("payout") or p.get("amount")
                if combo and amt:
                    got[str(combo).replace("=", "-")] = amt / 100
            if got:
                out[f"{int(jcd)}-{int(rno)}"] = (got, refunds)
    return out


def arr(v):
    return np.array([np.nan if x is None else float(x) for x in v]) if v else None


def rule_list():
    return [dict(t=t, minp=mp, maxo=mo, mino=mi, k=k)
            for t, mp, mo, mi, k in itertools.product([1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0], [0.0, 0.04, 0.07, 0.1], [30.0, 60.0, 100.0], [0.0, 10.0], [1, 2, 3])]


def pick(p, o, r):
    """research_exacta.py と同じ選び方: 期待値の高い順に k 点の中から、条件に合うものだけ"""
    ev = p * np.nan_to_num(o, nan=0)
    order = np.argsort(-ev)[: r["k"]]
    return [j for j in order if np.isfinite(o[j]) and ev[j] >= r["t"] and p[j] >= r["minp"] and r["mino"] < o[j] <= r["maxo"]]


def main():
    rows = []        # (日付, p[30], e0, e1, final, pay[30])
    days = []
    for pf in sorted((ROOT / "data" / "preds").rglob("*.json.gz")):
        ymd = pf.name[:8]
        pr = jload(pf); live = jload(ROOT / "docs" / "live" / f"{ymd}.json")
        fin = jload(ROOT / "data" / "odds" / ymd[:4] / f"{ymd}.json.gz")
        res = jload(ROOT / "data" / "openapi" / ymd[:4] / f"{ymd}.json.gz")
        if not pr or not live or not res:
            continue
        pays = payouts(res)
        n = 0
        for key, ex in pr["exacta"].items():
            lo = (live.get("odds") or {}).get(key) or {}
            if not lo.get("e") or key not in pays:
                continue
            p = np.array([ex.get(c, 0.0) for c in EXA])
            got, refunds = pays[key]
            pay = np.array([1.0 if any(int(b) in refunds for b in c.split("-")) else got.get(c, 0.0) for c in EXA])
            e1 = arr(lo["e"]) if lo.get("final") else None
            e0 = arr(lo.get("e0") or (None if lo.get("final") else lo["e"]))
            fo = arr((fin or {}).get(key, {}).get("e")) if fin else None
            rows.append((ymd, p, e0, e1, fo, pay)); n += 1
        if n:
            days.append(ymd)
    out = {"days": len(days), "first": days[0] if days else None, "last": days[-1] if days else None, "races": len(rows)}
    if not rows:
        (ROOT / "docs" / "data" / "drift.json").write_text(json.dumps(out, ensure_ascii=False))
        print("まだデータがありません")
        return

    # 1) オッズの動き（確定 ÷ 締切前）。全組と、10〜60倍の組
    def move(a, b, band=None):
        r = []
        for x in rows:
            s, f = x[a], x[b]
            if s is None or f is None:
                continue
            m = np.isfinite(s) & np.isfinite(f) & (s > 0)
            if band:
                m &= (s > band[0]) & (s <= band[1])
            r.append(f[m] / s[m])
        if not r:
            return None
        v = np.concatenate(r)
        return {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                "abs_log_median": round(float(np.median(np.abs(np.log(v)))), 3)}
    out["move"] = {"t1_to_final": move(3, 4), "t1_to_final_10_60": move(3, 4, (10, 60)), "early_to_final": move(2, 4), "early_to_t1": move(2, 3)}

    # 2) 今の買い方を3通りのオッズで選んだときの成績（払戻は実際の払戻）
    cur = None
    for x in reversed(sorted(days)):
        pr = jload(ROOT / "data" / "preds" / x[:4] / f"{x}.json.gz")
        if pr and pr.get("rule"):
            cur = pr["rule"]; break
    cur = cur or dict(t=1.3, minp=0.07, maxo=60.0, mino=10.0, k=1)
    cur = {k: cur.get(k, 0.0) for k in ("t", "minp", "maxo", "mino", "k")}

    def score(rule, src, need_all=True):
        bets, ret, sq, hits, races = 0, 0.0, 0.0, 0, 0
        for x in rows:
            if need_all and any(x[i] is None for i in (3, 4)):
                continue
            o = x[src]
            if o is None:
                continue
            races += 1
            for j in pick(x[1], o, rule):
                bets += 1; v = x[5][j]; ret += v; sq += v * v; hits += v > 0
        roi = ret / bets if bets else 0.0
        se = (max(sq / bets - roi * roi, 0) / bets) ** 0.5 if bets else 0.0
        return {"races": races, "bets": bets, "hits": int(hits), "roi": round(roi, 4), "se": round(se, 4)}
    out["current_rule"] = cur
    out["current"] = {"early": score(cur, 2), "t1": score(cur, 3), "final": score(cur, 4)}

    # 3) 約5分前（通知する時点）のオッズで選ぶ買い方の総当たり（たまったら学習し直しで使う）
    grid = []
    for r in rule_list():
        s = score(r, 2, need_all=False)
        if s["bets"] >= 100:
            grid.append({"rule": r, **s, "lb": round(s["roi"] - s["se"], 4)})
    grid.sort(key=lambda g: -g["lb"])
    out["grid_top"] = grid[:20]
    best = grid[0] if grid else None
    out["adopt"] = bool(best and len(days) >= MIN_DAYS_ADOPT and best["bets"] >= MIN_BETS_ADOPT and best["lb"] > 1.0)
    out["best_early"] = best
    (ROOT / "docs" / "data" / "drift.json").write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("days", "races", "move", "current", "adopt")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
