"""予想方法の総当たり比較（採点の部）: research.py で作った各月の予想を、確定オッズと払戻で採点する

  python tools/research_eval.py --work work/full
出力: docs/data/research.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research import (EXA, QUI, TRI, TRIO, load_odds, normalize_place, pl_from_strength,  # noqa: E402
                      tri_harville, tri_iter)

ROOT = Path(__file__).resolve().parent.parent
PAIRS_ALL = list(itertools.combinations(range(6), 2))
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 券種ごとの確率（3連単から）
def bet_probs(tri: np.ndarray) -> dict:
    R = len(tri)
    out = {"trifecta": tri}
    exa = np.zeros((R, 30)); trio = np.zeros((R, 20)); qui = np.zeros((R, 15)); qp = np.zeros((R, 15))
    win = np.zeros((R, 6)); place = np.zeros((R, 6))
    for k, (a, b, c) in enumerate(TRI):
        p = tri[:, k]
        exa[:, EXA.index((a, b))] += p
        trio[:, TRIO.index(tuple(sorted((a, b, c))))] += p
        qui[:, QUI.index(tuple(sorted((a, b))))] += p
        for i, j in ((a, b), (a, c), (b, c)):
            qp[:, QUI.index(tuple(sorted((i, j))))] += p
        win[:, a] += p
        place[:, a] += p; place[:, b] += p
    out.update({"exacta": exa, "trio": trio, "quinella": qui, "quinella_place": qp, "win": win, "place": place})
    return out


KEYS = {
    "trifecta": [f"{a+1}-{b+1}-{c+1}" for a, b, c in TRI],
    "exacta": [f"{a+1}-{b+1}" for a, b in EXA],
    "quinella": [f"{a+1}={b+1}" for a, b in QUI],
    "quinella_place": [f"{a+1}={b+1}" for a, b in QUI],
    "trio": [f"{a+1}={b+1}={c+1}" for a, b, c in TRIO],
    "win": [str(a + 1) for a in range(6)],
    "place": [str(a + 1) for a in range(6)],
}
ODDS_KEY = {"trifecta": "t", "exacta": "e", "quinella": "q", "trio": "f", "win": "w", "place": "p", "quinella_place": "k"}
LABEL = {"trifecta": "3連単", "trio": "3連複", "exacta": "2連単", "quinella": "2連複", "quinella_place": "拡連複", "win": "単勝", "place": "複勝"}
BOATS = {
    "trifecta": [(a, b, c) for a, b, c in TRI], "exacta": [(a, b) for a, b in EXA], "quinella": [(a, b) for a, b in QUI],
    "quinella_place": [(a, b) for a, b in QUI], "trio": [t for t in TRIO], "win": [(a,) for a in range(6)], "place": [(a,) for a in range(6)],
}


def payout_matrix(ids, races, bt):
    """各レース×各組み合わせの払戻（100円あたり。外れは0、返還は100）"""
    keys = KEYS[bt]
    kidx = {k: i for i, k in enumerate(keys)}
    pay = np.zeros((len(ids), len(keys)))
    for r, rid in enumerate(ids):
        rc = races.get(rid) or {}
        for p in (rc.get("payouts") or {}).get(bt, []):
            i = kidx.get(p["combo"])
            if i is not None and p.get("amount"):
                pay[r, i] = max(pay[r, i], p["amount"])
        refunds = [int(x) - 1 for x in (rc.get("refunds") or [])]
        if refunds:
            for i, bs in enumerate(BOATS[bt]):
                if any(b in refunds for b in bs):
                    pay[r, i] = 100
    return pay


def odds_matrix(ids, odds, bt):
    k = ODDS_KEY[bt]
    n = len(KEYS[bt])
    a = np.full((len(ids), n), np.nan)
    for r, rid in enumerate(ids):
        o = odds.get(rid)
        if not o or k not in o:
            continue
        v = o[k]
        if bt in ("place", "quinella_place"):
            a[r] = [x[0] if x and x[0] is not None else np.nan for x in v]   # 下限で見る（控えめ）
        else:
            a[r] = [np.nan if x is None else x for x in v]
    return a


def market_from_tri_odds(t_odds, present):
    inv = np.where(np.isfinite(t_odds) & (t_odds > 0), 1 / t_odds, 0)
    for k, (a, b, c) in enumerate(TRI):
        inv[:, k] *= present[:, a] & present[:, b] & present[:, c]
    s = inv.sum(1, keepdims=True)
    return inv / np.maximum(s, 1e-12), s[:, 0] > 0


# ---------------------------------------------------------------- 確率の採点
def prob_metrics(tri, place, present):
    R = len(tri)
    order = np.argsort(np.where(present, place, 99), 1)[:, :3]
    ok = (np.take_along_axis(place, order, 1) == np.array([1, 2, 3])).all(1)
    tri_idx = {t: i for i, t in enumerate(TRI)}
    truth = np.array([tri_idx.get(tuple(o), -1) for o in order])
    ok &= truth >= 0
    bp = bet_probs(tri)
    winner = order[:, 0]
    rows = np.arange(R)
    res = {"races": int(ok.sum())}
    res["win_top1"] = float((np.argmax(bp["win"], 1) == winner)[ok].mean())
    res["win_logloss"] = float(-np.log(np.clip(bp["win"][rows, winner], 1e-9, 1))[ok].mean())
    res["tri_logloss"] = float(-np.log(np.clip(tri[rows, np.maximum(truth, 0)], 1e-12, 1))[ok].mean())
    rk = np.argsort(-tri, 1)
    res["tri_top1"] = float((rk[:, 0] == truth)[ok].mean())
    res["tri_top5"] = float((rk[:, :5] == truth[:, None]).any(1)[ok].mean())
    t_exa = np.array([EXA.index((o[0], o[1])) for o in order])
    t_trio = np.array([TRIO.index(tuple(sorted(o))) for o in order])
    t_qui = np.array([QUI.index(tuple(sorted(o[:2]))) for o in order])
    res["exa_top1"] = float((np.argmax(bp["exacta"], 1) == t_exa)[ok].mean())
    res["trio_top1"] = float((np.argmax(bp["trio"], 1) == t_trio)[ok].mean())
    res["qui_top1"] = float((np.argmax(bp["quinella"], 1) == t_qui)[ok].mean())
    return res


# ---------------------------------------------------------------- 買い方の採点
EV_T = [1.0, 1.1, 1.2, 1.35, 1.5, 2.0]
MINP = [0.0, 0.02, 0.05, 0.1]
MAXO = [20.0, 50.0, 150.0]
MAXK = [1, 2, 3, 5]


def strategies_for(prob, odds, pay, valid):
    """1つの確率×券種について、買い方のルールを全部試す。戻り値: [(rule, mask)]"""
    ev = prob * np.where(np.isfinite(odds), odds, 0)
    o = np.where(np.isfinite(odds), odds, np.inf)
    base = valid[:, None] & np.isfinite(odds) & (prob > 0)
    rank_p = np.argsort(np.argsort(-prob, 1), 1)
    rank_ev = np.argsort(np.argsort(-ev, 1), 1)
    rules = []
    nC = prob.shape[1]
    for k in [1, 2, 3, 5, 10]:
        if k > nC // 2 and k > 1:
            continue
        for minp in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if minp and k > 3:
                continue
            m = base & (rank_p < k)
            if minp:   # 自信のあるレースだけ（本命の確率が minp 以上）
                m &= (prob.max(1) >= minp)[:, None]
            rules.append(({"kind": "top", "k": k, "conf": minp}, m))
    for t in EV_T:
        for minp in MINP:
            for maxo in MAXO:
                for k in MAXK:
                    m = base & (ev >= t) & (prob >= minp) & (o <= maxo) & (rank_ev < k)
                    rules.append(({"kind": "ev", "t": t, "minp": minp, "maxo": maxo, "k": k}, m))
    return rules


def score(mask, pay):
    n = int(mask.sum())
    if not n:
        return {"bets": 0, "races": 0, "hit": 0.0, "race_hit": 0.0, "roi": 0.0, "se": 0.0, "lb": -1.0, "profit": 0}
    r = pay[mask]
    ret = float(r.sum())
    hits = int((r > 0).sum())
    races = int(mask.any(1).sum())
    race_hit = int(((pay * mask) > 0).any(1).sum())
    roi = ret / (n * 100)
    se = float(r.std() / np.sqrt(n) / 100)   # 回収率のぶれ（標準誤差）
    return {"bets": n, "races": races, "hit": hits / n, "race_hit": race_hit / max(races, 1), "roi": roi, "se": se,
            "lb": roi - se, "profit": int(ret - n * 100)}


def rule_label(bt, r):
    if r["kind"] == "top":
        s = f"{LABEL[bt]} 確率の上位{r['k']}点"
        if r["conf"]:
            s += f"（本命が{int(r['conf'] * 100)}%以上のレースだけ）"
        return s
    s = f"{LABEL[bt]} 期待値{r['t']}以上"
    if r["minp"]:
        s += f"・確率{int(r['minp'] * 100)}%以上"
    if r["maxo"] < 1e8:
        s += f"・{int(r['maxo'])}倍以下"
    s += f"・1レース{r['k']}点まで"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work" / "full"))
    ap.add_argument("--min-bets", type=int, default=1500, help="開発期間で最低これだけ買った方法だけを選ぶ")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "research.json"))
    a = ap.parse_args()
    work = Path(a.work)
    P = pickle.load(open(work / "research_preds.pkl", "rb"))
    races = {}
    with open(work / "races.jsonl", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            races[r["id"]] = r
    dev_end = P["dev_end"]
    periods = {"dev": [m for m in P["months"] if m <= dev_end], "hold": [m for m in P["months"] if m > dev_end]}
    log("開発期間", periods["dev"], "本番テスト期間", periods["hold"])

    # 月ごとに、方法ごとの3連単の確率を作る
    blocks = []
    for pm in P["preds"]:
        ids, present, place = pm["ids"], pm["present"], pm["place"]
        dates = [f"{i[:4]}-{i[4:6]}-{i[6:8]}" for i in ids]
        odds = load_odds(dates)
        t_odds = odds_matrix(ids, odds, "trifecta")
        mkt, has_mkt = market_from_tri_odds(t_odds, present)
        g = pm["gbm"]; p1, p2, p3 = normalize_place(g["win"], g["top2"], g["top3"], present)
        gr = pm["gbm_rw"]; r1, r2, r3 = normalize_place(gr["win"], gr["top2"], gr["top3"], present)
        cl_p1, cl_tri = pl_from_strength(pm["clogit_s"], present)
        methods = {
            "gbm": tri_iter(p1, p2, p3, present),
            "gbm_harville": tri_harville(p1, present),
            "gbm_rw": tri_iter(r1, r2, r3, present),
            "clogit": cl_tri,
        }
        methods["ens"] = (methods["gbm"] + methods["clogit"]) / 2
        methods["ens3"] = (methods["gbm"] + methods["gbm_rw"] + methods["clogit"]) / 3
        methods["market"] = np.where(has_mkt[:, None], mkt, methods["gbm"])
        for al in (0.3, 0.5, 0.7):
            b = np.where(has_mkt[:, None], (methods["gbm"] ** al) * (np.maximum(mkt, 1e-9) ** (1 - al)), methods["gbm"])
            methods[f"blend{int(al * 10)}"] = b / b.sum(1, keepdims=True)
        blocks.append({"month": pm["month"], "ids": ids, "present": present, "place": place, "methods": methods, "odds": odds, "has_mkt": has_mkt})
        log(f"{pm['month']}: {len(ids)} レース（オッズあり {int(has_mkt.sum())}）")

    names = list(blocks[0]["methods"].keys())
    LABELS_M = {
        "gbm": "機械学習（勾配ブースティング・今の方法）", "gbm_harville": "機械学習＋ハービル式（1着の確率だけから3連単）",
        "gbm_rw": "機械学習・新しい日ほど重く（半減期90日）", "clogit": "条件付きロジット（レース内で比べる線形モデル）",
        "ens": "機械学習＋条件付きロジットの平均", "ens3": "3つの平均（今の方法・新しい日重視・ロジット）",
        "market": "市場（確定オッズから逆算）※締切後にしか分からない", "blend3": "機械学習30%＋オッズ70%", "blend5": "機械学習50%＋オッズ50%", "blend7": "機械学習70%＋オッズ30%",
    }

    # 1) 確率の方法の比較
    prob_table = []
    for nm in names:
        row = {"method": nm, "label": LABELS_M.get(nm, nm)}
        for per, months in periods.items():
            bs = [b for b in blocks if b["month"] in months]
            tri = np.concatenate([b["methods"][nm] for b in bs]); pl = np.concatenate([b["place"] for b in bs]); pr = np.concatenate([b["present"] for b in bs])
            for k, v in prob_metrics(tri, pl, pr).items():
                row[f"{per}_{k}"] = v
        prob_table.append(row)
        log(f"確率 {nm}: 開発 1着{row['dev_win_top1']:.3f} 3連単logloss {row['dev_tri_logloss']:.4f} / 本番 1着{row['hold_win_top1']:.3f} 3連単logloss {row['hold_tri_logloss']:.4f}")

    # 2) 買い方の比較（確定オッズで期待値を計算し、実際の払戻で採点）
    bet_types = ["win", "place", "exacta", "quinella", "quinella_place", "trio", "trifecta"]
    strat_rows = []
    for bt in bet_types:
        per_data = {}
        for per, months in periods.items():
            bs = [b for b in blocks if b["month"] in months]
            ids = np.concatenate([b["ids"] for b in bs])
            pay = np.concatenate([payout_matrix(b["ids"], races, bt) for b in bs])
            odds = np.concatenate([odds_matrix(b["ids"], b["odds"], bt) for b in bs])
            valid = np.concatenate([b["has_mkt"] for b in bs])
            month_of = np.concatenate([[b["month"]] * len(b["ids"]) for b in bs])
            per_data[per] = (ids, pay, odds, valid, month_of, bs)
        # 選ぶときに使うオッズ: 3連単はそのまま。ほかの券種は 3連単の市場の確率から出した「つり合いの倍率」
        # （単勝・複勝などは売り場が小さく、締切直前の大口で確定オッズが大きく歪むことがあり、
        #   それを見て選ぶと「後から分かる情報」を使うことになるため）
        sel_odds = {}
        for per in periods:
            if bt == "trifecta":
                sel_odds[per] = per_data[per][2]
            else:
                mk = np.concatenate([bet_probs(b["methods"]["market"])[bt] for b in per_data[per][5]])
                sel_odds[per] = np.where(mk > 0, 0.75 / np.maximum(mk, 1e-12), np.nan)
        for nm in names:
            if nm == "market":
                continue
            probs = {per: np.concatenate([bet_probs(b["methods"][nm])[bt] for b in per_data[per][5]]) for per in periods}
            dev_rules = strategies_for(probs["dev"], sel_odds["dev"], per_data["dev"][1], per_data["dev"][3])
            hold_rules = strategies_for(probs["hold"], sel_odds["hold"], per_data["hold"][1], per_data["hold"][3])
            for (rule, md), (_, mh) in zip(dev_rules, hold_rules):
                sd = score(md, per_data["dev"][1])
                if sd["bets"] < 50:
                    continue
                sh = score(mh, per_data["hold"][1])
                # 月ごとの回収率（安定しているか）
                by_m = {}
                for per in periods:
                    mm = md if per == "dev" else mh
                    pay, mo = per_data[per][1], per_data[per][4]
                    for m in sorted(set(mo)):
                        sel = mo == m
                        s = score(mm[sel], pay[sel])
                        if s["bets"]:
                            by_m[m] = round(s["roi"], 3)
                strat_rows.append({"bet": bt, "method": nm, "rule": rule, "label": rule_label(bt, rule),
                                   "dev": sd, "hold": sh, "months": by_m})
        log(f"買い方 {LABEL[bt]}: これまで {len(strat_rows)} 通り")

    # 3) 選ぶ: 開発期間の成績だけで決める
    def pick(rows, key, min_bets):
        cand = [r for r in rows if r["dev"]["bets"] >= min_bets]
        return max(cand, key=key) if cand else None

    # 回収率は「ぶれを引いた値」（回収率−標準誤差）で選ぶ（たまたま大穴が当たっただけの方法を選ばないため）
    best_roi = pick(strat_rows, lambda r: (r["dev"]["lb"], r["dev"]["profit"]), a.min_bets)
    # 券種ごとの最良（回収率）
    by_type = {}
    for bt in bet_types:
        rows = [r for r in strat_rows if r["bet"] == bt]
        b = pick(rows, lambda r: (r["dev"]["lb"], r["dev"]["profit"]), a.min_bets)
        if b:
            by_type[bt] = b
    # 的中重視: 1レース1点で、開発期間の的中率が高く回収率も悪すぎないもの
    hit_rows = [r for r in strat_rows if r["rule"]["kind"] == "top" and r["rule"]["k"] == 1 and r["rule"]["conf"] == 0]
    best_hit = {bt: max([r for r in hit_rows if r["bet"] == bt], key=lambda r: r["dev"]["hit"], default=None) for bt in bet_types}
    chosen_prob = min([r for r in prob_table if r["method"] != "market" and not r["method"].startswith("blend")], key=lambda r: r["dev_tri_logloss"])

    # ページの「おすすめの買い方」: 採用した確率（3つの平均）で、オッズを使わない買い方（確率の上位k点・自信のあるレースだけ）
    # 券種ごとに開発期間の「回収率−ぶれ」が一番良いもの。本番テスト期間の成績を添える
    TITLE = {"place": "複勝（当てやすさ重視）", "quinella_place": "拡連複", "quinella": "2連複", "exacta": "2連単", "trio": "3連複", "trifecta": "3連単（高配当）", "win": "単勝"}
    recommend = []
    for bt in ["place", "quinella_place", "quinella", "exacta", "trio", "trifecta", "win"]:
        rows = [r for r in strat_rows if r["bet"] == bt and r["method"] == chosen_prob["method"] and r["rule"]["kind"] == "top"]
        b = pick(rows, lambda r: (r["dev"]["lb"], r["dev"]["profit"]), a.min_bets)
        if b:
            recommend.append({**b, "title": TITLE[bt]})
    recommend.sort(key=lambda r: -r["dev"]["roi"])
    hold_days = len({rid[:8] for b in blocks if b["month"] in periods["hold"] for rid in b["ids"]})
    dev_days = len({rid[:8] for b in blocks if b["month"] in periods["dev"] for rid in b["ids"]})
    for r in by_type.values():
        r["method_label"] = LABELS_M.get(r["method"], r["method"])

    # 上位だけを残す（ファイルを小さく）
    top_rows = sorted([r for r in strat_rows if r["dev"]["bets"] >= a.min_bets], key=lambda r: -r["dev"]["lb"])[:60]
    # 方法（確率の出し方）ごとの最良と、その本番テスト期間の成績（方法の良し悪しを比べるため）
    by_method = {}
    for nm in names:
        b = pick([r for r in strat_rows if r["method"] == nm], lambda r: (r["dev"]["lb"], r["dev"]["profit"]), a.min_bets)
        if b:
            by_method[nm] = b
    out = {
        "when": time.strftime("%Y-%m-%d"),
        "months": P["months"], "dev": periods["dev"], "hold": periods["hold"],
        "rule": "各月のレースは、その月より前のデータだけで学習したモデルで予想（毎月学び直し）。開発期間で方法を選び、選ぶのに使っていない本番テスト期間で成績を確かめた。期待値は3連単の確定オッズ（3連単以外は3連単の市場から計算した倍率）で計算し、実際の払戻で採点。単勝・複勝などの確定オッズは締切直前の大口で大きく歪むことがあり、それを見て選ぶと実際には買えない成績になるため使っていない。実際の購入時は締切前のオッズで判断するので、成績は少し下がる見込み。",
        "prob_table": prob_table, "chosen_prob": chosen_prob["method"],
        "best_roi": best_roi, "best_by_type_top": by_type, "best_by_method": by_method, "best_hit": best_hit, "top": top_rows,
        "recommend": recommend[:4], "recommend_all": recommend, "hold_days": hold_days, "dev_days": dev_days,
        "n_strategies": len(strat_rows), "min_bets": a.min_bets,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, default=float))
    log("書き出し", a.out, f"（{len(strat_rows)} 通りを比較）")
    log("確率の方法で選ばれたもの:", chosen_prob["method"])
    if best_roi:
        log("回収率が一番（開発期間）:", best_roi["method"], best_roi["label"], "開発", best_roi["dev"], "本番", best_roi["hold"])
    for bt, r in by_type.items():
        log(f"  {LABEL[bt]}: {r['method']} {r['label']} 開発 回収{r['dev']['roi']:.3f}±{r['dev']['se']:.3f}({r['dev']['bets']}点) → 本番 回収{r['hold']['roi']:.3f}±{r['hold']['se']:.3f}({r['hold']['bets']}点) 的中{r['hold']['hit']:.3f}")
    log("方法ごとの最良:")
    for nm, r in by_method.items():
        log(f"  {nm}: {r['label']} 開発 回収{r['dev']['roi']:.3f}±{r['dev']['se']:.3f}({r['dev']['bets']}点) → 本番 回収{r['hold']['roi']:.3f}±{r['hold']['se']:.3f}({r['hold']['bets']}点)")


if __name__ == "__main__":
    main()
