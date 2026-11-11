"""公平な検証（その2）: 月ごとに「その月より前の成績だけで予想の混ぜ方と買い方を選び → その月を答え合わせ」を繰り返す

  python tools/fair_eval.py --preds fair/ --work work/full
入力: fair_fit.py が作った fair_YYYY-MM.pkl（各月を、その月より前だけで学習したモデルで予想したもの）
出力: docs/data/fair.json
決め事は全部その月より前のデータだけで行う（モデル・予想の混ぜ方・買い方・10倍以下を買うかどうか）
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research import load_odds  # noqa: E402
from research_eval import bet_probs, odds_matrix, payout_matrix  # noqa: E402
from research_exacta import geo, log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMBOS = {
    "展開モデル（大きめ）": {"tenkai_big": 1.0},
    "展開モデル（標準）": {"tenkai": 1.0},
    "機械学習": {"gbm": 1.0},
    "展開（標準＋大きめ）": {"tenkai": 0.5, "tenkai_big": 0.5},
    "展開（大きめ）＋展示0.15": {"tenkai_big": 0.85, "ce": 0.15},
    "展開（大きめ）＋機械学習0.25": {"tenkai_big": 0.75, "gbm": 0.25},
    "展開＋機械学習": {"tenkai_big": 0.5, "gbm": 0.5},
}
# 2026-11 に試して採用しなかった候補（公平な検証で 94〜102%。候補を増やすと選び方の当てずっぽうが増え、合計は 100.5% → 97.8% に下がった）:
#   展開モデル（特大: 1000本・学習率0.03）、新しい日重視（半減期180日）、それらと展示・大きめの組み合わせ
EV_T = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
MINP = [0.0, 0.02, 0.04, 0.07, 0.1]
MAXO = [15.0, 30.0, 60.0, 100.0]
MINO = [0.0, 10.0]
MAXK = [1, 2, 3]
EDGE = [1.0, 1.2]


def rule_list():
    return [dict(t=t, minp=mp, maxo=mo, mino=mi, k=k, edge=ed) for t, mp, mo, mi, k, ed in itertools.product(EV_T, MINP, MAXO, MINO, MAXK, EDGE)
            if not (ed > 1 and t > 1.2) and mi < mo]


def masks(prob, odds, valid, rules):
    o = np.where(np.isfinite(odds) & (odds > 0), odds, np.nan)
    ev = prob * np.nan_to_num(o, nan=0)
    inv = np.where(np.isfinite(o), 1 / o, 0)
    mkt = inv / np.maximum(inv.sum(1, keepdims=True), 1e-12)
    edge = prob / np.maximum(mkt, 1e-12)
    base = valid[:, None] & np.isfinite(o)
    rank = np.argsort(np.argsort(-ev, 1), 1)
    on = np.nan_to_num(o, nan=0)
    for r in rules:
        yield base & (ev >= r["t"]) & (prob >= r["minp"]) & (on <= r["maxo"]) & (on > r["mino"]) & (rank < r["k"]) & (edge >= r["edge"])


def label(r):
    s = f"期待値{r['t']}以上"
    if r["minp"]:
        s += f"・確率{r['minp'] * 100:g}%以上"
    if r["edge"] > 1:
        s += f"・市場の{r['edge']}倍以上"
    s += f"・{r['mino']:g}倍超〜{r['maxo']:g}倍" if r["mino"] else f"・{r['maxo']:g}倍以下"
    return s + f"・{r['k']}点まで"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="fair_YYYY-MM.pkl のあるフォルダ")
    ap.add_argument("--work", required=True)
    ap.add_argument("--first-test", default="2026-03", help="答え合わせを始める月（それより前は選ぶためだけに使う）")
    ap.add_argument("--min-bets-per-month", type=int, default=250)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "fair.json"))
    a = ap.parse_args()
    files = sorted(Path(a.preds).rglob("fair_*.pkl"))
    M = [pickle.load(open(f, "rb")) for f in files]
    M.sort(key=lambda b: b["month"])
    months = [b["month"] for b in M]
    log("月:", months)
    need = set(np.concatenate([b["ids"] for b in M]))
    races = {}
    for ln in open(Path(a.work) / "races.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        if r["id"] in need:
            races[r["id"]] = r
    combos = {cn: mix for cn, mix in COMBOS.items() if all(k in M[0] for k in mix)}
    # 試しに作った予想（--variants）があれば、それ単体と「＋展示0.15」も候補に入れる（比べるため）
    base_keys = {"tenkai", "tenkai_big", "ce", "gbm"}
    extra = [k for k in M[0] if k not in base_keys | {"month", "ids", "present", "place"} and all(k in b for b in M)]
    for k in extra:
        combos[f"[試] {k}"] = {k: 1.0}
        combos[f"[試] {k}＋展示0.15"] = {k: 0.85, "ce": 0.15}
        combos[f"[試] {k}＋展開（大きめ）"] = {k: 0.5, "tenkai_big": 0.5}

    rules = rule_list()
    cands = [(cn, i) for cn in combos for i in range(len(rules))]
    is_base = np.array([not cn.startswith("[試]") for cn, _ in cands])   # 試しの予想は比べるだけ（実際の買い方を選ぶ候補には入れない）
    stats = np.zeros((len(cands), len(M), 3))        # 点数・払戻合計・払戻の2乗合計（100円あたり）
    D = []
    for mi, b in enumerate(M):
        ids = b["ids"]
        eo = odds_matrix(ids, load_odds(sorted({f"{i[:4]}-{i[4:6]}-{i[6:8]}" for i in ids})), "exacta")
        pay = payout_matrix(ids, races, "exacta") / 100
        has = np.isfinite(eo).sum(1) >= 20
        probs = {cn: bet_probs(geo([b[k].astype(float) for k in mix], list(mix.values())))["exacta"] for cn, mix in combos.items()}
        D.append({"eo": eo, "pay": pay, "has": has, "probs": probs, "days": np.array([str(i)[:8] for i in ids])})
        ci = 0
        for cn in combos:
            for m in masks(probs[cn], eo, has, rules):
                x = pay[m]
                stats[ci, mi] = (m.sum(), x.sum(), (x ** 2).sum())
                ci += 1
        log(f"{b['month']}: {len(ids)} レース・オッズあり {has.sum()}")

    results, bets_all = [], []
    for mi, m in enumerate(months):
        if m < a.first_test:
            continue
        prior = slice(0, mi)
        n, s, s2 = stats[:, prior, 0].sum(1), stats[:, prior, 1].sum(1), stats[:, prior, 2].sum(1)
        roi = np.where(n > 0, s / np.maximum(n, 1), 0)
        se = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - roi ** 2, 0) / np.maximum(n, 1))
        lb = np.where((n >= a.min_bets_per_month * mi) & is_base, roi - se, -9)
        best = int(np.argmax(lb))
        cn, ri = cands[best]
        rule = rules[ri]
        d = D[mi]
        mask = next(masks(d["probs"][cn], d["eo"], d["has"], [rule]))
        x = d["pay"][mask]
        bets_all += list(zip(d["days"][np.nonzero(mask)[0]], x))
        results.append({"month": m, "chosen": {"prob": cn, "rule": rule, "label": label(rule)},
                        "prior": {"bets": int(n[best]), "roi": round(float(roi[best]), 4), "se": round(float(se[best]), 4)},
                        "test": {"bets": int(mask.sum()), "hits": int((x > 0).sum()), "roi": round(float(x.mean()), 4) if len(x) else None}})
        log(f"{m}: 選んだ {cn}・{label(rule)}（それまで {roi[best]:.3f}±{se[best]:.3f} {int(n[best])}点）→ この月 {x.mean() if len(x) else float('nan'):.3f}（{len(x)}点）")
    x = np.array([v for _, v in bets_all])
    days = np.array([d for d, _ in bets_all])
    rng = np.random.default_rng(0)
    ud = np.unique(days); bd = {dd: x[days == dd] for dd in ud}
    bs = np.array([np.concatenate([bd[dd] for dd in rng.choice(ud, len(ud))]).mean() for _ in range(5000)])
    total = {"bets": int(len(x)), "hits": int((x > 0).sum()), "roi": round(float(x.mean()), 4), "se": round(float(x.std() / np.sqrt(len(x))), 4),
             "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "p_below_100": round(float((bs < 1).mean()), 4),
             "days": int(len(ud))}
    log("公平な検証の合計:", json.dumps(total, ensure_ascii=False))
    # 参考: 今の買い方（後から選んだもの）をそのまま全部の月に当てはめた場合
    fixed = {"prob": "展開（大きめ）＋展示0.15", "rule": dict(t=1.3, minp=0.07, maxo=60.0, mino=10.0, k=1, edge=1.0)}
    ref = []
    for mi, m in enumerate(months):
        d = D[mi]
        mask = next(masks(d["probs"][fixed["prob"]], d["eo"], d["has"], [fixed["rule"]]))
        xx = d["pay"][mask]
        ref.append({"month": m, "bets": int(mask.sum()), "roi": round(float(xx.mean()), 4) if len(xx) else None})
    log("参考（今の買い方を全部の月に）:", ref)
    for fr in (dict(t=1.5, minp=0.07, maxo=30.0, mino=10.0, k=1, edge=1.0), dict(t=1.75, minp=0.04, maxo=60.0, mino=10.0, k=1, edge=1.0),
               dict(t=1.3, minp=0.07, maxo=60.0, mino=0.0, k=1, edge=1.0)):
        tb, tr_ = 0, 0.0; mon = []
        for mi, m in enumerate(months):
            d = D[mi]
            mask = next(masks(d["probs"][fixed["prob"]], d["eo"], d["has"], [fr]))
            xx = d["pay"][mask]; tb += len(xx); tr_ += xx.sum(); mon.append(round(float(xx.mean()), 2) if len(xx) else None)
        log(f"参考 {label(fr)}: 合計 {tr_ / max(tb, 1):.3f}（{tb}点） 月別 {mon}")
    # 次の月に使う買い方: 全部の月の成績で選ぶ（公平な検証と同じ選び方）
    n, s_, s2 = stats[:, :, 0].sum(1), stats[:, :, 1].sum(1), stats[:, :, 2].sum(1)
    roi = np.where(n > 0, s_ / np.maximum(n, 1), 0)
    se = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - roi ** 2, 0) / np.maximum(n, 1))
    lb = np.where((n >= a.min_bets_per_month * len(M)) & is_base, roi - se, -9)
    bi = int(np.argmax(lb)); cn, ri = cands[bi]
    nxt = {"prob": cn, "mix": combos[cn], "rule": rules[ri], "label": label(rules[ri]), "bets": int(n[bi]), "roi": round(float(roi[bi]), 4), "se": round(float(se[bi]), 4)}
    log("次の月に使う買い方:", json.dumps(nxt, ensure_ascii=False))
    # 予想の混ぜ方ごとの公平な成績（その月より前で買い方を選んだ場合）
    by_prob = {}
    for cn in combos:
        idx = [i for i, c in enumerate(cands) if c[0] == cn]
        xs = []
        for mi, m in enumerate(months):
            if m < a.first_test:
                continue
            nn, ss, ss2 = stats[idx, :mi, 0].sum(1), stats[idx, :mi, 1].sum(1), stats[idx, :mi, 2].sum(1)
            rr = np.where(nn > 0, ss / np.maximum(nn, 1), 0)
            ee = np.sqrt(np.maximum(ss2 / np.maximum(nn, 1) - rr ** 2, 0) / np.maximum(nn, 1))
            j = idx[int(np.argmax(np.where(nn >= a.min_bets_per_month * mi, rr - ee, -9)))]
            xs.append(stats[j, mi])
        xs = np.array(xs); tb = xs[:, 0].sum()
        lls, gap1, gapo = [], [], []
        for mi, m in enumerate(months):
            if m < a.first_test:
                continue
            b = M[mi]; e = D[mi]["probs"][cn]; eo = D[mi]["eo"]
            order = np.argsort(np.where(b["present"], b["place"], 99), 1)[:, :2]
            okk = (np.take_along_axis(b["place"], order, 1) == [1, 2]).all(1)
            tr = order[:, 0] * 5 + order[:, 1] - (order[:, 1] > order[:, 0])
            ll = -np.log(np.clip(e[np.arange(len(e)), tr], 1e-12, 1))
            lls.append(ll[okk])
            # 市場（確定オッズ）との差を、1号艇が勝ったレースとそれ以外で（正 = 市場のほうが正確）
            inv = np.where(np.isfinite(eo) & (eo > 0), 1 / eo, 0); mk = inv / np.maximum(inv.sum(1, keepdims=True), 1e-12)
            llm = -np.log(np.clip(mk[np.arange(len(mk)), tr], 1e-12, 1))
            okm = okk & (np.isfinite(eo).sum(1) >= 25)
            gap1.append((ll - llm)[okm & (order[:, 0] == 0)]); gapo.append((ll - llm)[okm & (order[:, 0] != 0)])
        by_prob[cn] = {"bets": int(tb), "roi": round(float(xs[:, 1].sum() / max(tb, 1)), 4), "logloss": round(float(np.concatenate(lls).mean()), 4),
                       "gap_win1": round(float(np.concatenate(gap1).mean()), 4), "gap_win_other": round(float(np.concatenate(gapo).mean()), 4)}
    log("混ぜ方ごと（公平）:", json.dumps(by_prob, ensure_ascii=False))
    out = {"next": nxt, "by_prob": by_prob, "months": results, "total": total, "reference_current_rule": ref, "first_test": a.first_test,
           "daily": [[dd, int(len(bd[dd])), round(float(bd[dd].sum()), 2)] for dd in ud]}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
