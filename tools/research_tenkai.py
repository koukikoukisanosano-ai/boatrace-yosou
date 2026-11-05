"""展開モデル（＋展示重視）の月ごとの予想し直しと、期待値の買い方（3連単・2連単）の比較

  python tools/research_tenkai.py --work work/full --exhibit-w 0.5
出力: work/full/tenkai_preds.pkl と docs/data/research_ev.json
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
import train as T  # noqa: E402
from research import EXA, TRI, load_odds, normalize_place, tri_iter  # noqa: E402
from research_eval import bet_probs, market_from_tri_odds, odds_matrix, payout_matrix, prob_metrics, score  # noqa: E402
from tenkai import COURSE, EXHIBIT, clf, load, pair_feats, pair_rows, race_tensor, scenario_probs, tri_tenkai, winner_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def geo(t1, t2, w):
    t = np.exp((1 - w) * np.log(np.maximum(t1, 1e-15)) + w * np.log(np.maximum(t2, 1e-15)))
    return t / t.sum(1, keepdims=True)


def predict_months(df, feats, months):
    out = []
    cef = [f for f in COURSE + EXHIBIT if f in feats]
    for m in months:
        tr = df[df.date < f"{m}-01"]
        te = df[df.date.str.startswith(m)]
        vc = T.venue_course_rates(tr)
        tr, te = T.fill_vc(tr, vc), T.fill_vc(te, vc)
        Sm = clf().fit(tr[feats].to_numpy(float), winner_class(tr))
        r2 = pair_rows(tr, feats, 2); N2 = clf().fit(r2[pair_feats(feats, 2)].to_numpy(float), r2.y)
        r3 = pair_rows(tr, feats, 3); N3 = clf().fit(r3[pair_feats(feats, 3)].to_numpy(float), r3.y)
        CE = {k: clf().fit(tr[cef].to_numpy(float), (tr.place <= n).astype(int)) for k, n in (("win", 1), ("top2", 2), ("top3", 3))}
        ids, present, place, X = race_tensor(te, feats)
        R = len(ids)
        S = scenario_probs(Sm, X, present)
        tm, _, _ = tri_tenkai(S, N2, N3, X, present, feats)
        Xc = X[:, :, [feats.index(f) for f in cef]].reshape(R * 6, -1)
        p = {k: c.predict_proba(Xc)[:, 1].reshape(R, 6) for k, c in CE.items()}
        tc = tri_iter(*normalize_place(p["win"], p["top2"], p["top3"], present), present)
        out.append({"month": m, "ids": ids, "present": present, "place": place, "tm": tm, "tc": tc})
        log(f"{m}: {R} レース")
    return out


EV_T = [1.0, 1.1, 1.2, 1.35, 1.5, 2.0]
MINP = [0.0, 0.01, 0.02, 0.05]
MAXO = [30.0, 60.0, 150.0]
MAXK = [1, 2, 3, 5]


def rules(prob, odds, valid):
    ev = prob * np.where(np.isfinite(odds), odds, 0)
    o = np.where(np.isfinite(odds), odds, np.inf)
    base = valid[:, None] & np.isfinite(odds) & (prob > 0)
    rank_ev = np.argsort(np.argsort(-ev, 1), 1)
    for t in EV_T:
        for mp in MINP:
            for mo in MAXO:
                for k in MAXK:
                    yield {"t": t, "minp": mp, "maxo": mo, "k": k}, base & (ev >= t) & (prob >= mp) & (o <= mo) & (rank_ev < k)


def label(bt, r):
    s = f"{'3連単' if bt == 'trifecta' else '2連単'} 期待値{r['t']}以上"
    if r["minp"]:
        s += f"・確率{r['minp'] * 100:g}%以上"
    return s + f"・{int(r['maxo'])}倍以下・1レース{r['k']}点まで"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--exhibit-ws", default="0,0.85", help="展示航走の重み（複数）")
    ap.add_argument("--alpha", default="0.3,0.5,0.7,1.0", help="予想の割合（残りは締切前オッズ）")
    ap.add_argument("--min-bets", type=int, default=3000)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "research_ev.json"))
    a = ap.parse_args()
    work = Path(a.work)
    pk = work / "tenkai_preds.pkl"
    df, feats = load(work)
    months_all = sorted({d[:7] for d in df.date})
    if sum(df.date.str.startswith(months_all[-1]).groupby(df.date).any()) < 20:
        months_all = months_all[:-1]          # 途中の月は使わない
    months = months_all[-8:]
    dev, hold = months[:4], months[4:]
    if a.reuse and pk.exists():
        blocks = pickle.load(open(pk, "rb"))
    else:
        blocks = predict_months(df, feats, months)
        pickle.dump(blocks, open(pk, "wb"))
    races = {json.loads(ln)["id"]: json.loads(ln) for ln in open(work / "races.jsonl", encoding="utf-8")}
    log("開発", dev, "本番テスト", hold)

    per = {}
    for name, ms in (("dev", dev), ("hold", hold)):
        bs = [b for b in blocks if b["month"] in ms]
        ids = np.concatenate([b["ids"] for b in bs])
        present = np.concatenate([b["present"] for b in bs]); place = np.concatenate([b["place"] for b in bs])
        tm = np.concatenate([b["tm"] for b in bs]); tc = np.concatenate([b["tc"] for b in bs])
        odds = load_odds([f"{i[:4]}-{i[4:6]}-{i[6:8]}" for i in ids])
        t_odds = odds_matrix(ids, odds, "trifecta"); e_odds = odds_matrix(ids, odds, "exacta")
        mkt, has = market_from_tri_odds(t_odds, present)
        per[name] = dict(ids=ids, present=present, place=place, tm=tm, tc=tc, t_odds=t_odds, e_odds=e_odds, mkt=mkt, has=has,
                         pay_t=payout_matrix(ids, races, "trifecta"), pay_e=payout_matrix(ids, races, "exacta"))

    # 確率の比較（展示の重み）
    prob_rows = []
    WS = [float(x) for x in a.exhibit_ws.split(",")]
    for w in WS:
        row = {"exhibit_w": w}
        for name in per:
            P = per[name]
            m = prob_metrics(geo(P["tm"], P["tc"], w), P["place"], P["present"])
            row.update({f"{name}_{k}": round(v, 4) for k, v in m.items()})
        prob_rows.append(row)
        log("展示の重み", w, {k: v for k, v in row.items() if k.startswith("hold")})

    # 期待値の買い方（3連単は3連単の確定オッズ、2連単は 2連単の確定オッズ と 3連単から計算した倍率 の両方で試す）
    rows = []
    for ew, al in itertools.product(WS, [float(x) for x in a.alpha.split(",")]):
        prob = {}
        for name, P in per.items():
            base = geo(P["tm"], P["tc"], ew)
            b = np.where(P["has"][:, None], (base ** al) * (np.maximum(P["mkt"], 1e-12) ** (1 - al)), base)
            prob[name] = b / b.sum(1, keepdims=True)
        for bt in ("trifecta", "exacta"):
            for src in (("own",) if bt == "trifecta" else ("own", "tri")):
                pr = {n: (prob[n] if bt == "trifecta" else bet_probs(prob[n])["exacta"]) for n in per}
                od = {}
                for n, P in per.items():
                    if bt == "trifecta":
                        od[n] = P["t_odds"]
                    elif src == "own":
                        od[n] = P["e_odds"]
                    else:
                        mk = bet_probs(P["mkt"])["exacta"]
                        od[n] = np.where(mk > 0, 0.75 / np.maximum(mk, 1e-12), np.nan)
                pay = {n: (P["pay_t"] if bt == "trifecta" else P["pay_e"]) for n, P in per.items()}
                rd = rules(pr["dev"], od["dev"], per["dev"]["has"])
                rh = rules(pr["hold"], od["hold"], per["hold"]["has"])
                for (rule, md), (_, mh) in zip(rd, rh):
                    sd = score(md, pay["dev"])
                    if sd["bets"] < a.min_bets:
                        continue
                    sh = score(mh, pay["hold"])
                    rows.append({"bet": bt, "odds_for_select": src, "exhibit_w": ew, "alpha": al, "rule": rule, "label": label(bt, rule), "dev": sd, "hold": sh})
        log(f"展示の重み {ew} 予想の割合 {al}: {len(rows)} 通り")

    def best(bt, src=None, ew=None):
        c = [r for r in rows if r["bet"] == bt and (src is None or r["odds_for_select"] == src) and (ew is None or r["exhibit_w"] == ew)]
        return max(c, key=lambda r: (r["dev"]["lb"], r["dev"]["profit"])) if c else None

    sweep_p = work / "exhibit_weight.json"
    out = {"when": time.strftime("%Y-%m-%d"), "dev": dev, "hold": hold,
           "hold_days": int(df[df.date.str[:7].isin(hold)].date.nunique()),
           "exhibit_sweep": json.loads(sweep_p.read_text()) if sweep_p.exists() else [], "exhibit_ws": WS, "prob": prob_rows,
           "best_trifecta": best("trifecta"), "best_exacta": best("exacta", "own"), "best_exacta_tri": best("exacta", "tri"),
           "by_w": {str(w): {"trifecta": best("trifecta", None, w), "exacta": best("exacta", "own", w), "exacta_tri": best("exacta", "tri", w)} for w in WS},
           "top": sorted(rows, key=lambda r: -r["dev"]["lb"])[:40], "n": len(rows)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, default=float))
    (work / "ev_rows.json").write_text(json.dumps(rows, ensure_ascii=False, default=float))
    for k in ("best_trifecta", "best_exacta", "best_exacta_tri"):
        r = out[k]
        if r:
            log(k, "展示", r["exhibit_w"], "予想", r["alpha"], r["label"], "開発", round(r["dev"]["roi"], 3), "±", round(r["dev"]["se"], 3), r["dev"]["bets"],
                "→ 本番", round(r["hold"]["roi"], 3), "±", round(r["hold"]["se"], 3), r["hold"]["bets"], "的中", round(r["hold"]["hit"], 3))


if __name__ == "__main__":
    main()
