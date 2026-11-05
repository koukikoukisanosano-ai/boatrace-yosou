"""仮想で機械的に買い続けた場合の損益（ページのグラフ用）: 採用した2連単の買い方で、開発・本番テスト期間の買い目と払戻を書き出す

  python tools/make_sim.py --work work/full
出力: docs/data/sim.json  {rule, dev, hold, bets: [[日付, 期間(0=開発 1=本番), オッズ, 100円あたりの払戻], …]}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research import load_odds  # noqa: E402
from research_eval import bet_probs, odds_matrix, payout_matrix  # noqa: E402
from research_exacta import geo, power  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "sim.json"))
    a = ap.parse_args()
    work = Path(a.work)
    R = json.loads((ROOT / "docs" / "data" / "research_exacta.json").read_text())
    best = R["best"]; rule = best["rule"]; mix = R["combos"][best["prob"]]
    P = pickle.load(open(work / "exacta_preds.pkl", "rb"))
    need = set(np.concatenate([P[s]["ids"] for s in P]))
    races = {}
    for ln in open(work / "races.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        if r["id"] in need:
            races[r["id"]] = r
    rows = []
    for si, s in enumerate(("dev", "hold")):
        b = P[s]; ids = b["ids"]
        eo = odds_matrix(ids, load_odds(sorted({f"{i[:4]}-{i[4:6]}-{i[6:8]}" for i in ids})), "exacta")
        pay = payout_matrix(ids, races, "exacta")
        e = bet_probs(geo([b[k] for k in mix], list(mix.values())))["exacta"]
        if best["gamma"] != 1:
            e = power(e, best["gamma"])
        o = np.where(np.isfinite(eo) & (eo > 0), eo, np.nan)
        ev = e * np.nan_to_num(o)
        inv = np.where(np.isfinite(o), 1 / o, 0); mkt = inv / np.maximum(inv.sum(1, keepdims=True), 1e-12)
        rank = np.argsort(np.argsort(-ev, 1), 1)
        m = (np.isfinite(eo).sum(1) >= 20)[:, None] & np.isfinite(o) & (ev >= rule["t"]) & (e >= rule["minp"]) & \
            (np.nan_to_num(o, nan=1e9) <= rule["maxo"]) & (rank < rule["k"]) & (e / np.maximum(mkt, 1e-12) >= rule.get("edge", 1))
        for r, c in zip(*np.nonzero(m)):
            rows.append([str(ids[r])[:8], si, round(float(o[r, c]), 1), round(float(pay[r, c]) / 100, 2), str(ids[r])])
    rows.sort(key=lambda x: x[4])
    out = {"rule": best["label"], "prob": best["prob"], "dev": R.get("dev"), "hold": R.get("hold"), "when": R.get("when"),
           "bets": [x[:4] for x in rows]}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    for si in (0, 1):
        z = [x for x in rows if x[1] == si]
        print(("開発", "本番")[si], len(z), "点 回収", round(sum(x[3] for x in z) / len(z), 3))


if __name__ == "__main__":
    main()
