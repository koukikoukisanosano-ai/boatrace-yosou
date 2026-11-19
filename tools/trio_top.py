"""確かめる: 予想の確率が一番高い3連複（上位3艇の組）を、期待値を見ずに買ったらどうなるか。
公平な検証の月ごとの予想（その月より前だけで学習）と、確定オッズ・実際の払い戻しで。
  python tools/trio_top.py <月ごとの予想のフォルダ>
"""
import gzip, itertools, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eval import bet_probs  # noqa: E402
from research_exacta import geo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRIO = list(itertools.combinations(range(1, 7), 3))
MIX = {"tenkai_big": 0.5, "gbm": 0.5}
rows = []
for f in sorted(Path(sys.argv[1]).rglob("fair_*.pkl")):
    b = pickle.load(open(f, "rb"))
    if b["month"] < "2026-01":
        continue
    pr = bet_probs(geo([b[k].astype(float) for k in MIX], list(MIX.values())))["trio"]
    cache = {}
    for i, rid in enumerate(b["ids"]):
        d, j, r = str(rid).split("-")
        if d not in cache:
            po = ROOT / "data" / "odds" / d[:4] / f"{d}.json.gz"
            pa = ROOT / "data" / "openapi" / d[:4] / f"{d}.json.gz"
            cache[d] = (json.loads(gzip.decompress(po.read_bytes())) if po.exists() else {},
                        (json.loads(gzip.decompress(pa.read_bytes())).get("programs") or {}).get("stadiums") or {} if pa.exists() else {})
        od, st = cache[d]
        fo = (od.get(f"{j}-{r}") or {}).get("f")
        rc = ((st.get(j) or {}).get("races") or {}).get(r) or {}
        pays = ((rc.get("result") or {}).get("payouts") or {}).get("trio") or []
        if not fo or not pays:
            continue
        pm = {p["combination"]: p["amount"] / 100 for p in pays}
        p = pr[i]
        order = np.argsort(-p)
        for rank, k in enumerate(order[:3]):
            a, bb, c = TRIO[k]
            o = fo[k]
            if not o:
                continue
            rows.append((b["month"], rank + 1, p[k], o, pm.get(f"{a}={bb}={c}", 0), p[k] * o))
A = np.array([r[1:] for r in rows], dtype=float)
rank, p, o, y, ev = A.T
print("RESULT 2026-01〜10 のレース数", int((rank == 1).sum()))
for k in (1, 2, 3):
    m = rank == k
    print(f"RESULT 確率{k}番目の3連複: {m.sum()}点 的中率{(y[m] > 0).mean() * 100:.1f}%（予想{p[m].mean() * 100:.1f}%） 平均オッズ{np.median(o[m]):.1f}倍 回収率{y[m].mean() * 100:.1f}%")
m = rank <= 3
print(f"RESULT 上位3つ全部: {m.sum()}点 回収率{y[m].mean() * 100:.1f}%  1レースどれか当たる率{(y[m].reshape(-1) > 0).sum() / (rank == 1).sum() * 100:.1f}%")
for lo, hi in ((0.5, 1), (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2)):
    mm = (rank == 1) & (p >= lo) & (p < hi)
    if mm.sum():
        print(f"RESULT 1番目の確率{int(lo * 100)}〜{int(hi * 100)}%: {mm.sum()}点 的中率{(y[mm] > 0).mean() * 100:.1f}% 回収率{y[mm].mean() * 100:.1f}%")
for t in (1.0, 1.2, 1.5):
    mm = (rank == 1) & (ev >= t)
    if mm.sum():
        print(f"RESULT 1番目のうち期待値{t}以上: {mm.sum()}点 的中率{(y[mm] > 0).mean() * 100:.1f}% 回収率{y[mm].mean() * 100:.1f}%")
