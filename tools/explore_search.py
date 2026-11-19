"""「普通は思いつかない買い方」を探す（探す期間と確かめる期間を分けて、偶然の当たりをふるい落とす）。

  探す期間:   2026-01〜2026-06（ここだけで候補を順位づけ）
  確かめる期間: 2026-07〜2026-10（探索では一度も見ない）
  2連単だけは 2024-11〜2025-12 でも確かめる（確定オッズがある）

候補 = 券種 × 選び方 × レースの条件。
  選び方:
    rank   予想の順位 i 位の目で、オッズの人気順位が j 位以下（予想は推すが市場は推さない）
    split  予想の1位と人気の1位が違うレースの、予想の1位（オッズの幅つき）
    ev     オッズの幅 × 期待値（予想の確率×オッズ）の下限
    ratio  オッズの幅 × 「予想の確率 ÷ 市場の確率」の下限
    fade   人気の1位を予想が低く見ている（予想の順位 j 位以下）レースで、予想の1位
  レースの条件: 全部・場ごと・R（前半/中盤/後半）・グレード・風5m以上・波5cm以上・A1の人数・1号艇の級別・
                1号艇が1コースでない・1号艇の展示タイム順位・夜（17時以降）・初日/最終日
探す期間で「回収率 − 2×誤差」が高い順に、似たものを除いて上位を確かめる期間に当てる。
  python tools/explore_search.py data/explore.npz explore_result.json
"""
import json, sys
from pathlib import Path
import numpy as np

S_FROM, S_TO, H_FROM, H_TO = "2026-01", "2026-06", "2026-07", "2026-10"
E_FROM, E_TO = "2024-11", "2025-12"
BTS = {"exacta": "2連単", "quinella": "2連複", "trio": "3連複", "quinella_place": "拡連複", "trifecta": "3連単", "win": "単勝"}
JNAME = "桐生 戸田 江戸川 平和島 多摩川 浜名湖 蒲郡 常滑 津 三国 びわこ 住之江 尼崎 鳴門 丸亀 児島 宮島 徳山 下関 若松 芦屋 福岡 唐津 大村".split()
MIN_BETS_S = 150


def rank_desc(x):
    """行ごとに大きい順の順位（1始まり）。NaN は最下位"""
    x = np.where(np.isfinite(x), x, -np.inf)
    order = np.argsort(-x, axis=1, kind="stable")
    r = np.empty_like(order); n, k = x.shape
    r[np.arange(n)[:, None], order] = np.arange(1, k + 1)[None, :]
    return r


def main():
    D = dict(np.load(sys.argv[1], allow_pickle=True))
    out_p = Path(sys.argv[2])
    month = D["month"].astype(str)
    N = len(month)
    seg = {"全部": np.ones(N, bool)}
    for j in range(1, 25):
        seg[f"{JNAME[j - 1]}"] = D["jcd"] == j
    rno = D["rno"]
    seg["1〜4R"] = rno <= 4; seg["5〜8R"] = (rno >= 5) & (rno <= 8); seg["9〜12R"] = rno >= 9; seg["12Rだけ"] = rno == 12
    g = D["grade"]
    seg["一般戦"] = g <= 1; seg["G3以上"] = g >= 2
    w = D["wind"]; seg["風5m以上"] = np.nan_to_num(w) >= 5; seg["無風〜2m"] = np.nan_to_num(w, nan=9) <= 2
    wv = D["wave"]; seg["波5cm以上"] = np.nan_to_num(wv) >= 5
    a1 = D["nA1"]; seg["A1が0人"] = a1 == 0; seg["A1が1人"] = a1 == 1; seg["A1が3人以上"] = a1 >= 3
    c1 = D["c1"]; seg["1号艇がA1"] = c1 == 1; seg["1号艇がB1・B2"] = c1 >= 3
    seg["1号艇が1コースでない"] = (D["course1"] > 1)
    ex = D["ex_rank1"]; seg["1号艇の展示1位"] = ex == 1; seg["1号艇の展示4位以下"] = ex >= 4
    seg["夜（17時以降）"] = D["hour"] >= 17; seg["朝（11時前）"] = (D["hour"] >= 0) & (D["hour"] < 11)
    dn = D["day"]; seg["初日"] = dn == 1; seg["最終日の可能性（6日目以降）"] = dn >= 6
    per = {"S": (month >= S_FROM) & (month <= S_TO), "H": (month >= H_FROM) & (month <= H_TO), "E": (month >= E_FROM) & (month <= E_TO)}

    cands = []   # (説明, 券種, mask N×K)
    stats = []
    for bt, name in BTS.items():
        p, o, y = D[f"p_{bt}"].astype(float), D[f"o_{bt}"].astype(float), D[f"y_{bt}"].astype(float)
        ok_row = np.isfinite(o).sum(1) >= max(3, int(o.shape[1] * 0.6))
        o = np.where(np.isfinite(o) & (o > 0), o, np.nan)
        inv = np.where(np.isfinite(o), 1 / o, 0)
        m = inv / np.maximum(inv.sum(1, keepdims=True), 1e-9)
        ev = p * np.nan_to_num(o)
        ratio = p / np.maximum(m, 1e-9)
        rp = rank_desc(p)                 # 予想の順位
        rm = rank_desc(-np.nan_to_num(o, nan=1e9))   # 人気の順位（オッズが低いほど上）
        valid = ok_row[:, None] & np.isfinite(o)
        sel = {}
        for i in (1, 2, 3):
            for j in (2, 3, 5, 8):
                sel[f"予想{i}位・人気{j}位以下"] = (rp == i) & (rm >= j)
        top_is_fav = ((rp == 1) & (rm == 1)).any(1)
        bands = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
        for lo, hi in bands:
            b = (o >= lo) & (o < hi)
            sel[f"予想と人気が割れた時の予想1位・{lo}〜{hi if hi < 1e9 else ''}倍"] = (rp == 1) & ~top_is_fav[:, None] & b
            for t in (1.0, 1.2, 1.5, 2.0):
                sel[f"{lo}〜{hi if hi < 1e9 else ''}倍・期待値{t}以上"] = b & (ev >= t)
            for rr in (1.3, 1.6, 2.0):
                sel[f"{lo}〜{hi if hi < 1e9 else ''}倍・予想÷市場{rr}倍以上"] = b & (ratio >= rr)
        fav_rank = np.where(rm == 1, rp, 0).max(1)   # 人気1位の目の、予想での順位
        for j in (3, 5):
            sel[f"人気1位を予想が{j}位以下に見る時の予想1位"] = (rp == 1) & (fav_rank >= j)[:, None]
        for sname, sm in sel.items():
            sm = sm & valid
            ret = np.where(sm, y, 0)
            rn, rs, rs2, rh = sm.sum(1).astype(float), ret.sum(1), (ret ** 2).sum(1), (ret > 0).sum(1)
            for gname, gm in seg.items():
                row = {"bt": name, "rule": sname, "seg": gname}
                for pk, pm in per.items():
                    mm = pm & gm
                    n_ = rn[mm].sum()
                    if n_ == 0:
                        row[pk] = None; continue
                    s_ = rs[mm].sum(); s2 = rs2[mm].sum()
                    roi = s_ / n_; se = np.sqrt(max(s2 / n_ - roi ** 2, 0) / n_)
                    nm = len(set(month[pm]))
                    row[pk] = {"n": int(n_), "roi": round(float(roi), 4), "se": round(float(se), 4), "hit": round(float(rh[mm].sum() / n_), 4),
                               "perday": round(float(n_ / max(nm * 30, 1)), 1)}
                stats.append(row)
    print(f"RESULT 候補の数 {len(stats)}", flush=True)
    good = [r for r in stats if r["S"] and r["S"]["n"] >= MIN_BETS_S]
    for r in good:
        r["score"] = r["S"]["roi"] - 2 * r["S"]["se"]
    good.sort(key=lambda r: -r["score"])
    # 似たもの（同じ券種・同じ条件で選び方だけ違う）を除く
    picked, seen = [], set()
    for r in good:
        key = (r["bt"], r["seg"])
        if key in seen:
            continue
        seen.add(key); picked.append(r)
        if len(picked) >= 40:
            break
    n_h_win = sum(1 for r in picked if r["H"] and r["H"]["roi"] > 1)
    print(f"RESULT 探す期間で基準を満たした候補 {len(good)} / 上位40（重複を除く）のうち、確かめる期間でも100%超え: {n_h_win}", flush=True)
    for r in picked:
        S, H, E = r["S"], r["H"], r.get("E")
        print("RESULT", f"{r['bt']}｜{r['rule']}｜{r['seg']}",
              f"探す {S['roi'] * 100:.0f}%±{S['se'] * 100:.0f}（{S['n']}点・的中{S['hit'] * 100:.0f}%）",
              f"確かめる {H['roi'] * 100:.0f}%±{H['se'] * 100:.0f}（{H['n']}点）" if H else "確かめる -",
              f"2024-25 {E['roi'] * 100:.0f}%（{E['n']}点）" if E and r["bt"] == "2連単" else "", flush=True)
    # 参考: 「全部のレース」で券種・選び方ごとに、探す期間と確かめる期間がどれだけ一致するか
    base = [r for r in stats if r["seg"] == "全部" and r["S"] and r["H"] and r["S"]["n"] >= 500]
    if base:
        a = np.array([[r["S"]["roi"], r["H"]["roi"]] for r in base])
        print(f"RESULT 参考: 全部のレースの選び方{len(base)}個で、探す期間と確かめる期間の回収率の相関 {np.corrcoef(a.T)[0, 1]:.2f}", flush=True)
    out_p.write_text(json.dumps({"picked": picked, "n_candidates": len(stats), "n_good": len(good),
                                 "all": [r for r in stats if r["S"] and r["S"]["n"] >= 100]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
