// core.js の確認（node tests/test_core.mjs）
import assert from 'node:assert/strict';
import { trifecta, placeProbs, FEATURES, buildFeatures, Snapshot, normalizeRace } from '../docs/js/core.js';

// 3連単の確率の合計は1、1着の確率と合う
const pp = { win: [0.5, 0.2, 0.12, 0.1, 0.05, 0.03], top2: [0.7, 0.4, 0.33, 0.28, 0.17, 0.12], top3: [0.8, 0.6, 0.52, 0.47, 0.35, 0.26] };
const tri = trifecta(pp);
assert.equal(tri.length, 120);
assert.ok(Math.abs(tri.reduce((a, t) => a + t.p, 0) - 1) < 1e-9);
const win1 = tri.filter((t) => t.idx[0] === 0).reduce((a, t) => a + t.p, 0);
assert.ok(Math.abs(win1 - 0.5) < 0.02, `1号艇の1着 ${win1}`);

// 直前情報なしのレースでも特徴量が作れる
const rc = { date: '2026-11-04', closed_at: '2026-11-04 15:20:00', racers: {}, preview: {}, result: {} };
for (let w = 1; w <= 6; w++) rc.racers[w] = { number: 4000 + w, national_win_rate: 5 + w / 10, rank_number: 3, average_start_timing: 0.16 };
const race = normalizeRace(rc, 1, 1);
assert.equal(race.preview, 'none');
const { X } = buildFeatures(race, new Snapshot(null), null);
assert.equal(X.length, 6);
assert.equal(X[0].length, FEATURES.length);
console.log('ok');
