// 予想ページと同じ計算で、過去のレースを予想し直して成績を出す（答え合わせ）
//   node tools/evaluate.mjs <model.json> <開始日 YYYY-MM-DD> [終了日] [--json 出力先]
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { normalizeDay, History, Snapshot, prepareModel, predictRace } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const args = process.argv.slice(2);
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;
const NO_PREVIEW = args.includes('--no-preview');
const EXW = args.includes('--exhibit-w') ? Number(args[args.indexOf('--exhibit-w') + 1]) : undefined;
const pos = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--json' && args[i - 1] !== '--exhibit-w');
const [modelPath, from, to = '2099-12-31'] = pos;
const model = prepareModel(JSON.parse(fs.readFileSync(modelPath, 'utf8')));

const hist = new History();
const warm = path.join(ROOT, 'data', 'warmup');
// Open API に2025年以前の日があれば、そちらを使う（PC版の下地と重ならないように）
const hasOldOpenApi = fs.readdirSync(path.join(ROOT, 'data', 'openapi')).some((y) => y < '2026');
if (fs.existsSync(warm) && !hasOldOpenApi) {
  for (const fn of fs.readdirSync(warm).filter((x) => x.endsWith('.jsonl.gz')).sort()) {
    for (const ln of zlib.gunzipSync(fs.readFileSync(path.join(warm, fn))).toString('utf8').split('\n')) if (ln) hist.addRace(JSON.parse(ln));
  }
}
const base = path.join(ROOT, 'data', 'openapi');
const files = [];
for (const y of fs.readdirSync(base).sort()) for (const fn of fs.readdirSync(path.join(base, y)).sort()) files.push(path.join(base, y, fn));

const K_TRI = [1, 3, 5, 10, 20];
const acc = {
  races: 0, top1: 0, ll: 0, tll: 0,
  tri: Object.fromEntries(K_TRI.map((k) => [k, { hit: 0, ret: 0, cost: 0 }])),
  ex: Object.fromEntries([1, 3, 5].map((k) => [k, { hit: 0, ret: 0, cost: 0 }])),
  trio: Object.fromEntries([1, 3, 5].map((k) => [k, { hit: 0, ret: 0, cost: 0 }])),
  qu: Object.fromEntries([1, 3].map((k) => [k, { hit: 0, ret: 0, cost: 0 }])),
  conf: {}, calib: {}, byVenue: {},
};
const bucket = (p, step, max) => Math.min(Math.floor(p / step) * step, max).toFixed(2);

function settle(slot, combos, payouts, refunds) {
  for (const c of combos) {
    slot.cost += 100;
    const nums = c.split(/[-=]/).map(Number);
    if (nums.some((x) => refunds.includes(x))) { slot.ret += 100; continue; }
    const hit = payouts.find((p) => p.combo === c);
    if (hit) { slot.hit += 1; slot.ret += hit.amount; }
  }
}

let lastDate = from;
for (const file of files) {
  const day = path.basename(file).slice(0, 8);
  const ymd = `${day.slice(0, 4)}-${day.slice(4, 6)}-${day.slice(6, 8)}`;
  const races = normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(file)).toString('utf8'))).filter((r) => r.hasResult);
  if (ymd >= from && ymd <= to) {
    const snap = new Snapshot(JSON.parse(JSON.stringify(hist.snapshotFor(races))));
    for (const race of races) {
      if (race.preview === 'none') continue;
      const pr = predictRace(NO_PREVIEW ? { ...race, preview: 'none' } : race, model, snap, { exhibitW: EXW });
      if (!pr) continue;
      const tp = race.payouts.trifecta || [];
      if (!tp.length) continue;
      acc.races++;
      if (race.date > lastDate) lastDate = race.date;
      const winner = race.boats.find((b) => b.place === 1);
      const iw = pr.boats.findIndex((b) => b.waku === winner.waku);
      const pw = iw >= 0 ? pr.probs.win[iw] : 1e-6;
      acc.ll += -Math.log(Math.max(pw, 1e-9));
      const tt = pr.trifecta.find((t) => t.combo === tp[0].combo);
      acc.tll += -Math.log(Math.max(tt ? tt.p : 0, 1e-12));
      const best = pr.probs.win.indexOf(Math.max(...pr.probs.win));
      if (pr.boats[best].waku === winner.waku) acc.top1++;
      const refunds = (race.refunds || []).map(Number);
      for (const k of K_TRI) settle(acc.tri[k], pr.trifecta.slice(0, k).map((t) => t.combo), tp, refunds);
      for (const k of [1, 3, 5]) settle(acc.ex[k], pr.exacta.slice(0, k).map((t) => t.combo), race.payouts.exacta || [], refunds);
      for (const k of [1, 3, 5]) settle(acc.trio[k], pr.trio.slice(0, k).map((t) => t.combo), race.payouts.trio || [], refunds);
      for (const k of [1, 3]) settle(acc.qu[k], pr.quinella.slice(0, k).map((t) => t.combo), race.payouts.quinella || [], refunds);
      // 自信度（本命の3連単の確率）別
      const b = bucket(pr.trifecta[0].p, 0.05, 0.3);
      const c = acc.conf[b] || (acc.conf[b] = { n: 0, hit: 0, ret: 0, psum: 0, top1: 0 });
      c.n++; c.psum += pr.trifecta[0].p;
      if (tp.some((p) => p.combo === pr.trifecta[0].combo)) { c.hit++; c.ret += tp.find((p) => p.combo === pr.trifecta[0].combo).amount; }
      if (pr.boats[best].waku === winner.waku) c.top1++;
      // 1着確率の当たり具合
      pr.probs.win.forEach((p, i) => {
        const cb = bucket(p, 0.1, 0.9);
        const x = acc.calib[cb] || (acc.calib[cb] = { n: 0, psum: 0, wins: 0 });
        x.n++; x.psum += p; x.wins += pr.boats[i].waku === winner.waku ? 1 : 0;
      });
      const v = acc.byVenue[race.venue] || (acc.byVenue[race.venue] = { n: 0, top1: 0 });
      v.n++; v.top1 += pr.boats[best].waku === winner.waku ? 1 : 0;
    }
  }
  for (const race of races) hist.addRace(race);
}

const pct = (a, b) => (b ? (100 * a / b).toFixed(1) : '-');
const out = {
  period: [from, lastDate], races: acc.races, top1: +(acc.top1 / acc.races).toFixed(4), winnerLogloss: +(acc.ll / acc.races).toFixed(4), triLogloss: +(acc.tll / acc.races).toFixed(4),
  trifecta: Object.fromEntries(Object.entries(acc.tri).map(([k, s]) => [k, { hit: +(s.hit / acc.races).toFixed(4), roi: +(s.ret / s.cost).toFixed(4) }])),
  exacta: Object.fromEntries(Object.entries(acc.ex).map(([k, s]) => [k, { hit: +(s.hit / acc.races).toFixed(4), roi: +(s.ret / s.cost).toFixed(4) }])),
  trio: Object.fromEntries(Object.entries(acc.trio).map(([k, s]) => [k, { hit: +(s.hit / acc.races).toFixed(4), roi: +(s.ret / s.cost).toFixed(4) }])),
  quinella: Object.fromEntries(Object.entries(acc.qu).map(([k, s]) => [k, { hit: +(s.hit / acc.races).toFixed(4), roi: +(s.ret / s.cost).toFixed(4) }])),
  confidence: Object.fromEntries(Object.entries(acc.conf).sort().map(([k, c]) => [k, { n: c.n, predicted: +(c.psum / c.n).toFixed(4), hit: +(c.hit / c.n).toFixed(4), roi: +(c.ret / c.n / 100).toFixed(4), top1: +(c.top1 / c.n).toFixed(4) }])),
  calibration: Object.fromEntries(Object.entries(acc.calib).sort().map(([k, c]) => [k, { n: c.n, predicted: +(c.psum / c.n).toFixed(4), actual: +(c.wins / c.n).toFixed(4) }])),
  byVenue: Object.fromEntries(Object.entries(acc.byVenue).map(([k, v]) => [k, { n: v.n, top1: +(v.top1 / v.n).toFixed(3) }])),
};
console.log(`期間 ${from}〜${lastDate} / ${acc.races} レース`);
console.log(`1着的中率 ${pct(acc.top1, acc.races)}% / 勝者logloss ${out.winnerLogloss} / 3連単logloss ${out.triLogloss}`);
for (const [name, o] of [['3連単', out.trifecta], ['2連単', out.exacta], ['3連複', out.trio], ['2連複', out.quinella]]) {
  console.log(`${name}: ` + Object.entries(o).map(([k, s]) => `上位${k}点 的中${(s.hit * 100).toFixed(1)}% 回収${(s.roi * 100).toFixed(0)}%`).join(' / '));
}
console.log('本命3連単の確率別: ' + Object.entries(out.confidence).map(([k, c]) => `${k}+ n=${c.n} 予想${(c.predicted * 100).toFixed(1)}% 実際${(c.hit * 100).toFixed(1)}% 回収${(c.roi * 100).toFixed(0)}%`).join(' | '));
console.log('1着確率の当たり具合: ' + Object.entries(out.calibration).map(([k, c]) => `${k}: 予想${(c.predicted * 100).toFixed(1)}% 実際${(c.actual * 100).toFixed(1)}% (n=${c.n})`).join(' | '));
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(out, null, 1));
