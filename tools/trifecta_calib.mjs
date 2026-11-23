// 3連単の的中率の確かめ: 「選んだ買い目の予想の確率の合計」が実際の的中率と合っているか
//   その日の朝に使っていたモデルで予想（当日より前のデータだけ）。docs/live のオッズ（判定時点の3連単）があれば混ぜた予想も
//   node tools/trifecta_calib.mjs <開始> <終了> > calib.jsonl
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { normalizeDay, Snapshot, prepareModel, predictRace } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const git = (...a) => execFileSync('git', a, { cwd: ROOT, maxBuffer: 1 << 28 }).toString();
const commits = git('log', '--format=%H %cI', '--', 'docs/model/model.json').trim().split('\n').map((l) => { const [h, t] = l.split(' '); return { h, t: new Date(t) }; });
const modelFor = (ymd) => { const at = new Date(`${ymd}T06:00:00+09:00`); const c = commits.find((x) => x.t < at); return c ? c.h : null; };
const cache = new Map();
const [from, to] = process.argv.slice(2);
for (let d = new Date(`${from}T00:00:00Z`); d <= new Date(`${to}T00:00:00Z`); d = new Date(d.getTime() + 86400000)) {
  const ymd = d.toISOString().slice(0, 10), compact = ymd.replace(/-/g, '');
  const dayFile = path.join(ROOT, 'data', 'openapi', ymd.slice(0, 4), `${compact}.json.gz`);
  const stateFile = path.join(ROOT, 'docs', 'data', 'state', `${compact}.json`);
  if (!fs.existsSync(dayFile) || !fs.existsSync(stateFile)) { console.error(`${ymd}: データなし`); continue; }
  const h = modelFor(ymd); if (!h) continue;
  if (!cache.has(h)) { try { cache.set(h, prepareModel(JSON.parse(git('show', `${h}:docs/model/model.json`)))); } catch (e) { cache.set(h, null); } }
  const model = cache.get(h); if (!model) continue;
  const stj = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  const snap = new Snapshot(stj.snapshot || stj);
  const lf = path.join(ROOT, 'docs', 'live', `${compact}.json`);
  const live = fs.existsSync(lf) ? (JSON.parse(fs.readFileSync(lf, 'utf8')).odds || {}) : {};
  const races = normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(dayFile)).toString('utf8'))).filter((r) => r.hasResult && r.boats.length === 6);
  let n = 0;
  for (const r of races) {
    const paid = (r.payouts.trifecta || [])[0]; if (!paid) continue;
    const o = live[`${r.jcd}-${r.rno}`];
    const out = { date: ymd, id: r.id, hit: paid.combo, amt: paid.amount };
    for (const [tag, odds] of [['m', null], ['o', o && o.t ? { t: o.t } : null]]) {
      if (tag === 'o' && !odds) continue;
      let pr; try { pr = predictRace({ ...r, odds }, model, snap); } catch (e) { pr = null; }
      if (!pr) continue;
      out[tag] = pr.trifecta.map((t) => [t.combo, +t.p.toFixed(5)]);
      if (tag === 'm') out.pos = pr.boats.map((b, i) => [b.waku, +pr.probs.win[i].toFixed(4), +pr.probs.top2[i].toFixed(4), +pr.probs.top3[i].toFixed(4)]);
    }
    if (!out.m) continue;
    out.place = Object.fromEntries(r.boats.filter((b) => b.place).map((b) => [b.waku, b.place]));
    if (o && o.t) out.t = o.t;
    console.log(JSON.stringify(out)); n++;
  }
  console.error(`${ymd}: ${n}`);
}
