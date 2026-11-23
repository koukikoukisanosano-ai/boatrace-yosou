// 外れ方の分析: その日の朝に実際に使っていた予想（その日の近況・その時点のモデル）で、
// 「誰がどの決まり手で勝つか」の予想と、実際の決まり手・勝った艇を並べて記録する（予想は当日より前のデータだけ）
//   node tools/miss_analysis.mjs <開始 YYYY-MM-DD> <終了 YYYY-MM-DD> > misses.jsonl
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { normalizeDay, Snapshot, prepareModel, predictRace } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const git = (...a) => execFileSync('git', a, { cwd: ROOT, maxBuffer: 1 << 28 }).toString();
const commits = git('log', '--format=%H %cI', '--', 'docs/model/model.json').trim().split('\n').map((l) => { const [h, t] = l.split(' '); return { h, t: new Date(t) }; });
const modelFor = (ymd) => {   // その日の朝6時（日本時間）より前の最後のモデル
  const at = new Date(`${ymd}T06:00:00+09:00`);
  const c = commits.find((x) => x.t < at);
  return c ? c.h : null;
};
const cache = new Map();
const [from, to] = process.argv.slice(2);
for (let d = new Date(`${from}T00:00:00Z`); d <= new Date(`${to}T00:00:00Z`); d = new Date(d.getTime() + 86400000)) {
  const ymd = d.toISOString().slice(0, 10), compact = ymd.replace(/-/g, '');
  const dayFile = path.join(ROOT, 'data', 'openapi', ymd.slice(0, 4), `${compact}.json.gz`);
  const stateFile = path.join(ROOT, 'docs', 'data', 'state', `${compact}.json`);
  if (!fs.existsSync(dayFile) || !fs.existsSync(stateFile)) { console.error(`${ymd}: データなし`); continue; }
  const h = modelFor(ymd);
  if (!h) continue;
  if (!cache.has(h)) {
    try { cache.set(h, prepareModel(JSON.parse(git('show', `${h}:docs/model/model.json`)))); } catch (e) { cache.set(h, null); }
  }
  const model = cache.get(h);
  if (!model) continue;
  const stj = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  const snap = new Snapshot(stj.snapshot || stj);
  const races = normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(dayFile)).toString('utf8'))).filter((r) => r.hasResult && r.boats.length >= 5);
  let n = 0;
  for (const r of races) {
    let pr;
    try { pr = predictRace(r, model, snap); } catch (e) { continue; }
    if (!pr || !pr.scenarios) continue;
    const win = r.boats.find((b) => b.place === 1);
    const second = r.boats.find((b) => b.place === 2);
    if (!win) continue;
    const idx = (w) => pr.boats.findIndex((x) => x.waku === w);
    const pv = r.boats.map((b) => b.exTime).filter((x) => typeof x === 'number');
    const exRank = (b) => (typeof b.exTime === 'number' ? 1 + pv.filter((x) => x < b.exTime).length : null);
    const b1 = r.boats.find((b) => b.waku === 1) || {};
    console.log(JSON.stringify({
      id: r.id, date: ymd, jcd: r.jcd, rno: r.rno, model: h.slice(0, 7),
      wind: r.wind ?? null, wave: r.wave ?? null, grade: r.grade ?? null,
      win: win.waku, winCourse: pr.course[idx(win.waku)] ?? null, second: second ? second.waku : null, tech: r.technique || 0,
      pwin: pr.boats.map((b, i) => [b.waku, +pr.probs.win[i].toFixed(4)]),
      scen: pr.scenarios.map((s) => [s.waku, s.course, s.tech, +s.p.toFixed(4)]),
      c1: pr.course[idx(1)] ?? null, rank1: b1.rank ?? null, ex1: exRank(b1), st1: b1.exST ?? null,
      top: pr.exacta && pr.exacta[0] ? pr.exacta[0].combo : null,
    }));
    n++;
  }
  console.error(`${ymd}: ${n} レース（モデル ${h.slice(0, 7)}）`);
}
