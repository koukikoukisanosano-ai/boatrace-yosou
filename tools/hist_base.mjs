// 選手・モーターの近況の「下地」（公式の過去データ 2016〜2024年8月を読み込んだ後の状態）
//   作る:   node tools/hist_base.mjs build <公式の過去データのフォルダ> [出力=data/hist_base.json.gz]
//   使う:   import { startHistory } from './hist_base.mjs'  → data/hist_base.json.gz があればそこから始める
// 学習（build_dataset.mjs --extra）で過去データを日付順に読み込んだのと同じ状態になる（その日の結果は日が終わってから足す）
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { normalizeDay, History } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
export const BASE_PATH = path.join(ROOT, 'data', 'hist_base.json.gz');

// 近況の始まり: 下地があればそこから。無ければ今まで通り（PC版で集めた下地 data/warmup）
export function startHistory({ log = console.error } = {}) {
  if (fs.existsSync(BASE_PATH)) {
    const s = JSON.parse(zlib.gunzipSync(fs.readFileSync(BASE_PATH)).toString('utf8'));
    log(`近況の下地: ${path.basename(BASE_PATH)}（${s.until} まで・選手 ${Object.keys(s.p).length} 人）`);
    return { hist: new History(s), fromBase: true };
  }
  const hist = new History();
  const warm = path.join(ROOT, 'data', 'warmup');
  // Open API に2025年以前の日があれば、そちらを使う（PC版の下地と重ならないように）
  const hasOldOpenApi = fs.readdirSync(path.join(ROOT, 'data', 'openapi')).some((y) => y < '2026');
  if (fs.existsSync(warm) && !hasOldOpenApi) {
    for (const fn of fs.readdirSync(warm).filter((x) => x.endsWith('.jsonl.gz')).sort()) {
      const lines = zlib.gunzipSync(fs.readFileSync(path.join(warm, fn))).toString('utf8').split('\n').filter(Boolean);
      for (const ln of lines) hist.addRace(JSON.parse(ln));
      log(`下地 ${fn}: ${lines.length} レース`);
    }
  }
  return { hist, fromBase: false };
}

// 公式の過去データの日別ファイル（Open API の最初の日より前だけ）
export function extraFiles(dir) {
  const base = path.join(ROOT, 'data', 'openapi');
  const first = fs.readdirSync(base).sort().flatMap((y) => fs.readdirSync(path.join(base, y)).filter((f) => f.endsWith('.json.gz')).sort())[0];
  const out = [];
  for (const y of fs.readdirSync(dir).sort()) {
    for (const fn of fs.readdirSync(path.join(dir, y)).sort()) if (fn.endsWith('.json.gz') && fn < first) out.push(path.join(dir, y, fn));
  }
  return out;
}

if (process.argv[2] === 'build') {
  const dir = path.resolve(process.argv[3]);
  const out = path.resolve(process.argv[4] || BASE_PATH);
  const hist = new History();
  const files = extraFiles(dir);
  let n = 0;
  for (const f of files) {
    const races = normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(f)).toString('utf8'))).filter((r) => r.hasResult && r.boats.length >= 5);
    for (const r of races) hist.addRace(r); // build_dataset.mjs と同じ（結果があり5艇以上のレース）
    n += races.length;
  }
  const until = path.basename(files[files.length - 1]).slice(0, 8);
  const body = JSON.stringify({ p: hist.p, m: hist.m, lastDay: hist.lastDay, until: `${until.slice(0, 4)}-${until.slice(4, 6)}-${until.slice(6, 8)}`, days: files.length, races: n });
  fs.writeFileSync(out, zlib.gzipSync(body, { level: 9 }));
  console.error(`下地を作りました: ${files.length} 日・${n} レース・選手 ${Object.keys(hist.p).length} 人 → ${out}（${(fs.statSync(out).size / 1e6).toFixed(1)} MB）`);
}
