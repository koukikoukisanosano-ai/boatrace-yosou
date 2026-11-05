// 過去データ → 学習用の表（1艇1行）を作る。予想ページと同じ core.js の計算を使う。
//   node tools/build_dataset.mjs [出力先ディレクトリ=work]
// 出力: work/features.tsv（特徴量＋正解）, work/races.jsonl（払戻）, work/state.json（最新の近況）
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { normalizeDay, History, Snapshot, buildFeatures, FEATURES, dayNumber } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const NO_PREVIEW = process.argv.includes('--no-preview'); // 展示前の予想用（直前情報を使わない）
const OUT = path.resolve(process.argv.slice(2).find((a) => !a.startsWith('--')) || path.join(ROOT, 'work'));
fs.mkdirSync(OUT, { recursive: true });

// 1) 近況の下地: PC版で集めた 2025年10〜12月の結果（あれば）
const hist = new History();
const warm = path.join(ROOT, 'data', 'warmup');
if (fs.existsSync(warm)) {
  for (const fn of fs.readdirSync(warm).filter((x) => x.endsWith('.jsonl.gz')).sort()) {
    const lines = zlib.gunzipSync(fs.readFileSync(path.join(warm, fn))).toString('utf8').split('\n').filter(Boolean);
    for (const ln of lines) hist.addRace(JSON.parse(ln));
    console.error(`下地 ${fn}: ${lines.length} レース`);
  }
}

// 2) Open API の日別データを日付順に
const files = [];
const base = path.join(ROOT, 'data', 'openapi');
for (const y of fs.readdirSync(base).sort()) {
  for (const fn of fs.readdirSync(path.join(base, y)).sort()) if (fn.endsWith('.json.gz')) files.push(path.join(base, y, fn));
}

const ft = fs.createWriteStream(path.join(OUT, 'features.tsv'));
const rj = fs.createWriteStream(path.join(OUT, 'races.jsonl'));
ft.write(['race', 'date', 'place', 'preview', ...FEATURES].join('\t') + '\n');
let nRaces = 0; let nRows = 0;
const fmt = (v) => (Number.isFinite(v) ? String(v) : ''); // 丸めない（予想ページと同じ値で学習するため）

for (const file of files) {
  const j = JSON.parse(zlib.gunzipSync(fs.readFileSync(file)).toString('utf8'));
  const races = normalizeDay(j).filter((r) => r.hasResult && r.boats.length >= 5);
  if (!races.length) continue;
  // その日に走る選手の近況を「前日まで」で書き出し → 予想ページと同じ形（Snapshot）で使う
  const snap = new Snapshot(JSON.parse(JSON.stringify(hist.snapshotFor(races))));
  for (const race of races) {
    if (race.preview === 'none') continue; // 直前情報が無いレースは学習に使わない
    const { boats, X } = buildFeatures(NO_PREVIEW ? { ...race, preview: 'none' } : race, snap, null);
    if (boats.length < 5) continue;
    boats.forEach((b, i) => {
      const place = b.place !== null ? b.place : 7;
      ft.write([race.id, race.date, place, race.preview === 'full' ? 1 : 0, ...X[i].map(fmt)].join('\t') + '\n');
      nRows++;
    });
    rj.write(JSON.stringify({ id: race.id, date: race.date, jcd: race.jcd, rno: race.rno, tech: race.technique, payouts: race.payouts, refunds: race.refunds, boats: boats.map((b) => b.waku) }) + '\n');
    nRaces++;
  }
  for (const race of races) hist.addRace(race); // その日が終わってから結果を取り込む
}
ft.end(); rj.end();
fs.writeFileSync(path.join(OUT, 'history.json'), JSON.stringify({ p: hist.p, m: hist.m, lastDay: hist.lastDay }));
console.error(`完了: ${nRaces} レース, ${nRows} 行 → ${OUT}`);
