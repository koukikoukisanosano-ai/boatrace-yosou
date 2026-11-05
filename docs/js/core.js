// 競艇予想のコア（データの整形・近況・特徴量・モデル・3連単の確率）
// スマホのブラウザと、学習データを作る Node の両方でこの同じファイルを使う（学習と予想で計算をそろえるため）

export const VENUES = {
  1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川', 6: '浜名湖', 7: '蒲郡', 8: '常滑',
  9: '津', 10: '三国', 11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島',
  17: '宮島', 18: '徳山', 19: '下関', 20: '若松', 21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
};
export const RANK_LABEL = { 1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2' };
const NaN_ = Number.NaN;

const num = (v) => (v === null || v === undefined || v === '' || !Number.isFinite(+v)) ? null : +v;
const nz = (v) => (v === null || v === 0) ? null : v; // 当地成績なし(0)は欠損扱い

// ------------------------------------------------------------------ 日付
export function dayNumber(dateStr) { // 'YYYY-MM-DD' or 'YYYYMMDD' -> 1970-01-01 からの日数
  const s = String(dateStr).replace(/-/g, '');
  return Math.round(Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8)) / 86400000);
}
export function dayString(n) {
  const d = new Date(n * 86400000);
  return d.toISOString().slice(0, 10);
}

// ------------------------------------------------------------------ データの整形（Boatrace Open API v1 → レース）
export function normalizeDay(j) {
  const out = [];
  const st = (j && j.programs && j.programs.stadiums) || {};
  for (const [jcd, s] of Object.entries(st)) {
    for (const [rno, rc] of Object.entries((s && s.races) || {})) {
      const r = normalizeRace(rc, +jcd, +rno);
      if (r) out.push(r);
    }
  }
  out.sort(raceOrder);
  return out;
}

export function raceOrder(a, b) {
  if (a.date !== b.date) return a.date < b.date ? -1 : 1;
  if (a.closedAt !== b.closedAt) return a.closedAt < b.closedAt ? -1 : 1;
  return a.jcd - b.jcd || a.rno - b.rno;
}

const MAJOR_PARTS = ['ピストン', 'シリンダ', 'キャブ', 'シャフト', 'ギヤ', '電気', 'キャリボ'];

export function normalizeRace(rc, jcd, rno) {
  if (!rc) return null;
  const pv = rc.preview || {};
  const res = rc.result || {};
  const boats = [];
  for (let w = 1; w <= 6; w++) {
    const x = (rc.racers || {})[w];
    if (!x) continue;
    const p = (pv.racers || {})[w] || {};
    const r = (res.racers || {})[w] || {};
    const placeNo = num(r.place_number);
    boats.push({
      waku: w, num: num(x.number), name: x.name || '', rank: num(x.rank_number), branch: x.branch_number_source || '',
      age: num(x.age), weight: num(x.weight), F: num(x.flying_count), L: num(x.late_count), avgST: num(x.average_start_timing),
      natWin: num(x.national_win_rate), nat2: num(x.national_top_2_percent), nat3: num(x.national_top_3_percent),
      locWin: nz(num(x.local_win_rate)), loc2: num(x.local_win_rate) ? num(x.local_top_2_percent) : null,
      loc3: num(x.local_win_rate) ? num(x.local_top_3_percent) : null,
      motorNo: num(x.motor_number), motor2: num(x.motor_top_2_percent), motor3: num(x.motor_top_3_percent),
      boatNo: num(x.boat_number), boat2: num(x.boat_top_2_percent), boat3: num(x.boat_top_3_percent),
      // 直前情報（スタート展示・展示航走）
      exCourse: num(p.course_number), exST: num(p.start_timing), exSTsrc: p.start_timing_source || '',
      exTime: num(p.exhibition_time), tilt: num(p.tilt_adjustment), exWeight: num(p.weight),
      weightAdj: num(p.weight_adjustment), propNew: p.propeller ? 1 : 0,
      parts: (p.parts || []).map((pt) => pt.number_source + (pt.quantity ? String(pt.quantity) : '')),
      // 結果
      course: num(r.course_number), st: num(r.start_timing), stSrc: r.start_timing_source || '',
      place: placeNo !== null && placeNo >= 1 && placeNo <= 6 ? placeNo : null, placeSrc: r.place_number_source || '',
    });
  }
  const pays = {};
  for (const [k, arr] of Object.entries(res.payouts || {})) {
    pays[k] = (arr || []).map((p) => ({ combo: p.combination, amount: num(p.amount) }));
  }
  const hasResult = boats.some((b) => b.place === 1);
  const race = {
    id: `${String(rc.date || '').replace(/-/g, '')}-${jcd}-${rno}`,
    date: rc.date, jcd, rno, closedAt: (rc.closed_at || '').slice(11, 16), venue: VENUES[jcd] || String(jcd),
    grade: num(rc.grade_number), gradeSrc: rc.grade_number_source || '', distance: num(rc.distance),
    day: num(rc.day_number), title: rc.title || '', subtitle: rc.subtitle || '',
    wind: num(pv.wind_speed), windDir: num(pv.wind_direction_number), wave: num(pv.wave_height),
    weather: num(pv.weather_number), airTemp: num(pv.air_temperature), waterTemp: num(pv.water_temperature),
    boats, payouts: pays, refunds: res.refunds || [], technique: num(res.technique_number), hasResult,
  };
  race.preview = previewState(race);
  return race;
}

// 直前情報の状態: 'full' 全艇そろう / 'partial' 一部（欠場艇を除けばそろう） / 'none' まだ
export function previewState(race) {
  const have = race.boats.filter((b) => b.exTime !== null && b.exCourse !== null);
  if (have.length === race.boats.length && have.length > 0) return 'full';
  if (have.length >= 3) return 'partial';
  return 'none';
}

// 予想に使う艇（欠場を除く）。直前情報がそろっていれば、展示の無い艇は欠場とみなす
export function activeBoats(race) {
  return race.boats.filter((b) => {
    if (b.placeSrc === '欠') return false;
    if (race.preview === 'partial' && b.exTime === null && b.exCourse === null) return false;
    return true;
  });
}

// ------------------------------------------------------------------ 近況（前日までの結果）
// 選手（登録番号）とモーター（場×番号）の成績を、日付順に取り込みながら持つ
const COURSE_PRIOR = [null, [0.55, 0.72, 0.82], [0.14, 0.40, 0.58], [0.12, 0.35, 0.55], [0.11, 0.30, 0.50], [0.06, 0.20, 0.38], [0.03, 0.12, 0.27]];
const shrink = (hits, n, prior, k) => (hits + k * prior) / (n + k);

export class History {
  constructor(state) {
    this.p = (state && state.p) || {}; // num -> {r: [[day, jcd, course, place(7=着外), st]], c: [[n,w,t2,t3]x6], v: {jcd: [n,w]}}
    this.m = (state && state.m) || {}; // 'jcd-motor' -> [[day, place], ...]
    this.lastDay = (state && state.lastDay) || 0;
  }

  addRace(race) {
    if (!race.hasResult) return;
    const d = dayNumber(race.date);
    // 展示タイムのレース内平均との差（選手・モーターの「普段の展示の出方」を覚える）
    const exs = race.boats.map((b) => b.exTime).filter((v) => Number.isFinite(v) && v > 0);
    const exMean = exs.length >= 4 ? exs.reduce((a, b) => a + b, 0) / exs.length : NaN_;
    for (const b of race.boats) {
      if (b.placeSrc === '欠' || b.num === null) continue;
      const relEx = Number.isFinite(b.exTime) && b.exTime > 0 && Number.isFinite(exMean) ? Math.round((b.exTime - exMean) * 1e4) / 1e4 : null;
      const place = b.place !== null ? b.place : 7;
      const course = b.course || b.exCourse || b.waku;
      const st = (b.st !== null && b.st >= 0) ? b.st : (b.st !== null && b.st < 0 ? -1 : null); // -1 = F
      let p = this.p[b.num];
      if (!p) p = this.p[b.num] = { r: [], c: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], v: {} };
      p.r.push([d, race.jcd, course, place, st, relEx]);
      if (p.r.length > 40) p.r.splice(0, p.r.length - 40);
      if (course >= 1 && course <= 6) {
        const c = p.c[course - 1];
        c[0] += 1; c[1] += place === 1; c[2] += place <= 2; c[3] += place <= 3;
      }
      const v = p.v[race.jcd] || (p.v[race.jcd] = [0, 0]);
      v[0] += 1; v[1] += place === 1;
      if (b.motorNo !== null) {
        const key = `${race.jcd}-${b.motorNo}`;
        const m = this.m[key] || (this.m[key] = []);
        m.push([d, place, relEx]);
        while (m.length && d - m[0][0] > 60) m.shift();
      }
    }
    this.lastDay = Math.max(this.lastDay, d);
  }

  // その日のレース用の近況（d より前の結果だけを使う）
  racer(num, d, jcd) {
    const p = this.p[num];
    const out = { n90: NaN_, avgFin: NaN_, win10: NaN_, top3_10: NaN_, venueWin: NaN_, setsuAvg: NaN_, setsuN: 0, daysOff: NaN_, stAvg: NaN_, stStd: NaN_, f180: NaN_, course: null, exRel: NaN_ };
    if (!p) return out;
    const recs = p.r.filter((r) => r[0] < d);
    if (!recs.length) return out;
    const last10 = recs.slice(-10);
    const fins = last10.map((r) => r[3]);
    out.n90 = recs.filter((r) => d - r[0] <= 90).length;
    out.avgFin = (fins.reduce((a, b) => a + b, 0) + 3 * 3.6) / (fins.length + 3);
    out.win10 = shrink(fins.filter((f) => f === 1).length, fins.length, 1 / 6, 3);
    out.top3_10 = shrink(fins.filter((f) => f <= 3).length, fins.length, 0.5, 3);
    // 今節（同じ場で、間を空けずに続いている出走）
    const setsu = [];
    for (let i = recs.length - 1; i >= 0; i--) {
      const r = recs[i];
      if (r[1] !== jcd || d - r[0] > 7) break;
      setsu.push(r[3]);
    }
    out.setsuN = setsu.length;
    out.setsuAvg = setsu.length ? setsu.reduce((a, b) => a + b, 0) / setsu.length : NaN_;
    out.daysOff = Math.min(d - recs[recs.length - 1][0], 60);
    const sts = last10.map((r) => r[4]).filter((s) => s !== null && s >= 0);
    if (sts.length >= 3) {
      const m = sts.reduce((a, b) => a + b, 0) / sts.length;
      out.stAvg = m;
      out.stStd = Math.sqrt(sts.reduce((a, b) => a + (b - m) * (b - m), 0) / sts.length);
    }
    out.f180 = recs.filter((r) => r[4] === -1 && d - r[0] <= 180).length;
    const rel = last10.map((r) => r[5]).filter((v) => v !== null && v !== undefined && Number.isFinite(v));
    if (rel.length >= 3) out.exRel = rel.reduce((a, b) => a + b, 0) / rel.length;
    const v = p.v[jcd] || [0, 0];
    out.venueWin = shrink(v[1], v[0], 1 / 6, 6);
    out.course = p.c;
    return out;
  }

  motor(jcd, motorNo, d) {
    const m = this.m[`${jcd}-${motorNo}`];
    const out = { avgFin: NaN_, win: NaN_, n: 0, exRel: NaN_ };
    if (!m) return out;
    const recs = m.filter((r) => r[0] < d && d - r[0] <= 45);
    if (!recs.length) return out;
    const rel = recs.map((r) => r[2]).filter((v) => v !== null && v !== undefined && Number.isFinite(v));
    if (rel.length >= 3) out.exRel = rel.reduce((a, b) => a + b, 0) / rel.length;
    const fins = recs.map((r) => r[1]);
    out.n = fins.length;
    out.avgFin = (fins.reduce((a, b) => a + b, 0) + 6 * 3.6) / (fins.length + 6);
    out.win = shrink(fins.filter((f) => f === 1).length, fins.length, 1 / 6, 6);
    return out;
  }

  // その日に走る選手・モーターの分だけを書き出す（スマホで読む量を減らす）
  snapshotFor(races) {
    const out = { r: {}, m: {} };
    for (const race of races) {
      const d = dayNumber(race.date);
      for (const b of race.boats) {
        if (b.num !== null && !out.r[b.num]) {
          const x = this.racer(b.num, d, race.jcd);
          out.r[b.num] = compactRacer(x, race.jcd);
        } else if (b.num !== null) {
          const x = this.racer(b.num, d, race.jcd);
          out.r[b.num].v = out.r[b.num].v || {};
          out.r[b.num].v[race.jcd] = [round4(x.venueWin), x.setsuN, round4(x.setsuAvg)];
        }
        if (b.motorNo !== null) {
          const k = `${race.jcd}-${b.motorNo}`;
          if (!out.m[k]) { const m = this.motor(race.jcd, b.motorNo, d); out.m[k] = [round4(m.avgFin), round4(m.win), m.n, round4(m.exRel)]; }
        }
      }
    }
    return out;
  }
}

const round4 = (x) => (Number.isFinite(x) ? Math.round(x * 1e6) / 1e6 : null);
function compactRacer(x, jcd) {
  const v = {}; v[jcd] = [round4(x.venueWin), x.setsuN, round4(x.setsuAvg)];
  return {
    a: [round4(x.n90), round4(x.avgFin), round4(x.win10), round4(x.top3_10), round4(x.daysOff), round4(x.stAvg), round4(x.stStd), round4(x.f180), round4(x.exRel)],
    c: x.course, v,
  };
}

// スナップショット（日別に書き出した近況）を History と同じ形で引けるようにする
export class Snapshot {
  constructor(snap) { this.s = snap || { r: {}, m: {} }; }
  racer(num, d, jcd) {
    const x = this.s.r[num];
    const out = { n90: NaN_, avgFin: NaN_, win10: NaN_, top3_10: NaN_, venueWin: NaN_, setsuAvg: NaN_, setsuN: 0, daysOff: NaN_, stAvg: NaN_, stStd: NaN_, f180: NaN_, course: null, exRel: NaN_ };
    if (!x) return out;
    const f = (v) => (v === null || v === undefined ? NaN_ : v);
    [out.n90, out.avgFin, out.win10, out.top3_10, out.daysOff, out.stAvg, out.stStd, out.f180, out.exRel] = [...x.a, null].slice(0, 9).map(f);
    out.course = x.c || null;
    const v = (x.v || {})[jcd];
    if (v) { out.venueWin = f(v[0]); out.setsuN = v[1] || 0; out.setsuAvg = f(v[2]); }
    else { out.venueWin = 1 / 6; out.setsuN = 0; out.setsuAvg = NaN_; }
    return out;
  }
  motor(jcd, motorNo) {
    const m = this.s.m[`${jcd}-${motorNo}`];
    if (!m) return { avgFin: NaN_, win: NaN_, n: 0, exRel: NaN_ };
    return { avgFin: m[0] === null ? NaN_ : m[0], win: m[1] === null ? NaN_ : m[1], n: m[2] || 0, exRel: m[3] === null || m[3] === undefined ? NaN_ : m[3] };
  }
}

// ------------------------------------------------------------------ 特徴量
export const FEATURES = [
  // レース
  'jcd', 'rno', 'grade', 'day', 'raceType', 'fixed', 'distance', 'wind', 'windSin', 'windCos', 'wave', 'weather', 'airTemp', 'waterTemp',
  // 艇・選手
  'waku', 'course', 'courseChanged', 'rank', 'age', 'weight', 'F', 'L', 'avgST',
  'natWin', 'nat2', 'nat3', 'locWin', 'loc2', 'loc3', 'motor2', 'motor3', 'boat2', 'boat3',
  // 直前情報
  'exTime', 'exST', 'exSTF', 'tilt', 'weightAdj', 'propNew', 'partsRing', 'partsMajor',
  // レース内の比較
  'exTimeRank', 'exTimeDiff', 'exTimeToBest', 'exSTRank', 'exSTDiff', 'natWinRank', 'natWinDiff',
  'motor2Rank', 'motor2Diff', 'avgSTRank', 'nBoats',
  // 場×コース
  'vcWin', 'vcTop2', 'vcTop3',
  // 展開（並びで隣・内・外と比べる）
  'exSTvsInner', 'exSTvsOuter', 'exSTvsC1', 'natWinVsInner', 'wallNatWin', 'innerMinNatWin', 'innerMeanNatWin',
  'outerThreat', 'outerMaxNatWin', 'c1NatWinGap', 'exTimeVsInner', 'c1ExST', 'c1CourseWin',
  // 近況
  'pN90', 'pAvgFin', 'pWin10', 'pTop3', 'pCourseWin', 'pCourseTop2', 'pCourseTop3', 'pCourseN', 'pVenueWin',
  'pSetsuAvg', 'pSetsuN', 'pDaysOff', 'pSTavg', 'pSTstd', 'pF180', 'stEst', 'stEstRank', 'stEstVsInner', 'stEstVsOuter',
  'mAvgFin', 'mWin', 'mN', 'pAvgFinRank', 'pCourseWinRank',
  // 普段の展示の出方（選手・モーターごとの、展示タイムのレース内平均との差の平均）と、今日の展示との差
  'pExRel', 'mExRel', 'exRelVsOwn', 'exRelVsMotor', 'exRelVsBoth',
];
export const FEATURE_INDEX = Object.fromEntries(FEATURES.map((f, i) => [f, i]));

function raceType(race) {
  const s = `${race.subtitle} `;
  if (s.includes('優勝戦') && !s.includes('準優')) return 3;
  if (s.includes('準優')) return 2;
  if (/特選|選抜|特賞|ドリーム|記念|特別/.test(s)) return 1;
  return 0;
}

function rankOf(vals, asc = true) { // 同順位は小さい方（method=min）。欠損は NaN
  return vals.map((v) => {
    if (!Number.isFinite(v)) return NaN_;
    let r = 1;
    for (const u of vals) if (Number.isFinite(u) && (asc ? u < v : u > v)) r++;
    return r;
  });
}
const mean = (arr) => { const a = arr.filter(Number.isFinite); return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN_; };
const minOf = (arr) => { const a = arr.filter(Number.isFinite); return a.length ? Math.min(...a) : NaN_; };
const maxOf = (arr) => { const a = arr.filter(Number.isFinite); return a.length ? Math.max(...a) : NaN_; };
const f = (v) => (v === null || v === undefined ? NaN_ : v);

// race: 整形済みレース / hist: History か Snapshot / vc: 場×コースの率 {jcd: {course: [win,top2,top3]}}
// 戻り値: { boats: 予想対象の艇, X: 各艇の特徴量（FEATURES の順の配列） }
export function buildFeatures(race, hist, vc) {
  const boats = activeBoats(race);
  const n = boats.length;
  const d = dayNumber(race.date);
  const hasPv = race.preview !== 'none';
  const wd = race.windDir;
  const ang = (wd && wd >= 1 && wd <= 16) ? (wd - 1) * 2 * Math.PI / 16 : null;
  const ws = f(race.wind);
  const windSin = ang === null ? (wd === 17 ? 0 : NaN_) : Math.sin(ang) * ws;
  const windCos = ang === null ? (wd === 17 ? 0 : NaN_) : Math.cos(ang) * ws;
  const rt = raceType(race);
  const fixed = /進入固定/.test(race.subtitle + race.title) ? 1 : 0;

  // コース: 直前情報があればスタート展示の進入、無ければ枠番
  const course = boats.map((b) => (hasPv && b.exCourse ? b.exCourse : b.waku));
  const exTime = boats.map((b) => (hasPv ? f(b.exTime) : NaN_));
  const exST = boats.map((b) => (hasPv ? f(b.exST) : NaN_));
  const exSTok = exST.map((v) => (Number.isFinite(v) && v >= 0 ? v : NaN_));
  const exSTforRank = exST.map((v) => (Number.isFinite(v) ? (v >= 0 ? v : 1.0) : NaN_)); // 展示Fは最下位扱い
  const natWin = boats.map((b) => f(b.natWin));
  const motor2 = boats.map((b) => f(b.motor2));
  const avgST = boats.map((b) => f(b.avgST));

  const hx = boats.map((b, i) => (b.num !== null ? hist.racer(b.num, d, race.jcd) : null));
  const mx = boats.map((b) => (b.motorNo !== null ? hist.motor(race.jcd, b.motorNo, d) : { avgFin: NaN_, win: NaN_, n: 0, exRel: NaN_ }));
  const pCourse = boats.map((b, i) => {
    const h = hx[i]; const c = course[i];
    const prior = COURSE_PRIOR[c] || [1 / 6, 1 / 3, 1 / 2];
    const cc = h && h.course ? h.course[c - 1] : null;
    if (!cc) return [prior[0], prior[1], prior[2], 0];
    return [shrink(cc[1], cc[0], prior[0], 6), shrink(cc[2], cc[0], prior[1], 6), shrink(cc[3], cc[0], prior[2], 6), cc[0]];
  });
  const stEst = boats.map((b, i) => (hx[i] && Number.isFinite(hx[i].stAvg) ? hx[i].stAvg : avgST[i]));
  const pAvgFin = hx.map((h) => (h ? h.avgFin : NaN_));
  const pCourseWin = pCourse.map((p) => p[0]);

  const exTimeRank = rankOf(exTime, true);
  const exSTRank = rankOf(exSTforRank, true);
  const natWinRank = rankOf(natWin, false);
  const motor2Rank = rankOf(motor2, false);
  const avgSTRank = rankOf(avgST, true);
  const stEstRank = rankOf(stEst, true);
  const pAvgFinRank = rankOf(pAvgFin, true);
  const pCourseWinRank = rankOf(pCourseWin, false);
  const exTimeMean = mean(exTime); const exTimeMin = minOf(exTime);
  const exSTMean = mean(exSTok); const natWinMean = mean(natWin); const motor2Mean = mean(motor2);

  const byCourse = {};
  course.forEach((c, i) => { if (!(c in byCourse)) byCourse[c] = i; });
  const c1 = byCourse[1];

  const X = boats.map((b, i) => {
    const c = course[i];
    const inner = []; for (let k = 1; k < c; k++) if (k in byCourse) inner.push(byCourse[k]);
    const outer = []; for (const k of [c + 1, c + 2]) if (k in byCourse) outer.push(byCourse[k]);
    const wi = byCourse[c - 1]; const wo = byCourse[c + 1];
    const h = hx[i] || {};
    const m = mx[i];
    const vr = (vc && vc[race.jcd] && vc[race.jcd][c]) || (vc && vc[0] && vc[0][c]) || [NaN_, NaN_, NaN_];
    const parts = b.parts || [];
    const row = {
      jcd: race.jcd, rno: race.rno, grade: f(race.grade), day: f(race.day), raceType: rt, fixed, distance: f(race.distance),
      wind: ws, windSin, windCos, wave: f(race.wave), weather: f(race.weather), airTemp: f(race.airTemp), waterTemp: f(race.waterTemp),
      waku: b.waku, course: c, courseChanged: c !== b.waku ? 1 : 0, rank: f(b.rank), age: f(b.age), weight: f(b.weight),
      F: f(b.F), L: f(b.L), avgST: avgST[i], natWin: natWin[i], nat2: f(b.nat2), nat3: f(b.nat3),
      locWin: f(b.locWin), loc2: f(b.loc2), loc3: f(b.loc3), motor2: motor2[i], motor3: f(b.motor3), boat2: f(b.boat2), boat3: f(b.boat3),
      exTime: exTime[i], exST: exST[i], exSTF: Number.isFinite(exST[i]) ? (exST[i] < 0 ? 1 : 0) : NaN_,
      tilt: hasPv ? f(b.tilt) : NaN_, weightAdj: hasPv ? f(b.weightAdj) : NaN_, propNew: hasPv ? b.propNew : NaN_,
      partsRing: hasPv ? parts.filter((p) => p.startsWith('リング')).length : NaN_,
      partsMajor: hasPv ? parts.filter((p) => MAJOR_PARTS.some((k) => p.startsWith(k))).length : NaN_,
      exTimeRank: exTimeRank[i], exTimeDiff: exTime[i] - exTimeMean, exTimeToBest: exTime[i] - exTimeMin,
      exSTRank: exSTRank[i], exSTDiff: exSTok[i] - exSTMean, natWinRank: natWinRank[i], natWinDiff: natWin[i] - natWinMean,
      motor2Rank: motor2Rank[i], motor2Diff: motor2[i] - motor2Mean, avgSTRank: avgSTRank[i], nBoats: n,
      vcWin: vr[0], vcTop2: vr[1], vcTop3: vr[2],
      exSTvsInner: wi !== undefined ? exSTok[i] - exSTok[wi] : NaN_,
      exSTvsOuter: wo !== undefined ? exSTok[i] - exSTok[wo] : NaN_,
      exSTvsC1: c1 !== undefined && c1 !== i ? exSTok[i] - exSTok[c1] : NaN_,
      natWinVsInner: wi !== undefined ? natWin[i] - natWin[wi] : NaN_,
      wallNatWin: wi !== undefined ? natWin[wi] : NaN_,
      innerMinNatWin: inner.length ? minOf(inner.map((k) => natWin[k])) : NaN_,
      innerMeanNatWin: inner.length ? mean(inner.map((k) => natWin[k])) : NaN_,
      outerThreat: outer.length ? minOf(outer.map((k) => exSTok[k])) - exSTok[i] : NaN_,
      outerMaxNatWin: outer.length ? maxOf(outer.map((k) => natWin[k])) : NaN_,
      c1NatWinGap: c1 !== undefined && c1 !== i ? natWin[c1] - natWin[i] : NaN_,
      exTimeVsInner: wi !== undefined ? exTime[i] - exTime[wi] : NaN_,
      c1ExST: c1 !== undefined ? exSTok[c1] : NaN_,
      c1CourseWin: c1 !== undefined ? pCourse[c1][0] : NaN_,
      pN90: f(h.n90), pAvgFin: pAvgFin[i], pWin10: f(h.win10), pTop3: f(h.top3_10),
      pCourseWin: pCourse[i][0], pCourseTop2: pCourse[i][1], pCourseTop3: pCourse[i][2], pCourseN: pCourse[i][3],
      pVenueWin: f(h.venueWin), pSetsuAvg: f(h.setsuAvg), pSetsuN: h.setsuN || 0, pDaysOff: f(h.daysOff),
      pSTavg: f(h.stAvg), pSTstd: f(h.stStd), pF180: f(h.f180), stEst: stEst[i], stEstRank: stEstRank[i],
      stEstVsInner: wi !== undefined ? stEst[i] - stEst[wi] : NaN_,
      stEstVsOuter: wo !== undefined ? stEst[i] - stEst[wo] : NaN_,
      mAvgFin: m.avgFin, mWin: m.win, mN: m.n, pAvgFinRank: pAvgFinRank[i], pCourseWinRank: pCourseWinRank[i],
      pExRel: f(h.exRel), mExRel: f(m.exRel),
      exRelVsOwn: exTime[i] - exTimeMean - f(h.exRel), exRelVsMotor: exTime[i] - exTimeMean - f(m.exRel),
      exRelVsBoth: exTime[i] - exTimeMean - f(h.exRel) - f(m.exRel),
    };
    return FEATURES.map((k) => { const v = row[k]; return (v === null || v === undefined || Number.isNaN(v)) ? NaN_ : +v; });
  });
  return { boats, X, course };
}

// ------------------------------------------------------------------ モデル（勾配ブースティング木）
// 書き出し形式: {base, trees: [[feature[], threshold[], left[], right[], missingLeft(0/1)[], value[]], ...]}
function evalTree(t, x) {
  const fe = t[0], th = t[1], le = t[2], ri = t[3], ml = t[4], va = t[5];
  let i = 0;
  while (le[i] !== 0) {
    const v = x[fe[i]];
    if (v !== v) i = ml[i] ? le[i] : ri[i];
    else i = v <= th[i] ? le[i] : ri[i];
  }
  return va[i];
}
export function rawScore(m, x) {
  let s = m.base;
  for (const t of m.trees) s += evalTree(t, x);
  return s;
}
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// model.json を読み込んだら一度だけ呼ぶ: モデルの特徴量の並びと FEATURES の対応を作る
// 展開モデルの特徴量: その艇自身・1着艇（w_）・2着艇（s_）・コース差（dcw, dcs）・決まり手（tech_w）
function compileSpec(names) {
  return names.map((n) => {
    if (n in FEATURE_INDEX) return [0, FEATURE_INDEX[n]];
    if (n.startsWith('w_') && n.slice(2) in FEATURE_INDEX) return [1, FEATURE_INDEX[n.slice(2)]];
    if (n.startsWith('s_') && n.slice(2) in FEATURE_INDEX) return [2, FEATURE_INDEX[n.slice(2)]];
    if (n === 'dcw') return [3, 0];
    if (n === 'dcs') return [4, 0];
    if (n === 'tech_w') return [5, 0];
    throw new Error(`未知の特徴量: ${n}`);
  });
}
function pairVec(spec, xc, xa, xb, t) {
  const ci = FEATURE_INDEX.course;
  const out = new Array(spec.length);
  for (let k = 0; k < spec.length; k++) {
    const [src, j] = spec[k];
    out[k] = src === 0 ? xc[j] : src === 1 ? xa[j] : src === 2 ? xb[j] : src === 3 ? xc[ci] - xa[ci] : src === 4 ? xc[ci] - xb[ci] : t;
  }
  return out;
}

export function prepareModel(json) {
  for (const key of Object.keys(json).filter((k) => k.startsWith('tenkai'))) {
    const T = json[key];
    if (!T || !T.S) continue;
    T.S.featureMap = T.S.features.map((n) => FEATURE_INDEX[n]);
    T.N2.spec = compileSpec(T.N2.features);
    T.N3.spec = compileSpec(T.N3.features);
  }
  for (const key of ['full', 'full_rw', 'pre', 'clogit', 'ce']) {
    const slot = json[key];
    if (!slot) continue;
    slot.featureMap = (slot.features || FEATURES).map((name) => {
      if (!(name in FEATURE_INDEX)) throw new Error(`未知の特徴量: ${name}`);
      return FEATURE_INDEX[name];
    });
  }
  return json;
}
function pick(x, map) { if (!map) return x; const out = new Array(map.length); for (let k = 0; k < map.length; k++) out[k] = x[map[k]]; return out; }

// 各艇の 1着・2着以内・3着以内 の確率（レース内で合計 1・2・3 になるよう調整）
export function placeProbs(model, Xfull) {
  const X = Xfull.map((x) => pick(x, model.featureMap));
  const n = X.length;
  const p1 = X.map((x) => sigmoid(rawScore(model.win, x)));
  const p2 = X.map((x) => sigmoid(rawScore(model.top2, x)));
  const p3 = X.map((x) => sigmoid(rawScore(model.top3, x)));
  const s1 = p1.reduce((a, b) => a + b, 0);
  const w = p1.map((v) => v / s1);
  const t2 = scaleTo(p2, Math.min(2, n));
  const t3 = scaleTo(p3, Math.min(3, n));
  for (let i = 0; i < n; i++) { t2[i] = Math.max(t2[i], w[i]); t3[i] = Math.max(t3[i], t2[i]); }
  return { win: w, top2: t2, top3: t3 };
}
function scaleTo(p, total) { // 合計を total に、各値は 1 以下に
  let q = p.slice();
  for (let it = 0; it < 20; it++) {
    const fixed = q.map((v) => v >= 0.999);
    const sumFixed = q.reduce((a, v, i) => a + (fixed[i] ? 1 : 0), 0);
    const sumFree = q.reduce((a, v, i) => a + (fixed[i] ? 0 : v), 0);
    if (sumFree <= 0) break;
    const k = (total - sumFixed) / sumFree;
    q = q.map((v, i) => (fixed[i] ? 0.999 : Math.min(0.999, v * k)));
    if (Math.abs(q.reduce((a, b) => a + b, 0) - total) < 1e-9) break;
  }
  return q;
}

// 1着・2着内・3着内の確率から3連単の確率を作る（2・3着の出方を反復で合わせる）
export function trifecta(pp, iters = 12) {
  const n = pp.win.length;
  const idx = [...Array(n).keys()];
  if (n < 3) return [];
  const s2t = idx.map((i) => Math.max(pp.top2[i] - pp.win[i], 1e-4));
  const s3t = idx.map((i) => Math.max(pp.top3[i] - pp.top2[i], 1e-4));
  const n2 = s2t.reduce((a, b) => a + b, 0); const n3 = s3t.reduce((a, b) => a + b, 0);
  for (let i = 0; i < n; i++) { s2t[i] /= n2; s3t[i] /= n3; }
  const w2 = s2t.slice(); const w3 = s3t.slice();
  const perms = [];
  for (const a of idx) for (const b of idx) if (b !== a) for (const c of idx) if (c !== a && c !== b) perms.push([a, b, c]);
  const build = () => {
    const sw2 = w2.reduce((x, y) => x + y, 0); const sw3 = w3.reduce((x, y) => x + y, 0);
    return perms.map(([a, b, c]) => pp.win[a] * w2[b] / (sw2 - w2[a]) * w3[c] / (sw3 - w3[a] - w3[b]));
  };
  let tri = build();
  for (let it = 0; it < iters; it++) {
    const m2 = new Array(n).fill(0); const m3 = new Array(n).fill(0);
    perms.forEach(([a, b, c], k) => { m2[b] += tri[k]; m3[c] += tri[k]; });
    for (let i = 0; i < n; i++) {
      w2[i] *= Math.pow(s2t[i] / Math.max(m2[i], 1e-9), 0.7);
      w3[i] *= Math.pow(s3t[i] / Math.max(m3[i], 1e-9), 0.7);
    }
    tri = build();
  }
  const tot = tri.reduce((a, b) => a + b, 0);
  return perms.map((p, k) => ({ idx: p, p: tri[k] / tot })).sort((x, y) => y.p - x.p);
}

// 3連単の確率から 2連単・3連複・2連複・拡連複・単勝・複勝 を作る（艇番で表す）
export function derived(tri, boats) {
  const ex = {}; const trio = {}; const qu = {}; const wide = {}; const win = {}; const place = {};
  const add = (o, k, p) => { o[k] = (o[k] || 0) + p; };
  for (const t of tri) {
    const [a, b, c] = t.idx.map((i) => boats[i].waku);
    add(ex, `${a}-${b}`, t.p);
    add(trio, [a, b, c].sort().join('='), t.p);
    add(qu, [a, b].sort().join('='), t.p);
    for (const [x, y] of [[a, b], [a, c], [b, c]]) add(wide, [x, y].sort().join('='), t.p);
    add(win, String(a), t.p);
    add(place, String(a), t.p); add(place, String(b), t.p);
  }
  const sorted = (o) => Object.entries(o).map(([k, p]) => ({ combo: k, p })).sort((x, y) => y.p - x.p);
  return { exacta: sorted(ex), trio: sorted(trio), quinella: sorted(qu), wide: sorted(wide), win: sorted(win), place: sorted(place) };
}

// ------------------------------------------------------------------ 3連単の確率（並び順を固定した配列）
function permsOf(n) {
  const out = [];
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (b !== a) for (let c = 0; c < n; c++) if (c !== a && c !== b) out.push([a, b, c]);
  return out;
}
function triArray(pp) { // 1着・2着内・3着内 → 3連単（固定の並び）
  const order = trifecta(pp);
  const n = pp.win.length; const perms = permsOf(n);
  const pos = new Map(perms.map((p, k) => [p.join(','), k]));
  const out = new Array(perms.length).fill(0);
  for (const t of order) out[pos.get(t.idx.join(','))] = t.p;
  return out;
}
function plArray(p1) { // 1着の確率だけから（Plackett-Luce）
  const perms = permsOf(p1.length);
  const out = perms.map(([a, b, c]) => p1[a] * p1[b] / Math.max(1 - p1[a], 1e-12) * p1[c] / Math.max(1 - p1[a] - p1[b], 1e-12));
  const s = out.reduce((x, y) => x + y, 0);
  return out.map((v) => v / s);
}
// 条件付きロジット: 標準化した特徴量の重み付きの和（欠けている値は平均扱い＋欠けている印）
export function clogitStrength(m, Xfull) {
  const map = m.featureMap; const F = map.length;
  return Xfull.map((x) => {
    let s = 0;
    for (let k = 0; k < F; k++) { const v = x[map[k]]; if (Number.isFinite(v)) s += m.w[k] * (v - m.mu[k]) / m.sd[k]; }
    for (let j = 0; j < m.miss.length; j++) { const v = x[map[m.miss[j]]]; if (!Number.isFinite(v)) s += m.w[F + j]; }
    return s;
  });
}
function softmax(s) { const mx = Math.max(...s); const e = s.map((v) => Math.exp(v - mx)); const t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); }

// 3連単の配列から各艇の 1着・2着内・3着内
function placeFromTri(tri, n) {
  const perms = permsOf(n);
  const win = new Array(n).fill(0), top2 = new Array(n).fill(0), top3 = new Array(n).fill(0);
  perms.forEach(([a, b, c], k) => { const p = tri[k]; win[a] += p; top2[a] += p; top2[b] += p; top3[a] += p; top3[b] += p; top3[c] += p; });
  return { win, top2, top3 };
}

// 締切前のオッズ（3連単）と混ぜる: 予想^α × 市場^(1-α)
export function blendOdds(tri, boats, tOdds, alpha) {
  const perms = permsOf(boats.length);
  const q = perms.map(([a, b, c]) => {
    const o = tOdds[`${boats[a].waku}-${boats[b].waku}-${boats[c].waku}`];
    return o && o > 0 ? 1 / o : 0;
  });
  const sq = q.reduce((x, y) => x + y, 0);
  if (!(sq > 0) || q.filter((v) => v > 0).length < perms.length * 0.9) return null;
  const out = tri.map((p, k) => Math.pow(Math.max(p, 1e-12), alpha) * Math.pow(Math.max(q[k] / sq, 1e-12), 1 - alpha));
  const s = out.reduce((x, y) => x + y, 0);
  return out.map((v) => v / s);
}

// 3連単のオッズ（data/odds と同じ並び: 1〜6 の順列の辞書順 120通り）→ {'1-2-3': 倍率}
export function oddsMap(t) {
  if (!t) return null;
  const out = {}; let k = 0;
  for (let a = 1; a <= 6; a++) for (let b = 1; b <= 6; b++) if (b !== a) for (let c = 1; c <= 6; c++) if (c !== a && c !== b) { out[`${a}-${b}-${c}`] = t[k]; k++; }
  return out;
}

// 2連単のオッズ（1〜6 の順列の辞書順 30通り）→ {'1-2': 倍率}
export function exactaMap(e) {
  if (!e) return null;
  const out = {}; let k = 0;
  for (let a = 1; a <= 6; a++) for (let b = 1; b <= 6; b++) if (b !== a) { out[`${a}-${b}`] = e[k]; k++; }
  return out;
}

// 期待値のある買い目: 予想（オッズを混ぜる前の base）に締切前オッズを割合 alpha で混ぜ、
// 期待値＝確率×オッズ が rule.t 以上・確率 rule.minp 以上・rule.maxo 倍以下のものを、期待値の高い順に rule.k 点まで
//   kind: 'trifecta'（3連単のオッズ）/ 'exacta'（2連単のオッズ）
export function evPicks(pr, race, rule, kind) {
  if (!pr || !pr.base || !race.odds) return null;
  const om = kind === 'exacta' ? exactaMap(race.odds.e) : oddsMap(race.odds.t);
  if (!om) return null;
  const alpha = rule.alpha ?? 1;
  const boats = pr.boats;
  const tri = alpha < 1 && race.odds.t ? (blendOdds(pr.base, boats, oddsMap(race.odds.t), alpha) || pr.base) : pr.base;
  const perms = permsOf(boats.length);
  const P = {};
  perms.forEach(([a, b, c], k) => {
    const key = kind === 'exacta' ? `${boats[a].waku}-${boats[b].waku}` : `${boats[a].waku}-${boats[b].waku}-${boats[c].waku}`;
    P[key] = (P[key] || 0) + tri[k];
  });
  // 確率の補正（γ乗して合計1に）: research_exacta.py と同じ
  if (rule.gamma && rule.gamma !== 1) {
    let z = 0; for (const k in P) { P[k] = Math.pow(Math.max(P[k], 1e-15), rule.gamma); z += P[k]; }
    for (const k in P) P[k] /= z;
  }
  // 市場の確率（オッズの逆数を合計1に）: 予想がそれより何倍高いか（edge）
  let zi = 0; for (const k in om) if (om[k] > 0) zi += 1 / om[k];
  const all = Object.entries(P).map(([combo, p]) => { const o = om[combo]; return { combo, p, odds: o, ev: o ? p * o : 0, edge: o > 0 && zi > 0 ? p / (1 / o / zi) : 0 }; })
    .filter((x) => x.odds > 0).sort((x, y) => y.ev - x.ev);
  // 期待値の高い順に k 点の中から、条件に合うものだけ（research_exacta.py と同じ選び方）
  const picks = all.slice(0, rule.k || 3).filter((x) => x.ev >= rule.t && x.p >= (rule.minp || 0) && x.odds <= (rule.maxo || 1e9) && x.odds > (rule.mino || 0) && x.edge >= (rule.edge || 0));
  return { picks, top: all.slice(0, 5), alpha };
}

// 買い目をフォーメーションの書き方にまとめる（例: 1-3-2,4,5 → "1-3-245"）。まとめられない分はそのまま
export function formation(combos) {
  const parts = combos.map((c) => c.split('-'));
  if (!parts.length) return [];
  const n = parts[0].length;
  // 最後の着を束ねる → 1つ前の着を束ねる → …
  let groups = parts.map((p) => p.map((x) => [x]));
  for (let pos = n - 1; pos >= 0; pos--) {
    const m = new Map();
    for (const g of groups) {
      const key = g.map((s, i) => (i === pos ? '*' : s.join(''))).join('|');
      if (!m.has(key)) m.set(key, g.map((s) => s.slice()));
      else { const t = m.get(key); for (const x of g[pos]) if (!t[pos].includes(x)) t[pos].push(x); }
    }
    groups = [...m.values()];
  }
  // 束ねた結果が元の買い目と同じ点数のときだけ使う（同じ艇が重なる組み合わせを避ける）
  const count = (g) => { let k = 0; const rec = (i, used) => { if (i === g.length) { k++; return; } for (const x of g[i]) if (!used.includes(x)) rec(i + 1, [...used, x]); }; rec(0, []); return k; };
  const total = groups.reduce((a, g) => a + count(g), 0);
  if (total !== combos.length) return combos;
  return groups.map((g) => g.map((s) => s.sort().join('')).join('-'));
}

// ------------------------------------------------------------------ 展開モデル
export const TECH_NAMES = ['', '逃げ', '差し', 'まくり', 'まくり差し', '抜き・恵まれ'];
function rawMulti(m, x) {
  const K = m.base.length; const s = m.base.slice();
  for (const it of m.trees) for (let k = 0; k < K; k++) s[k] += evalTree(it[k], x);
  return s;
}
// 戻り値: tri（3連単・固定の並び）, S[a][t]（a が決まり手 t で勝つ確率）, p2[a][t][b], p3[a][b][c]
export function tenkai(T, X) {
  const n = X.length;
  const S = X.map((x) => { const r = rawMulti(T.S, pick(x, T.S.featureMap)); const mx = Math.max(...r); const e = r.map((v) => Math.exp(v - mx)); const z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); });
  let tot = 0;
  for (let a = 0; a < n; a++) { S[a][0] = 0; for (let t = 1; t < 6; t++) tot += S[a][t]; }
  for (let a = 0; a < n; a++) for (let t = 0; t < 6; t++) S[a][t] /= tot;
  const p2 = [];
  for (let a = 0; a < n; a++) {
    p2[a] = [];
    for (let t = 1; t < 6; t++) {
      const v = new Array(n).fill(0); let z = 0;
      for (let b = 0; b < n; b++) if (b !== a) { v[b] = sigmoid(rawScore(T.N2, pairVec(T.N2.spec, X[b], X[a], null, t))); z += v[b]; }
      p2[a][t] = v.map((x) => x / z);
    }
  }
  const p3 = [];
  for (let a = 0; a < n; a++) {
    p3[a] = [];
    for (let b = 0; b < n; b++) {
      if (b === a) continue;
      const v = new Array(n).fill(0); let z = 0;
      for (let c = 0; c < n; c++) if (c !== a && c !== b) { v[c] = sigmoid(rawScore(T.N3, pairVec(T.N3.spec, X[c], X[a], X[b], 0))); z += v[c]; }
      p3[a][b] = v.map((x) => x / z);
    }
  }
  const tri = permsOf(n).map(([a, b, c]) => { let s2 = 0; for (let t = 1; t < 6; t++) s2 += S[a][t] * p2[a][t][b]; return s2 * p3[a][b][c]; });
  const z = tri.reduce((x, y) => x + y, 0);
  return { tri: tri.map((v) => v / z), S, p2, p3 };
}

// いくつかの3連単の確率を重み ws で掛け合わせる（重み付き幾何平均）
function geoMulti(ps, ws) {
  const out = ps[0].map((_, k) => { let l = 0; for (let i = 0; i < ps.length; i++) l += ws[i] * Math.log(Math.max(ps[i][k], 1e-15)); return l; });
  const mx = Math.max(...out);
  const e = out.map((v) => Math.exp(v - mx));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

// 2つの3連単の確率を重み w で掛け合わせる（幾何平均）
function geoMix(t1, t2, w) {
  const out = t1.map((p, k) => Math.exp((1 - w) * Math.log(Math.max(p, 1e-15)) + w * Math.log(Math.max(t2[k], 1e-15))));
  const z = out.reduce((a, b) => a + b, 0);
  return out.map((v) => v / z);
}

// まとめ: レース → 予想
//   直前情報あり: 3つの平均（機械学習・新しい日重視・条件付きロジット）。締切前のオッズがあれば混ぜる
//   展示前: 展示前のモデル
export function predictRace(race, model, hist, opts = {}) {
  const { boats, X, course } = buildFeatures(race, hist, model.vc);
  if (boats.length < 3) return null;
  const n = boats.length;
  const usePv = race.preview !== 'none';
  let tri;
  const parts = [];
  let scen = null; let exhibit = null;
  let betTri = null;
  if (usePv && model.mix) {
    // 回収率重視: いくつかのモデルを重み付きで掛け合わせる（オッズは混ぜない）。2連単の買い目はこの確率で選ぶ
    const tks = {};
    const get = (key) => {
      if (key.startsWith('tenkai')) return (tks[key] = tks[key] || tenkai(model[key], X)).tri;
      if (key === 'clogit') return plArray(softmax(clogitStrength(model.clogit, X)));
      return triArray(placeProbs(model[key], X));
    };
    const keys = Object.keys(model.mix);
    const ps = keys.map(get);
    tri = geoMulti(ps, keys.map((k) => model.mix[k]));
    betTri = tri;
    scen = tks.tenkai_big || tks.tenkai_xl || tks.tenkai_rw || tks.tenkai || (model.tenkai ? tenkai(model.tenkai, X) : null);
    if (model.ce) {
      const pce = placeProbs(model.ce, X);
      exhibit = pce.win;
      const ew = opts.exhibitW ?? 0;
      if (ew) tri = geoMix(tri, triArray(pce), ew);
    }
  } else if (usePv && model.tenkai) {
    // 展開モデル（＋必要なら3つの平均）に、枠＋展示航走だけのモデルを重み w で混ぜる
    const tk = tenkai(model.tenkai, X);
    scen = tk;
    tri = tk.tri;
    if (model.mix_ens && model.ensemble) {
      const ens = [];
      for (const key of model.ensemble) {
        const m = model[key]; if (!m) continue;
        ens.push(key === 'clogit' ? plArray(softmax(clogitStrength(m, X))) : triArray(placeProbs(m, X)));
      }
      const e = ens[0].map((_, k) => ens.reduce((a, p) => a + p[k], 0) / ens.length);
      tri = tri.map((p, k) => (1 - model.mix_ens) * p + model.mix_ens * e[k]);
    }
    if (model.ce) {
      const pce = placeProbs(model.ce, X);
      exhibit = pce.win;
      const ew = opts.exhibitW ?? model.exhibit_w;
      if (ew) tri = geoMix(tri, triArray(pce), ew);
    }
  } else if (usePv && model.ensemble) {
    for (const key of model.ensemble) {
      const m = model[key];
      if (!m) continue;
      if (key === 'clogit') parts.push(plArray(softmax(clogitStrength(m, X))));
      else parts.push(triArray(placeProbs(m, X)));
    }
    tri = parts[0].map((_, k) => parts.reduce((a, p) => a + p[k], 0) / parts.length);
  } else {
    tri = triArray(placeProbs(usePv ? model.full : (model.pre || model.full), X));
  }
  const base = betTri || tri;
  let withOdds = false;
  if (usePv && race.odds && race.odds.t && opts.blend !== false && (model.blend_alpha ?? 0.5) < 1) {
    const b = blendOdds(tri, boats, oddsMap(race.odds.t), model.blend_alpha ?? 0.5);
    if (b) { tri = b; withOdds = true; }
  }
  const perms = permsOf(n);
  const raw = perms.map((p, k) => ({ idx: p, p: tri[k] })).sort((x, y) => y.p - x.p);
  const trifectaList = raw.map((t) => ({ combo: t.idx.map((i) => boats[i].waku).join('-'), p: t.p }));
  const der = derived(raw, boats);
  // 展開シナリオ（誰が・どの決まり手で勝つか）と、その展開での本線（2・3着）
  let scenarios = null;
  if (scen) {
    const list = [];
    for (let a = 0; a < n; a++) for (let t = 1; t < 6; t++) {
      const p = scen.S[a][t];
      if (p < 0.02) continue;
      const follow = [];
      for (let b = 0; b < n; b++) if (b !== a) follow.push({ waku: boats[b].waku, combo: `${boats[a].waku}-${boats[b].waku}`, p: scen.p2[a][t][b] });
      follow.sort((x, y) => y.p - x.p);
      list.push({ waku: boats[a].waku, course: course[a], tech: t, name: TECH_NAMES[t], p, follow: follow.slice(0, 3) });
    }
    scenarios = list.sort((x, y) => y.p - x.p);
  }
  return { boats, course, X, probs: placeFromTri(tri, n), trifecta: trifectaList, ...der, withPreview: usePv, withOdds, scenarios, exhibit, base };
}
