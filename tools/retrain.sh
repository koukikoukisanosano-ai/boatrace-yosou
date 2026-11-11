#!/usr/bin/env bash
# 学習し直し: 過去データ → 学習用の表（直前情報あり・展示前）→ 検証 → 全期間で学習 → docs/model/model.json
#   MBRACE=data/mbrace があれば公式の過去データ（2016〜2024年8月）も読み込み、2020年1月から学習する（公平な検証で決めた形）
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WORK:-work}
mkdir -p "$W/full" "$W/pre"
EXTRA=()
if [ -n "${MBRACE:-}" ] && [ -d "$MBRACE" ]; then
  export TRAIN_FROM=${TRAIN_FROM:-2020-01-01}
  EXTRA=(--extra="$MBRACE" --rows-from="$TRAIN_FROM")
  echo "公式の過去データを使います: $MBRACE（学習は $TRAIN_FROM から）"
fi
node --max-old-space-size=10240 tools/build_dataset.mjs "$W/full" "${EXTRA[@]}"
node --max-old-space-size=10240 tools/build_dataset.mjs "$W/pre" --no-preview "${EXTRA[@]}"
python tools/train_all.py --work "$W"
ls -la docs/model/
