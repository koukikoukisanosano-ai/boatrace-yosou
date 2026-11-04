#!/usr/bin/env bash
# 学習し直し: 過去データ → 学習用の表（直前情報あり・展示前）→ 検証 → 全期間で学習 → docs/model/model.json
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WORK:-work}
mkdir -p "$W/full" "$W/pre"
node tools/build_dataset.mjs "$W/full"
node tools/build_dataset.mjs "$W/pre" --no-preview
python tools/train_all.py --work "$W"
ls -la docs/model/
