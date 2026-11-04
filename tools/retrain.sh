#!/usr/bin/env bash
# 学習し直し: 過去データ → 学習用の表 → 検証 → 全期間で学習 → docs/model/model.json
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WORK:-work}
mkdir -p "$W/full" "$W/pre"
node tools/build_dataset.mjs "$W/full"
node tools/build_dataset.mjs "$W/pre" --no-preview
python tools/train.py --work "$W/full" --slot full
VF=$(python -c "import json;print(json.load(open('$W/full/train_result.json'))['valid_from'])")
node tools/evaluate.mjs "$W/full/model_valid_full.json" "$VF" --json "$W/full/eval.json"
python tools/train.py --work "$W/pre" --slot pre --valid-from "$VF"
python tools/train.py --work "$W/full" --slot full --valid-from "$VF" --final
python tools/train.py --work "$W/pre" --slot pre --valid-from "$VF" --final
ls -la docs/model/
