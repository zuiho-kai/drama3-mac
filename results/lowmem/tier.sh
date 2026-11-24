#!/bin/zsh
# usage: tier.sh <id> <knobs-json> [manifest]   -> runs/<id>/{stats.json,quality.json}
set -e
cd ~/work/d3repo/drama3-mac/harness
id=$1; knobs=$2; man=${3:-manifest.json}
echo "{\"id\":\"$id\",\"build\":\"int4\",\"knobs\":$knobs,\"src\":\"src-base\"}" > runs/$id.json
D3_MANIFEST=$man ~/ai-voice/drama3-mac/.venv/bin/python -I bench.py runs/$id.json runs/$id
D3_MANIFEST=$man ~/ai-voice/voice-train/.venv-asr/bin/python -I quality.py runs/$id runs/int4-res > runs/$id/quality.log 2>&1 || D3_MANIFEST=$man ~/ai-voice/voice-train/.venv-asr/bin/python -I quality.py runs/$id >> runs/$id/quality.log 2>&1
python3 -c "import json;print('QUALITY',json.load(open('runs/$id/quality.json'))['summary'])"
