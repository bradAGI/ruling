#!/bin/zsh
# Start one open decision model's own /v1/systemone server, as its publisher (or its MLX port) ships it.
#   assets/bench/serve.sh pplx        # mlx-community/pplx-decider-v1.1-27b-4bit, port 8092
#   assets/bench/serve.sh decider-2b  # Mapika/decider-2b through decider-ai, port 8093
#   assets/bench/serve.sh decider-4b  # Mapika/decider-4b through decider-ai, port 8093
#   assets/bench/serve.sh bev         # avbiswas/bev-decider-0.4B through bev-decider, port 8094
# Each model gets its own pinned virtualenv under $ENVS (default runs/bench/envs), separate from ruling's.
set -eu
cd "$(dirname "$0")/../.."
ENVS=${ENVS:-runs/bench/envs}
snapshot() { uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))"; }
venv() { [ -x "$ENVS/$1/bin/python" ] || { uv venv --python 3.12 "$ENVS/$1"; VIRTUAL_ENV=$ENVS/$1 uv pip install "${@:2}"; } }
case "$1" in
  pplx)
    SNAP=$(snapshot mlx-community/pplx-decider-v1.1-27b-4bit)
    venv pplx -r "$SNAP/requirements.txt"
    # the snapshot is symlinks, so the script directory resolves to the blob store: name the import path
    PYTHONPATH=$SNAP exec "$ENVS/pplx/bin/python" "$SNAP/server.py" --model "$SNAP" --port "${2:-8092}" ;;
  decider-2b|decider-4b)
    SNAP=$(snapshot "Mapika/$1")
    venv decider "decider-ai[serve]==1.9.0"
    DECIDER_MODEL=$SNAP exec "$ENVS/decider/bin/uvicorn" decider.serve:app --host 127.0.0.1 --port "${2:-8093}" ;;
  bev)
    venv bev "bev-decider[serve]==0.2.1"
    exec "$ENVS/bev/bin/bev-decider" serve --port "${2:-8094}" ;;
  *) echo "unknown model $1" >&2; exit 2 ;;
esac
