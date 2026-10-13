#!/bin/zsh
# Score a /v1/systemone server with ruling's harness on the four held-out sets, then on JevBench's public items
# with JevBench's own typesafe adapter, unmodified. Output: runs/bench/<set>-<name>-r1.json and runs/bench/jevbench-<name>/.
#   assets/bench/score.sh <name> <port>      e.g. assets/bench/score.sh pplx-decider-v1.1-27b 8092
set -u
cd "$(dirname "$0")/../.."
NAME=$1; PORT=$2
export PYTHONUNBUFFERED=1 TYPESAFE_BASE_URL=http://127.0.0.1:$PORT
unset TYPESAFE_API_KEY
JEVBENCH=${JEVBENCH:-runs/bench/jevbench-src}
JEVBENCH_REV=7625ecb1b075cfe4d09d34141d353325825eea1d
typeset -A PATHS
PATHS=(typesafe102 runs/bench/eval/typesafe102.jsonl every runs/bench/eval/every.jsonl
       authored144 datasets/authored144.jsonl perturbations108 datasets/perturbations108.jsonl)
for set in typesafe102 every authored144 perturbations108; do
  [ -s ${PATHS[$set]} ] || { echo "missing ${PATHS[$set]}; see runs/bench/build-dataset.log"; exit 1; }
  out=runs/bench/${set}-${NAME}-r1
  [ -f ${out}.json ] && continue
  echo "=== ${NAME} ${set} $(date +%T)"
  uv run ruling eval ${PATHS[$set]} --model typesafe:${NAME} --output ${out}.json --predictions ${out}.predictions.jsonl || echo "FAILED ${set}"
done
[ -d "$JEVBENCH" ] || git clone -q https://github.com/fstandhartinger/jevbench "$JEVBENCH"
git -C "$JEVBENCH" checkout -q "$JEVBENCH_REV"
RUN=runs/bench/jevbench-${NAME}
mkdir -p "$RUN"
for tier in easy original hard; do
  [ -f "$RUN/$tier.summary.json" ] && continue
  echo "=== ${NAME} jevbench ${tier} $(date +%T)"
  ( cd "$JEVBENCH" && python3 -m jevbench.cli run --tasks datasets/public/$tier.jsonl \
      --adapter typesafe --endpoint "$TYPESAFE_BASE_URL" --key-env '' --model "$NAME" \
      --cost-basis local_server --reserve-usd 0 \
      --results "$OLDPWD/$RUN/$tier.results.jsonl" --raw-dir "$OLDPWD/$RUN/$tier.raw" --ledger "$OLDPWD/$RUN/ledger.jsonl" \
    && python3 -m jevbench.cli summarize --tasks datasets/public/$tier.jsonl \
      --results "$OLDPWD/$RUN/$tier.results.jsonl" --public-export "$OLDPWD/$RUN/$tier.summary.json" ) || echo "FAILED jevbench $tier"
done
echo "=== ${NAME} done $(date +%T)"
