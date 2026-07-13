#!/bin/bash
# GCEチューニングVMのセットアップ（冪等）。VM上で実行する。
# 前提: /tmp/tsuitate-bot.tar.gz にコードが転送済み（CLAUDE.md の手順参照）
#
# 使い方:
#   bash setup-tune.sh <サービス名> <ARENA_THREADS> <tune引数...> [-- ENV=VALUE ...]
# 例:
#   bash setup-tune.sh tune 14 "60 60 estimator_v5"
#   bash setup-tune.sh tune-rush 7 "40 40 estimator_v5" -- TUNE_LOG=tune-rush.jsonl "TUNE_CANDIDATE_LINE=居飛車速攻"
set -euo pipefail

SERVICE="$1"
THREADS="$2"
TUNE_ARGS="$3"
shift 3
EXTRA_ENV=()
if [ "${1:-}" = "--" ]; then
  shift
  EXTRA_ENV=("$@")
fi

sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl pkg-config libssl-dev > /dev/null

if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

cd "$HOME"
rm -rf tsuitate-bot
tar xzf /tmp/tsuitate-bot.tar.gz
cd tsuitate-bot
cargo build --release --bin tune 2>&1 | tail -1

ENV_LINES=""
for kv in "${EXTRA_ENV[@]}"; do
  ENV_LINES+="Environment=${kv}
"
done

# Spot停止→再起動後も自動で続きから回るよう systemd 常駐にする
# （tune は TUNE_LOG から再開する。完走後は「既に完了」で即終了ループになるので
#   回収が済んだら systemctl disable --now すること）
sudo tee "/etc/systemd/system/${SERVICE}.service" > /dev/null <<EOF
[Unit]
Description=tsuitate SPSA ${SERVICE}
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/tsuitate-bot
Environment=ARENA_THREADS=${THREADS}
${ENV_LINES}ExecStart=$HOME/tsuitate-bot/target/release/tune ${TUNE_ARGS}
Restart=on-success
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE}.service"
sudo systemctl restart "${SERVICE}.service"
echo "SETUP_DONE ${SERVICE}"
