#!/bin/bash
# GCE の GPU VM で R-NaD（DeepNash 路線 M2b）を常駐させるセットアップ（冪等）。VM 上で実行する。
# 設計は docs/rl-deepnash-design.md、運用は docs/arena-ops.md の「長時間ランは GCE で回す」。
#
# 前提（手元から gcloud compute scp で /tmp/ へ送っておく）:
#   /tmp/tsuitate-bot.tar.gz   … tsuitate-bot（target/ .git/ records/ を除く。setup-tune.sh と同じ作り方）
#   /tmp/tsuitate-nn-rnad.tar.gz … tsuitate-nn の rnad/ と out_rnad/bc/checkpoint.pt（初期化と錨）
#   VM は NVIDIA ドライバ入りのイメージ（Deep Learning VM など）で、nvidia-smi が通ること
#
# 使い方:
#   bash setup-rnad.sh <サービス名> "<train_rnad.py の引数>"
# 例:
#   bash setup-rnad.sh rnad "--init out_rnad/bc/checkpoint.pt --out out_rnad/rnad_gpu --steps 3000 --batch 256"
#
# Spot で止められたら instances start するだけ: systemd が train_rnad.py を起こし直し、
# <out>/checkpoint.pt から続きを再開する（--checkpoint-every ごとに書かれる）。
# **完走したら VM を自分で停止する**（GPU の課金を放置で積まないため。停止中はディスク代だけ）。
# 止めたくないときは AUTO_POWEROFF=0 bash setup-rnad.sh ... で入れる。
# 完走すると終了コード 0 で止まる（Restart=on-failure なので再起動ループにはならない）。
set -euo pipefail

SERVICE="$1"
TRAIN_ARGS="$2"
AUTO_POWEROFF="${AUTO_POWEROFF:-1}"
# 完走したら印を残し、以後の起動では走らせない（ConditionPathExists）。印が無いと、結果を
# 回収しようと VM を起こしたとき「完走済み → 即終了 → 自動停止」で落ちて SSH できない
# （2026-09-28 に実際に起きた。L4 の在庫切れと重なり、ディスクのスナップショット経由で回収した）
DONE_MARK="$HOME/.rnad-${SERVICE}.done"
rm -f "$DONE_MARK"
if [ "$AUTO_POWEROFF" = "1" ]; then
  # 完走（終了コード 0）のときだけ停止する。失敗は Restart=on-failure で起こし直す
  EXEC_START="/bin/bash -c '$HOME/tsuitate-nn/.venv/bin/python rnad/train_rnad.py ${TRAIN_ARGS} && touch ${DONE_MARK} && sudo /sbin/poweroff'"
else
  EXEC_START="/bin/bash -c '$HOME/tsuitate-nn/.venv/bin/python rnad/train_rnad.py ${TRAIN_ARGS} && touch ${DONE_MARK}'"
fi

nvidia-smi > /dev/null || { echo "nvidia-smi が通らない（GPU ドライバ未導入）" >&2; exit 1; }

sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl pkg-config libssl-dev python3-venv > /dev/null

if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

cd "$HOME"
rm -rf tsuitate-bot
tar xzf /tmp/tsuitate-bot.tar.gz
mkdir -p tsuitate-nn
# 学習の出力（out_rnad/ の checkpoint・log）は消さずに残す = 送り直しても再開できる
tar xzf /tmp/tsuitate-nn-rnad.tar.gz -C tsuitate-nn

cd "$HOME/tsuitate-nn"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
# torch の CUDA 版（ドライバはイメージのもの）
.venv/bin/pip install -q torch numpy maturin
VIRTUAL_ENV="$HOME/tsuitate-nn/.venv" .venv/bin/maturin develop --release \
  -m "$HOME/tsuitate-bot/rl-env/Cargo.toml" 2>&1 | tail -1
.venv/bin/python -c "import torch, tsuitate_rl; assert torch.cuda.is_available(), 'CUDA が見えない'; print('cuda', torch.cuda.get_device_name(0))"
# 参照実装との一致（環境が変わっても R-NaD の計算が同じであること）
.venv/bin/python rnad/tests/test_rnad.py

sudo tee "/etc/systemd/system/${SERVICE}.service" > /dev/null <<EOF
[Unit]
Description=tsuitate R-NaD ${SERVICE}
After=network.target
ConditionPathExists=!${DONE_MARK}

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/tsuitate-nn
Environment=PYTHONUNBUFFERED=1
ExecStart=${EXEC_START}
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE}.service"
sudo systemctl restart "${SERVICE}.service"
echo "SETUP_DONE ${SERVICE}"
