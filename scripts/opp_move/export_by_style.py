"""arena / 対人の記録を相手の系統ごとに分けて export_opp_move_data にかけ、1つの CSV にまとめる。

    python3 scripts/opp_move/export_by_style.py OUT.csv DIR [DIR ...]

先に `cargo build --release --bin export_opp_move_data` しておく。出力は tsuitate-nn の
train_opp_move_weighted.py と scripts/opp_move_eval.py が読む（docs/nn-direction.md の「opp_move NN の学び直し」）。

系統は記録の meta 行の baseline（arena）。meta が無い記録（対人）は "human"。
game_id は CSV ごとに 0 から振られるので、まとめるときに通し番号へずらす。列 style を足す。
"""

import csv
import io
import json
import pathlib
import subprocess
import sys
from collections import defaultdict

BIN = str(pathlib.Path(__file__).resolve().parents[2] / "target" / "release" / "export_opp_move_data")


def style_of(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "meta":
            return r.get("baseline") or "unknown"
        if r.get("type") not in ("match", "meta"):
            break
    return "human"


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    groups = defaultdict(list)
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("*.jsonl")):
            groups[style_of(p)].append(str(p))
    offset = 0
    with open(out, "w", newline="") as f:
        w = None
        for style in sorted(groups):
            res = subprocess.run([BIN, *groups[style]], capture_output=True, text=True, check=True)
            rows = list(csv.reader(io.StringIO(res.stdout)))
            head, body = rows[0], rows[1:]
            if w is None:
                w = csv.writer(f)
                w.writerow(head + ["style"])
            gmax = -1
            for r in body:
                g = int(r[0])
                gmax = max(gmax, g)
                w.writerow([g + offset] + r[1:] + [style])
            print(f"{style:20s} {len(groups[style]):4d} 局 → {gmax + 1:4d} 局 / {len(body):7d} 行", file=sys.stderr)
            offset += gmax + 1


if __name__ == "__main__":
    main()
