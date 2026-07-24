<script lang="ts">
  import type { Color, Role, Snapshot } from "./api";

  let {
    snapshot,
    flipped = false,
  }: {
    snapshot: Snapshot;
    flipped?: boolean;
  } = $props();

  const KANJI: Record<Role, string> = {
    pawn: "歩",
    lance: "香",
    knight: "桂",
    silver: "銀",
    gold: "金",
    bishop: "角",
    rook: "飛",
    king: "玉",
    tokin: "と",
    promotedlance: "杏",
    promotedknight: "圭",
    promotedsilver: "全",
    horse: "馬",
    dragon: "龍",
  };
  const PROMOTED = new Set<Role>([
    "tokin",
    "promotedlance",
    "promotedknight",
    "promotedsilver",
    "horse",
    "dragon",
  ]);
  const HAND_ORDER: Role[] = ["rook", "bishop", "gold", "silver", "knight", "lance", "pawn"];
  const RANKS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];

  // 下側に据える手番（既定 = 先手視点）
  const bottom: Color = $derived(flipped ? "gote" : "sente");
  const files = $derived(flipped ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : [9, 8, 7, 6, 5, 4, 3, 2, 1]);
  const ranks = $derived(flipped ? [...RANKS].reverse() : RANKS);

  function handText(hand: Partial<Record<Role, number>>): string {
    const parts: string[] = [];
    for (const role of HAND_ORDER) {
      const n = hand[role] ?? 0;
      if (n > 0) parts.push(KANJI[role] + (n > 1 ? `${n}` : ""));
    }
    return parts.length > 0 ? parts.join(" ") : "なし";
  }

  const topColor: Color = $derived(bottom === "sente" ? "gote" : "sente");
  const topHand = $derived(topColor === "sente" ? snapshot.handSente : snapshot.handGote);
  const bottomHand = $derived(bottom === "sente" ? snapshot.handSente : snapshot.handGote);

  function sideLabel(c: Color): string {
    const mark = c === "sente" ? "▲先手" : "△後手";
    const turn = snapshot.turn === c ? "（手番）" : "";
    const check = snapshot.inCheck[c === "sente" ? 0 : 1] ? " 王手!" : "";
    return `${mark}${turn}${check}`;
  }
</script>

<div class="board-wrap">
  <div class="hand" class:in-turn={snapshot.turn === topColor}>
    <span class="hand-label">{sideLabel(topColor)}</span>
    <span class="hand-pieces rotated">{handText(topHand)}</span>
  </div>

  <div class="board-grid-outer">
    <div class="file-labels">
      {#each files as f (f)}
        <span>{f}</span>
      {/each}
    </div>
    <div class="board-with-ranks">
      <div class="board-grid">
        {#each ranks as r (r)}
          {#each files as f (`${f}${r}`)}
            {@const sq = `${f}${r}`}
            {@const piece = snapshot.board[sq]}
            {@const isLast =
              snapshot.lastMove != null &&
              (snapshot.lastMove.to === sq || snapshot.lastMove.from === sq)}
            <div class="cell" class:last-move={isLast} title={sq}>
              {#if piece}
                <span
                  class="piece"
                  class:rotated={piece.color !== bottom}
                  class:promoted={PROMOTED.has(piece.role)}
                >
                  {KANJI[piece.role]}
                </span>
              {/if}
            </div>
          {/each}
        {/each}
      </div>
      <div class="rank-labels">
        {#each ranks as r (r)}
          <span>{r}</span>
        {/each}
      </div>
    </div>
  </div>

  <div class="hand" class:in-turn={snapshot.turn === bottom}>
    <span class="hand-label">{sideLabel(bottom)}</span>
    <span class="hand-pieces">{handText(bottomHand)}</span>
  </div>
</div>

<style>
  .board-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: fit-content;
  }

  .hand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 8px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    min-height: 30px;
  }

  .hand.in-turn {
    border-color: var(--accent);
  }

  .hand-label {
    color: var(--text-dim);
    white-space: nowrap;
  }

  .hand-pieces {
    font-size: 16px;
    letter-spacing: 2px;
  }

  .file-labels {
    display: grid;
    grid-template-columns: repeat(9, 46px);
    justify-items: center;
    color: var(--text-dim);
    font-size: 11px;
  }

  .board-with-ranks {
    display: flex;
    gap: 4px;
  }

  .rank-labels {
    display: grid;
    grid-template-rows: repeat(9, 46px);
    align-items: center;
    color: var(--text-dim);
    font-size: 11px;
  }

  .board-grid {
    display: grid;
    grid-template-columns: repeat(9, 46px);
    grid-template-rows: repeat(9, 46px);
    background: var(--board);
    border: 2px solid var(--board-line);
    width: fit-content;
  }

  .cell {
    border: 1px solid var(--board-line);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .cell.last-move {
    background: rgba(255, 214, 106, 0.45);
  }

  .piece {
    font-size: 26px;
    color: #1c1206;
    line-height: 1;
    user-select: none;
  }

  .piece.rotated,
  .hand-pieces.rotated {
    display: inline-block;
    transform: rotate(180deg);
  }

  .piece.promoted {
    color: #a01515;
  }
</style>
