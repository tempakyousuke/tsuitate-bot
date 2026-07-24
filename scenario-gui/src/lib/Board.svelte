<script lang="ts">
  import type { Color, Role, Snapshot } from "./api";

  let {
    snapshot,
    flipped = false,
    // ---- 対局モード用（省略時はリプレイと同じ表示専用盤） ----
    selected = null, // 選択中のマス
    targets = [], // 移動候補マス（ハイライト）
    danger = null, // 直前に自駒が取られたマス
    onCellClick = null,
    handSide = null, // 持ち駒をクリックできる側
    selectedHandRole = null,
    onHandClick = null,
  }: {
    snapshot: Snapshot;
    flipped?: boolean;
    selected?: string | null;
    targets?: string[];
    danger?: string | null;
    onCellClick?: ((sq: string) => void) | null;
    handSide?: Color | null;
    selectedHandRole?: Role | null;
    onHandClick?: ((role: Role) => void) | null;
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
  const targetSet = $derived(new Set(targets));

  function handEntries(hand: Partial<Record<Role, number>>): { role: Role; n: number }[] {
    return HAND_ORDER.filter((r) => (hand[r] ?? 0) > 0).map((r) => ({ role: r, n: hand[r]! }));
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

{#snippet hand(color: Color, rotated: boolean, pieces: Partial<Record<Role, number>>)}
  <div class="hand" class:in-turn={snapshot.turn === color}>
    <span class="hand-label">{sideLabel(color)}</span>
    <span class="hand-pieces" class:rotated>
      {#each handEntries(pieces) as h (h.role)}
        {#if handSide === color && onHandClick}
          <button
            class="hand-piece"
            class:selected={selectedHandRole === h.role}
            onclick={() => onHandClick?.(h.role)}
          >
            {KANJI[h.role]}{h.n > 1 ? h.n : ""}
          </button>
        {:else}
          <span class="hand-piece-static">{KANJI[h.role]}{h.n > 1 ? h.n : ""}</span>
        {/if}
      {:else}
        なし
      {/each}
    </span>
  </div>
{/snippet}

<div class="board-wrap">
  {@render hand(topColor, true, topHand)}

  <div class="board-grid-outer">
    <div class="file-labels">
      {#each files as f (f)}
        <span>{f}</span>
      {/each}
    </div>
    <div class="board-with-ranks">
      <div class="board-grid" class:clickable={onCellClick != null}>
        {#each ranks as r (r)}
          {#each files as f (`${f}${r}`)}
            {@const sq = `${f}${r}`}
            {@const piece = snapshot.board[sq]}
            {@const isLast =
              snapshot.lastMove != null &&
              (snapshot.lastMove.to === sq || snapshot.lastMove.from === sq)}
            <div
              class="cell"
              class:last-move={isLast}
              class:selected={selected === sq}
              class:target={targetSet.has(sq)}
              class:danger={danger === sq}
              title={sq}
              role="button"
              tabindex={onCellClick ? 0 : -1}
              onclick={() => onCellClick?.(sq)}
              onkeydown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") onCellClick?.(sq);
              }}
            >
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

  {@render hand(bottom, false, bottomHand)}
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
    display: flex;
    gap: 4px;
    align-items: center;
  }

  .hand-piece-static {
    display: inline-block;
  }

  button.hand-piece {
    font-size: 16px;
    padding: 1px 6px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
    color: inherit;
  }

  button.hand-piece.selected {
    border-color: var(--accent);
    outline: 1px solid var(--accent);
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

  .board-grid.clickable .cell {
    cursor: pointer;
  }

  .cell.last-move {
    background: rgba(255, 214, 106, 0.45);
  }

  .cell.selected {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }

  .cell.target {
    background: rgba(106, 169, 255, 0.35);
  }

  .cell.danger {
    background: rgba(255, 90, 90, 0.45);
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

  .hand-pieces.rotated {
    display: flex;
  }

  .piece.promoted {
    color: #a01515;
  }
</style>
