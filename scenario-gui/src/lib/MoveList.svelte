<script lang="ts">
  import type { MoveRow } from "./api";

  let {
    moves,
    ply,
    directivePly,
    target,
    onselect,
  }: {
    moves: MoveRow[];
    ply: number;
    directivePly: number | null;
    target: string | null;
    onselect: (ply: number) => void;
  } = $props();

  // 現在の ply の行が見えるように自動スクロール
  let listEl: HTMLDivElement | undefined = $state();
  $effect(() => {
    const el = listEl?.querySelector<HTMLElement>(`[data-ply="${ply}"]`);
    el?.scrollIntoView({ block: "nearest" });
  });
</script>

<div class="move-list" bind:this={listEl}>
  <button
    class="row"
    class:current={ply === 0}
    data-ply="0"
    onclick={() => onselect(0)}
  >
    <span class="no">0</span>
    <span class="usi">開始局面</span>
  </button>
  {#each moves as mv, i (i)}
    {#each mv.foulsBefore as foul, j (j)}
      <div class="row foul" data-ply={i}>
        <span class="no"></span>
        <span class="usi">✗ {foul}</span>
        <span class="tags">反則試行</span>
      </div>
    {/each}
    <button
      class="row"
      class:current={ply === i + 1}
      class:scenario-ply={directivePly != null && i + 1 === directivePly + 1}
      data-ply={i + 1}
      onclick={() => onselect(i + 1)}
    >
      <span class="no">{i + 1}</span>
      <span class="usi">{mv.side === "sente" ? "▲" : "△"}{mv.usi}</span>
      <span class="tags">
        {#if mv.usi === target}<span class="star">★注目手</span>{/if}
        {#if mv.capture}取{/if}
        {#if mv.givesCheck}王手{/if}
      </span>
    </button>
  {/each}
</div>

<style>
  .move-list {
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--panel);
  }

  .row {
    display: grid;
    grid-template-columns: 34px 1fr auto;
    gap: 6px;
    align-items: center;
    padding: 2px 8px;
    border: none;
    border-radius: 0;
    background: transparent;
    text-align: left;
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12.5px;
  }

  button.row:hover {
    background: var(--panel-2);
  }

  .row.current {
    background: rgba(106, 169, 255, 0.22);
  }

  .row.scenario-ply .no {
    color: var(--star);
  }

  .row.foul {
    color: var(--danger);
  }

  .no {
    color: var(--text-dim);
    text-align: right;
  }

  .tags {
    color: var(--text-dim);
    font-size: 11px;
    display: flex;
    gap: 6px;
  }

  .star {
    color: var(--star);
  }
</style>
