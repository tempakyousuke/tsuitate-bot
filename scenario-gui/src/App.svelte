<script lang="ts">
  import { onMount } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import Board from "./lib/Board.svelte";
  import MoveList from "./lib/MoveList.svelte";
  import AnalysisPanel from "./lib/AnalysisPanel.svelte";
  import PlayPanel from "./lib/PlayPanel.svelte";
  import {
    getEngines,
    listScenarios,
    loadKifu,
    type KifuData,
    type ScenarioInfo,
  } from "./lib/api";

  let mode = $state<"replay" | "play">("replay");
  let scenarios = $state<ScenarioInfo[]>([]);
  let engines = $state<string[]>([]);
  let selectedPath = $state("");
  let kifu = $state<KifuData | null>(null);
  let ply = $state(0);
  let flipped = $state(false);
  let loadError = $state("");

  const snapshot = $derived(kifu ? kifu.snapshots[ply] : null);

  onMount(async () => {
    engines = await getEngines();
    scenarios = await listScenarios();
  });

  async function loadPath(path: string) {
    loadError = "";
    try {
      const data = await loadKifu(path);
      kifu = data;
      selectedPath = path;
      // *scenario ply= があればそこ（=注目手を考えさせる局面）へ、無ければ先頭へ
      ply = data.directivePly != null ? Math.min(data.directivePly, data.totalPlies) : 0;
    } catch (e) {
      loadError = String(e);
      kifu = null;
      selectedPath = "";
    }
  }

  async function openFile() {
    const file = await open({
      multiple: false,
      filters: [{ name: "KIF", extensions: ["kif"] }],
    });
    if (typeof file === "string") await loadPath(file);
  }

  function clampPly(p: number): number {
    return Math.max(0, Math.min(p, kifu?.totalPlies ?? 0));
  }

  // 対局モードで書き出した kif をそのままリプレイで開く
  async function openExported(path: string) {
    mode = "replay";
    scenarios = await listScenarios();
    await loadPath(path);
  }

  function onKeydown(ev: KeyboardEvent) {
    if (mode !== "replay" || kifu == null) return;
    const tag = (ev.target as HTMLElement | null)?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (ev.key === "ArrowLeft") {
      ply = clampPly(ply - 1);
      ev.preventDefault();
    } else if (ev.key === "ArrowRight") {
      ply = clampPly(ply + 1);
      ev.preventDefault();
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

<main>
  <header>
    <nav class="mode-nav">
      <button class:active={mode === "replay"} onclick={() => (mode = "replay")}>
        リプレイ
      </button>
      <button class:active={mode === "play"} onclick={() => (mode = "play")}>対局</button>
    </nav>
    {#if mode === "replay"}
      <label>
        シナリオ
        <select
          value={selectedPath}
          onchange={(ev) => {
            const path = ev.currentTarget.value;
            if (path !== "") void loadPath(path);
          }}
        >
          <option value="" disabled>選択…</option>
          {#each scenarios.filter((s) => !s.archived) as s (s.path)}
            <option value={s.path}>{s.name}（{s.totalPlies}手）</option>
          {/each}
          {#each scenarios.filter((s) => s.archived) as s (s.path)}
            <option value={s.path}>archive/{s.name}（{s.totalPlies}手）</option>
          {/each}
        </select>
      </label>
      <button onclick={openFile}>.kif を開く…</button>
      {#if kifu}
        <span class="kifu-name">{kifu.name}</span>
        {#if kifu.desc}<span class="desc">{kifu.desc}</span>{/if}
      {/if}
    {/if}
  </header>

  <!-- 対局モードは非表示でもマウントしたままにする（進行中の対局状態を保つ） -->
  <div class="tab" style:display={mode === "play" ? "contents" : "none"}>
    <PlayPanel {engines} onOpenKifu={openExported} />
  </div>

  {#if mode === "replay" && loadError !== ""}
    <div class="error">{loadError}</div>
  {/if}

  {#if mode === "replay" && kifu && snapshot}
    <div class="content">
      <section class="left">
        <Board {snapshot} {flipped} />
        <div class="replay">
          <button onclick={() => (ply = 0)} disabled={ply === 0}>|◀</button>
          <button onclick={() => (ply = clampPly(ply - 1))} disabled={ply === 0}>◀</button>
          <input
            type="range"
            min="0"
            max={kifu.totalPlies}
            bind:value={ply}
            style="flex: 1"
          />
          <button onclick={() => (ply = clampPly(ply + 1))} disabled={ply === kifu.totalPlies}>
            ▶
          </button>
          <button onclick={() => (ply = kifu!.totalPlies)} disabled={ply === kifu.totalPlies}>
            ▶|
          </button>
        </div>
        <div class="replay-info">
          <span>
            {ply}手 / 全{kifu.totalPlies}手
            {#if snapshot.lastMove}（直前 {snapshot.lastMove.usi}）{/if}
          </span>
          {#if kifu.directivePly != null}
            <button onclick={() => (ply = clampPly(kifu!.directivePly!))}>
              シナリオ局面へ（ply={kifu.directivePly}）
            </button>
          {/if}
          <button onclick={() => (flipped = !flipped)}>
            視点: {flipped ? "△後手" : "▲先手"}（切替）
          </button>
        </div>
        <div class="replay-info dim">
          反則累計 ▲{snapshot.fouls[0]} △{snapshot.fouls[1]}
          {#if kifu.target}／ 注目手: {kifu.target}{/if}
        </div>
      </section>

      <section class="middle">
        <MoveList
          moves={kifu.moves}
          {ply}
          directivePly={kifu.directivePly}
          target={kifu.target}
          onselect={(p) => (ply = p)}
        />
      </section>

      <section class="right">
        <AnalysisPanel path={kifu.path} {ply} target={kifu.target} {engines} />
      </section>
    </div>
  {:else if mode === "replay"}
    <div class="placeholder">
      シナリオを選ぶか .kif ファイルを開いてください（scenarios/*.kif 形式、
      *illegal: 行・*scenario ディレクティブ対応）
    </div>
  {/if}
</main>

<style>
  main {
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 10px;
    gap: 10px;
  }

  header {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  header label {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-dim);
  }

  .mode-nav {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .mode-nav button {
    border: none;
    border-radius: 0;
    padding: 4px 14px;
  }

  .mode-nav button.active {
    background: var(--accent);
    color: #fff;
  }

  .kifu-name {
    font-weight: 600;
  }

  .desc {
    color: var(--text-dim);
    font-size: 12px;
  }

  .content {
    display: grid;
    grid-template-columns: auto 240px 1fr;
    gap: 12px;
    flex: 1;
    min-height: 0;
  }

  .left {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .middle {
    display: flex;
    min-height: 0;
  }

  .middle :global(.move-list) {
    flex: 1;
  }

  .right {
    display: flex;
    min-height: 0;
  }

  .right :global(.panel) {
    flex: 1;
  }

  .replay {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .replay-info {
    display: flex;
    gap: 10px;
    align-items: center;
    font-size: 13px;
  }

  .replay-info.dim {
    color: var(--text-dim);
  }

  .error {
    color: var(--danger);
    white-space: pre-wrap;
  }

  .placeholder {
    color: var(--text-dim);
    margin: auto;
  }
</style>
