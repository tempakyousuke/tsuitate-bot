<script lang="ts">
  import { onMount } from "svelte";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import {
    cancelEval,
    evalRanking,
    evalTally,
    type ProgressEvent,
    type RankingResult,
    type TallyResult,
    type TrialOutcome,
  } from "./api";

  let {
    path,
    ply,
    target,
    engines,
  }: {
    path: string;
    ply: number;
    target: string | null;
    engines: string[];
  } = $props();

  type Mode = "tally" | "ranking";
  let mode: Mode = $state("tally");
  let engine1 = $state("estimator");
  let engine2 = $state("");
  let trials = $state(10);
  let seed = $state(0);
  // TSUITATE_THINK_BUDGET_MS 相当（エンジン構築時にバックエンドが env へ反映する）
  let budgetMs = $state(2000);

  let running = $state(false);
  let currentRunId = $state(0);
  let currentEngine = $state("");
  let progressDone = $state(0);
  let progressTotal = $state(0);
  let liveOutcomes = $state<TrialOutcome[]>([]);
  let tallyResults = $state<{ ply: number; budgetMs: number; result: TallyResult }[]>([]);
  let rankingResult = $state<{ ply: number; budgetMs: number; result: RankingResult } | null>(
    null,
  );
  let error = $state("");

  let runCounter = 0;

  onMount(() => {
    let unlisten: UnlistenFn | undefined;
    listen<ProgressEvent>("eval-progress", (ev) => {
      if (ev.payload.runId !== currentRunId) return;
      progressDone = ev.payload.done;
      progressTotal = ev.payload.total;
      liveOutcomes = [...liveOutcomes, ev.payload.outcome].sort((a, b) => a.seed - b.seed);
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  });

  // 局面（ファイル・ply）が変わったら結果表示をクリアする
  $effect(() => {
    void path;
    void ply;
    tallyResults = [];
    rankingResult = null;
    error = "";
  });

  async function run() {
    error = "";
    tallyResults = [];
    rankingResult = null;
    running = true;
    const runPly = ply;
    const runBudget = budgetMs;
    try {
      if (mode === "ranking") {
        currentEngine = "estimator";
        const result = await evalRanking(path, runPly, "estimator", seed, runBudget);
        rankingResult = { ply: runPly, budgetMs: runBudget, result };
      } else {
        const engineList = engine2 !== "" && engine2 !== engine1 ? [engine1, engine2] : [engine1];
        for (const engine of engineList) {
          currentRunId = ++runCounter;
          currentEngine = engine;
          progressDone = 0;
          progressTotal = trials;
          liveOutcomes = [];
          const result = await evalTally(currentRunId, path, runPly, engine, trials, runBudget);
          tallyResults = [...tallyResults, { ply: runPly, budgetMs: runBudget, result }];
        }
      }
    } catch (e) {
      error = String(e);
    } finally {
      running = false;
      currentEngine = "";
    }
  }

  function cancel() {
    if (currentRunId > 0) void cancelEval(currentRunId);
  }

  function fmt(x: number, digits = 3): string {
    return x.toFixed(digits);
  }
</script>

<div class="panel">
  <div class="controls">
    <label>
      モード
      <select bind:value={mode} disabled={running}>
        <option value="tally">seed集計（全エンジン）</option>
        <option value="ranking">ランキング（estimatorのみ）</option>
      </select>
    </label>
    {#if mode === "tally"}
      <label>
        エンジン
        <select bind:value={engine1} disabled={running}>
          {#each engines as e (e)}
            <option value={e}>{e}</option>
          {/each}
        </select>
      </label>
      <label>
        比較
        <select bind:value={engine2} disabled={running}>
          <option value="">（なし）</option>
          {#each engines as e (e)}
            <option value={e}>{e}</option>
          {/each}
        </select>
      </label>
      <label>
        試行数
        <select bind:value={trials} disabled={running}>
          <option value={5}>5</option>
          <option value={10}>10</option>
          <option value={20}>20</option>
          <option value={40}>40</option>
        </select>
      </label>
    {:else}
      <label>
        seed
        <input type="number" bind:value={seed} min="0" disabled={running} style="width: 70px" />
      </label>
    {/if}
    <label>
      思考予算
      <select bind:value={budgetMs} disabled={running}>
        <option value={500}>500ms</option>
        <option value={900}>900ms（本番相当）</option>
        <option value={2000}>2000ms（既定）</option>
        <option value={5000}>5000ms</option>
        <option value={10000}>10000ms</option>
      </select>
    </label>
    {#if running}
      <button onclick={cancel}>キャンセル</button>
    {:else}
      <button onclick={run} disabled={path === ""}>▶ 実行</button>
    {/if}
  </div>

  <div class="hint">
    {ply}手まで再生した局面で {ply + 1} 手目を考えさせる（時間はエンジンの思考予算ぶんかかる）
  </div>

  {#if running}
    <div class="progress">
      <span>{currentEngine} 実行中 …</span>
      {#if mode === "tally" && progressTotal > 0}
        <progress value={progressDone} max={progressTotal}></progress>
        <span>{progressDone}/{progressTotal}</span>
      {/if}
    </div>
    {#if liveOutcomes.length > 0}
      <div class="live">
        {#each liveOutcomes as o (o.seed)}
          <div class="live-row">
            seed {o.seed}: {o.accepted}{o.accepted === target ? " ★" : ""}
            {#if o.fouls.length > 0}<span class="foul">反則 {o.fouls.join(", ")}</span>{/if}
          </div>
        {/each}
      </div>
    {/if}
  {/if}

  {#if error !== ""}
    <div class="error">{error}</div>
  {/if}

  {#if tallyResults.length > 0}
    <div class="results" class:compare={tallyResults.length > 1}>
      {#each tallyResults as { ply: runPly, budgetMs: runBudget, result } (result.engine)}
        <div class="tally">
          <div class="result-head">
            <b>{result.engine}</b>
            <span class="dim">
              {runPly + 1}手目 / 手番{result.side === "sente" ? "▲" : "△"} /
              予算{runBudget}ms / 追加反則 {result.totalFouls}
              {result.cancelled ? " / キャンセル済み（途中まで）" : ""}
            </span>
          </div>
          {#each result.tally as t (t.usi)}
            <div class="bar-row">
              <span class="bar-usi" class:is-target={t.usi === target}>
                {t.usi}{t.usi === target ? " ★" : ""}
              </span>
              <div class="bar-track">
                <div
                  class="bar"
                  style="width: {(100 * t.count) / Math.max(1, result.trials.length)}%"
                ></div>
              </div>
              <span class="bar-n">{t.count}/{result.trials.length}</span>
            </div>
          {/each}
        </div>
      {/each}
    </div>
  {/if}

  {#if rankingResult}
    {@const r = rankingResult.result}
    <div class="result-head">
      <b>{r.engine}</b>
      <span class="dim">
        {rankingResult.ply + 1}手目 / 手番{r.side === "sente" ? "▲" : "△"} / seed {r.seed} /
        予算{rankingResult.budgetMs}ms / 選択 {r.chosen} / 全{r.ranking.length}候補
      </span>
    </div>
    <div class="ranking-scroll">
      <table class="ranking">
        <thead>
          <tr>
            <th>#</th>
            <th>手</th>
            <th>score</th>
            <th>gain</th>
            <th>p_legal</th>
            <th>foul_cost</th>
            <th>adjust</th>
            <th>2手読み</th>
          </tr>
        </thead>
        <tbody>
          {#each r.ranking as c, i (c.usi)}
            <tr class:chosen={c.usi === r.chosen} class:is-target={c.usi === target}>
              <td>{i + 1}</td>
              <td>{c.usi}{c.usi === target ? " ★" : ""}</td>
              <td>{fmt(c.score)}</td>
              <td>{fmt(c.gain)}</td>
              <td>{fmt(c.p_legal)}</td>
              <td>{fmt(c.foul_cost)}</td>
              <td>{fmt(c.adjust)}</td>
              <td>{c.depth2 ? "○" : ""}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px;
    overflow-y: auto;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }

  .controls label {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--text-dim);
    white-space: nowrap;
  }

  .hint {
    color: var(--text-dim);
    font-size: 12px;
  }

  .progress {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .live {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12px;
    max-height: 120px;
    overflow-y: auto;
    color: var(--text-dim);
  }

  .foul {
    color: var(--danger);
    margin-left: 8px;
  }

  .error {
    color: var(--danger);
    white-space: pre-wrap;
  }

  .results.compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .result-head {
    display: flex;
    gap: 8px;
    align-items: baseline;
    flex-wrap: wrap;
  }

  .dim {
    color: var(--text-dim);
    font-size: 12px;
  }

  .bar-row {
    display: grid;
    grid-template-columns: 90px 1fr 52px;
    gap: 8px;
    align-items: center;
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12.5px;
    margin-top: 3px;
  }

  .bar-usi.is-target {
    color: var(--star);
  }

  .bar-track {
    background: var(--panel-2);
    border-radius: 3px;
    height: 14px;
  }

  .bar {
    background: var(--accent);
    height: 100%;
    border-radius: 3px;
    min-width: 2px;
  }

  .bar-n {
    color: var(--text-dim);
    text-align: right;
  }

  .ranking-scroll {
    overflow-y: auto;
    max-height: 340px;
  }

  table.ranking {
    border-collapse: collapse;
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12.5px;
    width: 100%;
  }

  table.ranking th,
  table.ranking td {
    padding: 2px 8px;
    text-align: right;
    border-bottom: 1px solid var(--border);
  }

  table.ranking th:nth-child(2),
  table.ranking td:nth-child(2) {
    text-align: left;
  }

  table.ranking thead th {
    color: var(--text-dim);
    position: sticky;
    top: 0;
    background: var(--panel);
  }

  tr.chosen {
    background: rgba(106, 169, 255, 0.18);
  }

  tr.is-target td {
    color: var(--star);
  }
</style>
