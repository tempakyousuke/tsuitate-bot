<script lang="ts">
  import { onMount } from "svelte";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import {
    cancelEval,
    evalKingBelief,
    evalRanking,
    evalTally,
    type KingBelief,
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
    onOverlay = null,
  }: {
    path: string;
    ply: number;
    target: string | null;
    engines: string[];
    /** 盤に重ねる確率マップを親へ渡す（null でクリア） */
    onOverlay?:
      | ((data: { squares: Record<string, number>; truth: string | null } | null) => void)
      | null;
  } = $props();

  type Mode = "tally" | "ranking" | "belief";
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
  let beliefResult = $state<{ ply: number; budgetMs: number; result: KingBelief } | null>(null);
  // 厳密整合の粒子だけで見るか、taint 込みの全粒子で見るか。
  // 終盤は厳密が全滅していることが普通にあるので既定は taint 込み
  let beliefSource = $state<"all" | "strict">("all");
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
    beliefResult = null;
    onOverlay?.(null);
    error = "";
  });

  // ビリーフ表示は「厳密のみ / taint込み」の切り替えに追随して盤へ流す
  $effect(() => {
    const b = beliefResult?.result;
    if (!b) return;
    const squares: Record<string, number> = {};
    for (const s of b.squares) {
      const v = beliefSource === "strict" ? s.strict : s.all;
      if (v > 0) squares[s.sq] = v;
    }
    onOverlay?.({ squares, truth: b.truth });
  });

  async function run() {
    error = "";
    tallyResults = [];
    rankingResult = null;
    beliefResult = null;
    onOverlay?.(null);
    running = true;
    const runPly = ply;
    const runBudget = budgetMs;
    try {
      if (mode === "belief") {
        currentEngine = "estimator（粒子）";
        const result = await evalKingBelief(path, runPly, trials, runBudget);
        beliefResult = { ply: runPly, budgetMs: runBudget, result };
      } else if (mode === "ranking") {
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
        <option value="belief">玉位置ビリーフ（盤に重ねる）</option>
      </select>
    </label>
    {#if mode === "belief"}
      <label>
        推定器数
        <select bind:value={trials} disabled={running}>
          <option value={2}>2</option>
          <option value={5}>5</option>
          <option value={10}>10</option>
          <option value={20}>20</option>
        </select>
      </label>
      <label>
        粒子
        <select bind:value={beliefSource} disabled={running}>
          <option value="all">taint込み全粒子</option>
          <option value="strict">厳密整合のみ</option>
        </select>
      </label>
    {:else if mode === "tally"}
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
    {#if mode === "belief"}
      {ply}手まで再生した局面で、{ply + 1} 手目を指す側の粒子が
      「相手玉はどこにいると思っているか」を盤に % で重ねる（赤枠 = 真実）
    {:else}
      {ply}手まで再生した局面で {ply + 1} 手目を考えさせる（時間はエンジンの思考予算ぶんかかる）
    {/if}
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

  {#if beliefResult}
    {@const b = beliefResult.result}
    {@const truthPct =
      b.truth == null
        ? null
        : (b.squares.find((s) => s.sq === b.truth)?.[
            beliefSource === "strict" ? "strict" : "all"
          ] ?? 0)}
    <div class="result-head">
      <b>玉位置ビリーフ</b>
      <span class="dim">
        {beliefResult.ply + 1}手目 / 手番{b.side === "sente" ? "▲" : "△"} /
        推定器{b.seeds}個 / 予算{beliefResult.budgetMs}ms /
        ユニーク粒子 {b.unique}（厳密 {b.strictUnique}）
      </span>
    </div>
    {#if beliefSource === "strict" && b.strictUnique === 0}
      <div class="hint warn">
        厳密整合の粒子が0個です。この局面の評価は taint 粒子と事前分布で走っているので、
        「taint込み全粒子」に切り替えてください
      </div>
    {/if}
    <div class="belief-scroll">
      <table class="ranking">
        <thead>
          <tr>
            <th>#</th>
            <th>マス</th>
            <th title="taint込みの全粒子での割合">全粒子</th>
            <th title="厳密整合の粒子だけでの割合">厳密</th>
          </tr>
        </thead>
        <tbody>
          {#each b.squares.slice(0, 12) as s, i (s.sq)}
            <tr class:truth-row={s.sq === b.truth}>
              <td>{i + 1}</td>
              <td>{s.sq}{s.sq === b.truth ? " ←真実" : ""}</td>
              <td>{(s.all * 100).toFixed(1)}%</td>
              <td>{(s.strict * 100).toFixed(1)}%</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if truthPct != null}
      <div class="hint">
        真実 {b.truth} への信念: <b>{(truthPct * 100).toFixed(1)}%</b>
        {#if truthPct < 0.2}（較正不良: 玉位置を外している）{/if}
      </div>
    {/if}
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
            <th title="gainのうち王手駒の除去期待値ぶん（王手中のみ）">除去EV</th>
            <th title="gainから引かれた捕獲の賭け分散ペナルティ p_hit(1−p_hit)×E[捕獲価値|hit]×w（王手外のみ）">賭けpen</th>
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
              <td>{c.checker_removal !== 0 ? fmt(c.checker_removal) : ""}</td>
              <td>{c.capture_bet_penalty !== 0 ? `−${fmt(c.capture_bet_penalty)}` : ""}</td>
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

  .hint.warn {
    color: #e0a030;
  }

  .belief-scroll {
    max-height: 260px;
    overflow: auto;
  }

  tr.truth-row {
    outline: 1px solid #d33;
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
