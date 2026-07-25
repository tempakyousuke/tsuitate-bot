<script lang="ts">
  import Board from "./Board.svelte";
  import {
    playBotMove,
    playExport,
    playHumanMove,
    playResign,
    playStart,
    playStartBots,
    type Color,
    type PlayView,
    type Role,
    type Snapshot,
  } from "./api";

  let { engines, onOpenKifu }: { engines: string[]; onOpenKifu: (path: string) => void } =
    $props();

  const DROP_LETTER: Partial<Record<Role, string>> = {
    pawn: "P",
    lance: "L",
    knight: "N",
    silver: "S",
    gold: "G",
    bishop: "B",
    rook: "R",
  };
  const REASON_JA: Record<string, string> = {
    checkmate: "詰み",
    stalemate: "ステイルメイト",
    resign: "投了",
    foul_limit: "反則10回",
  };

  const randomSeed = () => Math.floor(Math.random() * 1_000_000);

  // 設定（対人 / 観戦の2モード。予算は共通）
  let panelMode = $state<"human" | "watch">("human");
  let budgetMs = $state(900);
  // 対人
  let engine = $state("estimator");
  let humanColor = $state<Color>("sente");
  let seed = $state(randomSeed());
  // 観戦（bot vs bot）
  let senteEngine = $state("estimator");
  let goteEngine = $state("estimator");
  let senteSeed = $state(randomSeed());
  let goteSeed = $state(randomSeed());
  let watchFlipped = $state(false);

  // 対局状態
  let view = $state<PlayView | null>(null);
  let thinking = $state(false); // bot 思考中
  let busy = $state(false); // コマンド往復中
  let auto = $state(false); // 観戦の自動再生
  let autoIntervalMs = $state(300);
  let reveal = $state(false); // 真実（相手駒）を表示するデバッグトグル
  let log = $state<string[]>([]);
  let error = $state("");
  let selected = $state<{ kind: "board"; sq: string } | { kind: "hand"; role: Role } | null>(
    null,
  );
  let promo = $state<{ from: string; to: string } | null>(null);
  let logBox = $state<HTMLDivElement | undefined>();

  // 書き出し
  let exportName = $state("");
  let exportPly = $state("");
  let exportedPath = $state("");

  const over = $derived(view?.outcome != null);
  const watching = $derived(view != null && view.humanColor == null);
  const locked = $derived(thinking || busy || auto);
  const humanTurn = $derived(
    view != null &&
      view.humanColor != null &&
      !over &&
      !locked &&
      view.snapshot.turn === view.humanColor,
  );
  // 観戦は情報を隠す相手がいないので常に真実を出す
  const showTruth = $derived(reveal || over || watching);

  function sideLabel(v: PlayView, c: Color): string {
    const i = c === "sente" ? 0 : 1;
    return `${c === "sente" ? "▲先手" : "△後手"}（${v.names[i]}）`;
  }

  // 隠し盤面: 自駒だけ・自分の持ち駒だけ・相手の直前手は隠す（実対局と同じ情報）
  const displaySnapshot = $derived.by((): Snapshot | null => {
    if (!view) return null;
    const s = view.snapshot;
    const human = view.humanColor;
    if (showTruth || human == null) return s;
    const board: Snapshot["board"] = {};
    for (const [sq, p] of Object.entries(s.board)) {
      if (p.color === human) board[sq] = p;
    }
    // 直前手の着地マスにいる駒の色で「誰の手か」を判別し、相手の手は隠す
    const lastPiece = s.lastMove ? s.board[s.lastMove.to] : undefined;
    const lastMove = s.lastMove && lastPiece?.color === human ? s.lastMove : null;
    return {
      ...s,
      board,
      lastMove,
      handSente: human === "sente" ? s.handSente : {},
      handGote: human === "gote" ? s.handGote : {},
    };
  });

  const boardFlipped = $derived(
    view?.humanColor == null ? watchFlipped : view.humanColor === "gote",
  );

  const targets = $derived.by((): string[] => {
    if (!view || !humanTurn || !selected) return [];
    const sel = selected;
    if (sel.kind === "board") {
      return view.hints.filter((h) => h.from === sel.sq).map((h) => h.to);
    }
    return view.hints.filter((h) => h.from === null && h.role === sel.role).map((h) => h.to);
  });

  const statusText = $derived.by((): string => {
    const v = view;
    if (!v) return "";
    if (v.outcome) {
      const reason = REASON_JA[v.outcome.reason] ?? v.outcome.reason;
      if (v.outcome.winner == null) return `終局（${reason}）`;
      if (v.humanColor == null) return `${sideLabel(v, v.outcome.winner)}の勝ち（${reason}）`;
      return v.outcome.winner === v.humanColor
        ? `あなたの勝ち（${reason}）`
        : `botの勝ち（${reason}）`;
    }
    if (thinking) {
      return v.humanColor == null
        ? `${sideLabel(v, v.snapshot.turn)} 思考中…`
        : "bot 思考中…";
    }
    if (busy) return "…";
    if (v.humanColor == null) return `${sideLabel(v, v.snapshot.turn)} の手番`;
    return v.snapshot.turn === v.humanColor ? "あなたの手番です" : "botの手番です";
  });

  // イベントログは常に末尾へスクロール
  $effect(() => {
    void log.length;
    if (logBox) logBox.scrollTop = logBox.scrollHeight;
  });

  function stamp(): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  }

  function defaultExportName(): string {
    return panelMode === "watch"
      ? `watch-${senteEngine}-vs-${goteEngine}-${stamp()}`
      : `play-${engine}-${stamp()}`;
  }

  function applyView(v: PlayView) {
    view = v;
    log = [...log, ...v.events];
  }

  function resetForNewGame() {
    error = "";
    exportedPath = "";
    log = [];
    selected = null;
    promo = null;
    reveal = false;
    auto = false;
    exportPly = "";
  }

  async function start() {
    resetForNewGame();
    busy = true;
    try {
      const v =
        panelMode === "watch"
          ? await playStartBots(senteEngine, senteSeed, goteEngine, goteSeed, budgetMs)
          : await playStart(engine, humanColor, seed, budgetMs);
      log = [];
      applyView(v);
      exportName = defaultExportName();
      busy = false;
      // 対人で bot が先手なら1手指させる。観戦は操作を待つ
      if (v.humanColor != null && v.snapshot.turn !== v.humanColor) await botTurn();
    } catch (e) {
      error = String(e);
      busy = false;
    }
  }

  /** 手番側 bot に1手指させる。続行してよいなら true */
  async function botTurn(): Promise<boolean> {
    thinking = true;
    try {
      const v = await playBotMove();
      applyView(v);
      return v.outcome == null;
    } catch (e) {
      error = String(e);
      return false;
    } finally {
      thinking = false;
    }
  }

  async function stepBot() {
    error = "";
    await botTurn();
  }

  async function toggleAuto() {
    if (auto) {
      auto = false; // 実行中のループは次の判定で抜ける
      return;
    }
    error = "";
    auto = true;
    while (auto) {
      if (!(await botTurn())) break;
      if (!auto) break;
      if (autoIntervalMs > 0) {
        await new Promise((r) => setTimeout(r, autoIntervalMs));
      }
    }
    auto = false;
  }

  async function send(usi: string) {
    if (!view) return;
    selected = null;
    promo = null;
    busy = true;
    error = "";
    try {
      const v = await playHumanMove(usi);
      applyView(v);
      busy = false;
      if (!v.outcome && v.humanColor != null && v.snapshot.turn !== v.humanColor) {
        await botTurn();
      }
    } catch (e) {
      error = String(e);
      busy = false;
    }
  }

  function cellClick(sq: string) {
    if (!view || !humanTurn) return;
    promo = null;
    if (selected && targets.includes(sq)) {
      void commitTo(sq);
      return;
    }
    const piece = view.snapshot.board[sq];
    if (piece && piece.color === view.humanColor) {
      selected = { kind: "board", sq };
    } else {
      selected = null;
    }
  }

  function handClick(role: Role) {
    if (!view || !humanTurn) return;
    promo = null;
    selected =
      selected?.kind === "hand" && selected.role === role ? null : { kind: "hand", role };
  }

  async function commitTo(to: string) {
    if (!view || !selected) return;
    if (selected.kind === "hand") {
      const letter = DROP_LETTER[selected.role];
      if (letter) await send(`${letter}*${to}`);
      return;
    }
    const from = selected.sq;
    const hint = view.hints.find((h) => h.from === from && h.to === to);
    if (!hint) return;
    if (hint.promotion === "optional") {
      promo = { from, to };
      return;
    }
    await send(`${from}${to}${hint.promotion === "forced" ? "+" : ""}`);
  }

  async function resign() {
    busy = true;
    error = "";
    try {
      applyView(await playResign());
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  async function doExport() {
    error = "";
    exportedPath = "";
    const plyText = exportPly.trim();
    let ply: number | null = null;
    if (plyText !== "") {
      ply = Number(plyText);
      if (!Number.isInteger(ply) || ply < 0) {
        error = "書き出しの ply は 0 以上の整数で指定してください";
        return;
      }
    }
    try {
      exportedPath = await playExport(exportName, ply, null);
    } catch (e) {
      error = String(e);
    }
  }
</script>

<div class="play">
  <div class="setup">
    <nav class="sub-nav">
      <button
        class:active={panelMode === "human"}
        onclick={() => (panelMode = "human")}
        disabled={locked}
      >
        対人
      </button>
      <button
        class:active={panelMode === "watch"}
        onclick={() => (panelMode = "watch")}
        disabled={locked}
      >
        観戦（bot vs bot）
      </button>
    </nav>

    {#if panelMode === "human"}
      <label>
        エンジン
        <select bind:value={engine} disabled={locked}>
          {#each engines as e (e)}
            <option value={e}>{e}</option>
          {/each}
        </select>
      </label>
      <label>
        あなたの手番
        <select bind:value={humanColor} disabled={locked}>
          <option value="sente">▲先手</option>
          <option value="gote">△後手</option>
        </select>
      </label>
      <label>
        seed
        <input type="number" bind:value={seed} min="0" style="width: 90px" disabled={locked} />
      </label>
    {:else}
      <label>
        ▲先手
        <select bind:value={senteEngine} disabled={locked}>
          {#each engines as e (e)}
            <option value={e}>{e}</option>
          {/each}
        </select>
        <input
          type="number"
          bind:value={senteSeed}
          min="0"
          style="width: 84px"
          title="先手の seed"
          disabled={locked}
        />
      </label>
      <label>
        △後手
        <select bind:value={goteEngine} disabled={locked}>
          {#each engines as e (e)}
            <option value={e}>{e}</option>
          {/each}
        </select>
        <input
          type="number"
          bind:value={goteSeed}
          min="0"
          style="width: 84px"
          title="後手の seed"
          disabled={locked}
        />
      </label>
      <button
        onclick={() => {
          senteSeed = randomSeed();
          goteSeed = randomSeed();
        }}
        disabled={locked}
        title="両者の seed を引き直す"
      >
        seed 引き直し
      </button>
    {/if}

    <label>
      思考予算
      <select bind:value={budgetMs} disabled={locked}>
        <option value={500}>500ms</option>
        <option value={900}>900ms（本番相当）</option>
        <option value={2000}>2000ms</option>
        <option value={5000}>5000ms</option>
      </select>
    </label>
    <button onclick={start} disabled={locked}>
      {view ? (panelMode === "watch" ? "新しい観戦" : "新しい対局") : "対局開始"}
    </button>
  </div>

  {#if error !== ""}
    <div class="error">{error}</div>
  {/if}

  {#if view && displaySnapshot}
    <div class="game">
      <section class="board-col">
        <Board
          snapshot={displaySnapshot}
          flipped={boardFlipped}
          selected={selected?.kind === "board" ? selected.sq : null}
          {targets}
          danger={showTruth ? null : view.capturedSquare}
          onCellClick={humanTurn ? cellClick : null}
          handSide={humanTurn ? view.humanColor : null}
          selectedHandRole={selected?.kind === "hand" ? selected.role : null}
          onHandClick={handClick}
        />
        {#if promo}
          <div class="promo">
            {promo.from}→{promo.to}:
            <button onclick={() => promo && send(`${promo.from}${promo.to}+`)}>成</button>
            <button onclick={() => promo && send(`${promo.from}${promo.to}`)}>不成</button>
            <button onclick={() => (promo = null)}>キャンセル</button>
          </div>
        {/if}
        <div class="status">
          <b class:danger-text={over}>{statusText}</b>
          {#if thinking}<span class="spinner">⏳</span>{/if}
        </div>

        {#if watching}
          <div class="status">
            <button onclick={stepBot} disabled={over || locked}>1手進める</button>
            <button onclick={toggleAuto} disabled={over || busy}>
              {auto ? "■ 停止" : "▶ 自動再生"}
            </button>
            <label>
              間隔
              <select bind:value={autoIntervalMs}>
                <option value={0}>なし</option>
                <option value={300}>0.3秒</option>
                <option value={1000}>1秒</option>
                <option value={3000}>3秒</option>
              </select>
            </label>
            <button onclick={() => (watchFlipped = !watchFlipped)}>
              視点: {watchFlipped ? "△後手" : "▲先手"}（切替）
            </button>
          </div>
          <div class="status dim">
            {view.totalMoves}手 / 反則 ▲{view.snapshot.fouls[0]} △{view.snapshot.fouls[1]}
            / ▲{view.names[0]}（seed={view.seeds[0]}）vs △{view.names[1]}（seed={view
              .seeds[1]}）{view.budgetMs}ms
          </div>
        {:else}
          <div class="status dim">
            {view.totalMoves}手 / 反則 ▲{view.snapshot.fouls[0]} △{view.snapshot.fouls[1]} /
            bot={view.humanColor === "sente" ? view.names[1] : view.names[0]}（seed={view
              .humanColor === "sente"
              ? view.seeds[1]
              : view.seeds[0]}, {view.budgetMs}ms）
          </div>
          <div class="status">
            <label class="reveal">
              <input type="checkbox" bind:checked={reveal} disabled={over} />
              真実を表示（デバッグ。終局後は常に表示）
            </label>
            <button onclick={resign} disabled={over || locked}>投了する</button>
          </div>
        {/if}
      </section>

      <section class="side-col">
        <div class="log-head">観測イベント</div>
        <div class="log" bind:this={logBox}>
          {#each log as line, i (i)}
            <div class="log-line" class:foul-line={line.includes("反則")}>{line}</div>
          {/each}
        </div>

        <div class="export">
          <div class="log-head">kif 書き出し（scenarios/ へ保存）</div>
          <label>
            ファイル名
            <input type="text" bind:value={exportName} style="width: 260px" />
          </label>
          <label>
            シナリオply（任意）
            <input
              type="text"
              bind:value={exportPly}
              placeholder="例 32 = 33手目を考えさせる"
              style="width: 160px"
            />
          </label>
          <div class="export-actions">
            <button onclick={doExport} disabled={view.totalMoves === 0 || locked}>
              書き出す
            </button>
            {#if exportedPath !== ""}
              <button onclick={() => onOpenKifu(exportedPath)}>リプレイで開く</button>
            {/if}
          </div>
          {#if exportedPath !== ""}
            <div class="exported">保存済み: {exportedPath}</div>
          {/if}
        </div>
      </section>
    </div>
  {:else}
    <div class="placeholder">
      {#if panelMode === "human"}
        エンジンと手番を選んで「対局開始」。相手の駒は見えません（実対局と同じ）。
        候補ハイライトは自駒だけを考慮した移動先なので、そのまま指しても反則になることがあります。
        終局後（または途中でも）kif を書き出して、リプレイ・シナリオ実験に使えます。
      {:else}
        先手・後手のエンジンと seed を選んで「対局開始」。1手送りか自動再生で観戦できます
        （bot 同士も互いの盤面は見えません。観測ログは bot ごとに独立です）。
        気になる局面で止めて kif を書き出せば、そのままリプレイの候補手分析・
        玉位置ビリーフにかけられます。
      {/if}
    </div>
  {/if}
</div>

<style>
  .play {
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex: 1;
    min-height: 0;
  }

  .setup {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }

  .setup label,
  .export label,
  .reveal {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--text-dim);
    white-space: nowrap;
  }

  .sub-nav {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .sub-nav button {
    border: none;
    border-radius: 0;
    padding: 4px 12px;
  }

  .sub-nav button.active {
    background: var(--accent);
    color: #fff;
  }

  .game {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 14px;
    flex: 1;
    min-height: 0;
  }

  .board-col {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .side-col {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 0;
    min-width: 0;
  }

  .promo {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 6px 10px;
    background: var(--panel);
    border: 1px solid var(--accent);
    border-radius: 4px;
    width: fit-content;
  }

  .status {
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 13px;
    flex-wrap: wrap;
  }

  .status.dim {
    color: var(--text-dim);
  }

  .danger-text {
    color: var(--star);
  }

  .log-head {
    color: var(--text-dim);
    font-size: 12px;
  }

  .log {
    flex: 1;
    min-height: 120px;
    overflow-y: auto;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    font-size: 12.5px;
    font-family: ui-monospace, Menlo, monospace;
  }

  .log-line.foul-line {
    color: var(--danger);
  }

  .export {
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px;
  }

  .export-actions {
    display: flex;
    gap: 8px;
  }

  .exported {
    color: var(--text-dim);
    font-size: 12px;
    word-break: break-all;
  }

  .error {
    color: var(--danger);
    white-space: pre-wrap;
  }

  .placeholder {
    color: var(--text-dim);
    margin: auto;
    max-width: 560px;
    line-height: 1.7;
  }
</style>
