<script lang="ts">
  import Board from "./Board.svelte";
  import {
    playBotMove,
    playExport,
    playHumanMove,
    playResign,
    playStart,
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

  // 設定
  let engine = $state("estimator");
  let humanColor = $state<Color>("sente");
  let budgetMs = $state(900);
  let seed = $state(Math.floor(Math.random() * 1_000_000));

  // 対局状態
  let view = $state<PlayView | null>(null);
  let thinking = $state(false); // bot 思考中
  let busy = $state(false); // コマンド往復中
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
  const humanTurn = $derived(
    view != null && !over && !thinking && !busy && view.snapshot.turn === view.humanColor,
  );
  const showTruth = $derived(reveal || over);

  // 隠し盤面: 自駒だけ・自分の持ち駒だけ・相手の直前手は隠す（実対局と同じ情報）
  const displaySnapshot = $derived.by((): Snapshot | null => {
    if (!view) return null;
    const s = view.snapshot;
    if (showTruth) return s;
    const board: Snapshot["board"] = {};
    for (const [sq, p] of Object.entries(s.board)) {
      if (p.color === view.humanColor) board[sq] = p;
    }
    // 直前手の着地マスにいる駒の色で「誰の手か」を判別し、相手の手は隠す
    const lastPiece = s.lastMove ? s.board[s.lastMove.to] : undefined;
    const lastMove = s.lastMove && lastPiece?.color === view.humanColor ? s.lastMove : null;
    return {
      ...s,
      board,
      lastMove,
      handSente: view.humanColor === "sente" ? s.handSente : {},
      handGote: view.humanColor === "gote" ? s.handGote : {},
    };
  });

  const targets = $derived.by((): string[] => {
    if (!view || !humanTurn || !selected) return [];
    const sel = selected;
    if (sel.kind === "board") {
      return view.hints.filter((h) => h.from === sel.sq).map((h) => h.to);
    }
    return view.hints.filter((h) => h.from === null && h.role === sel.role).map((h) => h.to);
  });

  const statusText = $derived.by((): string => {
    if (!view) return "";
    if (view.outcome) {
      const reason = REASON_JA[view.outcome.reason] ?? view.outcome.reason;
      if (view.outcome.winner == null) return `終局（${reason}）`;
      return view.outcome.winner === view.humanColor
        ? `あなたの勝ち（${reason}）`
        : `botの勝ち（${reason}）`;
    }
    if (thinking) return "bot 思考中…";
    if (busy) return "…";
    return view.snapshot.turn === view.humanColor ? "あなたの手番です" : "botの手番です";
  });

  // イベントログは常に末尾へスクロール
  $effect(() => {
    void log.length;
    if (logBox) logBox.scrollTop = logBox.scrollHeight;
  });

  function defaultExportName(): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    return `play-${engine}-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  }

  function applyView(v: PlayView) {
    view = v;
    log = [...log, ...v.events];
  }

  async function start() {
    error = "";
    exportedPath = "";
    log = [];
    selected = null;
    promo = null;
    reveal = false;
    busy = true;
    try {
      const v = await playStart(engine, humanColor, seed, budgetMs);
      log = [];
      applyView(v);
      exportName = defaultExportName();
      exportPly = "";
      busy = false;
      if (v.snapshot.turn !== v.humanColor) await botTurn();
    } catch (e) {
      error = String(e);
      busy = false;
    }
  }

  async function botTurn() {
    thinking = true;
    try {
      applyView(await playBotMove());
    } catch (e) {
      error = String(e);
    } finally {
      thinking = false;
    }
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
      if (!v.outcome && v.snapshot.turn !== v.humanColor) await botTurn();
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
    <label>
      エンジン
      <select bind:value={engine} disabled={thinking || busy}>
        {#each engines as e (e)}
          <option value={e}>{e}</option>
        {/each}
      </select>
    </label>
    <label>
      あなたの手番
      <select bind:value={humanColor} disabled={thinking || busy}>
        <option value="sente">▲先手</option>
        <option value="gote">△後手</option>
      </select>
    </label>
    <label>
      思考予算
      <select bind:value={budgetMs} disabled={thinking || busy}>
        <option value={500}>500ms</option>
        <option value={900}>900ms（本番相当）</option>
        <option value={2000}>2000ms</option>
        <option value={5000}>5000ms</option>
      </select>
    </label>
    <label>
      seed
      <input type="number" bind:value={seed} min="0" style="width: 90px" disabled={thinking || busy} />
    </label>
    <button onclick={start} disabled={thinking || busy}>
      {view ? "新しい対局" : "対局開始"}
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
          flipped={view.humanColor === "gote"}
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
        <div class="status dim">
          {view.totalMoves}手 / 反則 ▲{view.snapshot.fouls[0]} △{view.snapshot.fouls[1]} /
          bot={view.engine}（seed={view.seed}, {view.budgetMs}ms）
        </div>
        <div class="status">
          <label class="reveal">
            <input type="checkbox" bind:checked={reveal} disabled={over} />
            真実を表示（デバッグ。終局後は常に表示）
          </label>
          <button onclick={resign} disabled={over || thinking || busy}>投了する</button>
        </div>
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
            <button onclick={doExport} disabled={view.totalMoves === 0 || thinking || busy}>
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
      エンジンと手番を選んで「対局開始」。相手の駒は見えません（実対局と同じ）。
      候補ハイライトは自駒だけを考慮した移動先なので、そのまま指しても反則になることがあります。
      終局後（または途中でも）kif を書き出して、リプレイ・シナリオ実験に使えます。
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
