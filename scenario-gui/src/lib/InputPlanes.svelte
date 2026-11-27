<script lang="ts">
  // 「入力プレーン」表示（対局モードとリプレイで共用）。ある席の観測から作った方策ネットの入力（エンコーダ v7 の全プレーン）を、
  // マスごとのプレーンは小さい盤、全面に同じ値を入れるプレーンは表で見せる。
  // 小さい盤をクリックすると、そのプレーンを左の盤に重ねる（親が overlay として Board に渡す）
  import { isBinaryPlane, planeKey as keyOf, type PlaneView, type PlanesOut } from "./api";

  let {
    planes,
    sides,
    side,
    onSide,
    flipped = false,
    selected = null,
    onSelect,
    loading = false,
    hiddenInfo = false,
  }: {
    planes: PlanesOut | null;
    /** 切り替えられる席（[キー, 表示名]。キーは親が決める: 色、リプレイでは「手番の席」も） */
    sides: [string, string][];
    side: string;
    onSide: (key: string) => void;
    /** 盤の向き（左の盤と同じ。true = 後手が下） */
    flipped?: boolean;
    /** 盤に重ねているプレーンのキー */
    selected?: string | null;
    onSelect: (key: string | null) => void;
    loading?: boolean;
    /** 対人で相手の席を見ている（隠れている駒の情報が出る） */
    hiddenInfo?: boolean;
  } = $props();

  const RANKS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
  const files = $derived(flipped ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : [9, 8, 7, 6, 5, 4, 3, 2, 1]);
  const ranks = $derived(flipped ? [...RANKS].reverse() : RANKS);

  // 空のプレーン（全マス 0）は小さい盤を出さず、番号だけ並べる
  let hideEmpty = $state(true);

  function tag(p: PlaneView): string {
    return p.index == null ? "参考" : `#${p.index}`;
  }

  type Group = { name: string; boards: PlaneView[]; empty: PlaneView[]; scalars: PlaneView[] };

  const groups = $derived.by((): Group[] => {
    const out: Group[] = [];
    for (const p of planes?.planes ?? []) {
      let g = out.find((x) => x.name === p.group);
      if (!g) {
        g = { name: p.group, boards: [], empty: [], scalars: [] };
        out.push(g);
      }
      if (p.scalar != null) g.scalars.push(p);
      else if (hideEmpty && Object.keys(p.cells).length === 0) g.empty.push(p);
      else g.boards.push(p);
    }
    return out;
  });

  const selectedPlane = $derived(
    planes?.planes.find((p) => keyOf(p) === selected) ?? null,
  );

  // 値 → 濃さ。小さい値（薄れた鮮度・1/候補数）も見えるように下駄を履かせる
  function alpha(v: number): number {
    return 0.18 + 0.82 * Math.min(1, Math.max(0, v));
  }

  function cellsTitle(p: PlaneView): string {
    const entries = Object.entries(p.cells);
    if (entries.length === 0) return `${tag(p)} ${p.label}: 全マス 0`;
    const shown = entries
      .slice(0, 24)
      .map(([sq, v]) => `${sq}=${fmt(v)}`)
      .join("  ");
    const more = entries.length > 24 ? ` …ほか${entries.length - 24}マス` : "";
    return `${tag(p)} ${p.label}（${p.version}）\n${shown}${more}`;
  }

  function fmt(v: number): string {
    if (v === 0 || v === 1) return String(v);
    return v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
</script>

<div class="planes">
  <div class="head">
    <div class="sub-nav">
      {#each sides as [c, name] (c)}
        <button class:active={side === c} onclick={() => onSide(c)}>{name}</button>
      {/each}
    </div>
    {#if planes}
      <span class="dim">
        {planes.moveNumber}手目・{planes.toMove ? "この席の手番" : "相手の手番"}
      </span>
    {/if}
    <label class="dim">
      <input type="checkbox" bind:checked={hideEmpty} />
      空のプレーンをたたむ
    </label>
    {#if loading}<span class="dim">更新中…</span>{/if}
    {#if hiddenInfo}<span class="warn">相手の席の観測です（見えていない駒の位置が出ます）</span>{/if}
  </div>

  <div class="head">
    {#if selectedPlane}
      <span>
        盤に重ねているプレーン: <b>{tag(selectedPlane)} {selectedPlane.label}</b>
        <span class="dim">
          {selectedPlane.scalar == null && Object.keys(selectedPlane.cells).length === 0
            ? "（この席では今は全マス 0）"
            : isBinaryPlane(selectedPlane.cells)
              ? "（色のついたマスが 1）"
              : "（盤の数字は値×100）"}
        </span>
      </span>
      <button onclick={() => onSelect(null)}>重ねるのをやめる</button>
    {:else}
      <span class="dim">小さい盤をクリックすると、そのプレーンを左の盤に重ねます。</span>
    {/if}
  </div>

  {#if !planes}
    <div class="dim">入力を読み込んでいます…</div>
  {:else}
    <div class="scroll">
      {#each groups as g (g.name)}
        <section>
          <h4>{g.name}</h4>
          {#if g.boards.length > 0}
            <div class="tiles">
              {#each g.boards as p (keyOf(p))}
                <button
                  class="tile"
                  class:active={selected === keyOf(p)}
                  title={cellsTitle(p)}
                  onclick={() => onSelect(selected === keyOf(p) ? null : keyOf(p))}
                >
                  <div class="mini" class:empty={Object.keys(p.cells).length === 0}>
                    {#each ranks as r (r)}
                      {#each files as f (`${f}${r}`)}
                        {@const v = p.cells[`${f}${r}`] ?? 0}
                        <span
                          class="mc"
                          style={v > 0 ? `background: rgba(106, 169, 255, ${alpha(v)})` : ""}
                        ></span>
                      {/each}
                    {/each}
                  </div>
                  <div class="cap">
                    <span class="num">{tag(p)}</span>
                    {p.label}
                  </div>
                </button>
              {/each}
            </div>
          {/if}
          {#if g.empty.length > 0}
            <div class="empties">
              空:
              {#each g.empty as p (keyOf(p))}
                <span class="chip" title={`${tag(p)} ${p.label}（${p.version}）`}>
                  {tag(p)} {p.label}
                </span>
              {/each}
            </div>
          {/if}
          {#if g.scalars.length > 0}
            <table>
              <tbody>
                {#each g.scalars as p (keyOf(p))}
                  <tr class:zero={p.scalar === 0}>
                    <td class="num">{tag(p)}</td>
                    <td>{p.label}</td>
                    <td class="val">{fmt(p.scalar ?? 0)}</td>
                    <td class="bar-cell">
                      <span class="bar" style={`width: ${Math.min(1, Math.max(0, p.scalar ?? 0)) * 100}%`}
                      ></span>
                    </td>
                    <td class="note">{p.note ?? ""}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </section>
      {/each}
    </div>
  {/if}
</div>

<style>
  .planes {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 0;
    flex: 1;
  }

  .head {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    font-size: 13px;
  }

  .head label {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .dim {
    color: var(--text-dim);
  }

  .warn {
    color: var(--star);
  }

  .sub-nav {
    display: flex;
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

  .scroll {
    overflow-y: auto;
    min-height: 0;
    flex: 1;
    padding-right: 6px;
  }

  section {
    margin-bottom: 14px;
  }

  h4 {
    margin: 0 0 6px;
    font-size: 13px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 3px;
  }

  .tiles {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .tile {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    width: 112px;
    padding: 6px 4px;
    background: var(--panel);
  }

  .tile.active {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }

  .mini {
    display: grid;
    grid-template-columns: repeat(9, 10px);
    grid-auto-rows: 10px;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }

  .mini.empty {
    opacity: 0.5;
  }

  .mc {
    background: var(--bg);
  }

  .cap {
    font-size: 11px;
    line-height: 1.3;
    text-align: center;
    word-break: break-all;
  }

  .num {
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }

  .empties {
    margin-top: 6px;
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    flex-wrap: wrap;
    gap: 4px 6px;
    align-items: center;
  }

  .chip {
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0 5px;
  }

  table {
    border-collapse: collapse;
    font-size: 12px;
    width: 100%;
    max-width: 640px;
  }

  td {
    padding: 2px 8px 2px 0;
    white-space: nowrap;
  }

  td.val {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  td.bar-cell {
    width: 120px;
  }

  .bar {
    display: block;
    height: 8px;
    background: var(--accent);
    border-radius: 2px;
    min-width: 0;
  }

  td.note {
    color: var(--text-dim);
  }

  tr.zero td {
    color: var(--text-dim);
  }
</style>
