// バックエンド（src-tauri/src/main.rs）の commands の型と invoke ラッパー

import { invoke } from "@tauri-apps/api/core";

export type Color = "sente" | "gote";
export type Role =
  | "pawn"
  | "lance"
  | "knight"
  | "silver"
  | "gold"
  | "bishop"
  | "rook"
  | "king"
  | "tokin"
  | "promotedlance"
  | "promotedknight"
  | "promotedsilver"
  | "horse"
  | "dragon";

export interface PieceOut {
  role: Role;
  color: Color;
}

export interface LastMove {
  usi: string;
  from: string | null;
  to: string;
}

export interface Snapshot {
  board: Record<string, PieceOut>;
  handSente: Partial<Record<Role, number>>;
  handGote: Partial<Record<Role, number>>;
  turn: Color;
  moveNumber: number;
  fouls: [number, number];
  inCheck: [boolean, boolean];
  lastMove: LastMove | null;
}

export interface MoveRow {
  usi: string;
  foulsBefore: string[];
  side: Color;
  givesCheck: boolean;
  capture: boolean;
}

export interface KifuData {
  path: string;
  name: string;
  totalPlies: number;
  directivePly: number | null;
  target: string | null;
  desc: string | null;
  snapshots: Snapshot[];
  moves: MoveRow[];
}

export interface ScenarioInfo {
  path: string;
  name: string;
  archived: boolean;
  totalPlies: number;
  directivePly: number | null;
  target: string | null;
  desc: string | null;
}

export interface TrialOutcome {
  seed: number;
  accepted: string;
  fouls: string[];
}

export interface ProgressEvent {
  runId: number;
  done: number;
  total: number;
  outcome: TrialOutcome;
}

export interface TallyEntry {
  usi: string;
  count: number;
}

export interface TallyResult {
  engine: string;
  side: Color;
  tally: TallyEntry[];
  totalFouls: number;
  trials: TrialOutcome[];
  cancelled: boolean;
}

// strategy.rs の CandidateScore（rename なしの snake_case）
export interface CandidateScore {
  usi: string;
  score: number;
  gain: number;
  p_legal: number;
  foul_cost: number;
  adjust: number;
  depth2: boolean;
  /** gain のうち王手駒の除去期待値ぶん（王手中の候補のみ非ゼロ） */
  checker_removal: number;
  /** gain から引かれた捕獲の賭け分散ペナルティ（正の値、王手外のみ） */
  capture_bet_penalty: number;
  /** gain に加算された詰めろ生成ボーナス */
  mate_threat: number;
  /** gain から引かれた被詰めろペナルティ（正の値） */
  mate_risk: number;
}

export interface RankingResult {
  engine: string;
  side: Color;
  seed: number;
  chosen: string;
  ranking: CandidateScore[];
}

// ---------- 玉位置ビリーフ（scenario_core::king_belief） ----------

export interface KingBeliefSquare {
  sq: string;
  /** taint 込みの全粒子での割合 [0,1] */
  all: number;
  /** 厳密整合の粒子だけでの割合 [0,1]。厳密が全滅していれば全て 0 */
  strict: number;
}

export interface KingBelief {
  /** 信念を持っている側（= その局面の手番側）。相手玉の位置についての分布 */
  side: Color;
  squares: KingBeliefSquare[];
  truth: string | null;
  unique: number;
  strictUnique: number;
  seeds: number;
  /** 王手宣言の履歴から健全に絞れる玉位置候補。ここに無いマスの信念はあり得ない */
  deduced: string[];
}

// ---------- 対局モード（play.rs） ----------

export interface PlayHint {
  from: string | null; // 打ちのときは null
  role: Role;
  to: string;
  promotion: "none" | "optional" | "forced";
}

export interface PlayOutcome {
  winner: Color | null;
  reason: string; // checkmate | stalemate | resign | foul_limit
}

export interface PlayView {
  /** [先手, 後手] の表示名（人間は "人間"、bot はエンジン名） */
  names: [string, string];
  /** [先手, 後手] の seed（人間側は null） */
  seeds: [number | null, number | null];
  /** [先手が bot か, 後手が bot か] */
  isBot: [boolean, boolean];
  budgetMs: number;
  /** 人間側の色。bot 同士の観戦では null */
  humanColor: Color | null;
  /** 真実の局面（隠す表示はフロント側で行う） */
  snapshot: Snapshot;
  /** 人間の手番のときだけ非空 */
  hints: PlayHint[];
  /** このコマンドで起きたイベント */
  events: string[];
  totalMoves: number;
  outcome: PlayOutcome | null;
  /** bot の直前の手で取られた人間の駒のマス */
  capturedSquare: string | null;
}

export const listScenarios = () => invoke<ScenarioInfo[]>("list_scenarios");
export const getEngines = () => invoke<string[]>("engines");
export const loadKifu = (path: string) => invoke<KifuData>("load_kifu", { path });
export const evalTally = (
  runId: number,
  path: string,
  ply: number,
  engine: string,
  trials: number,
  budgetMs: number,
) => invoke<TallyResult>("eval_tally", { runId, path, ply, engine, trials, budgetMs });
export const evalRanking = (
  path: string,
  ply: number,
  engine: string,
  seed: number,
  budgetMs: number,
) => invoke<RankingResult>("eval_ranking", { path, ply, engine, seed, budgetMs });
export const evalKingBelief = (
  path: string,
  ply: number,
  seeds: number,
  budgetMs: number,
) => invoke<KingBelief>("eval_king_belief", { path, ply, seeds, budgetMs });
/** 玉位置ビリーフの推定器キャッシュを捨てる（棋譜を読み直したとき） */
export const clearKingBeliefCache = () => invoke<void>("clear_king_belief_cache");
export const cancelEval = (runId: number) => invoke<void>("cancel_eval", { runId });
export const playStart = (
  engine: string,
  humanColor: Color,
  seed: number,
  budgetMs: number,
) => invoke<PlayView>("play_start", { engine, humanColor, seed, budgetMs });
/** bot 同士の対局（観戦モード）を開始する */
export const playStartBots = (
  senteEngine: string,
  senteSeed: number,
  goteEngine: string,
  goteSeed: number,
  budgetMs: number,
) =>
  invoke<PlayView>("play_start_bots", {
    senteEngine,
    senteSeed,
    goteEngine,
    goteSeed,
    budgetMs,
  });
export const playHumanMove = (usi: string) => invoke<PlayView>("play_human_move", { usi });
export const playBotMove = () => invoke<PlayView>("play_bot_move");
export const playResign = () => invoke<PlayView>("play_resign");
export const playView = () => invoke<PlayView>("play_view");
export const playExport = (fileName: string, ply: number | null, desc: string | null) =>
  invoke<string>("play_export", { fileName, ply, desc });
