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
}

export interface RankingResult {
  engine: string;
  side: Color;
  seed: number;
  chosen: string;
  ranking: CandidateScore[];
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
export const cancelEval = (runId: number) => invoke<void>("cancel_eval", { runId });
