//! 王手中の回避手選択のための制約推論。
//!
//! 対人対局の分析（records/ 2026-07-08）で、反則の86%が「王手中に攻め駒の位置が
//! 分からず回避候補を手探りで試す」ことによるバーストだった。粒子が枯渇する終盤に
//! 集中するため、粒子に依存しない専用の推論を用意する:
//!
//! - 王手駒の仮説 = 自玉を攻撃しうる（マス, 駒種）の全列挙。自駒の配置は既知なので、
//!   自駒に遮られる仮説は除外できる
//! - この手番で出した反則1つごとに「その手が合法になるはずだった仮説」を減衰させる
//!   （硬い消去にしないのは、反則の原因が別の隠れ駒でもありうるため）
//! - 粒子が生きていれば、粒子中の実際の王手駒に投票させて仮説を重み付けする
//! - 各回避候補の「仮説の下で王手を解消する確率」を返し、評価側が事前確率に使う
//!
//! 両王手・王手駒が紐つきの場合は単一駒仮説では表せないが、反則のたびに減衰が
//! かかるので数手で正しい回避に収束する（従来は同型の反則を繰り返していた）。

use std::collections::HashMap;

use crate::board::Coord;
use crate::model::GameModel;
use crate::observation::ObservationLog;
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{unpromote_role, Position, ShogiMove};

/// 王手駒になりうる駒種（玉は王手できない）
const CHECKER_ROLES: [Role; 13] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
    Role::Tokin,
    Role::Promotedlance,
    Role::Promotedknight,
    Role::Promotedsilver,
    Role::Horse,
    Role::Dragon,
];

/// 反則が仮説で説明できない（仮説の下では合法だったはず）ときの減衰係数。
/// 0にしない: 反則の真因が別の隠れ駒（経路封鎖・別の利き）の可能性があるため
const UNEXPLAINED_FOUL_DECAY: f64 = 0.15;

/// 「単独仮説では合法」だった手が実際にも合法だった率の実測値
/// （アリーナ400局・王手中2481決定、bin/analyze の拡張で計測 2026-07-29）。
/// legal_under は仮説の王手駒1枚しか置かないため、見えない支え駒・利きの
/// 被覆を無視する。この過信は**玉の手に集中する**（支え・被覆は玉の手に
/// しか効かない）: 玉で王手駒捕獲 73%（405件）・玉の後退 81%・玉の横/前進
/// 75% に対し、玉以外の捕獲 97%・打ち 100%。放置すると「支え付き
/// 王手駒を玉で取る手」を p=1.00 と断言して反則する健全性違反になる
/// （実測: p=1.00 断言の反則が400局で8件中6件が玉での王手駒捕獲）。
/// ユーザー指導の強い王手の構造②「王手駒は見えない駒に支えられ玉で
/// 取れない」が自玉の受け側の盲点になっていた。
///
/// **割引は玉での王手駒（仮説マス）捕獲だけに掛ける**: 玉の逃げ（75〜81%）
/// まで一律に割り引いた第1版は、較正は正しいのにアリーナ3シードのペア比較で
/// 全て悪化した（−1.0/−1.0/−3.5pt、2026-07-29）。逃げの割引は選択を
/// p_max 0.5〜0.9 の不確実な合駒・捕獲プローブへ押し出し、反則が増える
/// （ソルバー argmax シミュでも 611→630 と悪化を予告していた）。
/// 玉捕獲の割引は「玉以外の駒で取る手（97%正確）」への安全な誘導なので害がない
const KING_CAPTURE_LEGAL_PRIOR: f64 = 0.73;

/// 玉での王手駒捕獲の過信補正の強さ（0 = 従来挙動、1 = 実測事前確率をフルに適用）。
/// `TSUITATE_CHECK_KING_PRIOR_W` で上書き可（切り戻し・スイープ用。
/// 凍結版はこの名前を知らないので -f env= は候補側にだけ効く）
fn king_prior_w() -> f64 {
    static W: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_KING_PRIOR_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)
    })
}

/// 信念ネット占有による王手駒仮説の事前（`TSUITATE_CHECK_BELIEF_OCC_W`、
/// 既定 **0** = 無効、作業点 1.0）。
///
/// 仮説は自玉へ利きうる全（マス,駒種）なので、空きマスの幻仮説が生存すると
/// 正しい捕獲の `resolve_probability` が薄まる（kakutori の残ギャップ）。
/// pointwise ネットは「今どの駒が王手しているか」は表現できないが、
/// **空きマスに王手駒は居ない**という必要条件は占有較正（対数損失 0.36 /
/// AUC 0.89）で粒子より当たる。仮説重みへ ` (1−w)+w·p_occ ` を掛ける。
/// 床 `CHECK_BELIEF_OCC_FLOOR` で仮説を殺さない（健全性: 真の王手駒を
/// ネットが空きと見ても列挙から落とさない）。粒子が王手を投票できるときは
/// 掛けない（乗算だと床×投票が幻の高占有仮説に負ける）。ブラインドのとき
/// だけ、捕獲マスブーストの後に掛ける。直前の捕獲マスは観測で占有確定
/// なので乗らない。
/// 凍結版はこの名前を知らないので `-f env=` は候補側にだけ効く。
fn check_belief_occ_w() -> f64 {
    static W: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_BELIEF_OCC_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && *v >= 0.0)
            .unwrap_or(CHECK_BELIEF_OCC_W)
    })
}

/// 占有事前の既定。0 で従来（一様仮説）。1.0 は作業点（env でオン）
const CHECK_BELIEF_OCC_W: f64 = 0.0;
/// ネットが空きと見ても仮説を残す床（健全性）
const CHECK_BELIEF_OCC_FLOOR: f64 = 0.05;

/// マス占有 p と重み w から仮説への乗数を返す。w=0 は 1、w=1 は
/// `p.max(FLOOR)`。w は [0,1] にクリップ（負の乗数を出さない）。
/// テストと本番で同じ式を使う
fn occupancy_prior(p_occ: f64, w: f64) -> f64 {
    if w <= 0.0 {
        return 1.0;
    }
    let w = w.min(1.0);
    let p = p_occ.clamp(CHECK_BELIEF_OCC_FLOOR, 1.0);
    (1.0 - w) + w * p
}

/// **王手駒仮説のマス単位正規化**（`TSUITATE_CHECK_HYP_PRIOR`、既定 **0** =
/// 従来の一様、作業点 1.0）。
///
/// 仮説は（マス, 駒種）の直積で列挙され、重みは一様 1.0 だった。すると
/// **P(王手駒のマス) ∝ そのマスから王手になりうる駒種の個数**になる。
/// 自玉の隣接マスは金・と金・成香・成桂・成銀・馬・龍・飛…と 8〜10 通りが
/// 立つのに対し、離れた飛び駒のマスは飛/龍の 2 通りしか立たない。
/// つまり**隣接の幻仮説が距離のある真の王手駒より構造的に 4〜5 倍重い**。
/// 実測（arena-check-foul01、真の王手駒は 1一に打たれた飛車）: 2一の
/// 8仮説が質量 0.33 を占め、1一の 2仮説は 0.048 しかない。正しい合駒
/// N*2a の p_legal は 0.041 まで薄まり、幻の 3二/2二捕獲（gain 17）が
/// 7連続反則を生んだ。kakutori の「離れた角の捕獲が薄まる」残ギャップも同型。
///
/// w=1 で各マスの仮説の重み和を 1 に揃える（= P(マス) が一様、
/// P(駒種|マス) が在庫枚数×成り割引に比例）。w=0 は従来どおり。
/// **仮説は消さない**（重みは常に正）ので健全性は不変。
/// 凍結版はこの名前を知らないので `-f env=` は候補側にだけ効く
fn hyp_prior_w() -> f64 {
    static W: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_HYP_PRIOR")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(CHECK_HYP_PRIOR_W)
    })
}

/// マス単位正規化の既定（0 = 従来挙動）
const CHECK_HYP_PRIOR_W: f64 = 0.0;

/// **反則の証拠としての強さを手の種類で分ける**（`TSUITATE_CHECK_FOUL_EVIDENCE`、
/// 既定 **0** = 従来の一律、作業点 1.0）。
///
/// `observe_foul` は「その仮説の下では合法だったはずの手が反則になった」ことを
/// 一律 `UNEXPLAINED_FOUL_DECAY`（0.15）で減衰させていた。しかし
/// **単独仮説で合法だった手が実際にも合法だった率は手の種類で大きく違う**
/// （アリーナ400局・王手中2481決定の実測、KING_CAPTURE_LEGAL_PRIOR の doc）:
/// 玉で王手駒を取る 73% / 玉の逃げ 75〜81% / 玉以外の捕獲 97% / 打ち 100%。
/// つまり玉の手の反則は「別の隠れ駒が覆っていた」で説明できる余地が 2〜3 割
/// あるのに対し、打ち・合駒・非玉の捕獲の反則は**その仮説そのものを強く否定
/// する**（ほぼ確実に王手駒はそこではない）。一律 0.15 は前者を殺しすぎ・
/// 後者を殺し足りない。
///
/// w=1 で実測どおりの減衰（玉の逃げ 0.22 / 玉の捕獲 0.27 / 非玉の捕獲 0.03 /
/// 打ち・合駒 0.05）に切り替える。w=0 は従来。**resolve_probability 側は
/// 触らない**: 玉の逃げの解消確率まで割り引いた 2026-07-29 の第1版は
/// 較正が正しいのにアリーナ3シードで全て悪化した（不確実な合駒・捕獲プローブへ
/// の押し出し）。ここは「証拠の重み付け」だけを直す。
/// 凍結版はこの名前を知らないので `-f env=` は候補側にだけ効く
fn foul_evidence_w() -> f64 {
    static W: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_FOUL_EVIDENCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(CHECK_FOUL_EVIDENCE_W)
    })
}

/// 反則の証拠の強さを手の種類で分けるかどうかの既定（0 = 従来の一律）
const CHECK_FOUL_EVIDENCE_W: f64 = 0.0;

/// **同一手番で玉の手が反則になったあと、残りの玉の手の解消確率を割り引く**
/// （`TSUITATE_CHECK_ESCAPE_DECAY`、既定 **1.0** = 従来挙動、作業点 0.6）。
///
/// `legal_under` は仮説の王手駒1枚しか置かないので、**見えない第2の駒が
/// 逃げ先を覆っている**場合を表現できない（実測: 単独仮説で合法な玉の逃げが
/// 実際に合法だったのは 75〜81%）。一律に割り引いた 2026-07-29 の第1版は
/// 較正が正しいのにアリーナ3シードで全敗した（不確実な合駒・捕獲プローブへの
/// 押し出し）。
///
/// ここは**証拠が出たあとだけ**割り引く: この手番で玉の手が反則になったなら、
/// 「見えない駒が自玉の周りを覆っている」が確定した（仮説では説明できない
/// 反則なら仮説側は既に減衰しているが、玉の隣接マスの被覆は**相関する** =
/// 1枚の飛角金銀は複数の逃げ先を同時に覆う）。玉の手の解消確率へ
/// `decay^(この手番の玉の反則回数)` を掛け、合駒・捕獲へ早めに移らせる。
/// 反則0回の決定は完全に不変。
///
/// 発端は `bin/mine_check` の arena-check-foul01（玉 3一、真の王手駒は
/// 1一の飛車、逃げ先 3二/2二/2一 は見えない角と成香が覆っていて全滅）:
/// 3二 0.74 / 2一 0.51 / 2二 0.49 と高いまま3回焼き、正解の合駒 N*2a は
/// 0.04〜0.20 で最後まで浮かなかった。
/// 凍結版はこの名前を知らないので `-f env=` は候補側にだけ効く
fn escape_decay() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_ESCAPE_DECAY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(CHECK_ESCAPE_DECAY)
    })
}

/// `escape_decay` の既定（1.0 = 従来挙動）
const CHECK_ESCAPE_DECAY: f64 = 1.0;

/// 実測 75〜81% の中央付近（玉の逃げが単独仮説で合法だった率）
const KING_ESCAPE_LEGAL_PRIOR: f64 = 0.78;
/// 玉以外の駒での捕獲の反則は仮説をほぼ否定する（実測 97%）
const NONKING_CAPTURE_FOUL_DECAY: f64 = 0.03;
/// 打ち・合駒の反則も仮説をほぼ否定する（実測 100%。床として 0.05 を置く）
const QUIET_FOUL_DECAY: f64 = 0.05;

/// 成駒の事前割引（同じ基本駒種の在庫のうち「既に成っている」割合の近似）
const PROMOTED_PRIOR: f64 = 0.5;

/// 駒種の在庫事前: 相手が持ちうる枚数 n（基本駒種で数える）に成り割引を掛ける。
/// 歩 9 枚 > 金銀桂香 2 枚 > 飛角 1 枚 の序列がそのまま P(駒種|マス) になる
fn role_inventory_prior(role: Role, n: i32) -> f64 {
    let base = n.max(0) as f64;
    if unpromote_role(role) == role {
        base
    } else {
        base * PROMOTED_PRIOR
    }
}

/// 粒子のうち自玉が王手されているものが1つでもあれば投票できる。
/// 占有事前はこのとき掛けない（投票と乗算で喧嘩するため）
fn particles_vote_check(particles: &[(&Position, f64)], my_color: Color) -> bool {
    particles.iter().any(|(pos, _)| pos.in_check(my_color))
}

/// 粒子投票の強さ（全粒子が一致した仮説は一様仮説の 1+PARTICLE_VOTE_W 倍）
const PARTICLE_VOTE_W: f64 = 8.0;

/// 王手宣言と同時に自駒が取られたマス（= 直前の相手手の着地点）の仮説ブースト。
/// 「そこへ来た駒がそのまま王手している」は最有力の説明で、開き王手（動いた駒と
/// 別の駒が王手）は少数派。間違っていても反則の減衰で自然に解ける。
/// `TSUITATE_CHECK_CAPTURE_BOOST` で上書き可（切り戻し・スイープ用。1.0 で無効。
/// 凍結版はこの名前を知らない）
const CAPTURE_CHECKER_BOOST: f64 = 4.0;

/// 開き王手の幾何学判定による仮説除去（`prune_infeasible_discovered_checks`）。
/// `TSUITATE_CHECK_CAPTURE_PRUNE=0` で無効（切り戻し用。凍結版はこの名前を知らない）
fn capture_prune_enabled() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("TSUITATE_CHECK_CAPTURE_PRUNE").map_or(true, |v| v != "0"))
}

fn capture_checker_boost() -> f64 {
    static W: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_CAPTURE_BOOST")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && *v >= 1.0)
            .unwrap_or(CAPTURE_CHECKER_BOOST)
    })
}

/// 残存脅威（threat_of）の重み: 王手駒に攻撃されている自駒の交換価値に掛ける係数
const THREAT_MATERIAL_W: f64 = 0.5;
/// 残存脅威の重み: 自玉の隣接マス1つへの利き（逃げ場を縛り続ける圧力）の価値
const THREAT_KING_ZONE_W: f64 = 0.5;

struct Hypothesis {
    square: Coord,
    role: Role,
    weight: f64,
}

pub struct CheckSolver {
    /// 自駒＋持ち駒だけを置いたスパース盤面（手番=自分）。仮説の駒を載せて使う
    base: Position,
    my_color: Color,
    hypotheses: Vec<Hypothesis>,
    /// 仮説ごとの残存脅威（threat_of）の遅延キャッシュ。hypotheses と同じ並び
    threat_cache: Vec<Option<f64>>,
    /// 反則の証拠の強さを手の種類で分ける強さ（`foul_evidence_w`）。
    /// 構築時に env を1回読んで持つ（テストから明示指定できるように）
    foul_evidence: f64,
    /// 玉の手の解消確率の割引率（`escape_decay`）
    escape_decay: f64,
    /// この手番で反則になった玉の手の回数（`escape_decay` の指数）
    king_fouls: u32,
}

impl CheckSolver {
    /// 王手中の view から作る。自玉が見つからない等で推論できなければ None。
    /// particles はソフト救済の重みつき（strategy.rs の評価サンプルと同じ）
    pub fn new(
        view: &PlayerView,
        particles: &[(&Position, f64)],
        fouls_this_turn: &[ShogiMove],
        log: &ObservationLog,
    ) -> Option<CheckSolver> {
        Self::with_knobs(
            view,
            particles,
            fouls_this_turn,
            log,
            hyp_prior_w(),
            foul_evidence_w(),
            escape_decay(),
        )
    }

    /// ノブを明示指定して作る（`new` は env から読んで委譲する。テスト用）
    fn with_knobs(
        view: &PlayerView,
        particles: &[(&Position, f64)],
        fouls_this_turn: &[ShogiMove],
        log: &ObservationLog,
        prior_w: f64,
        foul_evidence: f64,
        escape_decay: f64,
    ) -> Option<CheckSolver> {
        let my_color = view.your_color;
        let mut base = Position::empty(my_color);
        for piece in &view.your_pieces {
            let sq = crate::board::parse_usi_square(&piece.square)?;
            base.set(
                sq,
                Some(crate::shogi::Piece {
                    color: my_color,
                    role: piece.role,
                }),
            );
        }
        for (&role, &count) in &view.your_hand {
            base.set_hand(my_color, role, count as u8);
        }
        base.king_square(my_color)?;

        // 位置が既知の敵駒（自駒が死んだマス = 敵駒がそこへ来た。取り返し済みは除く）を
        // 盤に載せる。回避先がこれらの利きに覆われているかを全仮説共通で判定できる
        // （対人実戦: 5三の既知の成駒が 4二/5二/6二 を覆っているのに順に試して4反則）。
        // **直近8手以内**の新鮮な情報に限定する: 古いマスは駒が動いて陳腐化しやすく、
        // 幻の駒が合法な逃げ場を塞ぐ害が実測で上回った（vs v5 アブレーション 2026-07-10）。
        // 駒種は不明なので粒子の多数決、なければ成駒の最頻・金動き（と金）で近似する
        for sq in known_enemy_squares(log, view.move_number.saturating_sub(8)) {
            if base.piece_at(sq).is_some() {
                continue;
            }
            let role =
                particle_majority_role(particles, my_color.other(), sq).unwrap_or(Role::Tokin);
            base.set(
                sq,
                Some(crate::shogi::Piece {
                    color: my_color.other(),
                    role,
                }),
            );
            // 近似駒種が王を攻撃してしまう（本物の王手駒と区別できない）配置は
            // 仮説列挙を壊すので載せない
            if base.in_check(my_color) {
                base.set(sq, None);
            }
        }

        let mut solver = CheckSolver {
            base,
            my_color,
            hypotheses: vec![],
            threat_cache: vec![],
            foul_evidence,
            escape_decay,
            king_fouls: 0,
        };
        solver.enumerate(&opponent_role_counts(view, log), prior_w);
        if solver.hypotheses.is_empty() {
            return None;
        }
        solver.threat_cache = vec![None; solver.hypotheses.len()];
        // 王手宣言と同時に自駒が取られた（= 直前の相手手がそのマスへ来た）なら、
        // そのマスの仮説を強める。粒子投票より前ではなく後でもよい（乗算なので
        // 順序は結果に影響しない）が、粒子が退化していて投票が無情報でも
        // このブーストだけで「来た駒が王手駒」の説明が支配的になる
        // （king-evade.kif の実測: 5二で金を取られた直後の王手で、taint 落ち時は
        // 5二の仮説が既知敵駒の近似駒種に塞がれて列挙されず、解消確率が
        // 7一 0.55 / 6二 0.53 とほぼ無差別だった）
        let last_opp_capture = log
            .events()
            .iter()
            .rev()
            .find_map(|e| match e {
                crate::observation::Observation::OpponentMoved {
                    captured_my_piece_at,
                    ..
                } => Some(
                    captured_my_piece_at
                        .as_deref()
                        .and_then(crate::board::parse_usi_square),
                ),
                _ => None,
            })
            .flatten();
        if let Some(sq) = last_opp_capture {
            // 捕獲マス以外の仮説は「開き王手」としてしか成立しない。自駒配置だけで
            // 開き王手が幾何学的に不可能な仮説は**除去**する（健全な演繹）
            if capture_prune_enabled() {
                solver.prune_infeasible_discovered_checks(sq);
            }
            for h in &mut solver.hypotheses {
                if h.square == sq {
                    h.weight *= capture_checker_boost();
                }
            }
        }
        solver.vote_by_particles(particles);
        // 信念ネット占有は**粒子が王手を投票できないときだけ**事前にする。
        // 乗算を投票の前に置くと、床 0.05 × 投票9倍 = 0.45 が占有1.0の幻仮説に
        // 負け、段階②の「王手中の再重み付けは有害」と同じ形になる
        // （700ms×3試行で kakutori 注目手 1/3→0/3）。直前の捕獲マスは
        // 観測で占有確定なので乗らない。
        let occ_w = check_belief_occ_w();
        if occ_w > 0.0 && !particles_vote_check(particles, my_color) {
            let ctx = crate::belief_features::BeliefContext::from_log(my_color, log);
            let occ = crate::belief_nn::board_occupancy(&ctx);
            solver.reweight_by_occupancy(&occ, occ_w, last_opp_capture);
        }
        for foul in fouls_this_turn {
            solver.observe_foul(foul);
        }
        Some(solver)
    }

    /// 自玉を攻撃しうる（マス, 駒種）を全列挙する。
    /// 相手が1枚も持ちえない駒種（総数制約）は仮説から外す。
    ///
    /// **既知の敵駒のマスも仮説の対象にする**（2026-07-29）: そのマスの駒種は
    /// 粒子の多数決による近似でしかなく、近似が「王手しない駒種」を返すと
    /// 真の王手駒マスが仮説集合から丸ごと消える（king-evade.kif: 5二で金を
    /// 取られた直後の王手なのに、taint 粒子の多数決が5二を非王手駒と近似して
    /// 仮説が列挙されなかった）。健全性（真の王手駒を仮説から落とさない）を
    /// 優先し、近似駒種は他の仮説の判定時の遮蔽・利き被覆にだけ使う
    fn enumerate(&mut self, opp_counts: &HashMap<Role, i32>, prior_w: f64) {
        let opp = self.my_color.other();
        let king = self
            .base
            .king_square(self.my_color)
            .expect("new で確認済み");
        for file in 1..=9i8 {
            for rank in 1..=9i8 {
                let sq = Coord { file, rank };
                let existing = self.base.piece_at(sq);
                if existing.is_some_and(|p| p.color == self.my_color) {
                    // 自駒のあるマスに王手駒はいない
                    continue;
                }
                if sq == king {
                    continue;
                }
                // このマスから王手になる駒種を先に集め、駒種事前（在庫枚数×成り
                // 割引）で**マス単位に正規化**する（`hyp_prior_w` の doc 参照）
                let mut roles: Vec<(Role, f64)> = vec![];
                for role in CHECKER_ROLES {
                    let Some(&n) = opp_counts.get(&unpromote_role(role)) else {
                        continue;
                    };
                    if n <= 0 {
                        continue;
                    }
                    self.base
                        .set(sq, Some(crate::shogi::Piece { color: opp, role }));
                    let checks = self.base.in_check(self.my_color);
                    self.base.set(sq, existing);
                    if checks {
                        roles.push((role, role_inventory_prior(role, n)));
                    }
                }
                let total: f64 = roles.iter().map(|(_, p)| *p).sum();
                for (role, p) in roles {
                    // w=0 で従来の一様（(マス,駒種)ごとに 1.0）、w=1 でマス単位
                    // 正規化（Σ_駒種 = 1）。間は線形補間
                    let normalized = if total > 0.0 { p / total } else { 1.0 };
                    let weight = (1.0 - prior_w) + prior_w * normalized;
                    self.hypotheses.push(Hypothesis {
                        square: sq,
                        role,
                        weight,
                    });
                }
            }
        }
    }

    /// **捕獲＋王手の開き王手判定**（2026-08-19、arena-check-flee02 が発端。
    /// ユーザー指摘「6七で取られたことは分かっているのだから同銀を調べるべき」）。
    ///
    /// 直前の相手手が自駒を S で取り、同時に王手宣言が出たなら、王手駒は
    /// (a) S へ来た駒そのもの、または (b) その駒が出て行った元マス O を通る線上の
    /// 飛び駒（開き王手）のどちらか。(b) は自駒の配置だけで可否が決まる:
    /// 仮説 Q（≠S）は、玉–Q が直線で、その間の全マスに自駒がなく、S 自身が
    /// その間に無く（来た駒が自ら線を塞ぐ）、間のどこかのマス O から S へ
    /// 何らかの駒種で動ける（隣接・桂跳び・自駒に遮られない直線）ときだけ残す。
    /// 隣接からの攻撃・桂・金銀などの非飛び駒の仮説は「前の手番から既に
    /// 王手だった」ことになるので不可（自分の直前の手は合法 = 王手ではなかった）。
    ///
    /// flee02 の実測: 玉6八・銀7八・桂5八・歩5七/8六で、6七の捕獲＋王手の
    /// 非6七仮説が 18% 残っていた（CAPTURE_CHECKER_BOOST ×4 は弱めるだけ）。
    /// 残り反則1回の foul_cost=60 では p_legal 0.82 の同銀（gain 14.7）が
    /// 0.989 の玉逃げ（1.26）と僅差になり 2〜3/10 で逃げていた。除去すれば
    /// 同銀は（隠れた飛車のピンを除き）確実に解消する手になる。
    /// 全仮説が消える場合は何もしない（健全性の最後の砦）。
    /// `TSUITATE_CHECK_CAPTURE_PRUNE=0` で無効（切り戻し用）
    fn prune_infeasible_discovered_checks(&mut self, s: Coord) {
        let king = self
            .base
            .king_square(self.my_color)
            .expect("new で確認済み");
        let my = self.my_color;
        let own_at = |c: Coord| self.base.piece_at(c).is_some_and(|p| p.color == my);
        let on_board = |c: Coord| (1..=9).contains(&c.file) && (1..=9).contains(&c.rank);
        // O から S へ何らかの相手駒が1手で動けるか（駒種不問の幾何学判定）
        let opp_knight_dr: i8 = if my.other() == Color::Sente { -2 } else { 2 };
        let reachable = |o: Coord| -> bool {
            let df = s.file - o.file;
            let dr = s.rank - o.rank;
            if df == 0 && dr == 0 {
                return false;
            }
            if df.abs() <= 1 && dr.abs() <= 1 {
                return true; // 隣接（玉・金・銀・成駒など）
            }
            if df.abs() == 1 && dr == opp_knight_dr {
                return true; // 桂
            }
            if df == 0 || dr == 0 || df.abs() == dr.abs() {
                // 直線（飛・角・香・竜・馬）: 間に自駒が無ければ可
                let sf = df.signum();
                let sr = dr.signum();
                let mut c = Coord {
                    file: o.file + sf,
                    rank: o.rank + sr,
                };
                while c != s {
                    if own_at(c) {
                        return false;
                    }
                    c = Coord {
                        file: c.file + sf,
                        rank: c.rank + sr,
                    };
                }
                return true;
            }
            false
        };
        let feasible = |q: Coord| -> bool {
            if q == s {
                return true;
            }
            let df = q.file - king.file;
            let dr = q.rank - king.rank;
            if !(df == 0 || dr == 0 || df.abs() == dr.abs()) {
                return false; // 直線上にない（桂など）= 開き王手になりえない
            }
            if df.abs().max(dr.abs()) < 2 {
                return false; // 隣接 = 前の手番から王手だったことになる
            }
            let sf = df.signum();
            let sr = dr.signum();
            let mut between = vec![];
            let mut c = Coord {
                file: king.file + sf,
                rank: king.rank + sr,
            };
            while c != q {
                if !on_board(c) {
                    return false;
                }
                between.push(c);
                c = Coord {
                    file: c.file + sf,
                    rank: c.rank + sr,
                };
            }
            // 間に自駒があれば遮られる。来た駒が間にいれば自ら塞ぐ
            if between.iter().any(|&b| own_at(b) || b == s) {
                return false;
            }
            between.iter().any(|&o| reachable(o))
        };
        let kept: Vec<Hypothesis> = self
            .hypotheses
            .iter()
            .filter(|h| feasible(h.square))
            .map(|h| Hypothesis {
                square: h.square,
                role: h.role,
                weight: h.weight,
            })
            .collect();
        if !kept.is_empty() {
            self.hypotheses = kept;
            self.threat_cache = vec![None; self.hypotheses.len()];
        }
    }

    /// 信念ネット占有で仮説重みを再配分する。同じマスの駒種仮説は同じ
    /// 乗数。w=0 は無変化。床付きなので仮説は消えない。
    /// `skip`（直前の捕獲マス）は観測で占有が確定しているので乗らない
    fn reweight_by_occupancy(&mut self, occ: &[f64; 81], w: f64, skip: Option<Coord>) {
        if w <= 0.0 {
            return;
        }
        for h in &mut self.hypotheses {
            if skip.is_some_and(|sq| sq == h.square) {
                continue;
            }
            let p = occ[crate::belief_features::sq_index(h.square)];
            h.weight *= occupancy_prior(p, w);
        }
    }

    /// 粒子中の実際の王手駒に投票させる（粒子が健全なら仮説が鋭くなる）。
    /// ソフト救済の粒子は重みぶんだけ薄く投票する
    fn vote_by_particles(&mut self, particles: &[(&Position, f64)]) {
        let opp = self.my_color.other();
        let mut voters = 0.0f64;
        let mut votes: Vec<f64> = vec![0.0; self.hypotheses.len()];
        for (pos, w) in particles {
            if !pos.in_check(self.my_color) {
                continue; // 王手を反映していない粒子は情報にならない
            }
            voters += w;
            for (i, h) in self.hypotheses.iter().enumerate() {
                if pos
                    .piece_at(h.square)
                    .is_some_and(|p| p.color == opp && p.role == h.role)
                {
                    // 粒子上でその駒が実際に王を攻撃しているかまでは見ない
                    // （enumerate 済みの仮説は自駒配置的に攻撃可能）
                    votes[i] += w;
                }
            }
        }
        if voters <= 0.0 {
            return;
        }
        for (h, &v) in self.hypotheses.iter_mut().zip(&votes) {
            h.weight *= 1.0 + PARTICLE_VOTE_W * (v / voters);
        }
    }

    /// 単独仮説（王手駒1枚だけの盤面）で合法だった手が、実際にも合法である
    /// 事前確率。**玉でその仮説の王手駒マスを取る手だけ**実測値で割り引く
    /// （定数の doc 参照。玉の逃げまで割り引くとアリーナで悪化する）。
    /// hyp_square = 判定対象の仮説の王手駒マス
    fn single_hyp_legal_prior(&self, mv: &ShogiMove, hyp_square: Coord) -> f64 {
        let ShogiMove::Board { from, to, .. } = *mv else {
            return 1.0;
        };
        if to != hyp_square || self.base.king_square(self.my_color) != Some(from) {
            return 1.0;
        }
        1.0 - king_prior_w() * (1.0 - KING_CAPTURE_LEGAL_PRIOR)
    }

    /// この手番の反則を観測: 仮説の下で合法だったはずの手が反則になった
    /// → その仮説の重みを減衰させる。
    /// 玉でその仮説の王手駒マスを取る手の反則は「仮説が正しくても隠れた
    /// 支えで反則になる」確率が高い（実測27%）ので、減衰を 1 − 事前確率
    /// まで緩めて真の仮説を殺しにくくする（他の手は従来の減衰のまま）
    fn observe_foul(&mut self, foul: &ShogiMove) {
        if let ShogiMove::Board { from, .. } = *foul {
            if self.base.king_square(self.my_color) == Some(from) {
                self.king_fouls += 1;
            }
        }
        for i in 0..self.hypotheses.len() {
            if self.legal_under(i, foul) {
                let decay = self.foul_decay_for(foul, self.hypotheses[i].square);
                self.hypotheses[i].weight *= decay;
            }
        }
    }

    /// 反則1件が「その仮説の下では合法だったはず」のときの減衰係数。
    /// `foul_evidence_w` が 0 なら従来どおり一律（玉での仮説マス捕獲だけ緩め）、
    /// 1 なら手の種類ごとの実測合法率に基づく（`foul_evidence_w` の doc）
    fn foul_decay_for(&self, foul: &ShogiMove, hyp_square: Coord) -> f64 {
        let legacy =
            UNEXPLAINED_FOUL_DECAY.max(1.0 - self.single_hyp_legal_prior(foul, hyp_square));
        let w = self.foul_evidence;
        if w <= 0.0 {
            return legacy;
        }
        let measured = match *foul {
            ShogiMove::Drop { .. } => QUIET_FOUL_DECAY,
            ShogiMove::Board { from, to, .. } => {
                if self.base.king_square(self.my_color) == Some(from) {
                    if to == hyp_square {
                        (1.0 - KING_CAPTURE_LEGAL_PRIOR).max(UNEXPLAINED_FOUL_DECAY)
                    } else {
                        1.0 - KING_ESCAPE_LEGAL_PRIOR
                    }
                } else if to == hyp_square || self.base.piece_at(to).is_some() {
                    NONKING_CAPTURE_FOUL_DECAY
                } else {
                    QUIET_FOUL_DECAY
                }
            }
        };
        (1.0 - w) * legacy + w * measured
    }

    /// 仮説 i の下で（他の隠れ駒を無視して）mv が合法か = 王手を解消するか。
    /// 仮説マスが既知の敵駒マス（近似駒種を載せてある）なら復元する
    fn legal_under(&mut self, i: usize, mv: &ShogiMove) -> bool {
        let h = &self.hypotheses[i];
        let piece = crate::shogi::Piece {
            color: self.my_color.other(),
            role: h.role,
        };
        let sq = h.square;
        let prev = self.base.piece_at(sq);
        self.base.set(sq, Some(piece));
        let legal = self.base.is_legal(mv);
        self.base.set(sq, prev);
        legal
    }

    /// 候補手が王手を解消する確率（仮説の重み付き割合）。
    /// 玉でその仮説の王手駒マスを取る手は「単独仮説で合法」でも隠れた支えで
    /// 反則になりうるので、実測の事前確率（single_hyp_legal_prior）を掛けて
    /// 過信を抑える（= p=1.00 の断言をしない。玉以外の捕獲は実測97%なので
    /// そのまま = 支え付き王手駒は玉以外の駒で取る序列へ自然に誘導される）
    pub fn resolve_probability(&mut self, mv: &ShogiMove) -> f64 {
        let mut total = 0.0;
        let mut resolved = 0.0;
        for i in 0..self.hypotheses.len() {
            let w = self.hypotheses[i].weight;
            total += w;
            if self.legal_under(i, mv) {
                resolved += w * self.single_hyp_legal_prior(mv, self.hypotheses[i].square);
            }
        }
        if total <= 0.0 {
            return 0.5; // 全仮説が死んだ（両王手など）: 情報なしに戻す
        }
        resolved / total * self.escape_factor(mv)
    }

    /// 玉の手の解消確率に掛ける割引（`escape_decay` の doc）。
    /// この手番で玉の手が反則になっていなければ 1.0（挙動不変）
    fn escape_factor(&self, mv: &ShogiMove) -> f64 {
        if self.king_fouls == 0 || self.escape_decay >= 1.0 {
            return 1.0;
        }
        let ShogiMove::Board { from, .. } = *mv else {
            return 1.0;
        };
        if self.base.king_square(self.my_color) != Some(from) {
            return 1.0;
        }
        self.escape_decay.powi(self.king_fouls as i32)
    }

    /// mv が「王手駒仮説のマスへ、自玉以外の駒で移動して、その仮説の下で
    /// 王手が解消する」手か = 王手駒を捕獲しに行く手か。
    ///
    /// かつては p_legal フロア（CHECK_CAPTURE_P_LEGAL_FLOOR）の発動条件
    /// だったが、フロアは removal_term（仮説条件付き期待値）に置き換えられた。
    /// 診断・テスト用に残している
    pub fn captures_checker(&mut self, mv: &ShogiMove) -> bool {
        let ShogiMove::Board { from, to, .. } = *mv else {
            return false;
        };
        if self.base.king_square(self.my_color) == Some(from) {
            return false;
        }
        for i in 0..self.hypotheses.len() {
            if self.hypotheses[i].square == to && self.legal_under(i, mv) {
                return true;
            }
        }
        false
    }

    /// 仮説 i の王手駒が（王手解消後も）盤に残った場合に自陣へ残す圧力
    /// （歩価値スケール）。攻撃されている自駒の交換価値と、自玉隣接マスへの
    /// 利き（逃げ場を縛り続ける圧力）の和。候補手にほぼ依存しないので
    /// 仮説ごとに1回だけ計算してキャッシュする（着手による自駒配置の変化は
    /// 無視する近似）
    fn threat_of(&mut self, i: usize) -> f64 {
        if let Some(t) = self.threat_cache[i] {
            return t;
        }
        let (sq, role) = {
            let h = &self.hypotheses[i];
            (h.square, h.role)
        };
        let opp = self.my_color.other();
        let prev = self.base.piece_at(sq);
        self.base
            .set(sq, Some(crate::shogi::Piece { color: opp, role }));
        let targets: Vec<(Coord, Role)> = self
            .base
            .pieces()
            .filter(|(_, p)| p.color == self.my_color && p.role != Role::King)
            .map(|(c, p)| (c, p.role))
            .collect();
        let mut t = 0.0;
        for (c, r) in targets {
            if self.base.attacks(sq, c) {
                t += THREAT_MATERIAL_W * crate::strategy::exchange_value(r);
            }
        }
        if let Some(king) = self.base.king_square(self.my_color) {
            for df in -1..=1i8 {
                for dr in -1..=1i8 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let a = Coord {
                        file: king.file + df,
                        rank: king.rank + dr,
                    };
                    if (1..=9).contains(&a.file)
                        && (1..=9).contains(&a.rank)
                        && self.base.attacks(sq, a)
                    {
                        t += THREAT_KING_ZONE_W;
                    }
                }
            }
        }
        self.base.set(sq, prev);
        self.threat_cache[i] = Some(t);
        t
    }

    /// 仮説条件付きの「王手駒の除去期待値」（歩価値スケール、非負）。
    /// mv が受理された（=王手を解消した）と条件付けた仮説の事後分布で、
    /// 王手駒のマスを取る未来の +（交換価値 + 回避された残存脅威 threat_of）を
    /// 平均する。捕獲は受理された未来では王手駒が消えており、玉逃げ等の解消手は
    /// 王手駒（1五角だったなら角）が盤に残って自陣を睨み続ける、という非対称を
    /// gain 側へ伝える。p_legal（resolve_probability）は合法性しか平均しない
    /// ため、粒子が真の王手駒を外している局面ではこの差が評価のどこにも
    /// 現れず、捕獲が玉逃げに完敗していた（kakutori.kif）。
    ///
    /// **正項のみ**にする理由: 王手駒が生き残る未来に −threat を課す対称形は
    /// 候補間の相対順序こそ同じだが、合法な解消手ほぼ全員の絶対水準を沈める。
    /// min 形式の combine_score では負の gain は p_legal で割り引かれず全額
    /// 効くため、ペナルティが反則コストの水位を越えると非合法寄りのプローブが
    /// 相対的に浮上する（実測: kakutori 20試行で追加反則 4→28 に爆発）。
    /// 王手中の候補はその手番内でしか比較されないので、全体を正側へ平行移動
    /// しても選択への副作用はない。
    ///
    /// **玉による捕獲は加点しない**（captures_checker と同じ除外）: 隣接マスへの
    /// 玉移動はすべて「そのマスの王手駒仮説の捕獲」になるため、加点すると
    /// 逃げ手全員が capture 並みに膨らんで相対差が消える（実測: kakutori で
    /// 3g2g の gain 0.3→10.4）。玉捕獲は相手駒に紐があれば反則になるだけで、
    /// 駒を失わずに王手駒を排除しに行く探索プローブの非対称な価値も持たない。
    /// mv がどの仮説の下でも受理されない・仮説が全滅している場合は None
    pub fn removal_term(&mut self, mv: &ShogiMove) -> Option<f64> {
        let to = match *mv {
            ShogiMove::Board { to, .. } => to,
            ShogiMove::Drop { to, .. } => to,
        };
        let from_king = match *mv {
            ShogiMove::Board { from, .. } => self.base.king_square(self.my_color) == Some(from),
            ShogiMove::Drop { .. } => false,
        };
        let mut legal_w = 0.0;
        let mut term = 0.0;
        for i in 0..self.hypotheses.len() {
            if !self.legal_under(i, mv) {
                continue;
            }
            let (w, sq, role) = {
                let h = &self.hypotheses[i];
                (h.weight, h.square, h.role)
            };
            legal_w += w;
            if sq == to && !from_king {
                term += w * (crate::strategy::exchange_value(role) + self.threat_of(i));
            }
        }
        if legal_w <= 1e-12 {
            return None;
        }
        Some(term / legal_w)
    }

    #[cfg(test)]
    fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }

    /// 仮説分布の診断ダンプ（`TSUITATE_DEBUG_CHECK=1` のときだけ呼ばれる）。
    /// (マスの USI, 駒種, 正規化重み) を重みの降順で返す。
    /// 「正しい解消手の p_legal が薄い」局面で、どの幻仮説が質量を
    /// 持っているかを見るために使う
    pub fn debug_hypotheses(&self) -> Vec<(String, Role, f64)> {
        let total: f64 = self.hypotheses.iter().map(|h| h.weight).sum();
        let mut out: Vec<(String, Role, f64)> = self
            .hypotheses
            .iter()
            .map(|h| {
                (
                    crate::board::make_usi_square(h.square),
                    h.role,
                    if total > 0.0 { h.weight / total } else { 0.0 },
                )
            })
            .collect();
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        out
    }
}

/// 位置が既知の敵駒のマス: 自駒が取られたマス（敵駒がそこへ来た事実）のうち、
/// その後に自分が取り返しておらず、かつ since_move 手目以降の新しいもの
fn known_enemy_squares(log: &ObservationLog, since_move: u32) -> Vec<Coord> {
    let mut map: HashMap<Coord, u32> = HashMap::new();
    for e in log.events() {
        match e {
            crate::observation::Observation::OpponentMoved {
                move_number,
                captured_my_piece_at: Some(sq),
            } => {
                if let Some(c) = crate::board::parse_usi_square(sq) {
                    map.insert(c, *move_number);
                }
            }
            crate::observation::Observation::MyMove {
                usi,
                captured: Some(_),
                ..
            } => {
                if let Some(ShogiMove::Board { to, .. }) = crate::shogi::parse_usi(usi) {
                    map.remove(&to);
                }
            }
            _ => {}
        }
    }
    let mut out: Vec<Coord> = map
        .into_iter()
        .filter(|(_, mn)| *mn >= since_move)
        .map(|(c, _)| c)
        .collect();
    out.sort_by_key(|c| (c.file, c.rank));
    out
}

/// 粒子の加重多数決でそのマスの敵駒の駒種を推定する（過半に満たなければ None）。
/// ソフト救済の粒子は重みぶんだけ薄く数える
fn particle_majority_role(particles: &[(&Position, f64)], opp: Color, sq: Coord) -> Option<Role> {
    if particles.is_empty() {
        return None;
    }
    let total: f64 = particles.iter().map(|(_, w)| w).sum();
    let mut counts: HashMap<Role, f64> = HashMap::new();
    for (pos, w) in particles {
        if let Some(p) = pos.piece_at(sq) {
            if p.color == opp {
                *counts.entry(p.role).or_default() += w;
            }
        }
    }
    let (role, n) = counts.into_iter().max_by(|(ra, a), (rb, b)| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| role_order(*rb).cmp(&role_order(*ra)))
    })?;
    if n * 2.0 > total {
        Some(role)
    } else {
        None
    }
}

fn role_order(role: Role) -> u8 {
    match role {
        Role::Pawn => 0,
        Role::Lance => 1,
        Role::Knight => 2,
        Role::Silver => 3,
        Role::Gold => 4,
        Role::Bishop => 5,
        Role::Rook => 6,
        Role::King => 7,
        Role::Tokin => 8,
        Role::Promotedlance => 9,
        Role::Promotedknight => 10,
        Role::Promotedsilver => 11,
        Role::Horse => 12,
        Role::Dragon => 13,
    }
}

/// 相手が盤上・持ち駒に持ちうる駒種の枚数（基本駒種で数える）。
/// = 初期枚数 + こちらが取られた枚数 − こちらが取った枚数（自分の持ち駒）
fn opponent_role_counts(view: &PlayerView, log: &ObservationLog) -> HashMap<Role, i32> {
    let mut counts: HashMap<Role, i32> = [
        (Role::Pawn, 9),
        (Role::Lance, 2),
        (Role::Knight, 2),
        (Role::Silver, 2),
        (Role::Gold, 2),
        (Role::Bishop, 1),
        (Role::Rook, 1),
    ]
    .into();
    for (_, role) in GameModel::from_log(view.your_color, log).lost_pieces() {
        *counts.entry(unpromote_role(*role)).or_default() += 1;
    }
    for (&role, &n) in &view.your_hand {
        *counts.entry(unpromote_role(role)).or_default() -= n as i32;
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::protocol::VisiblePiece;
    use crate::shogi::parse_usi;

    fn view_with(pieces: Vec<(&str, Role)>) -> PlayerView {
        let pieces = pieces
            .into_iter()
            .map(|(sq, role)| VisiblePiece {
                square: sq.into(),
                role,
            })
            .collect();
        let mut view = crate::strategy::tests::minimal_view(pieces, HashMap::new());
        view.you_in_check = true;
        view
    }

    fn mv(usi: &str) -> ShogiMove {
        parse_usi(usi).unwrap()
    }

    #[test]
    fn enumerates_plausible_checkers() {
        // 中央の裸玉: 隣接・桂・遠距離の攻撃仮説が多数列挙される
        let view = view_with(vec![("5e", Role::King)]);
        let solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        assert!(solver.hypothesis_count() > 50);
    }

    #[test]
    fn own_pieces_block_hypotheses() {
        // 5d に自分の歩がいれば「5c の香/飛が王手」仮説は成立しない
        let open = CheckSolver::new(
            &view_with(vec![("5e", Role::King)]),
            &[],
            &[],
            &ObservationLog::default(),
        )
        .unwrap();
        let blocked = CheckSolver::new(
            &view_with(vec![("5e", Role::King), ("5d", Role::Pawn)]),
            &[],
            &[],
            &ObservationLog::default(),
        )
        .unwrap();
        assert!(blocked.hypothesis_count() < open.hypothesis_count());
    }

    #[test]
    fn known_enemy_square_is_still_enumerated_as_checker_hypothesis() {
        // 5d で自駒が取られた直後の王手。粒子の多数決が「王手しない駒種」
        // （ここでは桂: 後手桂@5dは5eを攻撃しない）を返すと、旧実装は 5d に
        // 近似駒種を載せたまま仮説列挙から除外し、真の王手駒マスが仮説集合から
        // 消えていた（king-evade.kif の故障モードB）。健全性のため、既知の
        // 敵駒マスでも王手駒仮説は列挙する
        let view = view_with(vec![("5e", Role::King)]);
        let mut log = ObservationLog::default();
        log.record(crate::observation::Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("5d".into()),
        });
        let mut particle = Position::empty(Color::Sente);
        particle.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        particle.set(
            Coord { file: 5, rank: 4 },
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::Knight,
            }),
        );
        let particles = [(&particle, 1.0)];
        let solver = CheckSolver::new(&view, &particles, &[], &log).unwrap();
        let sq = Coord { file: 5, rank: 4 };
        assert!(
            solver.hypotheses.iter().any(|h| h.square == sq),
            "既知の敵駒マス 5d にも王手駒仮説が必要"
        );
    }

    #[test]
    fn capture_square_hypotheses_are_boosted() {
        // 王手宣言と同時に自駒が取られたマス（直前の相手手の着地点）の仮説は
        // CAPTURE_CHECKER_BOOST 倍される（開き王手より「来た駒が王手駒」が優勢）。
        // 占有事前が乗ったあとも正確な 4.0 倍にはならないので、支配だけ見る
        let view = view_with(vec![("5e", Role::King)]);
        let mut log = ObservationLog::default();
        log.record(crate::observation::Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("5d".into()),
        });
        let solver = CheckSolver::new(&view, &[], &[], &log).unwrap();
        let sq = Coord { file: 5, rank: 4 };
        let boosted = solver
            .hypotheses
            .iter()
            .filter(|h| h.square == sq)
            .map(|h| h.weight)
            .fold(f64::MIN, f64::max);
        let other = solver
            .hypotheses
            .iter()
            .filter(|h| h.square != sq)
            .map(|h| h.weight)
            .fold(f64::MIN, f64::max);
        assert!(
            boosted > other * 3.0,
            "捕獲マスの仮説がブーストされていない（5d={boosted:.2} 他={other:.2}）"
        );
    }

    /// 開き王手が自駒配置上不可能なら、捕獲マス以外の仮説は除去される
    /// （arena-check-flee02 の幾何: 玉6八・銀7八・桂5八・歩5七/8六、6七で捕獲＋王手）
    #[test]
    fn capture_check_prunes_geometrically_impossible_discovered_checks() {
        let view = view_with(vec![
            ("6h", Role::King),
            ("7h", Role::Silver),
            ("5h", Role::Knight),
            ("5g", Role::Pawn),
            ("8f", Role::Pawn),
        ]);
        let mut log = ObservationLog::default();
        log.record(crate::observation::Observation::OpponentMoved {
            move_number: 112,
            captured_my_piece_at: Some("6g".into()),
        });
        let solver = CheckSolver::new(&view, &[], &[], &log).unwrap();
        let s = Coord { file: 6, rank: 7 };
        assert!(
            solver.hypotheses.iter().all(|h| h.square == s),
            "6七以外の仮説が残っている: {:?}",
            solver
                .hypotheses
                .iter()
                .filter(|h| h.square != s)
                .map(|h| (h.square, h.role))
                .collect::<Vec<_>>()
        );
        // 7八の銀で6七を取る手は（仮説上）確実に解消する
        let mv = crate::shogi::parse_usi("7h6g").unwrap();
        let mut solver = solver;
        assert!((solver.resolve_probability(&mv) - 1.0).abs() < 1e-9);
    }

    /// 開き王手が可能な幾何では捕獲マス以外の仮説が残る（裸玉5五、5四で捕獲＋
    /// 王手: 4四から来た駒が斜線 1一–5五 を開けた、等）。同じ線上（5一の飛）は
    /// 来た駒が自ら塞ぐので残らない
    #[test]
    fn capture_check_keeps_feasible_discovered_checks() {
        let view = view_with(vec![("5e", Role::King)]);
        let mut log = ObservationLog::default();
        log.record(crate::observation::Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("5d".into()),
        });
        let solver = CheckSolver::new(&view, &[], &[], &log).unwrap();
        let has = |f: i8, r: i8| solver.hypotheses.iter().any(|h| h.square == Coord { file: f, rank: r });
        assert!(has(1, 1), "1一の角（4四経由の開き王手）は残るべき");
        assert!(!has(5, 1), "5一の飛は来た駒（5四）が自ら塞ぐので残らない");
        assert!(!has(4, 4), "隣接 4四 は前の手番から王手だったことになるので残らない");
    }

    #[test]
    fn fouls_shift_probability_toward_consistent_evasions() {
        // 裸玉 5e。縦の移動 5e5d と 5e5f が両方反則
        // → 「5筋の縦利き（飛/香/竜）」仮説が支配的になり、横へ逃げる手の
        //    解消確率が縦に留まる手より高くなる
        let view = view_with(vec![("5e", Role::King)]);
        let fouls = [mv("5e5d"), mv("5e5f")];
        let mut solver = CheckSolver::new(&view, &[], &fouls, &ObservationLog::default()).unwrap();
        let side = solver.resolve_probability(&mv("5e4e"));
        // 玉の逃げは割引かない（割引は玉での仮説マス捕獲のみ）ので、
        // 4e 仮説ぶんのわずかな割引を除き従来の閾値のまま
        assert!(
            side > 0.7,
            "縦筋仮説の下で横逃げは解消するはず（p={side:.2}）"
        );
    }

    #[test]
    fn only_king_capture_of_hypothesis_square_is_discounted() {
        // 割引の対象は「玉でその仮説の王手駒マスを取る手」だけ（実測73%）。
        // 玉の逃げ・玉以外の捕獲は割引かない（一律割引の第1版はアリーナ
        // 3シードのペア比較で全て悪化した）
        let view = view_with(vec![("5e", Role::King), ("4e", Role::Gold)]);
        let solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        let hyp = Coord { file: 5, rank: 4 }; // 5d
        let discounted = solver.single_hyp_legal_prior(&mv("5e5d"), hyp);
        assert!(
            (discounted - KING_CAPTURE_LEGAL_PRIOR).abs() < 1e-9,
            "玉で仮説マス 5d を取る手は実測事前確率へ割引（{discounted:.2}）"
        );
        assert_eq!(solver.single_hyp_legal_prior(&mv("5e4d"), hyp), 1.0);
        assert_eq!(solver.single_hyp_legal_prior(&mv("4e5d"), hyp), 1.0);
    }

    #[test]
    fn king_move_foul_decays_hypotheses_less_than_non_king_foul() {
        // 玉で仮説マスを取る手の反則は「仮説が正しくても隠れた支えで説明できる」
        // ので、その仮説の減衰は非玉の反則（0.15）より緩い（1 − 0.73 = 0.27）
        let view = view_with(vec![("5e", Role::King), ("4e", Role::Gold)]);
        // 4d に王手駒仮説がある。5e4d（玉で4dへ）の反則と 4e4d（金で4dへ）の
        // 反則で、4d 仮説の減衰量を比べる
        let base_p = {
            let mut s = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
            s.resolve_probability(&mv("4e4d"))
        };
        let after_king_foul = {
            let fouls = [mv("5e4d")];
            let mut s = CheckSolver::new(&view, &[], &fouls, &ObservationLog::default()).unwrap();
            s.resolve_probability(&mv("4e4d"))
        };
        let after_gold_foul = {
            let fouls = [mv("4e4d")];
            let mut s = CheckSolver::new(&view, &[], &fouls, &ObservationLog::default()).unwrap();
            s.resolve_probability(&mv("4e4d"))
        };
        // 金の反則は 4d 系仮説だけを 0.15 に選択的に減衰させるので鋭く下がる。
        // 玉の反則は（4d 系を含む）広い仮説を 0.23 で減衰させるので下がり方が緩い。
        // 注: after_king_foul は base_p より下がるとは限らない（広範な減衰は
        // 比率の分母も削るため）ので、比較は gold 側とだけ行う
        assert!(after_gold_foul < base_p);
        assert!(
            after_king_foul > after_gold_foul,
            "玉の手の反則 {after_king_foul:.3} は金の反則 {after_gold_foul:.3} より仮説を殺さない"
        );
    }

    #[test]
    fn captures_checker_detects_only_non_king_capture_on_hypothesis_square() {
        // 5d の後手歩は 5e 玉への王手駒仮説。4e 金で取る手だけを
        // 「王手駒捕獲」として扱い、玉捕獲・別マス移動・打ちは除外する
        let view = view_with(vec![
            ("5e", Role::King),
            ("4e", Role::Gold),
            ("1e", Role::Rook),
        ]);
        let mut solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        assert!(solver.captures_checker(&mv("4e5d")));
        assert!(!solver.captures_checker(&mv("5e5d")));
        assert!(!solver.captures_checker(&mv("1e1b")));
        assert!(!solver.captures_checker(&mv("P*5d")));
    }

    #[test]
    fn removal_term_prefers_capturing_the_checker_over_escaping() {
        // 5e 玉・4e 金。4e5d の捕獲は受理された未来（5d 王手駒仮説）では
        // 王手駒が消えているので大きな正、玉逃げ 5e5f は大半の受理仮説で
        // 王手駒が盤に残るのでほぼゼロになるはず
        let view = view_with(vec![("5e", Role::King), ("4e", Role::Gold)]);
        let mut solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        let capture = solver.removal_term(&mv("4e5d")).unwrap();
        let escape = solver.removal_term(&mv("5e5f")).unwrap();
        assert!(capture > 0.0, "捕獲は正の除去期待値のはず（{capture:.3}）");
        assert!(escape >= 0.0, "正項のみなので負にはならない（{escape:.3}）");
        assert!(capture > escape);
    }

    #[test]
    fn removal_term_is_none_when_no_hypothesis_accepts_the_move() {
        // どの王手駒仮説の下でも王手を解消しない手（その場に留まる別駒の横動き
        // 相当が無いので、離れた金の無関係な移動で代用）は None
        let view = view_with(vec![("5e", Role::King), ("1a", Role::Gold)]);
        let mut solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        assert!(solver.removal_term(&mv("1a1b")).is_none());
    }

    #[test]
    fn known_enemy_piece_rules_out_covered_escapes() {
        // 裸玉 5e。自駒が 5g で取られた → 敵駒（と金近似）が 5g にいると分かっている。
        // 後手のと金は 5g から 5f（後ろ）にも利くので、5e5f はどの王手駒仮説の
        // 下でも非合法（p=0）になるはず。4e への横逃げは生きている
        let mut view = view_with(vec![("5e", Role::King)]);
        view.move_number = 12; // 直近8手以内の捕獲情報として扱われる
        let mut log = ObservationLog::default();
        log.record(crate::observation::Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("5g".into()),
        });
        let mut solver = CheckSolver::new(&view, &[], &[], &log).unwrap();
        let covered = solver.resolve_probability(&mv("5e5f"));
        let side = solver.resolve_probability(&mv("5e4e"));
        assert!(
            covered == 0.0,
            "既知の敵駒に覆われたマスへの逃げは全仮説で非合法のはず（p={covered:.2}）"
        );
        assert!(
            side > 0.0,
            "覆われていない逃げは生きているはず（p={side:.2}）"
        );
    }

    #[test]
    fn particles_sharpen_hypotheses() {
        // 粒子が「5a の飛が王手」で一致していれば、5筋から外れる手の確率が上がり、
        // 5筋に留まる手（5d への合駒など）は下がる
        let view = view_with(vec![("5e", Role::King)]);
        let mut truth = Position::empty(Color::Sente);
        truth.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        truth.set(
            Coord { file: 5, rank: 1 },
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::Rook,
            }),
        );
        let particles: Vec<(&Position, f64)> = vec![(&truth, 1.0); 8];
        let mut solver =
            CheckSolver::new(&view, &particles, &[], &ObservationLog::default()).unwrap();
        let away = solver.resolve_probability(&mv("5e4d"));
        let stay = solver.resolve_probability(&mv("5e5d"));
        assert!(
            away > stay,
            "飛車仮説が支配的なら5筋から外れる手が有利（away={away:.2} stay={stay:.2}）"
        );
    }

    #[test]
    fn soft_particles_vote_with_reduced_weight() {
        // strict粒子（重み1.0）は「5aの飛」、soft粒子（重み0.1）は「2bの角」。
        // 重みが効いていれば飛仮説が支配的になり、5筋に留まる手の解消確率は
        // 重みを逆にしたソルバーより低くなる
        let view = view_with(vec![("5e", Role::King)]);
        let place = |sq: Coord, role: Role, pos: &mut Position| {
            pos.set(
                sq,
                Some(crate::shogi::Piece {
                    color: Color::Gote,
                    role,
                }),
            );
        };
        let mut rook = Position::empty(Color::Sente);
        place(Coord { file: 5, rank: 5 }, Role::King, &mut rook);
        rook.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        place(Coord { file: 5, rank: 1 }, Role::Rook, &mut rook);
        let mut bishop = Position::empty(Color::Sente);
        bishop.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        place(Coord { file: 2, rank: 2 }, Role::Bishop, &mut bishop);

        let rook_heavy: Vec<(&Position, f64)> = vec![(&rook, 1.0), (&bishop, 0.1)];
        let bishop_heavy: Vec<(&Position, f64)> = vec![(&rook, 0.1), (&bishop, 1.0)];
        let mut a = CheckSolver::new(&view, &rook_heavy, &[], &ObservationLog::default()).unwrap();
        let mut b =
            CheckSolver::new(&view, &bishop_heavy, &[], &ObservationLog::default()).unwrap();
        let stay_a = a.resolve_probability(&mv("5e5d"));
        let stay_b = b.resolve_probability(&mv("5e5d"));
        assert!(
            stay_a < stay_b,
            "重み付き投票なら飛重視側で5筋残留が不利（a={stay_a:.2} b={stay_b:.2}）"
        );
    }

    #[test]
    fn hypothesis_prior_normalizes_mass_per_square() {
        // 玉 5五・自駒なし。隣接マスは金・と金・成香…と多数の駒種が立つのに対し、
        // 離れた 5一（縦の飛び道）は飛/龍/香/成香の少数しか立たない。
        // w=0（従来）は（マス,駒種）一様なので隣接が構造的に重く、
        // w=1 はマス単位で正規化されるので両者が同格になる
        let view = view_with(vec![("5e", Role::King)]);
        let log = ObservationLog::default();
        let mass = |prior_w: f64, file: i8, rank: i8| -> f64 {
            let s = CheckSolver::with_knobs(&view, &[], &[], &log, prior_w, 0.0, 1.0).unwrap();
            let total: f64 = s.hypotheses.iter().map(|h| h.weight).sum();
            let sq = Coord { file, rank };
            s.hypotheses
                .iter()
                .filter(|h| h.square == sq)
                .map(|h| h.weight)
                .sum::<f64>()
                / total
        };
        // 隣接（5四）と遠方の飛び道（5一）の質量比
        let legacy = mass(0.0, 5, 4) / mass(0.0, 5, 1);
        let normalized = mass(1.0, 5, 4) / mass(1.0, 5, 1);
        assert!(
            legacy > 2.0,
            "従来は隣接マスが遠方より構造的に重い（実測比 {legacy:.2}）"
        );
        assert!(
            (normalized - 1.0).abs() < 1e-9,
            "w=1 ではマス単位の質量が揃う（比 {normalized:.3}）"
        );
    }

    #[test]
    fn foul_evidence_makes_quiet_fouls_stronger_refuters_than_king_escapes() {
        // 4四に王手駒仮説。玉の逃げの反則と、非玉の駒での捕獲の反則で、
        // 同じ仮説がどれだけ否定されるかを比べる（w=1 では実測どおり
        // 捕獲の反則のほうが強く否定する）
        let view = view_with(vec![("5e", Role::King), ("4e", Role::Gold)]);
        let log = ObservationLog::default();
        let p_after = |foul: &str, w: f64| -> f64 {
            let fouls = [mv(foul)];
            let mut s = CheckSolver::with_knobs(&view, &[], &fouls, &log, 0.0, w, 1.0).unwrap();
            s.resolve_probability(&mv("4e4d"))
        };
        // 玉の逃げ（5e5f）は 4d 仮説を殺さない方向、金の捕獲（4e4d）は殺す方向
        let escape_w1 = p_after("5e5f", 1.0);
        let capture_w1 = p_after("4e4d", 1.0);
        assert!(
            capture_w1 < escape_w1,
            "捕獲の反則 {capture_w1:.3} は玉の逃げの反則 {escape_w1:.3} より仮説を強く否定する"
        );
        // 従来（w=0）より、捕獲の反則の否定は強く、玉の逃げの否定は弱い
        assert!(capture_w1 < p_after("4e4d", 0.0));
        assert!(escape_w1 >= p_after("5e5f", 0.0) - 1e-12);
    }

    #[test]
    fn escape_decay_only_fires_after_a_king_foul_and_only_on_king_moves() {
        let view = view_with(vec![("5e", Role::King), ("4e", Role::Gold)]);
        let log = ObservationLog::default();
        let p = |fouls: &[ShogiMove], decay: f64, m: &str| -> f64 {
            let mut s =
                CheckSolver::with_knobs(&view, &[], fouls, &log, 0.0, 0.0, decay).unwrap();
            s.resolve_probability(&mv(m))
        };
        // 反則0回なら decay に依らず同じ（挙動不変）
        assert_eq!(p(&[], 1.0, "5e5f"), p(&[], 0.5, "5e5f"));
        // 玉の反則が1回あれば、以後の玉の手だけが割り引かれる
        let fouls = [mv("5e4d")];
        let king_plain = p(&fouls, 1.0, "5e5f");
        let king_decayed = p(&fouls, 0.5, "5e5f");
        assert!((king_decayed - king_plain * 0.5).abs() < 1e-9);
        // 非玉の手（金の捕獲）は割り引かない
        assert_eq!(p(&fouls, 1.0, "4e4d"), p(&fouls, 0.5, "4e4d"));
    }

    #[test]
    fn role_inventory_prior_discounts_promoted_roles() {
        assert_eq!(role_inventory_prior(Role::Pawn, 9), 9.0);
        assert_eq!(role_inventory_prior(Role::Tokin, 9), 9.0 * PROMOTED_PRIOR);
        assert_eq!(role_inventory_prior(Role::Rook, 1), 1.0);
        assert_eq!(role_inventory_prior(Role::Dragon, 1), PROMOTED_PRIOR);
    }

    #[test]
    fn occupancy_prior_is_identity_at_w0_and_floors_empty() {
        assert_eq!(occupancy_prior(0.0, 0.0), 1.0);
        assert_eq!(occupancy_prior(0.4, 0.0), 1.0);
        assert!((occupancy_prior(1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((occupancy_prior(0.0, 1.0) - CHECK_BELIEF_OCC_FLOOR).abs() < 1e-12);
        assert!((occupancy_prior(0.4, 1.0) - 0.4).abs() < 1e-12);
        // w=0.5 は一様と占有の中点
        assert!((occupancy_prior(0.4, 0.5) - 0.7).abs() < 1e-12);
        // w>1 は 1 にクリップ（負の乗数を出さない）
        assert!((occupancy_prior(0.4, 2.0) - occupancy_prior(0.4, 1.0)).abs() < 1e-12);
        if std::env::var("TSUITATE_CHECK_BELIEF_OCC_W").is_err() {
            assert!((check_belief_occ_w() - CHECK_BELIEF_OCC_W).abs() < 1e-12);
            assert_eq!(CHECK_BELIEF_OCC_W, 0.0);
        }
        assert!(!particles_vote_check(&[], Color::Sente));
    }

    #[test]
    fn occupancy_reweight_prefers_occupied_checker_squares() {
        // 5e 裸玉。5a（飛の筋）を占有、2b（角）を空き寄りにすると、
        // 5筋から外れる逃げ（4d）の解消確率が上がる
        let view = view_with(vec![("5e", Role::King)]);
        let mut solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        let mut occ = [0.2f64; 81];
        occ[crate::belief_features::sq_index(Coord { file: 5, rank: 1 })] = 0.9;
        occ[crate::belief_features::sq_index(Coord { file: 2, rank: 2 })] = 0.05;
        solver.reweight_by_occupancy(&occ, 1.0, None);
        let away = solver.resolve_probability(&mv("5e4d"));
        let stay = solver.resolve_probability(&mv("5e5d"));
        assert!(
            away > stay,
            "5a 占有なら飛仮説が強く、5筋から外れる手が有利（away={away:.2} stay={stay:.2}）"
        );
        // 床があるので仮説は残る
        assert!(
            solver.hypotheses.iter().all(|h| h.weight > 0.0),
            "占有事前は仮説を殺さない"
        );
    }

    #[test]
    fn occupancy_reweight_skips_observed_capture_square() {
        // 観測で確定した捕獲マスは占有が低くても重みが落ちない
        let view = view_with(vec![("5e", Role::King)]);
        let mut solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        let capture = Coord { file: 5, rank: 1 };
        let before: Vec<f64> = solver
            .hypotheses
            .iter()
            .filter(|h| h.square == capture)
            .map(|h| h.weight)
            .collect();
        assert!(!before.is_empty(), "5a の飛仮説がある");
        let mut occ = [0.9f64; 81];
        occ[crate::belief_features::sq_index(capture)] = 0.0;
        solver.reweight_by_occupancy(&occ, 1.0, Some(capture));
        let after: Vec<f64> = solver
            .hypotheses
            .iter()
            .filter(|h| h.square == capture)
            .map(|h| h.weight)
            .collect();
        assert_eq!(before, after, "捕獲マスは占有事前を乗らない");
        // 他マスは下がる
        assert!(
            solver
                .hypotheses
                .iter()
                .filter(|h| h.square != capture)
                .all(|h| h.weight < 1.0),
            "捕獲マス以外は占有事前が掛かる"
        );
    }
}
