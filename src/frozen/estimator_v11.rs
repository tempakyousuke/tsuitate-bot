//! estimator の凍結版 v11（2026-07-25 凍結）。
//!
//! v10 からの主な差分:
//! - **王手駒の除去期待値 checker_removal_w=1.0**: p_legal フロアを撤去し、
//!   仮説条件付きで王手駒を除去する非玉捕獲へ gain 内加点を与える。
//! - **捕獲の賭け分散ペナルティ capture_bet_var_w=2.5**: p_hit(1-p_hit) と
//!   捕獲価値に比例して五分の大駒捕獲賭けを凹割引する。王手中は無効。
//! - **opp_move NN en_prise_flee（25次元）**: 位置が既知の敵駒から当たりを
//!   付けられている駒を動かす相手手特徴量を追加。
//!
//! 凍結時の成績（GitHub Actions、match_seed=20260725、各200局、2026-07-25）:
//! vs v10 58.5%±6.8 / vs v9 63.0%±6.7 / vs v8 58.5%±6.8。
//! 反則・思考時間の悪化なし。
//!
//! 凍結後は編集しない（シード注入等の挙動を変えない追加のみ許容）。

use std::collections::{HashMap, HashSet, VecDeque};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

use crate::board::{
    dead_end_rank, drop_targets, make_usi_drop, make_usi_move, make_usi_square, move_targets,
    parse_usi_square, promotion_choice, Coord, Promotion,
};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::{
    parse_usi, piece_value, promote_role, unpromote_role, Piece, Position, ShogiMove,
};
use crate::strategy::Strategy;

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CandidateScore {
    usi: String,
    static_score: f64,
    static_gain: f64,
    score: f64,
    gain: f64,
    p_legal: f64,
    foul_cost: f64,
    adjust: f64,
    depth2: bool,
    checker_removal: f64,
    capture_bet_penalty: f64,
}

// ---------------------------------------------------------------------------
// 推定器（estimator.rs のコピー）
// ---------------------------------------------------------------------------

/// 粒子の目標数。1手あたりの計算量はこれ*候補手数に比例する
const TARGET_PARTICLES: usize = 512;
/// 1回の update での再生成リプレイ試行の上限（時間予算の担保）。
/// 複製よりリプレイのほうが粒子の多様性を保てるので多めに取る。
/// v6: 相手モデルのフィット（2026-07-09）で提案分布の打率が上がったぶん
/// 試行回数の効果が大きくなったので、思考予算の余り（平均360ms/目安1〜2秒）を
/// リプレイに振る
const REGEN_ATTEMPTS: usize = 320;
/// リプレイ中バックトラックの1決定点あたりの再サンプル回数
const BACKTRACK_ATTEMPTS: u32 = 4;
/// ソフト救済の累積回数の上限。超えた粒子は棄却する
/// （観測と何度も矛盾した粒子は近似としても信用できない）。
/// ソフト救済の発動閾値は target/4（apply_constraint 参照）
const INFO_MISS_CAP: u8 = 3;
/// 情報系ソフト救済1回あたりの観測尤度（logw へ ln(EPS_INFO) を課金）。
/// 評価側の較正（証拠数の勘定）にも同じ値を使うため pub。
/// C-7 P1 で評価側の soft_decay^penalty（旧0.6753）を置き換えた。
/// フィルタ超パラメータなので調整は SPSA でなくグリッド＋シナリオ目的で行う
pub const EPS_INFO: f64 = 0.1;
/// ESS がこの割合（対 現粒子数）を下回ったら systematic resampling
const ESS_THRESHOLD: f64 = 0.5;
/// 各粒子が保持する直近の相手決定点スナップショット数（若返りの巻き戻し窓）
const REJUV_SNAPSHOTS: usize = 8;
/// 若返りの巻き戻し深さの試行順（近い決定点から adaptive に広げる。
/// 固定深さだと「原因が窓の少し前」を拾えず、常に深いとコスト過剰）。
/// 主経路ではスナップショットを制約適用前に積むため、depth=1 は同じ決定点として
/// 常にスキップされる
const REJUV_DEPTHS: [usize; 4] = [1, 2, 4, 8];
/// 1つの巻き戻し深さあたりの再サンプル試行回数
const REJUV_TRIES: u32 = 3;
/// 若返り全体の時間予算（ms、スケール比例）。発動は厳密生存 < target/4 の
/// ターンだけなので、健全なターンのコストはゼロ
const REJUV_MS: f64 = 150.0;
/// 制約後読みガイドのブースト倍率（提案分布側。重み補正で正直に払うので
/// 分布は歪まない。needle 突破には複数決定点での連続命中が要るため強めに取る）
const GUIDE_BOOST: f64 = 24.0;
/// ガイドの後読み幅（決定点から先読みする制約数の上限）。
/// 24まで拡張して測定したが、kakunari continue の遂行率（14/20、
/// ブラインド玉攻め単体の既知基準13/20からノイズ内）・アリーナ vs v7
/// （48.5%±9.7%、有意勝ち越し未達）のいずれでも効果を確認できず、
/// guide_boost_factor 内の空盤BFS呼び出し回数だけが比例して増えて
/// kakunari continue の実行時間が約60%増加した（2026-07-19測定）。
/// 効果未確認・コストありのため、検証済みの8へ戻す
const GUIDE_HORIZON: usize = 8;

/// 診断用: TSUITATE_DISABLE_DEFEND_GUIDE=1 で「MyFoul由来のガイド」
/// （自玉移動反則→guide.attacks、打ちマス反則→guide.occupies）を
/// まとめて無効化できる（速度差の切り分け専用。一時的なフラグ）
fn defend_guide_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TSUITATE_DISABLE_DEFEND_GUIDE").is_ok_and(|v| v == "1"))
}

/// 全滅時に保持する棄却粒子（墓場）の上限
const GRAVEYARD_CAP: usize = 128;
/// 墓場スナップショットの有効期限（決定点からの制約数。これを超えたら stale）
const GRAVEYARD_MAX_SEGMENT: usize = 24;
/// 物理不整合の最後の砦（C-7 P3 / D4）: 完全全滅時だけ、棄却粒子を
/// logw += ln(EPS_PHYS) と phys_taint+1 で残す（TSUITATE_EPS_PHYS で上書き可、
/// 0 で無効）。嘘の盤面なので評価側は玉位置系の用途（王手ソルバーの投票）に限定。
/// 救済に回数上限は設けない（kakunari 型の多段 needle は4連続以上の全滅を起こし、
/// 上限があると結局ブラインドに落ちる）。深く汚れた粒子は ε の累積課金と
/// truncate の taint 優先淘汰で、修復・復活・リプレイが成功し次第自然に消える
const EPS_PHYS_DEFAULT: f64 = 0.01;

/// 若返り用のスナップショット: 相手決定点 cidx の適用**前**の状態
#[derive(Clone)]
struct Snap {
    /// この決定点の制約 index（constraints[cidx] が相手手）
    cidx: usize,
    pos: Position,
    logw: f64,
    miss: u8,
    taint: u8,
}

/// スナップショット付きの棄却粒子（若返り→ソフト救済→墓場の受け渡し用）。
/// (局面, info_miss, logw, 窓, phys_taint)
type Rejected = (Position, u8, f64, VecDeque<Snap>, u8);
/// 若返りの成功結果。(局面, info_miss, logw, 窓, phys_taint)
type Repaired = (Position, u8, f64, VecDeque<Snap>, u8);

/// 制約後読みガイド: 巻き戻し区間の再サンプルで満たしたい将来の状態条件
#[derive(Default)]
struct Guide {
    /// 後続 MyMove(to=X, captured=R) 由来: 「X に相手の R を立てる手」をブースト
    lands: Vec<(Coord, Role)>,
    /// 後続 OppMove(captured_at=X) 由来: 「X へ利きを作る手」をブースト
    /// （取り返しには X に利く駒が事前に必要）
    attacks: Vec<Coord>,
    /// 多段ガイド（C-7 追補）: `lands` と同じ future MyMove(to=X, captured=R) 由来
    /// だが、「今すぐ X に着地する」手ではなく「駒種 R を持つ駒が X へ**近づく**手」
    /// を弱くブーストする。needle が複数手先にある場合（kakunari c42 型の
    /// サイレント再配置）、1手先しか見ない lands/attacks では見つからない
    approach: Vec<(Role, Coord)>,
    /// 打ちマス反則ガイド: 後続 MyFoul(歩以外の打ち, 王手中でない) 由来:
    /// 「X に（駒種不明の）相手駒を置く手」をブースト。王手中でない打ちが
    /// 反則になる理由は二歩・行き所のない駒（自分の情報だけで既に候補から
    /// 除外済み）を除けば「着地マスに見えない相手駒がいる」でほぼ一意
    /// （lands と違い駒種は分からないので role を問わず着地だけを見る）。
    /// 歩打ちだけは打ち歩詰め（相手玉が見えないので自分からは判定不能）と
    /// いう別の反則理由がありうるため対象外にする
    occupies: Vec<Coord>,
}

impl Guide {
    fn is_empty(&self) -> bool {
        self.lands.is_empty()
            && self.attacks.is_empty()
            && self.approach.is_empty()
            && self.occupies.is_empty()
    }
}

fn shift_hist(hist: &mut VecDeque<Snap>, d: f64) {
    for s in hist.iter_mut() {
        s.logw += d;
    }
}

/// 観測列を推定に使える形に正規化した制約
#[derive(Debug, Clone)]
enum Constraint {
    /// 受理された自分の手（gives_check: 直後に相手玉へ王手宣言があったか）
    MyMove {
        mv: ShogiMove,
        captured: Option<Role>,
        gives_check: bool,
    },
    /// 反則になった自分の手（真の局面では非合法）
    MyFoul { mv: ShogiMove },
    /// 相手の着手（captured_at: 自駒が取られたマス、gives_check: 自玉への王手宣言、
    /// foul_count: この手番で相手がこの着手に至るまでに試みた反則の回数。
    /// 反則の中身は不明だが回数は Observation::OpponentFoul でリアルタイムに
    /// 観測できる。opp_move_weight の特徴量として使う。
    /// my_foul_count: 直前の自分手番で自分が試みた反則の回数。相手側も
    /// 反則宣言（回数のみ）を観測しているので、相手の指し手はこれに
    /// 反応しうる = my_foul_count_last_turn 特徴量の逆方向配線）
    OppMove {
        captured_at: Option<Coord>,
        gives_check: bool,
        foul_count: u32,
        my_foul_count: u32,
    },
}

pub struct Estimator {
    my_color: Color,
    particles: Vec<Position>,
    /// particles と同じ並びのソフト救済回数（0 = 全制約と厳密整合）。
    /// 尤度の課金（EPS_INFO）は logw 側で行い、これは回数の別勘定
    /// （上限管理と評価側の証拠数較正用）。リサンプリングでもリセットしない
    info_miss: Vec<u8>,
    /// particles と同じ並びの観測尤度の対数重み（SIR の重み更新）。
    /// 相手手の制約適用ごとに log(整合クラスの事前質量 / 全合法手の事前質量) を
    /// 累積する。「観測と整合する手はあるが、それが相手として指しにくい手しか
    /// ない粒子」（例: 幻の角の飛び込み王手でしか王手を説明できない粒子）を
    /// 粒子間で相対的に軽くする。リプレイ生成粒子も全制約ぶん累積するので
    /// 生存粒子と比較可能。絶対値に意味はなく、評価側が max を引いて正規化する
    logw: Vec<f64>,
    /// particles と同じ並びの若返り窓（直近の相手決定点スナップショット）
    hist: Vec<VecDeque<Snap>>,
    /// particles と同じ並びの物理不整合カウンタ（ε_phys の最後の砦で残した回数。
    /// 0 = 物理的に厳密。リサンプリングでもリセットしない。評価側は taint>0 を
    /// 通常サンプルから除外し、玉位置系の用途にだけ使う）
    phys_taint: Vec<u8>,
    /// ε_phys（TSUITATE_EPS_PHYS で上書き。0 = 最後の砦無効）
    eps_phys: f64,
    /// 思考予算に応じた粒子の目標数（スケール1.0で TARGET_PARTICLES）
    target: usize,
    /// リプレイ試行回数の上限（スケール比例）
    regen_attempts: usize,
    /// 通常リプレイの時間打ち切り（ms、スケール比例）
    regen_deadline_ms: u64,
    /// 全滅時に粘る時間の上限（ms、スケール比例）
    empty_deadline_ms: u64,
    /// 若返りの時間打ち切り（ms、スケール比例）
    rejuv_deadline_ms: u64,
    constraints: Vec<Constraint>,
    /// 自分が駒を取ったマス（= 相手は自駒がそこで死んだことを知っている）。
    /// 相手手の事前分布の threat_known 特徴量に使う。idx は制約列上の位置
    my_capture_idx: Vec<usize>,
    my_capture_sq: Vec<Coord>,
    /// 自分の手が触れたマス（from/to）。初期配置から動いていない自駒
    /// （相手が推論で狙ってくる = threat_home 特徴量）の判定に使う
    my_touched_idx: Vec<usize>,
    my_touched_sq: Vec<Coord>,
    /// ObservationLog の消化済みイベント数
    cursor: usize,
    /// この手番でここまでに観測した相手の反則回数（Observation::OpponentFoul
    /// の累積）。次の Constraint::OppMove が確定した時点でその制約へ焼き込み、
    /// 0へリセットする
    pending_opp_foul_count: u32,
    /// この手番でここまでに自分が試みた反則の回数（Constraint::MyFoul の累積）。
    /// 自分の着手（MyMove）確定時に last_my_foul_count へ移して0へ戻す
    pending_my_foul_count: u32,
    /// 直前の自分手番で自分が試みた反則の回数。次の Constraint::OppMove へ
    /// 焼き込む（相手は反則宣言の回数を観測している = my_foul_count_last_turn）
    last_my_foul_count: u32,
    /// 観測との矛盾（リプレイでも整合局面を作れない等）で信頼できなくなったら false
    healthy: bool,
    /// 直近の replenish で測った ESS（診断用）
    last_ess: f64,
    /// systematic resampling の累計回数（診断用）
    resamples: u64,
    /// logw の基準点（制約 index）。リサンプリングで logw が 0 に再ベースされた後、
    /// 初期リプレイの新粒子が絶対スケール（全制約の累積）を持つと生存粒子に対して
    /// 不当に軽くなるため、リプレイの logw はこの位置以降の累積だけを数える
    /// （それ以前の質量は「集団の典型と同じ」とみなす近似。ソフト救済の
    /// strict_dlw_median と同じ哲学）
    rebase_cidx: usize,
    /// 全滅時の棄却粒子の保管庫。以後のターンで制約列が伸びても、スナップショット
    /// からの若返りで復活を試み続けられる（全滅からの回復手段。stale になったら破棄）
    graveyard: Vec<Rejected>,
    /// 若返りで修復した粒子の累計（診断用）
    rejuv_repaired: u64,
    /// 墓場から復活した粒子の累計（診断用）
    revived: u64,
    /// TSUITATE_FILTER_DEBUG=1 のとき、リプレイ/若返りが失敗した制約 index の
    /// ヒストグラムを集める（needle の特定用）
    debug_fail: Option<std::collections::HashMap<usize, u32>>,
    /// 現在の自玉位置（自分の手でしか動かないので常に厳密に分かる）
    my_king: Coord,
    /// king_at[i] = 制約 index i を処理する直前の自玉位置。build_guide が
    /// 王手宣言との整合を確かめるたびに全体を舐め直さずに済むよう、
    /// 制約追加時にインクリメンタルに更新する（O(1) 参照用のキャッシュ）
    king_at: Vec<Coord>,
    /// 現在の自玉の被王手状態（直近の OppMove.gives_check で更新。MyMove が
    /// 受理された時点で必ず解消されているので false に戻す）
    in_check: bool,
    /// in_check_at[i] = 制約 index i を処理する直前の被王手状態。
    /// king_at と同じ O(1) 参照用のキャッシュ（打ちマス反則の理由の一意性判定に使う）
    in_check_at: Vec<bool>,
    rng: StdRng,
}

impl Estimator {
    pub fn new(my_color: Color) -> Self {
        Estimator::with_seed(my_color, rand::rng().random())
    }

    pub fn with_seed(my_color: Color, seed: u64) -> Self {
        Estimator::with_seed_and_scale(my_color, seed, 1.0)
    }

    /// 思考予算スケールつきで作る（1.0 = 従来基準。strategy.rs の
    /// TSUITATE_THINK_BUDGET_MS から渡される）。粒子数・リプレイ回数・
    /// 時間打ち切りがスケールに比例する
    pub fn with_scale(my_color: Color, scale: f64) -> Self {
        Estimator::with_seed_and_scale(my_color, rand::rng().random(), scale)
    }

    pub fn with_seed_and_scale(my_color: Color, seed: u64, scale: f64) -> Self {
        let scale = scale.clamp(0.25, 8.0);
        let target = ((TARGET_PARTICLES as f64 * scale) as usize).clamp(128, 4096);
        Estimator {
            my_color,
            particles: vec![Position::initial(); target],
            info_miss: vec![0; target],
            logw: vec![0.0; target],
            hist: vec![VecDeque::new(); target],
            phys_taint: vec![0; target],
            eps_phys: std::env::var("TSUITATE_EPS_PHYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(EPS_PHYS_DEFAULT),
            target,
            regen_attempts: (REGEN_ATTEMPTS as f64 * scale) as usize,
            regen_deadline_ms: (500.0 * scale) as u64,
            empty_deadline_ms: (900.0 * scale) as u64,
            rejuv_deadline_ms: (REJUV_MS * scale) as u64,
            constraints: vec![],
            my_capture_idx: vec![],
            my_capture_sq: vec![],
            my_touched_idx: vec![],
            my_touched_sq: vec![],
            cursor: 0,
            pending_opp_foul_count: 0,
            pending_my_foul_count: 0,
            last_my_foul_count: 0,
            healthy: true,
            last_ess: target as f64,
            resamples: 0,
            rebase_cidx: 0,
            graveyard: vec![],
            rejuv_repaired: 0,
            revived: 0,
            debug_fail: std::env::var("TSUITATE_FILTER_DEBUG")
                .is_ok_and(|v| v == "1")
                .then(std::collections::HashMap::new),
            my_king: Position::initial()
                .king_square(my_color)
                .expect("初期局面に玉がない"),
            king_at: vec![],
            in_check: false,
            in_check_at: vec![],
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 粒子の目標数（思考予算に応じてスケール済み）
    pub fn target(&self) -> usize {
        self.target
    }

    pub fn my_color(&self) -> Color {
        self.my_color
    }

    /// 現在の粒子集合。空なら推定は信頼できない（呼び出し側でフォールバック）
    pub fn particles(&self) -> &[Position] {
        &self.particles
    }

    /// particles() と同じ並びのソフト救済回数。評価側の証拠数較正に使う
    /// （尤度の減衰は logw 側に課金済みなので、重みには二重に掛けない）
    pub fn info_miss(&self) -> &[u8] {
        &self.info_miss
    }

    /// 直近の replenish で測った ESS（診断用）
    pub fn last_ess(&self) -> f64 {
        self.last_ess
    }

    /// systematic resampling の累計回数（診断用）
    pub fn resamples(&self) -> u64 {
        self.resamples
    }

    /// (若返りで修復した粒子, 墓場から復活した粒子) の累計（診断用）
    pub fn rejuv_stats(&self) -> (u64, u64) {
        (self.rejuv_repaired, self.revived)
    }

    /// TSUITATE_FILTER_DEBUG=1 のときの失敗制約ヒストグラム（(制約idx, 回数) を
    /// 回数降順で返す）。リプレイ・若返りがどの制約で死んでいるかの特定用
    pub fn fail_report(&self) -> Vec<(usize, u32)> {
        let Some(m) = &self.debug_fail else {
            return vec![];
        };
        let mut v: Vec<(usize, u32)> = m.iter().map(|(&k, &c)| (k, c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    fn note_fail(&mut self, i: usize) {
        if let Some(m) = &mut self.debug_fail {
            *m.entry(i).or_insert(0) += 1;
        }
    }

    /// particles() と同じ並びの観測尤度の対数重み。粒子間の相対値だけに意味が
    /// ある（評価側で max を引いて exp し正規化する）。複製粒子は同じ値を持つ
    pub fn log_weights(&self) -> &[f64] {
        &self.logw
    }

    /// particles() と同じ並びの物理不整合カウンタ（0 = 物理的に厳密）。
    /// 評価側は taint>0 を通常サンプルから除外し、玉位置系の用途にだけ使う
    pub fn phys_taint(&self) -> &[u8] {
        &self.phys_taint
    }

    pub fn healthy(&self) -> bool {
        self.healthy && !self.particles.is_empty()
    }

    /// ログの未消化イベントを取り込み、粒子を前進・棄却・補充する
    pub fn update(&mut self, log: &ObservationLog) {
        let events = log.events();
        while self.cursor < events.len() {
            // 相手の反則は中身不明だが回数は実戦でもリアルタイムに観測できる。
            // 次の相手着手（OppMove）が確定するまで累積し、そちらへ焼き込む
            if matches!(events[self.cursor], Observation::OpponentFoul { .. }) {
                self.pending_opp_foul_count += 1;
                self.cursor += 1;
                continue;
            }
            let (constraint, consumed) = self.normalize(&events[self.cursor..]);
            self.cursor += consumed;
            let Some(constraint) = constraint else {
                continue;
            };
            self.apply_constraint(&constraint);
            // king_at[idx] = この制約を処理する直前の自玉位置（build_guide の
            // O(1) 参照用。king_square_before の全体再走査を避けるため
            // インクリメンタルに維持する）
            self.king_at.push(self.my_king);
            if let Constraint::MyMove {
                mv: ShogiMove::Board { from, to, .. },
                ..
            } = &constraint
            {
                if *from == self.my_king {
                    self.my_king = *to;
                }
            }
            // in_check_at[idx] = この制約を処理する直前の被王手状態
            self.in_check_at.push(self.in_check);
            match &constraint {
                Constraint::OppMove { gives_check, .. } => {
                    self.in_check = *gives_check;
                    self.pending_opp_foul_count = 0;
                }
                Constraint::MyMove { .. } => {
                    self.in_check = false;
                    // 自分手番の反則回数を確定（次の OppMove が
                    // my_foul_count_last_turn として焼き込む）
                    self.last_my_foul_count = self.pending_my_foul_count;
                    self.pending_my_foul_count = 0;
                }
                Constraint::MyFoul { .. } => self.pending_my_foul_count += 1,
            }
            if let Constraint::MyMove { mv, captured, .. } = &constraint {
                let idx = self.constraints.len();
                let to = match *mv {
                    ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                };
                if captured.is_some() {
                    self.my_capture_idx.push(idx);
                    self.my_capture_sq.push(to);
                }
                if let ShogiMove::Board { from, .. } = *mv {
                    self.my_touched_idx.push(idx);
                    self.my_touched_sq.push(from);
                }
                self.my_touched_idx.push(idx);
                self.my_touched_sq.push(to);
            }
            self.constraints.push(constraint);
        }
        self.replenish();
    }

    /// 先頭イベントを制約へ正規化する。直後の Check イベントも一緒に消化する
    fn normalize(&self, events: &[Observation]) -> (Option<Constraint>, usize) {
        let head = &events[0];
        // 手の直後に王手宣言が続いているか（同じ着手の結果として扱う）
        let followed_by_check = |on: Color| -> bool {
            matches!(events.get(1), Some(Observation::Check { in_check }) if *in_check == on)
        };
        match head {
            Observation::MyMove { usi, captured, .. } => {
                let Some(mv) = parse_usi(usi) else {
                    return (None, 1);
                };
                let gives_check = followed_by_check(self.my_color.other());
                let consumed = if gives_check { 2 } else { 1 };
                (
                    Some(Constraint::MyMove {
                        mv,
                        captured: *captured,
                        gives_check,
                    }),
                    consumed,
                )
            }
            Observation::MyFoul { usi, .. } => match parse_usi(usi) {
                Some(mv) => (Some(Constraint::MyFoul { mv }), 1),
                None => (None, 1),
            },
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => {
                let captured_at = captured_my_piece_at
                    .as_deref()
                    .and_then(crate::board::parse_usi_square);
                let gives_check = followed_by_check(self.my_color);
                let consumed = if gives_check { 2 } else { 1 };
                (
                    Some(Constraint::OppMove {
                        captured_at,
                        gives_check,
                        foul_count: self.pending_opp_foul_count,
                        my_foul_count: self.last_my_foul_count,
                    }),
                    consumed,
                )
            }
            // 相手の反則は中身（どの手を試みたか）は分からないが、回数は
            // update() が pending_opp_foul_count へ累積し次の OppMove へ渡す
            // （opp_move_weight の特徴量。単独で現れた Check は手側で消化済みのはず）
            Observation::OpponentFoul { .. } | Observation::Check { .. } => (None, 1),
        }
    }

    fn apply_constraint(&mut self, constraint: &Constraint) {
        let my_color = self.my_color;
        // 今回の制約が constraints に積まれる位置（update が適用後に push する）
        let cidx = self.constraints.len();
        let particles = std::mem::take(&mut self.particles);
        let penalties = std::mem::take(&mut self.info_miss);
        let logws = std::mem::take(&mut self.logw);
        let hists = std::mem::take(&mut self.hist);
        let taints = std::mem::take(&mut self.phys_taint);
        let mut surv_pos = Vec::with_capacity(particles.len());
        let mut surv_pen = Vec::with_capacity(particles.len());
        let mut surv_logw = Vec::with_capacity(particles.len());
        let mut surv_hist = Vec::with_capacity(particles.len());
        let mut surv_taint = Vec::with_capacity(particles.len());
        // 棄却された粒子は適用前の局面を保持しておく（若返り・ソフト救済用。
        // apply_my_move / sample_opp_move は失敗時も局面を汚しうる）
        let mut failed: Vec<Rejected> = vec![];
        // 厳密生存者が今回の制約で得た対数重み増分（ソフト救済の課金基準に使う）
        let mut strict_dls: Vec<f64> = vec![];
        for ((((mut pos, pen), lw), mut hist), taint) in particles
            .into_iter()
            .zip(penalties)
            .zip(logws)
            .zip(hists)
            .zip(taints)
        {
            let backup = pos.clone();
            // 相手決定点なら適用前の状態をスナップショット（若返りの巻き戻し先）
            if matches!(constraint, Constraint::OppMove { .. }) {
                if hist.len() == REJUV_SNAPSHOTS {
                    hist.pop_front();
                }
                hist.push_back(Snap {
                    cidx,
                    pos: backup.clone(),
                    logw: lw,
                    miss: pen,
                    taint,
                });
            }
            // 自分の手・反則は決定的（尤度 0/1）なので重みは変えない。
            // 相手手は観測クラスの尤度（対数）を累積する
            let ok = match constraint {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, my_color, mv, *captured, Some(*gives_check))
                    .then_some(0.0),
                Constraint::MyFoul { mv } => foul_consistent(&pos, my_color, mv).then_some(0.0),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                    foul_count,
                    my_foul_count,
                } => sample_opp_move(
                    &mut pos,
                    my_color,
                    *captured_at,
                    Some(*gives_check),
                    *foul_count,
                    *my_foul_count,
                    &self.my_capture_sq,
                    &self.my_touched_sq,
                    &Guide::default(),
                    &mut self.rng,
                ),
            };
            if let Some(dlw) = ok {
                surv_pos.push(pos);
                surv_pen.push(pen);
                surv_logw.push(lw + dlw);
                surv_hist.push(hist);
                surv_taint.push(taint);
                strict_dls.push(dlw);
            } else {
                failed.push((backup, pen, lw, hist, taint));
            }
        }
        // 若返り（C-7 P2 / D3）: 厳密生存が薄いときは、棄却粒子を直近の
        // 相手決定点へ巻き戻して制約後読みガイド付きで引き直す。修復粒子は
        // 厳密整合（info_miss/phys_taint はスナップショット時点の値へ戻る）。
        // ゲートは**厳密生存数**（info_miss=0 かつ phys_taint=0）で判定する
        // （codex レビュー指摘: ソフト/taint は独立証拠ではない）。
        // 完全全滅（生存ゼロ）のときは予算を regen_deadline 級へ引き上げる
        // （どうせブラインドになるならリプレイ予算を前借りして修復に使う）
        let strict_count = |pens: &[u8], taints: &[u8]| -> usize {
            pens.iter()
                .zip(taints)
                .filter(|&(&m, &t)| m == 0 && t == 0)
                .count()
        };
        if strict_count(&surv_pen, &surv_taint) < self.target / 4 && !failed.is_empty() {
            let budget_ms = if surv_pos.is_empty() {
                self.regen_deadline_ms
            } else {
                self.rejuv_deadline_ms
            };
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(budget_ms);
            let (repaired, still) =
                self.rejuvenate_batch(failed, cidx, Some(constraint), self.target, deadline);
            for (pos, pen, lw, hist, taint) in repaired {
                self.rejuv_repaired += 1;
                surv_pos.push(pos);
                surv_pen.push(pen);
                surv_logw.push(lw);
                surv_hist.push(hist);
                surv_taint.push(taint);
            }
            failed = still;
        }
        // ソフト救済: 若返り後も厳密整合の生存が少ないときだけ、情報系の制約を
        // 緩和して棄却粒子を info_miss+1 で生かす（物理制約は緩和しない）
        let mut graveyard_candidates: Vec<Rejected> = vec![];
        if strict_count(&surv_pen, &surv_taint) < self.target / 4 {
            // ソフト粒子の観測尤度: 本当は P(観測|粒子)=0 だが近似として生かす
            // ので、「典型的な厳密生存者と同じ増分」（中央値）を課す。緩和クラスの
            // r（≈1）をそのまま使うと、観測を説明できない粒子のほうが正直に
            // 小さい r を払った厳密粒子より重くなってしまう。厳密生存者がいない
            // ときだけ緩和クラスの r で代用する（全員ソフトなら相対値として無害で、
            // 後からリプレイされる厳密粒子は正直な累積 r を持つので比較もできる）
            let strict_dlw_median = (!strict_dls.is_empty()).then(|| {
                strict_dls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                strict_dls[strict_dls.len() / 2]
            });
            for (mut pos, pen, lw, hist, taint) in failed {
                if pen >= INFO_MISS_CAP {
                    continue;
                }
                if let Some(dlw) = self.apply_soft(&mut pos, constraint) {
                    surv_pos.push(pos);
                    surv_pen.push(pen + 1);
                    // 観測を説明できなかった近似粒子の課金: 典型的な厳密生存者と
                    // 同じ増分（中央値）に加えて、情報系ソフトの尤度 EPS_INFO を払う
                    // （旧: 評価側の soft_decay^penalty。C-7 P1 で logw へ統合）
                    surv_logw.push(lw + strict_dlw_median.unwrap_or(dlw) + EPS_INFO.ln());
                    surv_hist.push(hist);
                    surv_taint.push(taint);
                } else {
                    graveyard_candidates.push((pos, pen, lw, hist, taint));
                }
            }
        }
        // ε_phys の最後の砦（C-7 P3 / D4）: 完全全滅（ソフトも含め生存ゼロ）の
        // ときだけ、物理不整合の棄却粒子を強制適用して phys_taint+1 で残す。
        // 狙いは信念の連続性（玉位置などの大域情報）で、盤面は観測と厳密整合
        // しない「嘘」— 評価側は taint>0 を通常サンプルから除外し、
        // 玉位置系の用途（王手ソルバーの投票）にだけ使う。
        //
        // エポック正規化（codex P3 レビュー指摘への対応）: wipe をまたぐと
        // 旧スケールの logw（全制約の累積）と、rebase 後の新規リプレイ粒子
        // （rebase_cidx 以降のみ課金 ≈ 0 基準）が混在してしまう。wipe 時点で
        // 生き残る taint 粒子・墓場エントリの logw とスナップショットを
        // **共通定数（候補内の max logw）だけシフト**して新エポックの 0 基準へ
        // 揃える。共通シフトなので相対重みは保存され、以後の若返り修復・
        // 墓場復活はどちらも「スナップショット値から再出発」の一本の規約で
        // 新規リプレイ粒子と比較可能になる
        let complete_wipe = surv_pos.is_empty();
        let epoch_shift = if complete_wipe {
            graveyard_candidates
                .iter()
                .map(|(_, _, lw, _, _)| *lw)
                .fold(f64::MIN, f64::max)
        } else {
            0.0
        };
        let epoch_shift = if epoch_shift == f64::MIN {
            0.0
        } else {
            epoch_shift
        };
        if complete_wipe && self.eps_phys > 0.0 && !graveyard_candidates.is_empty() {
            for (pos, pen, lw, hist, taint) in &graveyard_candidates {
                if surv_pos.len() >= self.target {
                    break;
                }
                let mut forced = pos.clone();
                force_apply(&mut forced, my_color, constraint);
                let mut h = hist.clone();
                shift_hist(&mut h, -epoch_shift);
                surv_pos.push(forced);
                surv_pen.push(*pen);
                surv_logw.push(lw - epoch_shift + self.eps_phys.ln());
                surv_hist.push(h);
                surv_taint.push(taint.saturating_add(1));
            }
        }
        // 厳密全滅なら棄却粒子を墓場へ保管する（以後のターンで復活を試みる。
        // 物理的にはスナップショット時点まで整合していた歴史なので嘘ではないが、
        // snap.miss > 0 のものは情報観測に info_miss 分だけ汚染されている —
        // miss は復活後も維持されるので較正は保たれる）。
        // 完全全滅（ソフトもゼロ。taint 救済は数えない）のときだけ logw の
        // 基準点を今へ再ベースし、墓場エントリも同じエポックへシフトする
        if strict_count(&surv_pen, &surv_taint) == 0 && !graveyard_candidates.is_empty() {
            if complete_wipe {
                for (_, _, lw, hist, _) in graveyard_candidates.iter_mut() {
                    *lw -= epoch_shift;
                    shift_hist(hist, -epoch_shift);
                }
            }
            graveyard_candidates.sort_by_key(|(_, pen, _, _, taint)| (*taint, *pen));
            graveyard_candidates.truncate(GRAVEYARD_CAP);
            self.graveyard = graveyard_candidates;
            if complete_wipe {
                self.rebase_cidx = cidx;
            }
        }
        if self.debug_fail.is_some() {
            let strict = strict_count(&surv_pen, &surv_taint);
            let taint_n = surv_taint.iter().filter(|&&t| t > 0).count();
            let soft = surv_pen.len() - strict - taint_n;
            let kind = match constraint {
                Constraint::MyMove {
                    captured,
                    gives_check,
                    ..
                } => {
                    format!("MyMove(cap={captured:?},chk={gives_check})")
                }
                Constraint::MyFoul { .. } => "MyFoul".into(),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                    ..
                } => {
                    format!("OppMove(cap_at={captured_at:?},chk={gives_check})")
                }
            };
            eprintln!(
                "    [c{cidx}] {kind}: 厳密{strict} soft{soft} taint{taint_n} 墓場{} 修復累計{}",
                self.graveyard.len(),
                self.rejuv_repaired,
            );
        }
        self.particles = surv_pos;
        self.info_miss = surv_pen;
        self.logw = surv_logw;
        self.hist = surv_hist;
        self.phys_taint = surv_taint;
    }

    /// 情報系の制約（王手宣言の一致・自分の反則の説明）だけを緩和した適用。
    /// 物理的な制約（自手の合法性・取った駒種・取られたマス）は緩和しない。
    /// 成功時は対数重みの増分（緩和クラスでの観測尤度）を返す
    fn apply_soft(&mut self, pos: &mut Position, constraint: &Constraint) -> Option<f64> {
        match constraint {
            Constraint::MyMove { mv, captured, .. } => {
                apply_my_move(pos, self.my_color, mv, *captured, None).then_some(0.0)
            }
            // 粒子上では合法だった手が実際は反則だった: この粒子は反則を
            // 説明できないが、盤面自体は生かす（反則手は実行されていない）
            Constraint::MyFoul { .. } => Some(0.0),
            Constraint::OppMove {
                captured_at,
                foul_count,
                my_foul_count,
                ..
            } => sample_opp_move(
                pos,
                self.my_color,
                *captured_at,
                None,
                *foul_count,
                *my_foul_count,
                &self.my_capture_sq,
                &self.my_touched_sq,
                &Guide::default(),
                &mut self.rng,
            ),
        }
    }

    /// depth-major の若返りバッチ: **浅い巻き戻しを全粒子に先に試し、だめなら
    /// 深くする**。1粒子に深い試行を使い切るより、多様な粒子の浅い修復を
    /// 先に広く拾うほうが予算効率がよい（kakunari c42 の教訓: 深い巻き戻しが
    /// 必要な needle では、粒子ごとの深さ内訳よりバッチ全体の深さ配分が効く）。
    /// max 件修復するか deadline で打ち切り、(修復済み, 未修復) を返す
    fn rejuvenate_batch(
        &mut self,
        failed: Vec<Rejected>,
        upto: usize,
        current: Option<&Constraint>,
        max: usize,
        deadline: std::time::Instant,
    ) -> (Vec<Repaired>, Vec<Rejected>) {
        let mut repaired = vec![];
        let mut pool: Vec<Option<Rejected>> = failed.into_iter().map(Some).collect();
        // deadline まで depth スイープを周回する（1周の固定試行で打ち切ると
        // 予算が余る。毎周 rng が進むので同じ粒子でも別の経路を引ける）
        'outer: loop {
            let mut attempts = 0usize;
            for &depth in &REJUV_DEPTHS {
                for slot in pool.iter_mut() {
                    if repaired.len() >= max || std::time::Instant::now() > deadline {
                        break 'outer;
                    }
                    let Some(f) = slot else { continue };
                    let hist = &f.3;
                    if depth > hist.len() {
                        continue;
                    }
                    let snap = &hist[hist.len() - depth];
                    if snap.cidx == upto || upto - snap.cidx > GRAVEYARD_MAX_SEGMENT {
                        continue;
                    }
                    attempts += 1;
                    for _ in 0..REJUV_TRIES {
                        if let Some(out) = self.replay_segment(snap, hist, upto, current) {
                            repaired.push(out);
                            *slot = None;
                            break;
                        }
                    }
                }
            }
            // 試行対象がもう無い（全修復 or 全 stale）なら周回しても無駄
            if attempts == 0 {
                break;
            }
        }
        let still: Vec<Rejected> = pool.into_iter().flatten().collect();
        (repaired, still)
    }

    /// 巻き戻し区間のリプレイ: snap の状態から constraints[snap.cidx..upto] と
    /// current（upto 位置の未登録制約。None なら constraints のみ）を再適用する。
    /// 相手決定点は制約後読みガイド付きでサンプルし、重み補正（ln r + ln p/g）を
    /// logw へ累積する。logw はスナップショット値から再出発する（旧セグメントの
    /// 累積は捨てる = 二重計上なし）。成功時は新しいスナップショット窓も返す
    fn replay_segment(
        &mut self,
        snap: &Snap,
        hist: &VecDeque<Snap>,
        upto: usize,
        current: Option<&Constraint>,
    ) -> Option<Repaired> {
        let mut pos = snap.pos.clone();
        let mut lw = snap.logw;
        let miss = snap.miss;
        let taint = snap.taint;
        // 巻き戻し先より前のスナップショットは有効（snap.cidx のエントリは
        // 「この決定の適用前」の状態なので、引き直し後もそのまま正しい）。
        // wipe をまたぐエントリはエポック正規化済みなのでそのまま使える
        let mut new_hist: VecDeque<Snap> = hist
            .iter()
            .filter(|s| s.cidx <= snap.cidx)
            .cloned()
            .collect();
        let end = upto + usize::from(current.is_some());
        for i in snap.cidx..end {
            let c: &Constraint = if i < upto {
                &self.constraints[i]
            } else {
                current.expect("i == upto は current がある場合のみ")
            };
            let ok = match c {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, self.my_color, mv, *captured, Some(*gives_check)),
                Constraint::MyFoul { mv } => foul_consistent(&pos, self.my_color, mv),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                    foul_count,
                    my_foul_count,
                } => {
                    if i > snap.cidx {
                        if new_hist.len() == REJUV_SNAPSHOTS {
                            new_hist.pop_front();
                        }
                        new_hist.push_back(Snap {
                            cidx: i,
                            pos: pos.clone(),
                            logw: lw,
                            miss,
                            taint,
                        });
                    }
                    let k = self.my_capture_idx.partition_point(|&j| j < i);
                    let t = self.my_touched_idx.partition_point(|&j| j < i);
                    let guide = self.build_guide(i, upto, current);
                    match sample_opp_move(
                        &mut pos,
                        self.my_color,
                        *captured_at,
                        Some(*gives_check),
                        *foul_count,
                        *my_foul_count,
                        &self.my_capture_sq[..k],
                        &self.my_touched_sq[..t],
                        &guide,
                        &mut self.rng,
                    ) {
                        Some(dlw) => {
                            lw += dlw;
                            true
                        }
                        None => false,
                    }
                }
            };
            if !ok {
                self.note_fail(i);
                return None;
            }
        }
        Some((pos, miss, lw, new_hist, taint))
    }

    /// 制約後読みガイド: 決定点 i の後（最大 GUIDE_HORIZON 制約先まで）から
    /// 状態条件を集める。
    /// - MyMove(to=X, captured=R) → 「X に相手の R が立つ」（lands）
    /// - OppMove(captured_at=X) → 「X へ利きを作る」（attacks。取り返しには
    ///   X に利く駒が事前に必要 — kakunari の同桂成の型）
    /// - MyFoul(自玉が X への移動を試みて反則) → 「X は他の相手駒に守られて
    ///   いる」ことが確定する（自玉の移動は経路遮蔽が起きないので、反則の
    ///   理由は必ず「移動先が相手の利きにある」）。「X へ利きを作る」という
    ///   意味では attacks と同じブースト対象なので同じ場に積む（新しい
    ///   フィールドは作らない。窓探索実験で確認した mover/defender 構成の
    ///   考え方を、既存の重み付きサンプリングへ再利用する形）
    /// - MyFoul(打ち to=X, 王手中でない) → 「X に相手駒がいる」ことがほぼ確定
    ///   する（二歩・行き所のない駒は自分の情報だけで候補から除外済みなので、
    ///   残る理由は着地マスの占有がほぼ全て。王手中は「合駒のはずが実は違う
    ///   ラインだった」という別説明があるので除外する）→ occupies
    /// upto 位置には current（未登録の制約）が入る。None なら constraints のみ
    fn build_guide(&self, i: usize, upto: usize, current: Option<&Constraint>) -> Guide {
        let mut guide = Guide::default();
        // king_at は O(1) 参照（king_square_before の全体再走査版は廃止）。
        // i が未記録の最新位置なら self.my_king が正しい値
        let mut king = self.king_at.get(i).copied().unwrap_or(self.my_king);
        for j in (i + 1)..=(i + GUIDE_HORIZON) {
            let c = match j.cmp(&upto) {
                std::cmp::Ordering::Less => &self.constraints[j],
                std::cmp::Ordering::Equal => match current {
                    Some(c) => c,
                    None => break,
                },
                std::cmp::Ordering::Greater => break,
            };
            match c {
                Constraint::MyMove {
                    mv,
                    captured: Some(role),
                    ..
                } => {
                    let to = match *mv {
                        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                    };
                    guide.lands.push((to, *role));
                    guide.approach.push((*role, to));
                }
                Constraint::OppMove {
                    captured_at: Some(at),
                    ..
                } => guide.attacks.push(*at),
                Constraint::MyFoul {
                    mv: ShogiMove::Board { from, to, .. },
                } if *from == king && !defend_guide_disabled() => {
                    guide.attacks.push(*to);
                }
                Constraint::MyFoul {
                    mv: ShogiMove::Drop { to, role },
                } if *role != Role::Pawn
                    && !self.in_check_at.get(j).copied().unwrap_or(self.in_check)
                    && !defend_guide_disabled() =>
                {
                    // 歩打ちだけは打ち歩詰め（自分からは判定不能: 相手玉の位置が
                    // 見えないので王手/詰みの成否を検証できない）という別の反則
                    // 理由がありうるため除外する（codex 指摘、2026-07-19）
                    guide.occupies.push(*to);
                }
                _ => {}
            }
            // ガイド窓の中でも自玉が動く可能性があるので追跡を続ける
            if let Constraint::MyMove {
                mv: ShogiMove::Board { from, to, .. },
                ..
            } = c
            {
                if *from == king {
                    king = *to;
                }
            }
        }
        guide
    }

    /// 粒子が減っていたら、制約列のリプレイ（多様性）と生存粒子の複製（安価）で補充。
    /// 枯渇時は時間予算いっぱいまでリプレイで粘る（観測が正しい限り整合局面は必ず存在する）。
    /// リプレイ1回のコストは手数に比例するため、回数と時間の両方で打ち切る
    fn replenish(&mut self) {
        let start = std::time::Instant::now();
        let regen_deadline = start + std::time::Duration::from_millis(self.regen_deadline_ms);
        // 墓場からの復活（C-7 P2）: 厳密生存がゼロなら、保管してある棄却粒子の
        // スナップショットから若返りを試みる。初期局面からの前向きリプレイより
        // 成功率がはるかに高い（巻き戻し幅が窓に収まるため）。制約列が伸びて
        // stale になったエントリは破棄する。
        // logw のスケール: ソフトが生きている（集団のスケールが継続している）
        // ときは snap.logw から再出発（rebase なし）、完全全滅後の再建なら
        // rebase 規約（0 起点、rebase_cidx 以降のみ課金）で揃える
        let strict0 = self
            .info_miss
            .iter()
            .zip(&self.phys_taint)
            .filter(|&(&m, &t)| m == 0 && t == 0)
            .count();
        if strict0 < self.target / 4 && !self.graveyard.is_empty() {
            let n = self.constraints.len();
            self.graveyard.retain(|(_, _, _, hist, _)| {
                hist.back()
                    .is_some_and(|s| n - s.cidx <= GRAVEYARD_MAX_SEGMENT)
            });
            let graveyard = std::mem::take(&mut self.graveyard);
            // 全滅時（ブラインド確定）は empty 予算まで使って復活に賭ける。
            // 復活の成功機会は全滅直後の数ターンに集中する（セグメントが
            // 伸びるほど needle が累積して通らなくなる）
            let budget_ms = if self.particles.is_empty() {
                self.empty_deadline_ms
            } else {
                self.regen_deadline_ms
            };
            let deadline = start + std::time::Duration::from_millis(budget_ms);
            let (repaired, still) =
                self.rejuvenate_batch(graveyard, n, None, self.target / 4, deadline);
            for (pos, pen, lw, hist, taint) in repaired {
                self.revived += 1;
                self.particles.push(pos);
                self.info_miss.push(pen);
                self.logw.push(lw);
                self.hist.push(hist);
                self.phys_taint.push(taint);
            }
            // 修復できなかった分は墓場に残す（stale で自然消滅）
            self.graveyard = still;
        }
        // リプレイの目標は「厳密整合の粒子数」。ソフト粒子で頭数が足りていても
        // 厳密粒子が薄ければリプレイで置き換えにいく（ソフトはあくまで近似）
        let mut strict = self
            .info_miss
            .iter()
            .zip(&self.phys_taint)
            .filter(|&(&m, &t)| m == 0 && t == 0)
            .count();
        if strict < self.target {
            for _ in 0..self.regen_attempts {
                if strict >= self.target || std::time::Instant::now() > regen_deadline {
                    break;
                }
                if let Some((pos, lw, hist)) = self.replay_once() {
                    self.particles.push(pos);
                    self.info_miss.push(0);
                    self.logw.push(lw);
                    self.hist.push(hist);
                    self.phys_taint.push(0);
                    strict += 1;
                }
            }
        }
        let deadline = start + std::time::Duration::from_millis(self.empty_deadline_ms);
        while self.particles.is_empty() && std::time::Instant::now() < deadline {
            if let Some((pos, lw, hist)) = self.replay_once() {
                self.particles.push(pos);
                self.info_miss.push(0);
                self.logw.push(lw);
                self.hist.push(hist);
                self.phys_taint.push(0);
            }
        }
        // ラッチしない: 粒子が戻れば健全に戻る（呼び出し側は毎手 update する）
        self.healthy = !self.particles.is_empty();
        if self.particles.is_empty() {
            return;
        }
        // 溢れの整理: info_miss 昇順（厳密優先）→ logw 降順で target まで絞る
        if self.particles.len() > self.target {
            let mut quints: Vec<(u8, u8, f64, Position, VecDeque<Snap>)> =
                std::mem::take(&mut self.info_miss)
                    .into_iter()
                    .zip(std::mem::take(&mut self.phys_taint))
                    .zip(std::mem::take(&mut self.logw))
                    .zip(std::mem::take(&mut self.particles))
                    .zip(std::mem::take(&mut self.hist))
                    .map(|((((pen, taint), lw), pos), hist)| (taint, pen, lw, pos, hist))
                    .collect();
            quints.sort_by(|a, b| {
                (a.0, a.1)
                    .cmp(&(b.0, b.1))
                    .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            });
            quints.truncate(self.target);
            for (taint, pen, lw, pos, hist) in quints {
                self.info_miss.push(pen);
                self.phys_taint.push(taint);
                self.logw.push(lw);
                self.particles.push(pos);
                self.hist.push(hist);
            }
        }
        // ESS 監視（C-7 P1 / D2）: 重みが退化していたら systematic resampling で
        // 質量を複製数へ実現し logw をリセットする。退化していないが頭数が
        // 足りないときは質量保存の分割複製で埋める（logw の相対値 = 評価側の
        // 重み付けを崩さずに、次の相手手サンプルで分岐する多様性の種を蒔く）
        let m = self.particles.len();
        let max_lw = self.logw.iter().copied().fold(f64::MIN, f64::max);
        let ws: Vec<f64> = self.logw.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let total: f64 = ws.iter().sum();
        let sum2: f64 = ws.iter().map(|w| w * w).sum();
        self.last_ess = if sum2 > 0.0 {
            total * total / sum2
        } else {
            0.0
        };
        if self.last_ess < m as f64 * ESS_THRESHOLD {
            self.systematic_resample(&ws, total);
            self.resamples += 1;
        } else if m < self.target {
            self.split_fill(&ws, total);
        }
    }

    /// systematic resampling: 正規化重み比例で target 個へ複製し logw を
    /// リセットする（質量が複製数へ実現される）。低分散・O(n)。
    /// info_miss は各コピーへ引き継ぐ（較正・上限管理はカウンタが担う）。
    /// スナップショットの logw もリセットと同じ量シフトする（相対会計の保存:
    /// 巻き戻し時に「旧セグメント分を新セグメント分へ差し替える」が成り立つ）
    fn systematic_resample(&mut self, ws: &[f64], total: f64) {
        let m = self.particles.len();
        let want = self.target;
        let step = total / want as f64;
        let mut u = self.rng.random_range(0.0..step);
        let mut new_pos = Vec::with_capacity(want);
        let mut new_miss = Vec::with_capacity(want);
        let mut new_hist = Vec::with_capacity(want);
        let mut new_taint = Vec::with_capacity(want);
        let mut i = 0usize;
        let mut cum = ws[0];
        for _ in 0..want {
            while cum < u && i + 1 < m {
                i += 1;
                cum += ws[i];
            }
            new_pos.push(self.particles[i].clone());
            new_miss.push(self.info_miss[i]);
            new_taint.push(self.phys_taint[i]);
            let mut h = self.hist[i].clone();
            shift_hist(&mut h, -self.logw[i]);
            new_hist.push(h);
            u += step;
        }
        self.particles = new_pos;
        self.info_miss = new_miss;
        self.logw = vec![0.0; want];
        self.hist = new_hist;
        self.phys_taint = new_taint;
        // 以後の新規リプレイ粒子はここ以降の累積だけを課金する（スケール整合）
        self.rebase_cidx = self.constraints.len();
    }

    /// 質量保存の分割複製: 重み比例で複製先を選び、同一個体群（元+コピー）で
    /// exp(logw) を等分する。指紋ごとの合計質量が変わらないため、評価側の
    /// multiplicity 畳み込みと二重に効かない（旧複製埋めの後継）。
    /// スナップショットの logw も同じ量シフトする（相対会計の保存）
    fn split_fill(&mut self, ws: &[f64], total: f64) {
        let m = self.particles.len();
        let mut cum = Vec::with_capacity(m);
        let mut acc = 0.0f64;
        for &w in ws {
            acc += w;
            cum.push(acc);
        }
        let mut copies = vec![0usize; m];
        for _ in m..self.target {
            let t = self.rng.random_range(0.0..total);
            let i = cum.partition_point(|&c| c < t).min(m - 1);
            copies[i] += 1;
        }
        for (i, &c) in copies.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let d = -(((c + 1) as f64).ln());
            self.logw[i] += d;
            shift_hist(&mut self.hist[i], d);
            let share = self.logw[i];
            for _ in 0..c {
                self.particles.push(self.particles[i].clone());
                self.info_miss.push(self.info_miss[i]);
                self.logw.push(share);
                self.hist.push(self.hist[i].clone());
                self.phys_taint.push(self.phys_taint[i]);
            }
        }
    }

    /// 制約列を最初からリプレイして整合する粒子を1つ作る。
    ///
    /// 相手手のサンプルは確率的なので、後続の制約（自分の手の合法性・反則・
    /// 取られたマス・王手宣言）と矛盾して失敗しうる。全部やり直すと手数に対して
    /// 成功率が指数的に落ちるため、失敗したら直近の決定点（相手手）まで戻って
    /// 引き直す限定バックトラックにする。ステップ予算で最悪時間を抑える。
    /// 相手決定点は制約後読みガイド付きでサンプルする（C-7 P2。重み補正で
    /// 正直に払うので分布は歪まない）。成功時は若返り窓（直近決定点の
    /// スナップショット）も返す
    fn replay_once(&mut self) -> Option<(Position, f64, VecDeque<Snap>)> {
        let n = self.constraints.len();
        let step_budget = n * 4 + 32;
        let mut steps = 0usize;
        let mut pos = Position::initial();
        let mut lw = 0.0f64;
        // 決定点スタック: (制約index, 適用前の局面, 適用前の対数重み, 再試行回数)
        let mut stack: Vec<(usize, Position, f64, u32)> = vec![];
        let mut i = 0;
        while i < n {
            steps += 1;
            if steps > step_budget {
                return None;
            }
            let ok = match &self.constraints[i] {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, self.my_color, mv, *captured, Some(*gives_check)),
                Constraint::MyFoul { mv } => foul_consistent(&pos, self.my_color, mv),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                    foul_count,
                    my_foul_count,
                } => {
                    // バックトラックで戻ってきた再訪なら積み直さない
                    let is_retry = stack.last().is_some_and(|(j, _, _, _)| *j == i);
                    if !is_retry {
                        stack.push((i, pos.clone(), lw, 0));
                    }
                    // この時点までに自分が駒を取ったマス／触れたマス
                    let k = self.my_capture_idx.partition_point(|&j| j < i);
                    let t = self.my_touched_idx.partition_point(|&j| j < i);
                    let guide = self.build_guide(i, n, None);
                    match sample_opp_move(
                        &mut pos,
                        self.my_color,
                        *captured_at,
                        Some(*gives_check),
                        *foul_count,
                        *my_foul_count,
                        &self.my_capture_sq[..k],
                        &self.my_touched_sq[..t],
                        &guide,
                        &mut self.rng,
                    ) {
                        Some(dlw) => {
                            // logw は再ベース点以降だけ課金する（リサンプリングで
                            // 0 に再ベースされた生存粒子とのスケール合わせ。
                            // それ以前の質量は「集団の典型と同じ」とみなす近似）
                            if i >= self.rebase_cidx {
                                lw += dlw;
                            }
                            true
                        }
                        None => false,
                    }
                }
            };
            if ok {
                i += 1;
                continue;
            }
            self.note_fail(i);
            // 失敗: 直近の決定点に戻って引き直す。試行を使い切った点はさらに前へ
            loop {
                let Some((j, snapshot, snapshot_lw, attempts)) = stack.pop() else {
                    return None;
                };
                // 失敗した制約自身が決定点なら、同じ局面からの再試行は無意味
                // （整合候補ゼロは決定的）なのでさらに前へ戻る
                if j == i {
                    continue;
                }
                if attempts + 1 < BACKTRACK_ATTEMPTS {
                    pos = snapshot.clone();
                    lw = snapshot_lw;
                    stack.push((j, snapshot, snapshot_lw, attempts + 1));
                    i = j;
                    break;
                }
            }
        }
        // スタックには全決定点の適用前状態が積まれている（成功時は pop されない）
        // ので、末尾 REJUV_SNAPSHOTS 件がそのまま若返り窓になる
        let hist: VecDeque<Snap> = stack
            .iter()
            .rev()
            .take(REJUV_SNAPSHOTS)
            .rev()
            .map(|(j, p, l, _)| Snap {
                cidx: *j,
                pos: p.clone(),
                logw: *l,
                miss: 0,
                taint: 0,
            })
            .collect();
        Some((pos, lw, hist))
    }
}

/// 受理された自分の手を粒子に適用する。粒子と観測が矛盾したら false。
/// gives_check が None のときは王手宣言との一致を検査しない（ソフト救済用）
fn apply_my_move(
    pos: &mut Position,
    my_color: Color,
    mv: &ShogiMove,
    captured: Option<Role>,
    gives_check: Option<bool>,
) -> bool {
    if pos.turn() != my_color || !pos.is_legal(mv) {
        return false;
    }
    let actual = pos.play_unchecked(mv).map(unpromote_role);
    if actual != captured {
        return false;
    }
    gives_check.is_none_or(|gc| pos.in_check(my_color.other()) == gc)
}

/// 反則になった手との整合: 粒子上でも非合法であること
fn foul_consistent(pos: &Position, my_color: Color, mv: &ShogiMove) -> bool {
    pos.turn() == my_color && !pos.is_legal(mv)
}

/// 物理不整合の粒子への制約の強制適用（ε_phys の最後の砦専用）。
/// 自分側の状態（自駒配置・持ち駒・手番）は真実と同期させ、相手側は
/// 分かる範囲（取られた自駒 → 相手の持ち駒）だけ反映する。結果の盤面は
/// 観測と厳密整合しない近似なので、評価側は玉位置系の用途に限定すること
fn force_apply(pos: &mut Position, my_color: Color, constraint: &Constraint) {
    match constraint {
        Constraint::MyMove { mv, captured, .. } => {
            // 盤面: 自駒を強制移動（to の相手駒は盤から消えるだけ）。
            // 持ち駒: **観測された captured（真実）**だけを加える — 粒子上の
            // 嘘の駒種で自分の持ち駒を汚さない（codex P3 レビュー指摘）。
            // 合法時の play_unchecked も同じ理由で使わない（粒子上の to の駒種が
            // 真実と違うと持ち駒がズレる）
            match *mv {
                ShogiMove::Board { from, to, promote } => {
                    if let Some(mut p) = pos.piece_at(from).filter(|p| p.color == my_color) {
                        if promote {
                            if let Some(pr) = promote_role(p.role) {
                                p.role = pr;
                            }
                        }
                        pos.set(from, None);
                        pos.set(to, Some(p));
                    }
                }
                ShogiMove::Drop { role, to } => {
                    // 真実では打てた手なので必ず置く（粒子の持ち駒は saturating）
                    let h = pos.hand_count(my_color, role);
                    pos.set_hand(my_color, role, h.saturating_sub(1));
                    pos.set(
                        to,
                        Some(Piece {
                            color: my_color,
                            role,
                        }),
                    );
                }
            }
            if let Some(r) = captured {
                pos.set_hand(my_color, *r, pos.hand_count(my_color, *r) + 1);
            }
            pos.set_turn(my_color.other());
        }
        // 反則は指されていないので盤面維持（説明できない、は情報系の嘘として飲む）
        Constraint::MyFoul { .. } => {}
        Constraint::OppMove { captured_at, .. } => {
            // 幽霊取り: 取られた自駒だけ盤から除き、相手の持ち駒へ移す
            // （どの相手駒が来たかは分からないので相手駒は置かない）
            if let Some(sq) = captured_at {
                if let Some(p) = pos.piece_at(*sq).filter(|p| p.color == my_color) {
                    let r = unpromote_role(p.role);
                    let opp = my_color.other();
                    pos.set_hand(opp, r, pos.hand_count(opp, r) + 1);
                    pos.set(*sq, None);
                }
            }
            pos.set_turn(my_color);
        }
    }
}

/// 動かした駒（着地点）が対象マスのどれかへ新たに利きを付けたか。
/// 「新たに」= 移動元からは利いていなかった（打ちは常に新規）。
/// **定義は bin/fit_opp の newly_threatens と一致させること**（学習と推論の整合）
fn newly_threatens(pos: &Position, next: &Position, mv: &ShogiMove, targets: &[Coord]) -> bool {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    targets.iter().any(|&s| {
        if s == to || !next.attacks(to, s) {
            return false;
        }
        match *mv {
            ShogiMove::Board { from, .. } => !pos.attacks(from, s),
            ShogiMove::Drop { .. } => true,
        }
    })
}

/// 観測と整合する相手の合法手をサンプルして適用する。整合手がなければ None。
/// 成功時は対数重みの増分 Δlogw = ln(r) + ln(p_class/g_class) を返す
/// （SIR の重み更新。r = 整合クラスの素の事前質量 / 全合法手の素の事前質量、
/// p/g = クラス内での素の事前分布／ガイド付き提案分布における選択手の確率。
/// guide が空なら g = p で補正は 0、従来の ln(r) に一致する）。
/// - gives_check: None なら王手宣言との一致を検査しない（ソフト救済用）
/// - known_squares: 自分が駒を取ったマス（相手は自駒がそこで死んだことを知っている）
/// - my_touched: 自分の手が触れたマス（初期配置のまま動いていない自駒の判定用。
///   相手はそれらを推論で狙ってくる = 飛車頭への歩打ち等）
/// - guide: 制約後読みガイド（若返り・リプレイ用）。該当手を GUIDE_BOOST 倍した
///   提案分布から選ぶ。マスクはしない（成功しうる素の経路を提案の台から消すと
///   補正が定義できない）
#[allow(clippy::too_many_arguments)]
fn sample_opp_move(
    pos: &mut Position,
    my_color: Color,
    captured_at: Option<Coord>,
    gives_check: Option<bool>,
    foul_count_this_turn: u32,
    my_foul_count_last_turn: u32,
    known_squares: &[Coord],
    my_touched: &[Coord],
    guide: &Guide,
    rng: &mut StdRng,
) -> Option<f64> {
    let opp = my_color.other();
    if pos.turn() != opp {
        return None;
    }
    // 初期配置から動いていない自駒のマス（粒子内の実配置と突き合わせる）
    let initial = Position::initial();
    let homes: Vec<Coord> = initial
        .pieces()
        .filter(|(sq, p)| {
            p.color == my_color
                && !my_touched.contains(sq)
                && pos
                    .piece_at(*sq)
                    .is_some_and(|cur| cur.color == my_color && cur.role == p.role)
        })
        .map(|(sq, _)| sq)
        .collect();

    // (手, 素の重み w, ガイド後の提案重み g)
    let mut candidates: Vec<(ShogiMove, f64, f64)> = vec![];
    let mut total_mass = 0.0f64;
    for mv in pos.legal_moves() {
        // 取られたマスとの整合（取りがなかったなら自駒のあるマスへは来ていない）
        let to_capture = match mv {
            ShogiMove::Board { to, .. } => pos
                .piece_at(to)
                .filter(|p| p.color == my_color)
                .map(|p| (to, p.role)),
            ShogiMove::Drop { .. } => None,
        };
        let capture_ok = match (captured_at, to_capture) {
            (Some(at), Some((to, _))) => at == to,
            (None, None) => true,
            _ => false,
        };
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        // 分母（total_mass）には全合法手の重みが要るが、王手判定はクラス判定に
        // しか使わないので capture_ok の短絡で省く（in_check は比較的重い）
        let consistent = capture_ok && gives_check.is_none_or(|gc| next.in_check(my_color) == gc);
        let threat_known = newly_threatens(pos, &next, &mv, known_squares);
        let threat_home = newly_threatens(pos, &next, &mv, &homes);
        let (is_king, flee) = match mv {
            ShogiMove::Board { from, to, .. } => {
                let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
                (is_king, is_king && flees_danger(from, to, known_squares))
            }
            ShogiMove::Drop { .. } => (false, false),
        };
        let w = opp_move_weight(
            pos,
            opp,
            &mv,
            threat_known,
            threat_home,
            is_king,
            flee,
            moved_is_minor(pos, &mv),
            deep_unsupported(&next, &mv, opp),
            hangs_on_landing(pos, &next, &mv, opp),
            foul_count_this_turn,
            my_foul_count_last_turn,
            moved_from_known_attacked(pos, &mv, opp, known_squares),
        );
        total_mass += w;
        if consistent {
            let g = w * guide_boost_factor(pos, &next, &mv, guide, opp);
            candidates.push((mv, w, g));
        }
    }
    // 選択はガイド後の提案分布 g から。補正はクラス内確率の比 p/g で払う
    let idx = weighted_choice_idx(candidates.iter().map(|(_, _, g)| *g), rng)?;
    let class_mass: f64 = candidates.iter().map(|(_, w, _)| w).sum();
    let guide_mass: f64 = candidates.iter().map(|(_, _, g)| g).sum();
    let (chosen, w_c, g_c) = &candidates[idx];
    pos.play_unchecked(chosen);
    // weighted_choice_idx が成功した時点で class_mass > 0、total_mass ≥ class_mass
    let r = (class_mass / total_mass).min(1.0);
    Some(r.ln() + (w_c / class_mass).ln() - (g_c / guide_mass).ln())
}

/// 多段ガイドの接近ブースト倍率（GUIDE_BOOST より弱め。「向かっている」だけで
/// 確定ではないため、exact landing/attacks ほど強くは信じない）
const GUIDE_APPROACH_BOOST: f64 = 3.0;

/// ガイド条件に合う手のブースト倍率（1.0 = ブーストなし）:
/// - lands: マス sq に（成りを剥がした）駒種 role を立てる手 → GUIDE_BOOST。
///   取得駒の観測（captured）は unpromote 済みの駒種なので、成り駒も剥がして照合
/// - occupies: マス sq に駒種を問わず着地する手（打ちマス反則由来）→ GUIDE_BOOST
/// - attacks: 着地点から対象マスへ利きを作る手（取り返しの事前準備）→ GUIDE_BOOST
/// - approach（多段ガイド）: 駒種が一致し、空盤上の最短手数（deduce の BFS）が
///   目的地へ真に縮む手 → GUIDE_APPROACH_BOOST。1手先しか見ない lands/attacks
///   では拾えない「複数手先の目的地への接近」を弱くブーストする
fn guide_boost_factor(
    pos: &Position,
    next: &Position,
    mv: &ShogiMove,
    guide: &Guide,
    mover: Color,
) -> f64 {
    if guide.is_empty() {
        return 1.0;
    }
    let (to, role, from) = match *mv {
        ShogiMove::Board { from, to, .. } => match pos.piece_at(from) {
            Some(p) => (to, unpromote_role(p.role), Some(from)),
            None => return 1.0,
        },
        ShogiMove::Drop { to, role } => (to, role, None),
    };
    if guide.lands.iter().any(|&(sq, r)| sq == to && r == role) {
        return GUIDE_BOOST;
    }
    if guide.occupies.iter().any(|&sq| sq == to) {
        return GUIDE_BOOST;
    }
    if guide
        .attacks
        .iter()
        .any(|&sq| sq != to && next.attacks(to, sq))
    {
        return GUIDE_BOOST;
    }
    if let Some(from) = from {
        for &(r, target) in &guide.approach {
            if r != role || target == to {
                continue; // target==to は既に lands で処理済み（二重ブースト回避）
            }
            let before = crate::deduce::distance_empty_board(role, mover, from, target, false)
                .into_iter()
                .chain(crate::deduce::distance_empty_board(
                    role, mover, from, target, true,
                ))
                .min();
            let after = crate::deduce::distance_empty_board(role, mover, to, target, false)
                .into_iter()
                .chain(crate::deduce::distance_empty_board(
                    role, mover, to, target, true,
                ))
                .min();
            if let (Some(b), Some(a)) = (before, after) {
                if a < b {
                    return GUIDE_APPROACH_BOOST;
                }
            }
        }
    }
    1.0
}

/// 露見マス（自分が駒を取った=相手に通知されたマス）での取り返しブースト。
/// 事前分布のフィットでは駒取りは観測条件で絞られるため学習されていない。
/// 対人実戦では露見駒の回収はほぼ必ず実行されるので予測では強く優先する
const PREDICT_RECAPTURE_BOOST: f64 = 8.0;

/// 相手の応手を事前分布モデルで1手サンプルする（2手読み用の予測）。
/// sample_opp_move と同じ尤度モデルだが、これから指される手の予測なので
/// 観測（取られたマス・王手宣言）による絞り込みは行わない。
/// known_squares / my_touched の意味は sample_opp_move と同じ
pub fn predict_opp_reply<R: Rng>(
    pos: &Position,
    my_color: Color,
    known_squares: &[Coord],
    my_touched: &[Coord],
    my_foul_count_this_turn: u32,
    rng: &mut R,
) -> Option<ShogiMove> {
    weighted_choice(
        &opp_reply_weights(
            pos,
            my_color,
            known_squares,
            my_touched,
            my_foul_count_this_turn,
        ),
        rng,
    )
}

/// 相手の全合法応手と方策重み（事前分布モデル＋露見マスの取り返しブースト）。
/// 2手読みの期待値評価用: サンプルせず重み付き平均を取れる。
/// my_foul_count_this_turn: この手番でここまでに自分が試みた反則の回数
/// （相手は応手時にこれを反則宣言として観測している = 応手予測の
/// my_foul_count_last_turn 特徴量）
pub fn opp_reply_weights(
    pos: &Position,
    my_color: Color,
    known_squares: &[Coord],
    my_touched: &[Coord],
    my_foul_count_this_turn: u32,
) -> Vec<(ShogiMove, f64)> {
    let opp = my_color.other();
    if pos.turn() != opp {
        return vec![];
    }
    let initial = Position::initial();
    let homes: Vec<Coord> = initial
        .pieces()
        .filter(|(sq, p)| {
            p.color == my_color
                && !my_touched.contains(sq)
                && pos
                    .piece_at(*sq)
                    .is_some_and(|cur| cur.color == my_color && cur.role == p.role)
        })
        .map(|(sq, _)| sq)
        .collect();
    let mut candidates: Vec<(ShogiMove, f64)> = vec![];
    for mv in pos.legal_moves() {
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        let threat_known = newly_threatens(pos, &next, &mv, known_squares);
        let threat_home = newly_threatens(pos, &next, &mv, &homes);
        let (is_king, flee) = match mv {
            ShogiMove::Board { from, to, .. } => {
                let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
                (is_king, is_king && flees_danger(from, to, known_squares))
            }
            ShogiMove::Drop { .. } => (false, false),
        };
        // 2手読み予測はまだ起きていない相手の応手を当てるので、相手手番の
        // 反則回数は未知（観測なし）。既定値0（実データの最頻値）を使う。
        // 一方こちらの反則回数（my_foul_count_this_turn）は既知: 相手は
        // 応手時にこの手番の反則宣言を観測済みのはず
        let mut w = opp_move_weight(
            pos,
            opp,
            &mv,
            threat_known,
            threat_home,
            is_king,
            flee,
            moved_is_minor(pos, &mv),
            deep_unsupported(&next, &mv, opp),
            hangs_on_landing(pos, &next, &mv, opp),
            0,
            my_foul_count_this_turn,
            moved_from_known_attacked(pos, &mv, opp, known_squares),
        );
        if let ShogiMove::Board { to, .. } = mv {
            let captures_mine = pos.piece_at(to).is_some_and(|p| p.color == my_color);
            if captures_mine && known_squares.contains(&to) {
                w *= PREDICT_RECAPTURE_BOOST;
            }
        }
        candidates.push((mv, w));
    }
    candidates
}

/// 動かす駒種（移動前の役）が歩・香・桂の小駒か。
/// **定義は bin/fit_opp の moved_is_minor と一致させること**
fn moved_is_minor(pos: &Position, mv: &ShogiMove) -> bool {
    let role = match *mv {
        ShogiMove::Board { from, .. } => pos.piece_at(from).map(|p| p.role),
        ShogiMove::Drop { role, .. } => Some(role),
    };
    matches!(role, Some(Role::Pawn | Role::Lance | Role::Knight))
}

/// 相手の利きがあるマスへの紐なし着地か（取りは除く = 交換ではなく差し出し）。
/// 利き・紐とも着地後の盤面（next）で判定する（開き駒の利きを含む）。
/// 相手の玉の利きも数える（紐がなければ玉に取られる）。銀以上の駒での該当は
/// 実質タダの駒捨てで人間はほぼ指さない（馬@62 のような幻の飛び込み王手の
/// 過大評価を抑える）。**定義は opp_move_features::hangs_on_landing と一致させること**
fn hangs_on_landing(pos: &Position, next: &Position, mv: &ShogiMove, mover: Color) -> bool {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    if pos.piece_at(to).is_some() {
        return false; // 取り（交換の文脈）は対象外
    }
    let opp = mover.other();
    let attacked = next
        .pieces()
        .any(|(sq, p)| p.color == opp && next.attacks(sq, to));
    attacked
        && !next
            .pieces()
            .any(|(sq, p)| p.color == mover && sq != to && next.attacks(sq, to))
}

/// 敵陣（成れる3段）への紐なし着地か。着地点に自分の別の駒の利きが無い。
/// **定義は opp_move_features::deep_unsupported と一致させること**
fn deep_unsupported(next: &Position, mv: &ShogiMove, mover: Color) -> bool {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    let deep = match mover {
        Color::Sente => to.rank <= 3,
        Color::Gote => to.rank >= 7,
    };
    deep && !next
        .pieces()
        .any(|(sq, p)| p.color == mover && sq != to && next.attacks(sq, to))
}

/// チェビシェフ距離（玉の歩数）
fn dist(a: Coord, b: Coord) -> i8 {
    (a.file - b.file).abs().max((a.rank - b.rank).abs())
}

/// 玉の移動が危険地点集合（自分が駒を取ったマス = 相手にとっての露見地点）から
/// 遠ざかる手か。**定義は opp_move_features::flees_danger と一致させること**
fn flees_danger(from: Coord, to: Coord, danger: &[Coord]) -> bool {
    let near = |sq: Coord| danger.iter().map(|&d| dist(sq, d)).min();
    match (near(from), near(to)) {
        (Some(a), Some(b)) => b > a,
        _ => false,
    }
}

/// 位置が既知の敵駒（known 上に立つ mover の敵駒）から当たりを付けられている
/// マスの駒（玉以外）を動かす手か（en-prise 回避）。
/// **定義は opp_move_features::moved_from_known_attacked と一致させること**
fn moved_from_known_attacked(
    pos: &Position,
    mv: &ShogiMove,
    mover: Color,
    known: &[Coord],
) -> bool {
    let ShogiMove::Board { from, .. } = *mv else {
        return false;
    };
    if pos.piece_at(from).is_some_and(|p| p.role == Role::King) {
        return false;
    }
    known.iter().any(|&s| {
        s != from && pos.piece_at(s).is_some_and(|p| p.color != mover) && pos.attacks(s, from)
    })
}

/// 相手の手の尤度づけ。2026-07-21、NN段階①-a: bin/fit_opp の12特徴量
/// 線形フィット（旧実装、パープレキシティ24.2）を1隠れ層MLP
/// （`opp_move_nn::opp_move_nn_forward`）へ置き換えた。
/// 2026-07-22、①-b: 駒種特化ブロック（駒種one-hot・成駒・移動距離・
/// 初期配置マスからの移動）を追加して13→23特徴量。kakutoriで露呈した
/// 「角・飛の長距離移動を表現できない」欠陥と、home_lance_move の
/// 駒種横断への一般化（未観測の駒は初期配置のまま）が狙い。
/// 旧実装で別立てだった home_lance の-1.3加点は、NNが from_home×lance を
/// 直接表現できるようになったため二重計上を避けて廃止した。
/// 呼び出し頻度が1手の意思決定あたり最大10万回超のオーダーのため、
/// ONNX等の推論クレートは使わず手書きforward pass（外部依存ゼロ、
/// 数百FLOP）にしている（詳細は`opp_move_nn.rs`のモジュールコメント）
#[allow(clippy::too_many_arguments)]
fn opp_move_weight(
    pos: &Position,
    opp: Color,
    mv: &ShogiMove,
    threat_known: bool,
    threat_home: bool,
    is_king_move: bool,
    king_flee: bool,
    moved_minor: bool,
    deep_unsup: bool,
    hang: bool,
    foul_count_this_turn: u32,
    my_foul_count_last_turn: u32,
    en_prise_flee: bool,
) -> f64 {
    let (advance, is_drop, promotes) = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let advance = match opp {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            (advance, false, promote)
        }
        ShogiMove::Drop { .. } => (0.0, true, false),
    };
    let pt = piece_type_features(pos, mv, opp);
    let features = [
        advance,
        (promotes && moved_minor) as u8 as f64,
        (promotes && !moved_minor) as u8 as f64,
        is_drop as u8 as f64,
        threat_known as u8 as f64,
        threat_home as u8 as f64,
        is_king_move as u8 as f64,
        king_flee as u8 as f64,
        (deep_unsup && moved_minor) as u8 as f64,
        (deep_unsup && !moved_minor) as u8 as f64,
        (hang && moved_minor) as u8 as f64,
        (hang && !moved_minor) as u8 as f64,
        f64::from(foul_count_this_turn),
        pt[0],
        pt[1],
        pt[2],
        pt[3],
        pt[4],
        pt[5],
        pt[6],
        pt[7],
        pt[8],
        pt[9],
        f64::from(my_foul_count_last_turn),
        en_prise_flee as u8 as f64,
    ];
    // クランプ: NNは訓練データの分布から外れた入力（リプレイの仮説探索中に
    // 現れる、実戦ではまれな特徴量の組み合わせ）に対して極端なlogitを出しうる
    // （旧線形モデルは係数が小さく手作りなので自然に有界だった）。診断で
    // 反則中の王手駒探索（kakutori.kif）の粒子再生成コストが2〜3倍以上に
    // 悪化する事例を確認したため、外挿時の暴走を防ぐ安全弁として導入
    let s = opp_move_nn_forward(&features).clamp(-15.0, 15.0);
    s.exp()
}

fn weighted_choice<R: Rng>(candidates: &[(ShogiMove, f64)], rng: &mut R) -> Option<ShogiMove> {
    let total: f64 = candidates.iter().map(|(_, w)| w).sum();
    if candidates.is_empty() || total <= 0.0 {
        return None;
    }
    let mut t = rng.random_range(0.0..total);
    for (mv, w) in candidates {
        t -= w;
        if t <= 0.0 {
            return Some(*mv);
        }
    }
    candidates.last().map(|(mv, _)| *mv)
}

/// 重み比例で index を選ぶ（weighted_choice の index 版）
fn weighted_choice_idx<R: Rng>(
    weights: impl Iterator<Item = f64> + Clone,
    rng: &mut R,
) -> Option<usize> {
    let total: f64 = weights.clone().sum();
    if total <= 0.0 {
        return None;
    }
    let mut t = rng.random_range(0.0..total);
    let mut last = None;
    for (i, w) in weights.enumerate() {
        t -= w;
        last = Some(i);
        if t <= 0.0 {
            return Some(i);
        }
    }
    last
}

/// synth_particle が棄却サンプリングで試す回数の上限
const SYNTH_ATTEMPTS: u32 = 64;

/// C-8 MVP（直接盤面合成）: 履歴の指し手列を再現せず、既知の制約
/// （自分側は真実そのまま・相手の持ち駒は既知・相手の盤上駒の役割別内訳は
/// 初期20枚から取られた駒を引いて既知）だけを満たす盤面を直接サンプルする。
///
/// **意図的に最小版**: テンポ収支・負の証拠・配置事前分布はまだ実装しない。
/// 相手の残り駒は「取られる前の役割（成りを剥がした生駒）」で配置し、
/// 空きマスは二歩・行き所のない駒の配置合法性だけを守って一様ランダムに選ぶ。
/// 成り（どの駒が成っているか・どこに成ったか）は一切推定しない —
/// これは意図的で、「単純な配置サンプルだけでどこまで再現できるか」を
/// 検証するための基準線（bin/synth_check で確認する）。
///
/// **手番側の静的合法性**（cursor の C-8 設計レビュー指摘。deduce.rs の
/// 部品で実装）: `you_in_check`（今まさに自玉が王手されているか。観測から
/// 厳密に分かる）と矛盾する配置は棄却して引き直す。手番はこちらなので、
/// 王手されているならその通り、されていないなら相手の駒が誰も自玉に
/// 利いていない、という静的な整合性だけを見る（経路・履歴は見ない）。
/// 玉位置バイアス等の事前分布は後続フェーズで追加する
pub fn synth_particle(
    my_color: Color,
    model: &GameModel,
    you_in_check: bool,
    rng: &mut StdRng,
) -> Option<Position> {
    let opp = my_color.other();
    for _ in 0..SYNTH_ATTEMPTS {
        if let Some(pos) = synth_particle_once(my_color, model, rng) {
            let actually_in_check = pos.king_square(my_color).is_some_and(|k| {
                pos.pieces()
                    .any(|(sq, p)| p.color == opp && pos.attacks(sq, k))
            });
            if actually_in_check == you_in_check {
                return Some(pos);
            }
        }
    }
    None
}

/// 玉の配置事前分布の減衰率（本国からのチェビシェフ距離 1 につき exp(-λ)）。
/// **注意**: 当初 likelihood.rs の FITTED_THETA（king_advance に上限なし）を
/// そのまま生成分布として流用したところ、盤の隅（本国から最遠）に確率が
/// 集中する誤った挙動になった（実測: 1a・8a等の隅に上位が集中）。
/// FITTED_THETA は「候補粒子群の中で真実を判別する」識別モデルであり、
/// 候補群自体が指し手の連鎖で自然に生成される（＝隅は元々出現しにくい）
/// という前提の上に成り立つ相対的な重みなので、一様な全マスに対する
/// 生成分布としては使えない。代わりに「本国から離れるほど単調に減衰する」
/// 素直な事前分布に置き換えた。
/// λ は kakunari 1点（診断的中率）への簡易スイープで選定
/// （0.15→5.2% / 0.35→6.0% / 0.5→7.2% / 0.8→8.8% / 1.2→8.7%、0.8-1.2で頭打ち）。
/// **1シナリオだけへの過学習リスクに注意**——他のシナリオでの再検証が必要
const KING_DISTANCE_DECAY: f64 = 0.8;

/// 玉の配置事前分布スコア: 本国からのチェビシェフ距離だけで単調減衰する
fn king_placement_score(king_home: Coord, candidate: Coord) -> f64 {
    let dist = (candidate.file - king_home.file)
        .abs()
        .max((candidate.rank - king_home.rank).abs());
    -KING_DISTANCE_DECAY * f64::from(dist)
}

fn synth_particle_once(my_color: Color, model: &GameModel, rng: &mut StdRng) -> Option<Position> {
    let opp = my_color.other();
    let mut pos = Position::empty(my_color);
    for p in model.my_pieces() {
        let sq = parse_usi_square(&p.square)?;
        pos.set(
            sq,
            Some(Piece {
                color: my_color,
                role: p.role,
            }),
        );
    }
    for (role, n) in model.my_hand() {
        pos.set_hand(my_color, role, n as u8);
    }
    for (role, n) in model.opponent_hand() {
        pos.set_hand(opp, role, n as u8);
    }

    // 相手の盤上駒（生駒ベースの役割）の残り枚数 = 初期配置 − 取られた駒
    let mut counts: HashMap<Role, i32> = HashMap::new();
    for (_, p) in Position::initial().pieces() {
        if p.color == opp {
            *counts.entry(p.role).or_insert(0) += 1;
        }
    }
    for (_, role) in model.lost_pieces() {
        *counts.entry(unpromote_role(*role)).or_insert(0) -= 1;
    }
    let mut remaining: Vec<Role> = vec![];
    for (&role, &c) in &counts {
        for _ in 0..c.max(0) {
            remaining.push(role);
        }
    }

    // 空きマスの初期プール
    let mut empties: Vec<Coord> = (1..=9)
        .flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
        .filter(|&sq| pos.piece_at(sq).is_none())
        .collect();

    // 玉だけ先に配置事前分布で重み付きサンプリングする（taint に頼らない
    // 玉位置ビリーフ。他の駒は依然として一様ランダム — 意図的な最小拡張）
    if let Some(king_idx) = remaining.iter().position(|&r| r == Role::King) {
        remaining.remove(king_idx);
        let king_home = Position::initial().king_square(opp);
        let placed = king_home.and_then(|home| {
            let weights: Vec<f64> = empties
                .iter()
                .map(|&sq| king_placement_score(home, sq).exp())
                .collect();
            weighted_choice_idx(weights.into_iter(), rng)
        });
        match placed {
            Some(i) => {
                let sq = empties.remove(i);
                pos.set(
                    sq,
                    Some(Piece {
                        color: opp,
                        role: Role::King,
                    }),
                );
            }
            // 万一重み付きサンプリングが失敗したら通常の一様配置へ戻す
            None => remaining.push(Role::King),
        }
    }

    // 残りの駒をシャッフルして順に置く（二歩・行き所のない駒だけ回避）
    empties.shuffle(rng);
    remaining.shuffle(rng);
    let mut ei = 0usize;
    for role in remaining {
        let mut placed = false;
        while ei < empties.len() {
            let sq = empties[ei];
            ei += 1;
            if role == Role::Pawn
                && pos
                    .pieces()
                    .any(|(s, p)| p.color == opp && p.role == Role::Pawn && s.file == sq.file)
            {
                continue; // 二歩
            }
            if dead_end_rank(role, sq.rank, opp) {
                continue;
            }
            pos.set(sq, Some(Piece { color: opp, role }));
            placed = true;
            break;
        }
        if !placed {
            return None;
        }
    }
    Some(pos)
}

// ---------------------------------------------------------------------------
// 相手手モデルの駒種特化特徴量（opp_move_features.rs のコピー）
// ---------------------------------------------------------------------------

fn piece_type_features(pos: &Position, mv: &ShogiMove, mover: Color) -> [f64; 10] {
    let (role_raw, dist, from_home) = match *mv {
        ShogiMove::Board { from, to, .. } => {
            let Some(p) = pos.piece_at(from) else {
                return [0.0; 10];
            };
            let dist = (from.file - to.file).abs().max((from.rank - to.rank).abs());
            (p.role, f64::from(dist), is_home_square(p.role, mover, from))
        }
        ShogiMove::Drop { role, .. } => (role, 0.0, false),
    };
    let base = unpromote_role(role_raw);
    let one_hot = |r: Role| (base == r) as u8 as f64;
    [
        one_hot(Role::Pawn),
        one_hot(Role::Lance),
        one_hot(Role::Knight),
        one_hot(Role::Silver),
        one_hot(Role::Gold),
        one_hot(Role::Bishop),
        one_hot(Role::Rook),
        (role_raw != base) as u8 as f64,
        dist,
        from_home as u8 as f64,
    ]
}

/// マス sq がその駒種（成っていない駒）の初期配置マスか。
/// 「まだ初期配置マスに立っている＝未動」の近似（実際は一度動いて戻った
/// 可能性もあるが、旧home_lance_moveと同じ近似を全駒種へ一般化した）
fn is_home_square(role: Role, mover: Color, sq: Coord) -> bool {
    let home_rank = |sente: i8, gote: i8| match mover {
        Color::Sente => sente,
        Color::Gote => gote,
    };
    match role {
        Role::Pawn => sq.rank == home_rank(7, 3),
        Role::Lance => sq.rank == home_rank(9, 1) && (sq.file == 1 || sq.file == 9),
        Role::Knight => sq.rank == home_rank(9, 1) && (sq.file == 2 || sq.file == 8),
        Role::Silver => sq.rank == home_rank(9, 1) && (sq.file == 3 || sq.file == 7),
        Role::Gold => sq.rank == home_rank(9, 1) && (sq.file == 4 || sq.file == 6),
        Role::King => sq.rank == home_rank(9, 1) && sq.file == 5,
        Role::Bishop => match mover {
            Color::Sente => sq.file == 8 && sq.rank == 8,
            Color::Gote => sq.file == 2 && sq.rank == 2,
        },
        Role::Rook => match mover {
            Color::Sente => sq.file == 2 && sq.rank == 8,
            Color::Gote => sq.file == 8 && sq.rank == 2,
        },
        _ => false, // 成駒は初期配置に存在しない
    }
}

// ---------------------------------------------------------------------------
// opp_move NN（opp_move_nn.rs のコピー）
// ---------------------------------------------------------------------------

// opp_move_weight のNN版（NN段階①-a、①-bで駒種特化ブロックを追加し
// 13→23特徴量、逆方向反則特徴量 my_foul_count_last_turn で23→24特徴量、
// en-prise回避 en_prise_flee で24→25特徴量）。
// tsuitate-nn/train_opp_move.pyで学習し、重み配列は
// export_opp_move_weights.pyで生成、forward passはここに手書き
// （呼び出し頻度が1手あたり最大10万回超のオーダーのため、
// ONNX/推論クレートは使わずLinear->ReLU->Linearを直接計算する）。

// AUTO-GENERATED BEGIN (export_opp_move_weights.py)
// 学習: seed=1 hidden=16 val_nll=2.6172 val_top1=0.327 (6512決定点)
// 再生成: tsuitate-nn/export_opp_move_weights.py --data data/opp_move_data.csv --out ../tsuitate-bot/src/opp_move_nn.rs
pub const OPP_MOVE_NN_MEAN: [f64; 25] = [
    2.38061994e-01,
    3.11661744e-03,
    4.56332741e-03,
    6.62361443e-01,
    2.38059148e-01,
    2.39054576e-01,
    3.40469666e-02,
    9.21870302e-03,
    3.05992197e-02,
    7.91552365e-02,
    1.04215659e-01,
    1.47170156e-01,
    1.18647330e-01,
    1.40912086e-01,
    1.43285796e-01,
    1.38246745e-01,
    1.52198613e-01,
    1.43919945e-01,
    1.20687872e-01,
    1.26701981e-01,
    3.12932506e-02,
    4.65916365e-01,
    1.82032526e-01,
    1.52912602e-01,
    1.46772638e-02,
];
pub const OPP_MOVE_NN_STD: [f64; 25] = [
    7.35523283e-01,
    5.56570552e-02,
    6.72520995e-02,
    4.75926906e-01,
    4.30719286e-01,
    4.31239307e-01,
    1.79841042e-01,
    9.51848775e-02,
    1.70250565e-01,
    2.68571347e-01,
    3.07153672e-01,
    3.53657365e-01,
    4.28291649e-01,
    3.47634375e-01,
    3.49573135e-01,
    3.44627529e-01,
    3.56994629e-01,
    3.50290954e-01,
    3.26263338e-01,
    3.33068997e-01,
    1.73040435e-01,
    8.72717083e-01,
    3.83507073e-01,
    5.65936506e-01,
    1.19507775e-01,
];
pub const OPP_MOVE_NN_W1: [[f64; 25]; 16] = [
    [
        8.25573862e-01,
        -2.49286201e-02,
        2.10578702e-02,
        1.48935974e+00,
        -1.02386820e+00,
        -3.35646160e-02,
        -5.01761556e-01,
        -1.09204546e-01,
        -2.08352432e-01,
        6.11023009e-01,
        5.04491866e-01,
        6.70185089e-01,
        -1.07614422e+00,
        -6.14011884e-02,
        1.14402644e-01,
        4.96964669e-03,
        -5.82918346e-01,
        -2.53121018e-01,
        4.63353276e-01,
        6.96933448e-01,
        -1.12394166e+00,
        2.20366091e-01,
        -8.90867710e-01,
        -3.60206813e-01,
        -3.51312041e-01,
    ],
    [
        1.15195882e+00,
        5.48958294e-02,
        -3.69493961e-01,
        -4.16241318e-01,
        -2.90731639e-01,
        6.73045397e-01,
        -7.46071860e-02,
        -2.72190213e-01,
        -4.50586855e-01,
        -1.86106920e-01,
        -2.86244661e-01,
        8.01477954e-02,
        1.82702914e-01,
        -1.45597741e-01,
        -1.62287399e-01,
        -5.67440331e-01,
        1.29445359e-01,
        1.97799146e-01,
        4.59172159e-01,
        4.93924171e-01,
        4.05841202e-01,
        -1.35603392e+00,
        1.39492646e-01,
        1.22571262e-02,
        2.69856393e-01,
    ],
    [
        -8.20275664e-01,
        -1.34829119e-01,
        -1.13143949e-02,
        1.13434064e+00,
        6.14616752e-01,
        -8.86239886e-01,
        -4.52608049e-01,
        -1.93555444e-01,
        -5.30202627e-01,
        -9.49155092e-02,
        -1.16111077e-01,
        2.17123255e-01,
        9.95176584e-02,
        2.94362843e-01,
        -2.48341590e-01,
        1.52195506e-02,
        2.14738697e-01,
        8.13540161e-01,
        -2.88692296e-01,
        -5.84246516e-01,
        6.44098282e-01,
        4.51885730e-01,
        -8.88692200e-01,
        2.96436399e-01,
        1.59165412e-01,
    ],
    [
        6.30731583e-01,
        -5.44718623e-01,
        -9.46641624e-01,
        -1.14240301e+00,
        -9.20839071e-01,
        -8.46767426e-01,
        -7.03858078e-01,
        -8.99398103e-02,
        1.07720459e+00,
        4.74931538e-01,
        -7.83380210e-01,
        -7.39762723e-01,
        -2.34435663e-01,
        -5.84813893e-01,
        2.91904509e-01,
        -6.38456464e-01,
        7.57457733e-01,
        2.62352705e-01,
        1.83651492e-01,
        3.24837118e-01,
        -1.15849078e+00,
        7.65222132e-01,
        -6.37602985e-01,
        -7.14948028e-02,
        6.65809587e-02,
    ],
    [
        -5.11225402e-01,
        -4.40564081e-02,
        -5.50667085e-02,
        7.51743972e-01,
        -9.28734958e-01,
        -2.39153162e-01,
        -2.80856103e-01,
        -2.65311807e-01,
        -3.64106983e-01,
        5.13640989e-04,
        1.70227915e-01,
        -8.45490582e-03,
        -1.81827873e-01,
        -2.44984493e-01,
        8.04747164e-01,
        5.36726296e-01,
        3.04283202e-01,
        4.96581376e-01,
        -1.36154330e+00,
        -8.11300099e-01,
        -3.57373565e-01,
        -2.72698021e+00,
        2.05026460e+00,
        -6.60800189e-02,
        -6.75510019e-02,
    ],
    [
        -3.49894494e-01,
        -1.69989869e-01,
        -1.05395846e-01,
        7.22017229e-01,
        2.96186976e-04,
        4.51399565e-01,
        -1.70240730e-01,
        -6.57356232e-02,
        7.60375202e-01,
        1.36108920e-01,
        -5.88729084e-01,
        -8.66383851e-01,
        -3.01456243e-01,
        -3.99302214e-01,
        -2.68451929e-01,
        6.32265285e-02,
        5.53079307e-01,
        -5.79838276e-01,
        4.90527809e-01,
        3.33974719e-01,
        -1.92262173e-01,
        -1.24522221e+00,
        1.72612524e+00,
        -7.50005007e-01,
        1.19623899e-01,
    ],
    [
        1.53140414e+00,
        3.18689585e-01,
        -3.88936877e-01,
        -7.89035022e-01,
        -3.46002758e-01,
        -1.02861178e+00,
        9.85183492e-02,
        -2.74218572e-03,
        -4.48143303e-01,
        -6.00695252e-01,
        6.06766224e-01,
        4.70420808e-01,
        -1.75035134e-01,
        -5.40069759e-01,
        -2.24353373e-02,
        3.31106246e-01,
        5.06068528e-01,
        1.07520223e-01,
        -5.58090925e-01,
        3.05080503e-01,
        -5.85354328e-01,
        -1.10294139e+00,
        -6.98330939e-01,
        -1.65755749e-01,
        2.53315061e-01,
    ],
    [
        1.02122188e+00,
        2.86930680e-01,
        -3.75988260e-02,
        7.17293859e-01,
        2.75237560e-01,
        1.45661163e+00,
        -5.43118954e-01,
        -2.08362415e-01,
        -4.29253280e-01,
        -8.58444273e-01,
        -5.96079767e-01,
        -1.73485398e+00,
        -6.11122966e-01,
        6.18585110e-01,
        -6.59531653e-01,
        7.32244626e-02,
        -1.01262681e-01,
        1.73866943e-01,
        -3.24793100e-01,
        1.17309086e-01,
        -1.21811606e-01,
        3.45124006e-01,
        1.04377747e-01,
        -1.01333514e-01,
        -4.24961627e-01,
    ],
    [
        1.26082802e+00,
        -3.98809522e-01,
        3.21009010e-01,
        -1.05207193e+00,
        -1.03638321e-01,
        -9.85222906e-02,
        9.53634456e-03,
        -2.08066344e-01,
        -5.44752479e-01,
        1.43988237e-01,
        -6.76987052e-01,
        8.22536767e-01,
        -9.50575992e-02,
        1.35039702e-01,
        9.94637430e-01,
        -3.26237321e-01,
        -1.33680433e-01,
        2.16403112e-01,
        -5.06597102e-01,
        -7.54189789e-02,
        -6.66193366e-02,
        -1.07953310e+00,
        -9.68161285e-01,
        -4.37795281e-01,
        -4.63809341e-01,
    ],
    [
        -5.85371077e-01,
        3.82398546e-01,
        2.83418626e-01,
        5.54684162e-01,
        3.79017442e-02,
        6.41887486e-02,
        -3.27678651e-01,
        -3.62043679e-02,
        4.11790788e-01,
        9.82220709e-01,
        1.54726371e-01,
        -4.90335912e-01,
        -6.14698455e-02,
        5.37848413e-01,
        -3.06595683e-01,
        -2.24155098e-01,
        -2.05250308e-01,
        9.04533863e-01,
        -5.87063313e-01,
        -2.35056162e-01,
        -6.42888844e-01,
        -1.41116178e+00,
        1.81320548e-01,
        -9.99843143e-03,
        8.16172287e-02,
    ],
    [
        7.36639321e-01,
        1.85382023e-01,
        -1.30178720e-01,
        1.39630902e+00,
        5.25575876e-01,
        1.16393781e+00,
        -3.57145250e-01,
        -1.24048978e-01,
        -3.75284165e-01,
        -4.00086164e-01,
        -2.78146237e-01,
        5.16313791e-01,
        2.75031030e-01,
        -7.78030396e-01,
        7.33166710e-02,
        -5.34381330e-01,
        3.44885319e-01,
        6.70131564e-01,
        7.38953233e-01,
        3.88846882e-02,
        -3.60746868e-02,
        1.04820657e+00,
        -1.69869125e+00,
        -1.60682365e-01,
        -5.00139415e-01,
    ],
    [
        -1.20493448e+00,
        -2.16021270e-01,
        -7.29201853e-01,
        -1.22397339e+00,
        -4.21979398e-01,
        -1.20413232e+00,
        5.44167638e-01,
        7.54300207e-02,
        1.30443826e-01,
        -9.22104478e-01,
        -1.98875785e-01,
        -1.26660550e+00,
        -1.11263597e+00,
        2.85497993e-01,
        -2.02376787e-02,
        -2.28807330e-01,
        -4.92755324e-01,
        1.93011299e-01,
        -1.02666557e-01,
        2.80604102e-02,
        5.85281365e-02,
        -8.39462876e-01,
        6.87177002e-01,
        -1.29640505e-01,
        -7.05859542e-01,
    ],
    [
        6.48559868e-01,
        1.09248430e-01,
        2.72675864e-02,
        -7.46895134e-01,
        -2.77491003e-01,
        5.48850894e-01,
        -3.99668425e-01,
        1.73919752e-01,
        -7.55141258e-01,
        -2.43159272e-02,
        2.56304413e-01,
        -1.84347034e-01,
        3.72190863e-01,
        -1.01885104e+00,
        8.36019099e-01,
        -9.81831491e-01,
        -2.69272625e-01,
        8.45673680e-02,
        1.02110529e+00,
        3.81198406e-01,
        -3.69307667e-01,
        -1.99047506e+00,
        -1.09089899e+00,
        5.58570385e-01,
        -2.94330418e-02,
    ],
    [
        -9.03229535e-01,
        -2.89736271e-01,
        -3.91533107e-01,
        8.12587082e-01,
        -8.71305883e-01,
        7.68079102e-01,
        4.63343188e-02,
        -2.14192212e-01,
        2.01364443e-01,
        -2.64667332e-01,
        4.26000863e-01,
        -3.22425276e-01,
        -8.78881887e-02,
        -1.70971346e+00,
        3.76412794e-02,
        2.38288894e-01,
        5.18223941e-01,
        -1.33797038e+00,
        1.11222243e+00,
        7.10938573e-01,
        -7.31459200e-01,
        7.01082423e-02,
        -1.88672078e+00,
        -9.34743807e-02,
        -3.77393156e-01,
    ],
    [
        -1.31778049e+00,
        -8.25247988e-02,
        -5.87121189e-01,
        -3.35239142e-01,
        -1.05638444e+00,
        8.51878703e-01,
        1.27285942e-01,
        -1.30774483e-01,
        -5.58953464e-01,
        -4.89436507e-01,
        1.02300191e+00,
        4.88829255e-01,
        -7.12934434e-02,
        -7.02052057e-01,
        4.34947312e-01,
        3.37357521e-01,
        4.11144465e-01,
        -4.15363967e-01,
        3.99293691e-01,
        -1.86541546e-02,
        -1.02294910e+00,
        3.19768786e-01,
        7.43351460e-01,
        -1.04950830e-01,
        -2.90650785e-01,
    ],
    [
        1.86802104e-01,
        -5.52378476e-01,
        -7.07535744e-01,
        5.88803947e-01,
        -1.08550692e+00,
        -4.95291203e-02,
        -3.65242809e-01,
        -9.27071050e-02,
        2.86133081e-01,
        1.88664109e-01,
        1.61665693e-01,
        7.34124422e-01,
        -8.89881074e-01,
        -4.79745157e-02,
        -2.38417313e-01,
        -8.60413849e-01,
        2.62400270e-01,
        -2.93917954e-01,
        5.00544131e-01,
        9.20612216e-01,
        -5.74721694e-01,
        5.08344054e-01,
        -7.35321820e-01,
        -1.43353462e-01,
        3.73052031e-01,
    ],
];
pub const OPP_MOVE_NN_B1: [f64; 16] = [
    2.93858886e-01,
    -7.57773519e-01,
    -7.69762576e-01,
    -1.33660758e+00,
    6.05705917e-01,
    -1.19830525e+00,
    -1.27117205e+00,
    -1.53919324e-01,
    -9.28404868e-01,
    -1.04345310e+00,
    2.24867195e-01,
    -6.41137183e-01,
    -6.23404860e-01,
    -5.71554601e-01,
    -2.08442163e+00,
    1.90033570e-01,
];
pub const OPP_MOVE_NN_W2: [f64; 16] = [
    -2.05562353e-01,
    9.55694541e-02,
    -1.83556929e-01,
    -3.16798031e-01,
    -5.08648336e-01,
    3.22915882e-01,
    1.80140346e-01,
    1.84866697e-01,
    2.03519419e-01,
    1.81837097e-01,
    -1.93816170e-01,
    -2.68392771e-01,
    2.61301726e-01,
    -3.76972497e-01,
    4.21299011e-01,
    -1.68915391e-01,
];
pub const OPP_MOVE_NN_B2: f64 = 3.33522039e-04;
// AUTO-GENERATED END

/// 学習時と同じ正規化 + Linear(23→16) → ReLU → Linear(16→1) のforward pass。
/// 出力はlogit（Sigmoidではない）。呼び出し側は `clamp(-15, 15)` してから
/// `exp(logit)` として使う
pub fn opp_move_nn_forward(features: &[f64; 25]) -> f64 {
    let mut x = [0.0f64; 25];
    for i in 0..25 {
        x[i] = (features[i] - OPP_MOVE_NN_MEAN[i]) / OPP_MOVE_NN_STD[i];
    }
    let mut h = [0.0f64; 16];
    for j in 0..16 {
        let mut s = OPP_MOVE_NN_B1[j];
        for i in 0..25 {
            s += OPP_MOVE_NN_W1[j][i] * x[i];
        }
        h[j] = s.max(0.0); // ReLU
    }
    let mut out = OPP_MOVE_NN_B2;
    for j in 0..16 {
        out += OPP_MOVE_NN_W2[j] * h[j];
    }
    out
}

// ---------------------------------------------------------------------------
// 粒子尤度モデル（likelihood.rs のコピー）
// ---------------------------------------------------------------------------

pub const PARTICLE_FEATURES: usize = 8;

pub const FEATURE_NAMES: [&str; PARTICLE_FEATURES] = [
    "king_moved",   // 相手玉が初期位置から動いた
    "king_advance", // 相手玉の前進量（段。負=後退はない）
    "king_shift",   // 相手玉の横ずれ量（筋）
    "pawn_advance", // 相手の歩（と金含む）の平均前進量
    "pieces_home",  // 初期位置に残っている相手駒の割合（0..1）
    "at_my_death",  // 直近で自駒が死んだマスに相手駒がいる（取った駒は残留しがち）
    "in_my_camp",   // 自陣（3段）内の相手駒数
    "past_mid",     // 中央線を越えて自分側にいる相手駒数（歩・玉以外）
];

/// フィット済み係数（bin/fit_particles の出力を反映する）。
/// 2026-07-16 フィット（CI run 29468501253、600局・6157決定点、
/// 実効候補数 59.3→32.9、真実が上位半分に入る率 77.9%）。
/// 主な補正: 実際の相手は粒子の想定より歩を突き駒を展開している
/// （pawn_advance / pieces_home）、玉は想定ほど動かない（king_moved）、
/// 大駒の中央線越えは過大評価だった（past_mid）
pub const FITTED_THETA: [f64; PARTICLE_FEATURES] = [
    -0.815, // king_moved
    0.543,  // king_advance
    0.248,  // king_shift
    2.532,  // pawn_advance
    -2.051, // pieces_home
    -0.073, // at_my_death
    -0.050, // in_my_camp
    -1.377, // past_mid
];

/// 推論時に観測から分かる文脈
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticleCtx {
    /// 直近で自駒が取られたマス（相手の駒がそこに着地した）
    pub opp_landed_last: Option<Coord>,
}

/// 相手側の前進量（段）: 初期配置側から自分側へ何段進んだか
fn advance_of(rank: i8, home_rank: i8, opp: Color) -> f64 {
    match opp {
        Color::Gote => f64::from(rank - home_rank),
        Color::Sente => f64::from(home_rank - rank),
    }
}

/// 粒子の特徴量。my_color は自分（観測者）の色
pub fn particle_features(
    pos: &Position,
    my_color: Color,
    ctx: &ParticleCtx,
) -> [f64; PARTICLE_FEATURES] {
    let opp = my_color.other();
    let initial = Position::initial();

    // 玉の3特徴
    let king_home = initial.king_square(opp);
    let king = pos.king_square(opp);
    let (king_moved, king_advance, king_shift) = match (king, king_home) {
        (Some(k), Some(h)) => (
            f64::from(k != h),
            advance_of(k.rank, h.rank, opp).max(0.0),
            f64::from((k.file - h.file).abs()),
        ),
        _ => (1.0, 0.0, 0.0),
    };

    // 歩（と金含む）の平均前進量。相手歩の初期段: 後手=3段目 / 先手=7段目
    let pawn_home = match opp {
        Color::Gote => 3,
        Color::Sente => 7,
    };
    let mut pawn_adv = 0.0;
    let mut pawns = 0.0;
    // 初期位置に残っている相手駒（種類まで一致）の数
    let mut home_count = 0.0;
    let mut in_my_camp = 0.0;
    let mut past_mid = 0.0;
    for (sq, p) in pos.pieces() {
        if p.color != opp {
            continue;
        }
        if matches!(p.role, Role::Pawn | Role::Tokin) {
            pawn_adv += advance_of(sq.rank, pawn_home, opp).max(0.0);
            pawns += 1.0;
        }
        // 自陣3段（自分側の端から3段）
        let in_camp = match my_color {
            Color::Sente => sq.rank >= 7,
            Color::Gote => sq.rank <= 3,
        };
        if in_camp {
            in_my_camp += 1.0;
        }
        // 中央線越え（歩・玉以外）
        let past = match my_color {
            Color::Sente => sq.rank >= 6,
            Color::Gote => sq.rank <= 4,
        };
        if past && !matches!(p.role, Role::Pawn | Role::Tokin | Role::King) {
            past_mid += 1.0;
        }
    }
    for (sq, p) in initial.pieces() {
        if p.color == opp
            && pos
                .piece_at(sq)
                .is_some_and(|cur| cur.color == opp && cur.role == p.role)
        {
            home_count += 1.0;
        }
    }

    let at_my_death = ctx
        .opp_landed_last
        .map(|s| f64::from(pos.piece_at(s).is_some_and(|p| p.color == opp)))
        .unwrap_or(0.0);

    [
        king_moved,
        king_advance,
        king_shift,
        if pawns > 0.0 { pawn_adv / pawns } else { 0.0 },
        home_count / 20.0,
        at_my_death,
        in_my_camp,
        past_mid,
    ]
}

/// θ·φ（対数重み）。重みは exp(θ·φ) で、呼び出し側が平均1へ正規化する
pub fn particle_log_weight(
    features: &[f64; PARTICLE_FEATURES],
    theta: &[f64; PARTICLE_FEATURES],
) -> f64 {
    features.iter().zip(theta).map(|(f, t)| f * t).sum()
}

// ---------------------------------------------------------------------------
// 王手ソルバー（check.rs のコピー）
// ---------------------------------------------------------------------------

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

/// 粒子投票の強さ（全粒子が一致した仮説は一様仮説の 1+PARTICLE_VOTE_W 倍）
const PARTICLE_VOTE_W: f64 = 8.0;

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
        };
        solver.enumerate(&opponent_role_counts(view, log));
        if solver.hypotheses.is_empty() {
            return None;
        }
        solver.threat_cache = vec![None; solver.hypotheses.len()];
        solver.vote_by_particles(particles);
        for foul in fouls_this_turn {
            solver.observe_foul(foul);
        }
        Some(solver)
    }

    /// 自玉を攻撃しうる（マス, 駒種）を全列挙する。
    /// 相手が1枚も持ちえない駒種（総数制約）は仮説から外す
    fn enumerate(&mut self, opp_counts: &HashMap<Role, i32>) {
        let opp = self.my_color.other();
        let king = self
            .base
            .king_square(self.my_color)
            .expect("new で確認済み");
        for file in 1..=9i8 {
            for rank in 1..=9i8 {
                let sq = Coord { file, rank };
                if self.base.piece_at(sq).is_some() {
                    // 自駒・既知の敵駒のあるマスに（新たな）王手駒はいない
                    // （既知の敵駒が王手していたなら以前から王手宣言されているはず）
                    continue;
                }
                if sq == king {
                    continue;
                }
                for role in CHECKER_ROLES {
                    if opp_counts
                        .get(&unpromote_role(role))
                        .is_none_or(|&n| n <= 0)
                    {
                        continue;
                    }
                    self.base
                        .set(sq, Some(crate::shogi::Piece { color: opp, role }));
                    let checks = self.base.in_check(self.my_color);
                    self.base.set(sq, None);
                    if checks {
                        self.hypotheses.push(Hypothesis {
                            square: sq,
                            role,
                            weight: 1.0,
                        });
                    }
                }
            }
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

    /// この手番の反則を観測: 仮説の下で合法だったはずの手が反則になった
    /// → その仮説の重みを減衰させる
    fn observe_foul(&mut self, foul: &ShogiMove) {
        for i in 0..self.hypotheses.len() {
            if self.legal_under(i, foul) {
                self.hypotheses[i].weight *= UNEXPLAINED_FOUL_DECAY;
            }
        }
    }

    /// 仮説 i の下で（他の隠れ駒を無視して）mv が合法か = 王手を解消するか
    fn legal_under(&mut self, i: usize, mv: &ShogiMove) -> bool {
        let h = &self.hypotheses[i];
        let piece = crate::shogi::Piece {
            color: self.my_color.other(),
            role: h.role,
        };
        let sq = h.square;
        self.base.set(sq, Some(piece));
        let legal = self.base.is_legal(mv);
        self.base.set(sq, None);
        legal
    }

    /// 候補手が王手を解消する確率（仮説の重み付き割合）
    pub fn resolve_probability(&mut self, mv: &ShogiMove) -> f64 {
        let mut total = 0.0;
        let mut resolved = 0.0;
        for i in 0..self.hypotheses.len() {
            let w = self.hypotheses[i].weight;
            total += w;
            if self.legal_under(i, mv) {
                resolved += w;
            }
        }
        if total <= 0.0 {
            return 0.5; // 全仮説が死んだ（両王手など）: 情報なしに戻す
        }
        resolved / total
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
                t += THREAT_MATERIAL_W * exchange_value(r);
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
        self.base.set(sq, None);
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
                term += w * (exchange_value(role) + self.threat_of(i));
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

// ---------------------------------------------------------------------------
// 定跡ブック（opening.rs のコピー）
// ---------------------------------------------------------------------------

/// 組み込みの定跡ライン（joseki.json が見つからないときのフォールバック）。
/// 正本は joseki.json（tools/joseki-editor.html で編集・エクスポートする）
const BUILTIN_LINES: [&[&str]; 4] = [
    // 居飛車速攻（所有者定跡: 基本中の基本）。2六歩〜2三歩成まで一直線。
    // 最後の歩成で駒取りが発生し、その観測でブックを抜けて通常思考に戻る
    &["2g2f", "2f2e", "2e2d", "2d2c+"],
    // 玉を右に逃がして金銀で蓋をする（仮ライン）
    &["5i4h", "4h3h", "7i6h", "5g5f"],
    // 中住まい風（仮ライン）
    &["5i5h", "3i4h", "7i6h", "5g5f"],
    // 左に囲う（仮ライン）
    &["5i6h", "7i7h", "6h7i", "5g5f"],
];

/// 定跡ラインの読み込み（プロセス内で1回だけ）。
/// TSUITATE_JOSEKI（既定 joseki.json）の {"lines":[{"name","moves":[usi...]}]} を読む。
/// パースできない手を含むラインは警告してスキップする
fn load() -> &'static (Vec<String>, Vec<Vec<String>>) {
    static LOADED: std::sync::OnceLock<(Vec<String>, Vec<Vec<String>>)> =
        std::sync::OnceLock::new();
    LOADED.get_or_init(|| {
        let path = std::env::var("TSUITATE_JOSEKI").unwrap_or_else(|_| "joseki.json".into());
        if let Ok(content) = std::fs::read_to_string(&path) {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(v) => {
                    let mut names = vec![];
                    let mut lines = vec![];
                    for line in v["lines"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                        let moves: Vec<String> = line["moves"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|m| m.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        if moves.is_empty() || moves.iter().any(|u| parse_usi(u).is_none()) {
                            eprintln!("定跡ラインを解釈できずスキップ: {:?}", line["name"]);
                            continue;
                        }
                        names.push(line["name"].as_str().unwrap_or("?").to_string());
                        lines.push(moves);
                    }
                    if !lines.is_empty() {
                        return (names, lines);
                    }
                    eprintln!("{path} に有効なラインがないため組み込み定跡を使います");
                }
                Err(e) => eprintln!("{path} をパースできません（組み込み定跡を使用）: {e}"),
            }
        }
        (
            (1..=BUILTIN_LINES.len())
                .map(|i| format!("組み込み{i}"))
                .collect(),
            BUILTIN_LINES
                .iter()
                .map(|l| l.iter().map(|s| s.to_string()).collect())
                .collect(),
        )
    })
}

fn lines() -> &'static Vec<Vec<String>> {
    &load().1
}

fn line_names() -> &'static Vec<String> {
    &load().0
}

/// USI手を点対称にミラーする（先手ライン → 後手用）
fn mirror_usi(usi: &str) -> Option<String> {
    let mv = parse_usi(usi)?;
    let flip = |c: crate::board::Coord| crate::board::Coord {
        file: 10 - c.file,
        rank: 10 - c.rank,
    };
    let mirrored = match mv {
        ShogiMove::Board { from, to, promote } => ShogiMove::Board {
            from: flip(from),
            to: flip(to),
            promote,
        },
        ShogiMove::Drop { role, to } => ShogiMove::Drop { role, to: flip(to) },
    };
    Some(mirrored.to_usi())
}

pub struct OpeningBook {
    /// 対局開始時に選んだライン（自色向けにミラー済み）
    line: Vec<String>,
    /// ブックから抜けたら true（以後戻らない）
    exited: bool,
}

impl OpeningBook {
    /// 指定インデックスのラインに固定したブック（定跡特化チューニング用）
    pub fn with_line(my_color: Color, index: usize) -> Self {
        let all = lines();
        let raw = &all[index % all.len()];
        let line = raw
            .iter()
            .filter_map(|usi| match my_color {
                Color::Sente => Some(usi.clone()),
                Color::Gote => mirror_usi(usi),
            })
            .collect();
        OpeningBook {
            line,
            exited: false,
        }
    }

    /// ライン名（joseki.json の name）からインデックスを引く
    pub fn line_index(name: &str) -> Option<usize> {
        line_names().iter().position(|n| n == name)
    }

    pub fn new(my_color: Color) -> Self {
        // ランダム選択（対局をまたいで人間に順番を読まれないため）。
        // SPSA（bin/tune）は with_seed で決定論的に選ぶ（共通乱数法）
        Self::with_line(my_color, rand::rng().random_range(0..lines().len()))
    }

    /// シードから決定論的にラインを選ぶ（SPSA の f+/f− 評価で
    /// 同じ対局番号に同じ定跡を割り当てるための共通乱数法用）
    pub fn with_seed(my_color: Color, seed: u64) -> Self {
        Self::with_line(my_color, (seed % lines().len() as u64) as usize)
    }

    /// ブックの次の一手。None ならブックを抜けた（通常思考へ）
    pub fn next(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        if self.exited {
            return None;
        }
        // 静かな序盤でなくなったら抜ける
        let quiet = log.events().iter().all(|e| match e {
            Observation::MyMove { captured, .. } => captured.is_none(),
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => captured_my_piece_at.is_none(),
            Observation::MyFoul { .. } | Observation::Check { .. } => false,
            Observation::OpponentFoul { .. } => true, // 相手の反則は情報にならない
        });
        if !quiet || view.you_in_check {
            self.exited = true;
            return None;
        }
        // 自分が何手指したか = ラインの進行位置
        let my_moves = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { .. }))
            .count();
        let Some(usi) = self.line.get(my_moves) else {
            self.exited = true; // ライン消化完了
            return None;
        };
        if foul_tried.contains(usi.as_str()) {
            self.exited = true;
            return None;
        }
        // 自分の駒が想定位置にいるか（自分に見える範囲の妥当性チェック）
        let playable = match parse_usi(usi) {
            Some(ShogiMove::Board { from, to, .. }) => {
                let from_ok = view
                    .your_pieces
                    .iter()
                    .any(|p| parse_usi_square(&p.square) == Some(from));
                let to_free = !view
                    .your_pieces
                    .iter()
                    .any(|p| parse_usi_square(&p.square) == Some(to));
                from_ok && to_free
            }
            _ => false, // 定跡ラインに打ちは想定しない
        };
        if !playable {
            self.exited = true;
            return None;
        }
        Some(usi.clone())
    }
}

// ---------------------------------------------------------------------------
// 局面価値の特徴量（value_features.rs のコピー）
// ---------------------------------------------------------------------------

pub const VALUE_FEATURES: usize = 16;

pub const VALUE_FEATURE_NAMES: [&str; VALUE_FEATURES] = [
    "material_diff",         // 自分の駒価値合計（盤上+持ち駒） − 相手の同値
    "my_hand_value",         // 自分の持ち駒価値合計
    "opp_hand_value",        // 相手の持ち駒価値合計
    "king_pressure_on_me",   // 自玉周囲8マスへの相手の利き数
    "king_pressure_on_opp",  // 相手玉周囲8マスへの自分の利き数
    "drop_check_danger_me",  // 自玉への打ち込み王手の受け入れ面積（相手持ち駒基準）
    "drop_check_danger_opp", // 相手玉への同（自分の持ち駒基準）
    "my_in_check",           // 自分が王手されている
    "opp_in_check",          // 相手が王手されている
    "my_pieces_in_opp_camp", // 敵陣3段にいる自分の駒数（歩・と金・玉除く）
    "opp_pieces_in_my_camp", // 自陣3段にいる相手の駒数（歩・と金・玉除く）
    "my_max_hanging",        // 相手の利きが当たり自分の紐が無い自分の駒の最大価値
    "opp_max_hanging",       // 同、相手側（=自分が取れる駒の最大価値）
    "my_max_exchange_loss",  // 相手に取られた場合の最悪交換損失（取り返しの補償を差し引いた後）
    "opp_max_exchange_loss", // 同、相手側（=自分が仕掛けられる最悪の交換損失）
    "ply_progress",          // 手数を100で割った進行度（局面フェーズの粗い指標）
];

fn camp_rank_range(owner: Color) -> std::ops::RangeInclusive<i8> {
    // owner の敵陣（盤の奥3段）
    match owner {
        Color::Sente => 1..=3,
        Color::Gote => 7..=9,
    }
}

fn board_value(pos: &Position, color: Color) -> f64 {
    pos.pieces()
        .filter(|(_, p)| p.color == color)
        .map(|(_, p)| piece_value(p.role))
        .sum()
}

fn hand_value(pos: &Position, color: Color) -> f64 {
    pos.hand_map(color)
        .iter()
        .map(|(r, n)| piece_value(*r) * f64::from(*n))
        .sum()
}

fn material_sum(pos: &Position, color: Color) -> f64 {
    board_value(pos, color) + hand_value(pos, color)
}

/// `color` の駒（歩・と金・玉除く）のうち、`color` から見た敵陣（盤の奥3段）に
/// いる枚数。攻め込みの深さ（自分が攻めているなら my_pieces、相手が攻めて
/// いるなら opp_pieces として呼ぶ）
fn pieces_in_enemy_camp(pos: &Position, color: Color) -> f64 {
    let range = camp_rank_range(color);
    pos.pieces()
        .filter(|(sq, p)| {
            p.color == color
                && !matches!(p.role, Role::Pawn | Role::Tokin | Role::King)
                && range.contains(&sq.rank)
        })
        .count() as f64
}

/// `color` の駒（玉除く）のうち、相手の利きが当たっていて自分の紐が無い
/// （取り返せない）駒の最大価値。33手目5八四金（scenarios/gold-check.kif）の
/// ような「利きが確定している駒への無防備な接近」を捉えるための特徴量
/// （元々の12特徴量にはこれが無く、まさに動機となった局面を判別できなかった）
fn max_hanging_value(pos: &Position, color: Color) -> f64 {
    let opp = color.other();
    pos.pieces()
        .filter(|(sq, p)| {
            p.color == color
                && p.role != Role::King
                && pos.is_attacked(*sq, opp)
                && !pos.is_attacked(*sq, color)
        })
        .map(|(_, p)| piece_value(p.role))
        .fold(0.0, f64::max)
}

/// マス sq を攻撃している `by` 側の駒のうち、最も安い exchange_value（取り返す/
/// 取られる際に実際に使われるはずの駒。攻撃側は損を最小化するため最安の駒で
/// 取る）。1枚も無ければ None
///
/// 近似: `attacks()`（利きの有無）だけを見ており、ピンで動けない駒や
/// 取ると自玉が王手になる駒も攻撃駒に数える（既存の`max_hanging_value`と
/// 同じ近似方針）。厳密な合法性チェックは局面ごとに指し手を構築する必要があり
/// コストが高いため、学習データの特徴量としては許容範囲としている
/// （codexレビュー指摘、2026-07-20。pairwiseの教師信号としてのノイズ源になる
/// 可能性は残る）
fn min_attacker_exchange_value(pos: &Position, sq: crate::board::Coord, by: Color) -> Option<f64> {
    pos.pieces()
        .filter(|(from, p)| p.color == by && pos.attacks(*from, sq))
        .map(|(_, p)| exchange_value(p.role))
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| a.min(v)))
        })
}

/// `color` の駒（歩・と金・玉除く。歩は打ち歩詰め等の特殊性が強く exchange_value の
/// 前提が崩れやすいため除外）のうち、相手に取られた場合の最悪の交換損失
/// （取り返せるなら相手の攻め駒の exchange_value を補償として差し引く）。
/// kakudo局面（scenarios/kakudo.kif、R*2d vs P*2h）のような「取られる駒の
/// 価値の高さ」を、single hangingでは表現できない紐つき交換でも捉えるための特徴量
/// （2026-07-20、codexレビュー指摘: max_hanging_valueは紐なしの即取りしか
/// 表せず、飛車を切って角を得る/歩を切って角を得るの損得差を区別できない）
fn max_exchange_loss(pos: &Position, color: Color) -> f64 {
    let opp = color.other();
    pos.pieces()
        .filter(|(_, p)| p.color == color && !matches!(p.role, Role::King | Role::Pawn))
        .filter_map(|(sq, p)| {
            // 相手は損を最小化するため最安の攻め駒で取ってくる想定
            let attacker = min_attacker_exchange_value(pos, sq, opp)?;
            let loss = exchange_value(p.role);
            // 取り返せる（sq を自分の他の駒も攻撃している）なら、取り返して
            // 得る相手の攻め駒の価値ぶんを補償として差し引く
            let can_recapture = min_attacker_exchange_value(pos, sq, color).is_some();
            let comp = if can_recapture { attacker } else { 0.0 };
            Some((loss - comp).max(0.0))
        })
        .fold(0.0, f64::max)
}

/// 局面特徴量。`me` は評価する側（手番側とは限らない。学習データ書き出し側で
/// 手番ごとに `me` を指定して両方の視点を作れる）
pub fn value_features(pos: &Position, me: Color) -> [f64; VALUE_FEATURES] {
    let opp = me.other();
    [
        material_sum(pos, me) - material_sum(pos, opp),
        hand_value(pos, me),
        hand_value(pos, opp),
        king_zone_pressure(pos, me, opp),
        king_zone_pressure(pos, opp, me),
        drop_check_danger(pos, me),
        drop_check_danger(pos, opp),
        f64::from(pos.in_check(me)),
        f64::from(pos.in_check(opp)),
        pieces_in_enemy_camp(pos, me),
        pieces_in_enemy_camp(pos, opp),
        max_hanging_value(pos, me),
        max_hanging_value(pos, opp),
        max_exchange_loss(pos, me),
        max_exchange_loss(pos, opp),
        f64::from(pos.move_number()) / 100.0,
    ]
}

pub const TRANSITION_FEATURES: usize = 6;

pub const TRANSITION_FEATURE_NAMES: [&str; TRANSITION_FEATURES] = [
    "moved_piece_value",          // 直前に着手された駒（動いた/打たれた駒）の価値
    "moved_piece_hanging_value",  // 同、紐なしで即取られる状態なら価値、そうでなければ0
    "moved_piece_exchange_loss", // 同、紐つきでも駒種の交換で損する額（取り返しの補償を差し引いた後）
    "captured_value",            // その着手で取った相手駒の価値（打つ手・非取りなら0）
    "net_capture_then_recapture", // captured_value − moved_piece_exchange_loss（この一手の実質損得）
    "gives_check",                // その着手で相手に王手をかけたか
];

/// 直前の着手（`mv`）固有の特徴量。`max_hanging_value`/`max_exchange_loss`は
/// 盤面全体でのworst-caseを返すため、無関係などこか別の駒のリスクが大きいと
/// その着手自体が生むリスクの差がmaxに埋もれて消える（kakudo局面 R*2d vs P*2h
/// で実際に発生・codexレビューで指摘、2026-07-20）。この関数は着手で動いた/
/// 打たれた駒**だけ**に絞ることでその埋没を避ける。`mover` は着手した側
pub fn transition_features(
    before: &Position,
    mv: &ShogiMove,
    after: &Position,
    mover: Color,
) -> [f64; TRANSITION_FEATURES] {
    let opp = mover.other();
    let to = match *mv {
        ShogiMove::Board { to, .. } => to,
        ShogiMove::Drop { to, .. } => to,
    };
    let moved_role = after
        .piece_at(to)
        .expect("着手直後は to に自駒があるはず")
        .role;
    let moved_value = piece_value(moved_role);

    let hanging = if after.is_attacked(to, opp) && !after.is_attacked(to, mover) {
        moved_value
    } else {
        0.0
    };

    let exchange_loss = min_attacker_exchange_value(after, to, opp).map_or(0.0, |attacker| {
        let loss = exchange_value(moved_role);
        let can_recapture = min_attacker_exchange_value(after, to, mover).is_some();
        let comp = if can_recapture { attacker } else { 0.0 };
        (loss - comp).max(0.0)
    });

    // exchange_value に揃える（captured_value - exchange_loss = net の両辺が
    // 同じ「持ち駒化後の実質価値」基準でないと差し引きの意味がズレる。
    // codexレビュー指摘、2026-07-20: ここだけpiece_valueのままだと、と金等
    // 成駒を取った際の得を過大評価し、net_capture_then_recaptureが歪む）
    let captured_value = match *mv {
        ShogiMove::Board { to, .. } => before.piece_at(to).map_or(0.0, |p| exchange_value(p.role)),
        ShogiMove::Drop { .. } => 0.0,
    };

    [
        moved_value,
        hanging,
        exchange_loss,
        captured_value,
        captured_value - exchange_loss,
        f64::from(after.in_check(opp)),
    ]
}

// ---------------------------------------------------------------------------
// value NN（value_nn.rs のコピー、新定石1536局 seed2 で再学習）
// ---------------------------------------------------------------------------

// 粒子上のvalueネット（NN段階③、フェーズ2統合）。学習はtsuitate-nnの
// train.py（勝敗回帰 + pairwise補助loss）、重み配列はexport_value_weights.pyで
// 生成、forward passはここに手書き（evaluate()の粒子ループから
// 候補×粒子のオーダーで呼ばれるホットパスのため、ONNX/推論クレートは使わない）。

// AUTO-GENERATED BEGIN (export_value_weights.py)
// 学習メタ: 新1512局（v8/v9/v10教師・my_foul世代）pairwise w=20 m=0.1 seed2、gold-check/kakudo両正解
// 再生成: tsuitate-nn/export_value_weights.py --model-dir out/ --out ../tsuitate-bot/src/value_nn.rs
pub const VALUE_NN_MEAN: [f64; 22] = [
    -1.63345611e+00,
    9.35212994e+00,
    9.56934166e+00,
    1.22547507e+00,
    9.47085202e-01,
    2.34852123e+00,
    2.36038876e+00,
    1.10207133e-01,
    0.00000000e+00,
    5.68652570e-01,
    6.63207352e-01,
    3.03020501e+00,
    3.08056808e+00,
    3.03737712e+00,
    3.06964016e+00,
    4.06791687e-01,
    4.85278654e+00,
    1.41801727e+00,
    1.51651990e+00,
    1.51441383e+00,
    -2.10602186e-03,
    1.20798633e-01,
];
pub const VALUE_NN_STD: [f64; 22] = [
    2.47431564e+01,
    1.06621513e+01,
    1.07356567e+01,
    1.62186837e+00,
    1.38975203e+00,
    3.32845902e+00,
    3.36344934e+00,
    3.13116640e-01,
    1.00000000e+00,
    9.00985539e-01,
    9.34475720e-01,
    3.45852613e+00,
    3.53573751e+00,
    3.37159991e+00,
    3.42780900e+00,
    2.70778894e-01,
    3.47164917e+00,
    2.91156840e+00,
    2.78290653e+00,
    2.73320365e+00,
    3.81144977e+00,
    3.25968981e-01,
];
pub const VALUE_NN_W1: [[f64; 22]; 64] = [
    [
        3.30794752e-01,
        -1.17048256e-01,
        1.48148403e-01,
        -4.50666770e-02,
        1.14774793e-01,
        -9.15898010e-02,
        3.85619141e-02,
        -4.94516253e-01,
        2.85007152e-39,
        -6.14257455e-02,
        5.34150340e-02,
        -4.74167019e-02,
        1.58869281e-01,
        3.52872312e-02,
        -1.73772741e-02,
        1.42754391e-01,
        -1.07256593e-02,
        -7.21680894e-02,
        -5.44915125e-02,
        1.49211809e-01,
        -1.84651986e-01,
        5.17519796e-03,
    ],
    [
        -4.06070706e-03,
        -2.17663907e-02,
        -5.13783609e-03,
        -3.74774472e-03,
        -1.07121924e-02,
        1.31066963e-02,
        3.49287726e-02,
        2.35445285e-03,
        2.62622810e-39,
        -1.55229680e-02,
        2.01113168e-02,
        3.31247598e-02,
        -9.85066965e-02,
        -4.04681861e-02,
        2.21725479e-02,
        -4.46586646e-02,
        2.24438142e-02,
        -2.59570718e-01,
        -3.21911424e-01,
        2.88719326e-01,
        7.61605382e-01,
        -8.67652074e-02,
    ],
    [
        9.42098200e-02,
        -8.89915973e-02,
        1.45500377e-01,
        4.06506099e-03,
        9.73220766e-02,
        1.09379319e-02,
        1.46286178e-03,
        -8.92822444e-02,
        4.23784185e-39,
        -2.38246173e-02,
        -5.18758893e-02,
        -7.73780271e-02,
        -8.02416131e-02,
        1.16000012e-01,
        -3.74984033e-02,
        1.98896855e-01,
        1.94807742e-02,
        -1.03321306e-01,
        -1.05347641e-01,
        6.72577471e-02,
        1.83463901e-01,
        1.94039524e-01,
    ],
    [
        -6.09096736e-02,
        -1.12441599e-01,
        -4.62665446e-02,
        -2.91321147e-02,
        -4.69862893e-02,
        8.35622777e-04,
        -9.02034491e-02,
        1.73723385e-01,
        1.63965149e-37,
        1.84916388e-02,
        2.55837440e-02,
        -3.09733562e-02,
        -1.08793013e-01,
        -7.82969519e-02,
        1.68694165e-02,
        -2.28835158e-02,
        8.11910629e-03,
        7.01997131e-02,
        1.57483265e-01,
        -3.53392005e-01,
        -2.31770173e-01,
        -1.16764894e-02,
    ],
    [
        -6.14332110e-02,
        -1.08721353e-01,
        -4.04897109e-02,
        -2.15376690e-02,
        -1.09981820e-01,
        4.65145707e-03,
        -6.57646880e-02,
        4.63298261e-02,
        -1.69194458e-39,
        -5.62808365e-02,
        -3.28920558e-02,
        -5.64246364e-02,
        4.10204101e-03,
        -2.53553316e-03,
        -1.22767597e-01,
        4.03246805e-02,
        2.72641145e-03,
        8.60286802e-02,
        1.48328096e-01,
        -2.32582211e-01,
        -3.12020808e-01,
        5.87624451e-03,
    ],
    [
        -5.52149229e-02,
        -4.22147755e-03,
        -7.17498921e-03,
        -5.91213070e-02,
        -4.02116030e-02,
        6.12593815e-03,
        -6.41276911e-02,
        1.52405664e-01,
        6.59797626e-38,
        5.29111922e-02,
        5.72768562e-02,
        1.81321539e-02,
        -1.42058536e-01,
        2.01486275e-02,
        6.24896660e-02,
        -1.29102111e-01,
        -4.15831581e-02,
        4.08301502e-03,
        -1.25244454e-01,
        -3.81061077e-01,
        -2.51229465e-01,
        -1.07345181e-02,
    ],
    [
        -1.55854315e-01,
        -5.36464043e-02,
        -6.49416773e-03,
        1.34884492e-01,
        -1.40302122e-01,
        2.58825608e-02,
        -1.91465523e-02,
        9.83247086e-02,
        -2.51158928e-40,
        2.45798957e-02,
        -2.12751441e-02,
        -2.08828002e-02,
        -1.68999992e-02,
        2.67707314e-02,
        5.43629751e-03,
        -1.31122068e-01,
        8.19797367e-02,
        -1.37178853e-01,
        2.00193170e-02,
        -2.23184284e-02,
        -1.15752406e-01,
        -7.79122263e-02,
    ],
    [
        -7.70934373e-02,
        6.72871172e-02,
        -1.03979252e-01,
        7.39492252e-02,
        2.70691584e-03,
        6.32928014e-02,
        2.78943172e-03,
        2.41070136e-01,
        5.34659403e-39,
        -7.24865869e-02,
        2.85606366e-02,
        7.65516609e-02,
        -1.88557327e-01,
        -2.31366530e-02,
        3.68594821e-03,
        4.50852439e-02,
        -4.41277996e-02,
        -2.32949406e-02,
        7.74439722e-02,
        -6.68923184e-03,
        -1.12429686e-01,
        -5.36892377e-02,
    ],
    [
        -3.53611529e-01,
        1.49344921e-01,
        1.05441354e-01,
        1.04901515e-01,
        -3.63085493e-02,
        3.47535498e-02,
        -3.72089855e-02,
        -2.36883666e-02,
        1.14012446e-39,
        6.00248165e-02,
        1.54301990e-02,
        -3.08524296e-02,
        -9.09200758e-02,
        1.52090296e-01,
        1.42933875e-01,
        -4.32220370e-01,
        7.33474270e-02,
        -3.78004760e-02,
        -3.64631438e-03,
        -5.35292551e-02,
        2.05939841e-02,
        -3.33732292e-02,
    ],
    [
        2.91753650e-01,
        1.26031429e-01,
        1.02283470e-01,
        -1.29870445e-01,
        4.04722728e-02,
        -9.65941772e-02,
        8.51211138e-04,
        -3.31139416e-02,
        3.97776366e-39,
        7.39718676e-02,
        1.35923982e-01,
        -8.08900073e-02,
        1.97134446e-03,
        2.90889964e-02,
        1.19007379e-01,
        -3.90467614e-01,
        4.99252044e-02,
        5.84867932e-02,
        -1.42350690e-02,
        -4.01187427e-02,
        6.47998378e-02,
        2.30003800e-02,
    ],
    [
        -6.25903457e-02,
        -4.19996455e-02,
        6.40135854e-02,
        7.10001215e-02,
        -7.11869597e-02,
        1.38404872e-02,
        -7.60007501e-02,
        2.39575744e-01,
        2.26621351e-39,
        -1.47498459e-01,
        -7.21171722e-02,
        2.38644667e-02,
        7.57977273e-03,
        -1.25237713e-02,
        1.52934846e-02,
        -6.96432367e-02,
        -4.82298285e-02,
        -1.27572287e-02,
        5.89479096e-02,
        -1.41993597e-01,
        -2.13013634e-01,
        7.65611418e-03,
    ],
    [
        -9.71943662e-02,
        8.63825679e-02,
        2.11816952e-02,
        -2.43252385e-02,
        -1.87096354e-02,
        7.77536184e-02,
        -1.01684943e-01,
        6.70577586e-02,
        5.97433091e-39,
        -3.11879534e-02,
        9.63797122e-02,
        -8.84234309e-02,
        -1.19386993e-01,
        -2.12650970e-02,
        -7.17830509e-02,
        -1.36939494e-03,
        1.44412428e-01,
        1.11943342e-01,
        3.11438628e-02,
        -1.54899418e-01,
        -7.98815265e-02,
        5.85697703e-02,
    ],
    [
        -3.38391401e-02,
        -1.23703361e-01,
        3.08251916e-03,
        -5.90493996e-03,
        9.33107957e-02,
        3.04464786e-03,
        4.68261018e-02,
        -1.32713129e-03,
        -3.20960407e-40,
        2.71797571e-02,
        -1.23450980e-02,
        1.30998604e-02,
        -2.01887667e-01,
        1.88275576e-02,
        -1.96247086e-01,
        1.02627508e-01,
        -4.38368954e-02,
        -1.37356110e-02,
        -2.47642547e-01,
        9.09056067e-02,
        3.41450483e-01,
        -1.29574174e-02,
    ],
    [
        1.89870410e-02,
        -7.10524851e-04,
        3.38304825e-02,
        -1.89236235e-02,
        -3.77558242e-03,
        1.54523142e-02,
        -2.02474110e-02,
        1.42284343e-02,
        1.72741285e-39,
        -1.23037882e-02,
        4.55556586e-02,
        4.16553160e-03,
        5.03437687e-03,
        1.91471016e-03,
        1.88799780e-02,
        -1.51761668e-03,
        1.94435734e-02,
        1.56361435e-03,
        -3.59796435e-01,
        4.37259644e-01,
        7.63500512e-01,
        -1.98176689e-02,
    ],
    [
        -1.63604598e-02,
        -3.94819528e-02,
        -5.62367309e-03,
        -1.28084263e-02,
        2.48810370e-03,
        2.15823739e-03,
        -5.80121316e-02,
        4.46537659e-02,
        1.07181116e-40,
        -3.12478859e-02,
        2.23515965e-02,
        3.63121033e-02,
        -1.11749135e-01,
        -4.74185497e-02,
        4.79951948e-02,
        -1.73604954e-02,
        -2.78551057e-02,
        -2.80425102e-01,
        -3.31711978e-01,
        1.45420343e-01,
        3.86370182e-01,
        3.20075490e-02,
    ],
    [
        -3.43284488e-01,
        1.61163539e-01,
        -2.89062760e-03,
        2.29525734e-02,
        -1.55201375e-01,
        -1.56075628e-02,
        3.68883908e-02,
        -6.60778210e-02,
        -1.68359845e-39,
        -4.62863445e-02,
        4.04689945e-02,
        -1.23495296e-01,
        -1.49559140e-01,
        8.90970081e-02,
        1.85222059e-01,
        -2.20690116e-01,
        4.47419323e-02,
        -2.52030715e-02,
        -3.01374570e-02,
        1.66025177e-01,
        -8.09359327e-02,
        -3.38574350e-02,
    ],
    [
        -5.96827678e-02,
        3.64410621e-03,
        4.25221678e-03,
        -1.12628134e-03,
        -7.22538540e-03,
        -5.21044433e-03,
        4.23487928e-03,
        -4.80801146e-03,
        -6.90819404e-39,
        -1.84485335e-02,
        1.36459935e-02,
        -4.43058051e-02,
        -6.74521253e-02,
        4.03660424e-02,
        3.92517298e-02,
        -3.50712575e-02,
        -1.80543698e-02,
        -1.82802573e-01,
        -3.66163790e-01,
        2.92884529e-01,
        6.08404458e-01,
        -5.71979135e-02,
    ],
    [
        -5.17335646e-02,
        2.54008919e-02,
        -1.68558005e-02,
        -2.71803443e-03,
        1.10621909e-02,
        3.35195884e-02,
        -9.86422785e-03,
        2.38767569e-03,
        -2.03960953e-39,
        1.23353060e-02,
        -2.47206166e-03,
        1.40664289e-02,
        4.54249280e-03,
        -3.34144346e-02,
        -7.96618126e-03,
        -3.31827812e-02,
        1.40878940e-02,
        4.58809882e-02,
        -4.37321782e-01,
        4.16015565e-01,
        7.63232708e-01,
        -7.87869766e-02,
    ],
    [
        2.30391994e-01,
        3.70211042e-02,
        1.48465425e-01,
        -1.54523015e-01,
        3.99353635e-03,
        -1.75726749e-02,
        -1.83778058e-03,
        1.48812920e-01,
        -2.10225598e-39,
        9.08297896e-02,
        8.39442760e-03,
        -7.11520687e-02,
        -2.50990950e-02,
        4.47575643e-04,
        -1.38371542e-01,
        -1.42985642e-01,
        9.20144767e-02,
        6.14757240e-02,
        -1.17564857e-01,
        1.28085511e-02,
        1.41269878e-01,
        3.72700691e-02,
    ],
    [
        -5.91976866e-02,
        4.33952287e-02,
        -1.43770315e-02,
        9.45801195e-03,
        1.81726590e-02,
        -3.76001000e-02,
        -2.44460884e-04,
        -2.73023709e-03,
        -3.12431684e-39,
        3.62764224e-02,
        6.21772185e-03,
        -5.31966565e-03,
        1.00478968e-02,
        -2.92806439e-02,
        2.19585635e-02,
        -2.83964653e-03,
        -3.96188796e-02,
        8.06315616e-02,
        3.61472070e-02,
        -4.98309553e-01,
        -2.87273407e-01,
        -2.26450518e-01,
    ],
    [
        -2.98837759e-02,
        -1.45841511e-02,
        -1.41566796e-02,
        -1.07334899e-02,
        -7.13069597e-03,
        1.00597488e-02,
        3.49087119e-02,
        3.64799090e-02,
        5.53983730e-39,
        -2.89849006e-03,
        2.15170830e-02,
        6.35897415e-03,
        -2.92571690e-02,
        -5.86813688e-03,
        1.77788846e-02,
        -3.66436541e-02,
        -2.26258617e-02,
        -7.89231136e-02,
        -4.09494579e-01,
        3.64255339e-01,
        5.06963789e-01,
        -7.38064572e-02,
    ],
    [
        -8.31049234e-02,
        -7.51457587e-02,
        1.04310893e-01,
        5.30951209e-02,
        -3.81349362e-02,
        -1.66233592e-02,
        -1.32854551e-01,
        1.75533384e-01,
        -5.16424447e-39,
        2.65884213e-02,
        -8.37881863e-03,
        -2.75444426e-02,
        -8.97751749e-02,
        -6.83399737e-02,
        -1.19499192e-01,
        -2.06921339e-01,
        -2.66889781e-02,
        2.00698171e-02,
        -1.38789881e-03,
        -3.36334435e-03,
        -1.00667262e-02,
        3.69452313e-02,
    ],
    [
        -2.55283624e-01,
        1.65379837e-01,
        1.73878316e-02,
        2.16681119e-02,
        -7.85837173e-02,
        -2.11921278e-02,
        -5.36259555e-04,
        1.46298617e-01,
        -2.85202493e-39,
        -7.09351059e-03,
        7.46823801e-03,
        -4.40935865e-02,
        -3.43387909e-02,
        -6.03724830e-02,
        3.91125008e-02,
        -2.04121023e-01,
        -9.61760874e-04,
        3.50357592e-02,
        -1.31770805e-03,
        -2.03144118e-01,
        -8.41390714e-02,
        5.14482334e-02,
    ],
    [
        -6.49417341e-02,
        1.12235136e-02,
        -2.80003875e-01,
        1.18039370e-01,
        -2.76003666e-02,
        -6.48929849e-02,
        4.55272272e-02,
        -4.25289907e-02,
        2.09533917e-39,
        -6.66651279e-02,
        2.22263932e-02,
        2.62649469e-02,
        4.13992107e-02,
        2.36557536e-02,
        -2.31820270e-02,
        3.64794284e-01,
        -5.60468994e-02,
        -9.05907527e-02,
        2.42532473e-02,
        3.76172811e-02,
        -1.95113242e-01,
        -1.48540258e-01,
    ],
    [
        1.95391595e-01,
        -1.72465771e-01,
        9.44094211e-02,
        6.07303008e-02,
        9.38493684e-02,
        -3.48338559e-02,
        3.79646309e-02,
        -3.43790948e-02,
        -6.13204845e-39,
        1.07626189e-02,
        3.76379527e-02,
        -1.31632704e-02,
        -1.82929635e-01,
        -4.71376255e-03,
        -3.08981657e-01,
        7.73110166e-02,
        -4.08034883e-02,
        5.77459671e-03,
        -2.27100030e-01,
        1.11715645e-01,
        3.37757796e-01,
        2.31654756e-02,
    ],
    [
        -1.36990666e-01,
        2.46516280e-02,
        4.03819643e-02,
        6.00192919e-02,
        -2.35307187e-01,
        -3.73771563e-02,
        -1.42807826e-01,
        1.47737644e-03,
        -6.58910044e-38,
        1.20001741e-01,
        4.10183668e-02,
        -1.50183849e-02,
        -3.77163179e-02,
        -1.64875165e-01,
        2.30491757e-02,
        -2.33866721e-01,
        1.95728540e-02,
        -3.32039706e-02,
        -9.49942768e-02,
        1.68162286e-02,
        -1.36651382e-01,
        1.08355964e-02,
    ],
    [
        -9.03216600e-02,
        4.47557569e-02,
        6.68203384e-02,
        3.71210277e-02,
        -2.01229513e-01,
        -5.51042147e-03,
        -1.96840703e-01,
        2.14988515e-02,
        -1.72086038e-39,
        9.87335667e-02,
        -2.72735879e-02,
        4.94009778e-02,
        -4.34226505e-02,
        -6.45866692e-02,
        5.73494695e-02,
        -2.10180148e-01,
        -1.31687209e-01,
        -5.36740497e-02,
        7.99952745e-02,
        -1.26056559e-03,
        -1.67273298e-01,
        -1.70868225e-02,
    ],
    [
        1.10737316e-01,
        -5.09198010e-02,
        1.42319515e-01,
        -7.41136000e-02,
        2.46809609e-02,
        4.21901979e-02,
        1.67606436e-02,
        2.24245355e-01,
        3.13702101e-39,
        -5.91643155e-03,
        -2.29192711e-02,
        2.06112154e-02,
        -1.36784196e-01,
        -5.15821464e-02,
        -4.78026867e-02,
        8.19182396e-02,
        1.71076447e-01,
        -5.91241680e-02,
        2.69269268e-03,
        1.52005449e-01,
        5.38189374e-02,
        -7.82139450e-02,
    ],
    [
        -1.39152512e-01,
        1.00506715e-01,
        -1.61692679e-01,
        1.49905875e-01,
        2.98377275e-02,
        -6.39106706e-02,
        2.99816895e-02,
        5.95186241e-02,
        1.84831127e-39,
        -1.50512084e-02,
        1.42423920e-02,
        -1.18059665e-02,
        -2.47671027e-02,
        1.98063981e-02,
        -3.62687223e-02,
        1.29520893e-01,
        -1.90708842e-02,
        -6.32316917e-02,
        9.33033228e-02,
        -1.99100286e-01,
        -8.72668028e-02,
        -1.20957240e-01,
    ],
    [
        -1.12039588e-01,
        7.62086958e-02,
        -1.95252150e-01,
        1.66310638e-01,
        2.61864904e-02,
        4.52644899e-02,
        4.97043356e-02,
        4.39511649e-02,
        -2.75765729e-39,
        8.77410267e-03,
        5.90701541e-03,
        -6.53729681e-03,
        -9.27057639e-02,
        2.61907130e-02,
        6.61699772e-02,
        9.14374590e-02,
        -1.35470182e-01,
        -1.34786619e-02,
        1.14089414e-01,
        4.65693548e-02,
        -8.40996653e-02,
        -7.86075443e-02,
    ],
    [
        -2.80714706e-02,
        9.36765447e-02,
        -1.55608162e-01,
        1.20872498e-01,
        -5.29779755e-02,
        4.33689654e-02,
        -6.12426996e-02,
        7.13272719e-03,
        2.10320606e-39,
        1.62291015e-03,
        -1.25361811e-02,
        1.50226743e-03,
        -5.60763925e-02,
        -4.92806267e-03,
        4.22109663e-02,
        1.57492068e-02,
        -1.99439257e-01,
        6.16797730e-02,
        6.45060688e-02,
        -1.30618528e-01,
        4.34879214e-02,
        2.20394116e-02,
    ],
    [
        -1.25327781e-01,
        7.10898489e-02,
        -4.13128622e-02,
        5.69555834e-02,
        -5.63466027e-02,
        -3.86689864e-02,
        -1.31479660e-02,
        1.20035730e-01,
        2.03172863e-39,
        1.59467794e-02,
        4.90365457e-03,
        1.10466070e-01,
        -1.06729351e-01,
        -6.34402111e-02,
        5.83615191e-02,
        4.62399349e-02,
        -6.11236058e-02,
        -1.37988580e-02,
        1.44046068e-01,
        -9.30353627e-02,
        -1.40803441e-01,
        -6.04100861e-02,
    ],
    [
        -4.13679242e-01,
        1.52421787e-01,
        -2.00901449e-01,
        8.76108781e-02,
        -1.38246849e-01,
        9.65809599e-02,
        -1.71739683e-02,
        5.97973131e-02,
        -2.27883640e-39,
        9.29209813e-02,
        2.98844464e-02,
        -3.00251003e-02,
        -2.05518171e-01,
        1.11101002e-01,
        1.64559081e-01,
        -1.66697055e-01,
        1.09325655e-01,
        -5.54264411e-02,
        -1.19800409e-02,
        1.00767180e-01,
        7.91961502e-04,
        1.77500527e-02,
    ],
    [
        -5.78590855e-02,
        -4.90657128e-02,
        -9.80023202e-03,
        -9.79574546e-02,
        -5.21640666e-02,
        1.96313094e-02,
        9.12667066e-03,
        2.68017530e-01,
        -1.43676533e-40,
        4.41945978e-02,
        6.60997406e-02,
        1.75708793e-02,
        -1.83097705e-01,
        7.48160854e-03,
        9.54801366e-02,
        -5.35167642e-02,
        -1.42817674e-02,
        3.00448462e-02,
        4.62859645e-02,
        -7.12975189e-02,
        -3.87819149e-02,
        3.13756079e-03,
    ],
    [
        -1.42484412e-01,
        1.39026165e-01,
        -4.39945795e-02,
        6.39194846e-02,
        -3.39293927e-01,
        6.86215833e-02,
        -2.57662445e-01,
        -2.57945098e-02,
        -4.68393260e-39,
        -1.91698316e-02,
        -3.39641422e-02,
        -6.83616549e-02,
        -4.60270531e-02,
        1.03338286e-01,
        1.07827395e-01,
        -3.62982638e-02,
        8.58999491e-02,
        -9.50284377e-02,
        1.22060634e-01,
        -4.22531813e-02,
        6.37762994e-02,
        -5.75111900e-03,
    ],
    [
        -6.57699555e-02,
        1.01184137e-01,
        -5.98524809e-02,
        5.39906546e-02,
        6.20341189e-02,
        3.24588344e-02,
        -9.34686437e-02,
        2.50744641e-01,
        -3.71524160e-39,
        -2.25828011e-02,
        1.56157569e-03,
        -1.54807307e-02,
        -8.20936859e-02,
        -2.16326453e-02,
        6.39170483e-02,
        -3.98503616e-02,
        1.87232926e-01,
        -2.59745102e-02,
        7.54038198e-03,
        8.07202607e-03,
        -6.53969496e-02,
        3.17489691e-02,
    ],
    [
        -8.33575875e-02,
        1.65183675e-02,
        -7.29889572e-02,
        2.74618939e-02,
        -1.76874995e-02,
        1.72383301e-02,
        -6.64160773e-02,
        2.43595261e-02,
        6.19893803e-40,
        1.02832049e-01,
        -1.39301014e-03,
        -9.36662033e-02,
        -4.26582098e-02,
        -1.18329721e-02,
        3.10429428e-02,
        3.63650210e-02,
        9.54556372e-03,
        3.47520523e-02,
        1.85451731e-01,
        -3.87959898e-01,
        -2.67649919e-01,
        -2.14792356e-01,
    ],
    [
        -6.04065955e-02,
        -3.62993851e-02,
        -6.01834208e-02,
        -2.31959913e-02,
        2.77885105e-02,
        9.73237620e-05,
        -1.03197200e-02,
        3.04789133e-02,
        -3.06155408e-39,
        5.74299544e-02,
        -3.07182278e-02,
        7.17085204e-04,
        -1.78625702e-03,
        3.24307196e-03,
        -4.91269194e-02,
        2.81823669e-02,
        1.02019068e-02,
        1.82200279e-02,
        -3.07212323e-01,
        3.05517524e-01,
        5.77706695e-01,
        -6.98973164e-02,
    ],
    [
        -2.20311329e-01,
        9.60770324e-02,
        -1.41333356e-01,
        6.21046536e-02,
        1.03358589e-02,
        8.57453570e-02,
        1.02406610e-02,
        9.95384976e-02,
        -2.20360209e-39,
        5.76981790e-02,
        -4.48492356e-03,
        -2.38380246e-02,
        3.57929766e-02,
        8.35240930e-02,
        -8.71231183e-02,
        -1.60910264e-01,
        -3.24386135e-02,
        -2.71366048e-03,
        -8.81720111e-02,
        -3.84823620e-01,
        -1.84292763e-01,
        -1.45643756e-01,
    ],
    [
        2.05761448e-01,
        8.13520849e-02,
        9.80459377e-02,
        -8.13043341e-02,
        1.44771129e-01,
        -8.07369710e-04,
        1.69490818e-02,
        -2.71633953e-01,
        1.68666028e-39,
        3.40209194e-02,
        8.22926313e-02,
        -9.99763831e-02,
        -1.58593014e-01,
        2.31332593e-02,
        -1.78084120e-01,
        -1.05191484e-01,
        -4.87575959e-03,
        3.38055864e-02,
        -1.89275682e-01,
        1.54483035e-01,
        3.86662669e-02,
        3.67642976e-02,
    ],
    [
        4.42719549e-01,
        -9.24835280e-02,
        1.98733404e-01,
        -1.93164553e-02,
        7.12199882e-02,
        -7.31793121e-02,
        3.47127840e-02,
        -4.44202155e-01,
        -3.63877274e-39,
        -3.26464735e-02,
        1.40674040e-01,
        -6.38407022e-02,
        9.23975930e-02,
        -1.34073501e-03,
        1.74446218e-02,
        -9.95365754e-02,
        -1.01795597e-02,
        -6.58790395e-02,
        1.10421993e-01,
        -1.13060642e-02,
        4.59254012e-02,
        -1.17848804e-02,
    ],
    [
        2.71314263e-01,
        -2.29657069e-02,
        6.52314350e-02,
        -2.46294767e-01,
        4.06294428e-02,
        -1.03208877e-01,
        3.66505086e-02,
        1.99474439e-01,
        9.72358202e-40,
        3.81156728e-02,
        1.12374060e-01,
        -8.91744569e-02,
        -1.00984275e-02,
        2.76821014e-02,
        -1.11360863e-01,
        -1.74989343e-01,
        6.01424426e-02,
        2.81972103e-02,
        -1.22775398e-02,
        1.82403624e-02,
        1.46115959e-01,
        -6.98565990e-02,
    ],
    [
        -1.37929723e-01,
        8.07013139e-02,
        -2.91216820e-01,
        1.97552666e-01,
        -1.28908420e-03,
        -1.42136635e-02,
        -1.02052782e-02,
        7.46357292e-02,
        4.14989383e-38,
        -2.00014282e-02,
        -2.09125672e-02,
        7.75767898e-04,
        3.50631704e-03,
        -2.56450530e-02,
        -5.72135225e-02,
        2.49379113e-01,
        1.32507592e-01,
        -6.13318086e-02,
        -2.93373540e-02,
        -1.50552571e-01,
        -1.20949306e-01,
        -4.25793007e-02,
    ],
    [
        -1.80969074e-01,
        1.62960459e-02,
        2.43868325e-02,
        5.22576421e-02,
        -2.03103885e-01,
        -4.09282297e-02,
        -1.19999543e-01,
        8.77711549e-02,
        3.55072355e-39,
        1.86143368e-02,
        -7.07133347e-03,
        -2.10249443e-02,
        -9.14245960e-04,
        7.01083848e-03,
        6.57305866e-02,
        -1.07689314e-01,
        -3.22875269e-02,
        -1.16485402e-01,
        1.79431707e-01,
        -7.70998970e-02,
        -7.43672624e-02,
        -6.86395317e-02,
    ],
    [
        1.18649885e-01,
        -7.73583725e-02,
        6.84899390e-02,
        -2.99435109e-02,
        1.09579816e-01,
        -1.89874619e-02,
        -1.15420008e-02,
        -3.93491834e-01,
        -9.71696453e-38,
        5.58820646e-03,
        -3.33901681e-02,
        -3.91734242e-02,
        1.75965011e-01,
        3.28056365e-02,
        6.05550967e-02,
        1.56759113e-01,
        1.18956640e-01,
        6.88519627e-02,
        -1.09490789e-01,
        -2.10378366e-03,
        -7.48472214e-02,
        -8.49103183e-03,
    ],
    [
        1.69866234e-01,
        8.62410367e-02,
        9.78530720e-02,
        -1.73500523e-01,
        2.24457860e-01,
        -8.84906296e-03,
        3.42231952e-02,
        -2.24051833e-01,
        4.58079984e-39,
        -2.17448417e-02,
        -7.40606571e-04,
        -8.91610608e-02,
        -7.12653026e-02,
        7.23295053e-03,
        6.45419061e-02,
        -9.73858833e-02,
        2.16930639e-02,
        -1.23372614e-01,
        2.59373561e-02,
        1.65968120e-01,
        -3.76107218e-03,
        1.08975079e-02,
    ],
    [
        1.65726990e-01,
        -1.19037870e-02,
        1.05977818e-01,
        -2.33100265e-01,
        -1.86471101e-02,
        -1.75725445e-01,
        6.46888092e-02,
        2.05978125e-01,
        -4.26086658e-39,
        -1.74519494e-02,
        -6.14214875e-02,
        -1.81157403e-02,
        -7.00732395e-02,
        -1.99033935e-02,
        -1.02098286e-01,
        -2.53539570e-02,
        5.31681702e-02,
        8.03743675e-03,
        -5.68855330e-02,
        1.10457517e-01,
        1.63069889e-02,
        -3.00556086e-02,
    ],
    [
        -4.47729416e-02,
        9.67637524e-02,
        5.11681437e-02,
        2.40245461e-02,
        4.70922068e-02,
        -2.90201213e-02,
        9.79052018e-03,
        -8.40206891e-02,
        -5.43197935e-40,
        1.72977932e-02,
        -1.62152499e-01,
        -1.20490827e-02,
        -2.66497403e-01,
        4.66581294e-03,
        -7.42093772e-02,
        -1.71980653e-02,
        1.07182097e-03,
        -6.85511455e-02,
        -2.58625656e-01,
        1.11045092e-01,
        2.73346156e-01,
        -2.70208512e-02,
    ],
    [
        -4.87618744e-02,
        9.16447584e-03,
        -7.62365684e-02,
        2.43334775e-03,
        -5.76517964e-03,
        2.41682176e-02,
        3.31189549e-05,
        7.03810621e-03,
        -5.26800501e-39,
        2.13801619e-02,
        -3.50797810e-02,
        1.60383601e-02,
        6.62158430e-02,
        -8.88968818e-03,
        -6.15302771e-02,
        1.47448005e-02,
        -8.76892265e-03,
        3.04416269e-02,
        -2.98255831e-02,
        -5.34516573e-01,
        -3.31029683e-01,
        -1.84575424e-01,
    ],
    [
        1.53265268e-01,
        -2.88677990e-01,
        1.18241452e-01,
        1.99500956e-02,
        1.31168023e-01,
        1.69726256e-02,
        6.73465207e-02,
        -3.79985452e-01,
        -3.70812861e-39,
        -2.87054516e-02,
        8.90463358e-04,
        -9.56644416e-02,
        2.75387503e-02,
        4.77797352e-02,
        1.48413032e-02,
        2.44483069e-01,
        5.11193573e-02,
        -1.25699282e-01,
        1.01431936e-01,
        1.27540214e-03,
        -1.49257034e-02,
        6.58061057e-02,
    ],
    [
        -2.39692345e-01,
        5.53826094e-02,
        -2.25079209e-01,
        7.36998692e-02,
        -4.02072482e-02,
        -6.58006687e-03,
        -9.49755162e-02,
        1.24277323e-01,
        -4.44941409e-39,
        3.17572579e-02,
        -1.97562631e-02,
        -6.50796816e-02,
        -1.19345836e-01,
        4.56697419e-02,
        9.31362957e-02,
        -4.31461968e-02,
        -9.57283005e-03,
        1.68672521e-02,
        -1.42421797e-02,
        -1.58869162e-01,
        -1.03017271e-01,
        6.27237186e-02,
    ],
    [
        3.79890623e-03,
        -5.98974340e-02,
        2.04094741e-02,
        7.35910535e-02,
        -1.93299148e-02,
        -3.80874798e-02,
        -2.33522207e-02,
        9.70086362e-03,
        -2.39123596e-39,
        3.06704398e-02,
        4.97063482e-03,
        2.09335145e-02,
        -5.11423051e-02,
        -2.68207435e-02,
        4.24173020e-04,
        -6.83259359e-03,
        -2.00182833e-02,
        -1.80893809e-01,
        -4.15050477e-01,
        2.82234609e-01,
        3.97542417e-01,
        -6.02139235e-02,
    ],
    [
        -1.65635094e-01,
        5.02296425e-02,
        -2.14959145e-01,
        9.65455770e-02,
        -2.65525691e-02,
        4.28722724e-02,
        3.92626338e-02,
        1.82609081e-01,
        -6.21235126e-39,
        2.31704824e-02,
        -1.23320417e-02,
        -2.59203557e-02,
        -1.22093977e-02,
        3.86793423e-03,
        -1.09203989e-02,
        1.30201921e-01,
        1.37252659e-01,
        -1.30115703e-01,
        1.43901065e-01,
        7.20608886e-03,
        -1.03802778e-01,
        -4.94662449e-02,
    ],
    [
        -4.15983498e-01,
        2.12422326e-01,
        -4.36237492e-02,
        5.31193241e-02,
        -1.61387742e-01,
        4.31887172e-02,
        -5.21091148e-02,
        4.27226350e-02,
        -4.03530938e-39,
        1.82463408e-01,
        1.14882194e-01,
        -1.13191202e-01,
        -1.56669408e-01,
        5.33471331e-02,
        1.85861334e-01,
        -3.47018123e-01,
        9.09313709e-02,
        3.33668850e-03,
        1.46487504e-01,
        -5.42579889e-02,
        1.09135225e-01,
        1.33992452e-02,
    ],
    [
        -1.08817220e-01,
        -1.48605853e-01,
        1.43327145e-02,
        5.94605431e-02,
        3.95893026e-03,
        2.00082287e-02,
        -3.90074067e-02,
        -1.38992965e-01,
        3.17142009e-39,
        -1.18015558e-02,
        -8.46886635e-02,
        8.07307884e-02,
        4.88696583e-02,
        9.01132729e-03,
        2.73398664e-02,
        3.60320717e-01,
        6.67284951e-02,
        -3.76367569e-02,
        -1.29012764e-02,
        1.15553729e-01,
        1.22557946e-01,
        9.10442621e-02,
    ],
    [
        -1.41285077e-01,
        4.81627323e-02,
        -3.46615538e-02,
        -2.44255234e-02,
        -1.10077895e-01,
        3.47236171e-02,
        -2.91847698e-02,
        1.61158025e-01,
        -1.97956530e-39,
        -1.09702021e-01,
        -2.03233995e-02,
        -5.26874177e-02,
        -1.34758754e-02,
        3.01254238e-03,
        5.92869334e-02,
        -1.35532990e-01,
        -1.56316869e-02,
        1.27363885e-02,
        -6.87620267e-02,
        -3.03902775e-01,
        -2.54943371e-01,
        4.44428474e-02,
    ],
    [
        -1.46038041e-01,
        1.30516827e-01,
        -1.30431637e-01,
        4.13627736e-02,
        5.56976236e-02,
        -1.58129353e-02,
        3.17776129e-02,
        4.68298681e-02,
        -3.37718535e-40,
        1.57319941e-03,
        4.65862341e-02,
        5.77400364e-02,
        2.24920772e-02,
        -3.02676149e-02,
        5.63077210e-03,
        2.08167776e-01,
        -1.22885518e-01,
        5.16184559e-03,
        5.92165515e-02,
        -1.65169582e-01,
        -1.69462457e-01,
        -1.47785917e-01,
    ],
    [
        3.66744936e-01,
        -1.74042329e-01,
        2.47302502e-01,
        8.57425183e-02,
        1.27623484e-01,
        3.17690074e-02,
        -4.30110283e-02,
        -1.67581797e-01,
        -5.14372946e-39,
        -2.92561483e-02,
        5.99738862e-03,
        -1.16854794e-01,
        1.48466537e-02,
        5.44826873e-02,
        5.12253260e-03,
        -8.96807462e-02,
        -1.27767846e-01,
        -4.87805381e-02,
        5.73361069e-02,
        7.38519579e-02,
        4.57124412e-02,
        2.18208693e-02,
    ],
    [
        -2.06685722e-01,
        8.94212127e-02,
        5.80366626e-02,
        -1.11431539e-01,
        -1.51744019e-02,
        3.22304331e-02,
        -1.76357459e-02,
        8.92058089e-02,
        -4.30889889e-39,
        6.36590719e-02,
        9.56019685e-02,
        -3.66204903e-02,
        2.37665270e-02,
        3.99450511e-02,
        -5.50294816e-02,
        -1.46395728e-01,
        -6.35443777e-02,
        -2.66693830e-02,
        9.72016528e-02,
        -1.64702430e-01,
        -3.19513083e-02,
        -8.04689080e-02,
    ],
    [
        -1.62868798e-02,
        2.59202402e-02,
        -1.97853241e-02,
        -1.46997152e-02,
        -2.48614643e-02,
        -1.66675728e-02,
        9.82065336e-04,
        1.49894739e-02,
        3.08906297e-39,
        4.76755686e-02,
        2.54919324e-02,
        -3.32841724e-02,
        1.51600512e-02,
        7.51331449e-03,
        -8.25837106e-02,
        -5.03281131e-02,
        -1.65361613e-02,
        -5.00442125e-02,
        -2.73042053e-01,
        3.22523624e-01,
        6.76349699e-01,
        -3.28660756e-02,
    ],
    [
        -2.55196512e-01,
        3.76529172e-02,
        -1.30013630e-01,
        4.25749011e-02,
        -1.57254357e-02,
        -1.18287988e-02,
        -3.56946848e-02,
        1.67685021e-02,
        -1.53431672e-39,
        -4.28883806e-02,
        -5.25208972e-02,
        -2.64701410e-03,
        -1.69575587e-02,
        -5.09354621e-02,
        1.00296944e-01,
        5.36005050e-02,
        2.97499262e-02,
        -9.60078984e-02,
        -1.61662865e-02,
        2.22076073e-01,
        2.49424979e-01,
        8.37189183e-02,
    ],
    [
        3.18298340e-01,
        1.40243992e-02,
        1.57593936e-01,
        1.37130590e-02,
        5.68149798e-02,
        -1.47584721e-01,
        -2.31164191e-02,
        -8.41285661e-02,
        -6.67921907e-39,
        1.76997762e-02,
        1.83771759e-01,
        -8.14558491e-02,
        -6.09051734e-02,
        6.85105845e-02,
        1.26087382e-01,
        -3.34066421e-01,
        -9.80660245e-02,
        -1.91267088e-01,
        -1.18202679e-01,
        4.83977199e-02,
        4.48334962e-02,
        2.16041863e-01,
    ],
    [
        -1.14240073e-01,
        -6.17122687e-02,
        6.53027669e-02,
        -3.24446410e-02,
        -6.84135780e-02,
        -3.80305457e-03,
        -7.54766092e-02,
        2.07899600e-01,
        3.21725516e-40,
        -1.07352838e-01,
        -9.36319958e-03,
        4.13556024e-02,
        4.41469811e-02,
        -3.58738266e-02,
        2.38798968e-02,
        -1.52459130e-01,
        -6.40069023e-02,
        3.84902805e-02,
        1.04762211e-01,
        -2.14042857e-01,
        -1.37795940e-01,
        4.22344394e-02,
    ],
    [
        1.17514871e-01,
        -2.24299923e-01,
        7.22853914e-02,
        -2.15428159e-01,
        1.35922790e-01,
        8.93213823e-02,
        -1.02751190e-02,
        -3.18681709e-02,
        -6.39259888e-39,
        7.38538727e-02,
        -2.22180970e-02,
        -1.07860543e-01,
        5.86299412e-03,
        -2.45380551e-02,
        -6.50509894e-02,
        2.17358512e-03,
        -2.09465638e-01,
        -1.26182158e-02,
        -7.55328825e-03,
        7.79239908e-02,
        4.55333218e-02,
        -2.22439244e-02,
    ],
];
pub const VALUE_NN_B1: [f64; 64] = [
    -3.65703292e-02,
    -4.16467451e-02,
    -2.77594835e-01,
    -2.38352269e-01,
    -2.73710877e-01,
    -2.66689628e-01,
    -1.02199443e-01,
    -2.57409275e-01,
    -1.22304827e-01,
    -1.46683604e-01,
    -1.41067132e-01,
    -2.42054313e-01,
    -3.07945669e-01,
    1.16243839e-01,
    -1.10166609e-01,
    -1.48996890e-01,
    -7.82494340e-03,
    1.29473045e-01,
    -2.36121058e-01,
    -1.40208155e-01,
    5.04277647e-03,
    -2.77666181e-01,
    -1.09912843e-01,
    -3.72957200e-01,
    -3.26754481e-01,
    -2.63595432e-01,
    -3.11753035e-01,
    -2.65471488e-01,
    -2.97864020e-01,
    -1.94330797e-01,
    -1.30942181e-01,
    -1.51780143e-01,
    -1.69625372e-01,
    -5.97459041e-02,
    -1.42739788e-01,
    -2.46726632e-01,
    -7.60841444e-02,
    4.80519347e-02,
    -1.86252415e-01,
    -2.27251232e-01,
    -1.04582220e-01,
    -2.63198912e-01,
    -2.99130797e-01,
    -2.15445980e-01,
    5.79024181e-02,
    1.79785639e-02,
    -1.31595880e-01,
    -2.39314958e-01,
    -1.60454705e-01,
    -2.50877857e-01,
    -1.81872562e-01,
    -2.57457513e-02,
    -4.31711376e-01,
    -1.68807298e-01,
    -3.44951540e-01,
    -2.58378923e-01,
    -4.15343851e-01,
    -1.63572609e-01,
    -1.53722659e-01,
    7.29337260e-02,
    -1.46569312e-01,
    -2.25334793e-01,
    -1.87704265e-01,
    -2.29018226e-01,
];
pub const VALUE_NN_W2: [[f64; 64]; 32] = [
    [
        -3.83398240e-03,
        -1.97012238e-02,
        4.69089821e-02,
        -2.53843993e-01,
        -1.93137378e-01,
        -2.37270787e-01,
        8.71641263e-02,
        4.53386307e-02,
        -9.52842683e-02,
        2.09339123e-05,
        -5.12426943e-02,
        -1.44633114e-01,
        8.45876932e-02,
        -5.98623604e-03,
        3.32588330e-02,
        -1.35613561e-01,
        1.02450633e-02,
        -6.57703727e-03,
        1.36793628e-01,
        -3.01744401e-01,
        -7.57094892e-03,
        1.18591033e-01,
        -7.71099105e-02,
        -1.31416261e-01,
        -3.26141343e-02,
        1.84446722e-01,
        1.17968306e-01,
        -4.02268954e-02,
        -1.16922095e-01,
        -8.54066685e-02,
        6.26571523e-03,
        3.41962883e-03,
        -1.25238895e-01,
        -3.89717445e-02,
        -1.02276066e-02,
        1.07085016e-02,
        -3.17166895e-01,
        -1.65328737e-02,
        -2.85537332e-01,
        6.50194362e-02,
        1.78053677e-02,
        4.70198989e-02,
        -5.68341613e-02,
        -7.76319439e-03,
        -2.62590796e-02,
        1.02455346e-02,
        3.00576258e-02,
        9.22060460e-02,
        -3.72303426e-01,
        6.62357137e-02,
        -5.60374707e-02,
        2.96685696e-02,
        -5.59626743e-02,
        -7.48304501e-02,
        -4.25335281e-02,
        -1.78709254e-01,
        -1.18750229e-01,
        1.26248345e-01,
        -1.04469389e-01,
        -5.08877784e-02,
        -2.30118960e-01,
        9.25858691e-02,
        -8.60592350e-02,
        3.38529944e-02,
    ],
    [
        7.74791911e-02,
        -1.59416161e-02,
        5.84408082e-02,
        -8.25005025e-02,
        -8.77329856e-02,
        -5.42515926e-02,
        -3.71257886e-02,
        -2.97647864e-02,
        -1.12154350e-01,
        1.77147761e-01,
        -9.14943069e-02,
        -5.89134842e-02,
        1.36128262e-01,
        1.62141547e-02,
        -1.75445750e-02,
        -5.27932905e-02,
        -8.46560951e-03,
        8.45025294e-03,
        9.21034142e-02,
        -1.07030449e-02,
        -1.22960219e-02,
        2.75939815e-02,
        -4.96788695e-02,
        -7.33476058e-02,
        1.43869177e-01,
        2.01450419e-02,
        1.83437509e-03,
        8.53967369e-02,
        -6.76340982e-02,
        -4.28415798e-02,
        1.55931599e-02,
        -5.96433133e-02,
        -8.72815549e-02,
        3.81611027e-02,
        -2.22409870e-02,
        -2.04444360e-02,
        -2.46412605e-02,
        7.85664364e-04,
        3.23067084e-02,
        5.23717366e-02,
        6.86735511e-02,
        1.22767590e-01,
        -9.32713225e-02,
        -5.63386977e-02,
        6.24600751e-03,
        8.34121257e-02,
        7.28893057e-02,
        3.59679386e-02,
        -1.04974592e-02,
        1.01447232e-01,
        -5.39299697e-02,
        3.79783213e-02,
        -1.34604007e-01,
        -1.45003363e-01,
        1.45613804e-01,
        -7.37912804e-02,
        -6.56821951e-02,
        1.03329115e-01,
        -4.82292026e-02,
        1.36446888e-02,
        -2.04194635e-02,
        1.66033939e-01,
        -5.95685244e-02,
        7.31805786e-02,
    ],
    [
        9.70261730e-03,
        -2.61404878e-03,
        -1.03794120e-01,
        1.42809525e-02,
        -1.07683251e-02,
        4.61937115e-03,
        1.16196023e-02,
        1.90840419e-02,
        1.79196149e-01,
        -1.47012487e-01,
        5.01212142e-02,
        -2.80055944e-02,
        -1.19078442e-01,
        4.12048139e-02,
        -7.95833883e-04,
        1.23539999e-01,
        8.20721779e-03,
        1.48933325e-02,
        -5.91179319e-02,
        7.59957209e-02,
        1.92977972e-02,
        -9.79008451e-02,
        9.06349048e-02,
        1.13007106e-01,
        -9.27824751e-02,
        6.96180314e-02,
        7.21413121e-02,
        -4.38942090e-02,
        7.53072277e-02,
        1.10555030e-01,
        1.27843440e-01,
        7.00371340e-02,
        5.80120124e-02,
        -2.78786905e-02,
        9.47372243e-02,
        -7.57089555e-02,
        -2.62336303e-02,
        -3.14960852e-02,
        5.05198538e-02,
        -5.70355728e-02,
        -7.68403634e-02,
        -1.46789715e-01,
        1.83004960e-01,
        7.43603036e-02,
        1.73897799e-02,
        -1.04145266e-01,
        -1.38251960e-01,
        -9.07096118e-02,
        1.47328321e-02,
        -1.62442178e-01,
        4.01086137e-02,
        1.93369854e-02,
        4.09155376e-02,
        1.43615827e-01,
        -6.15027174e-02,
        9.16284416e-03,
        -3.26168388e-02,
        -5.79569489e-02,
        2.30688415e-02,
        4.10305569e-04,
        9.03418809e-02,
        -8.30813348e-02,
        -5.47294170e-02,
        -1.90528333e-02,
    ],
    [
        7.88648203e-02,
        4.65073995e-03,
        8.83664861e-02,
        -9.93617997e-02,
        -1.05444707e-01,
        -5.01923300e-02,
        -3.50291980e-03,
        -1.15561187e-02,
        -1.08389430e-01,
        6.93658739e-02,
        -9.75025073e-02,
        -2.01210082e-02,
        7.54328221e-02,
        -1.33385938e-02,
        4.21473868e-02,
        -1.30247667e-01,
        2.14771871e-02,
        9.95456986e-03,
        1.28252625e-01,
        -3.49381492e-02,
        1.19089633e-02,
        1.24791218e-03,
        -2.25425810e-02,
        -1.05104335e-01,
        1.32695287e-01,
        -1.53355869e-02,
        -3.48579623e-02,
        4.63795215e-02,
        -7.15976357e-02,
        -7.31973723e-02,
        -3.15997400e-03,
        -1.56538729e-02,
        -1.58286497e-01,
        3.81697551e-03,
        -9.34391990e-02,
        2.52335370e-02,
        -4.08087298e-02,
        -6.69751018e-02,
        9.23961867e-03,
        7.34301582e-02,
        7.33729675e-02,
        6.54182583e-02,
        -1.09058268e-01,
        -8.89183059e-02,
        -4.65929657e-02,
        -1.11410385e-02,
        2.62897201e-02,
        2.36084443e-02,
        -3.23235169e-02,
        6.11241795e-02,
        -5.11692427e-02,
        5.16079813e-02,
        -9.77878720e-02,
        -1.69143736e-01,
        8.52485597e-02,
        -9.08473805e-02,
        -8.68676603e-02,
        7.98415169e-02,
        -2.43009329e-02,
        2.41896808e-02,
        -9.07669589e-02,
        1.18233219e-01,
        -6.06597662e-02,
        3.26007344e-02,
    ],
    [
        1.05731390e-01,
        -3.53421830e-02,
        6.28072694e-02,
        -1.48485482e-01,
        -2.69710898e-01,
        -6.17446518e-03,
        1.24137715e-01,
        1.53880909e-01,
        -1.53525993e-01,
        4.87672985e-02,
        4.84468304e-02,
        -1.71271250e-01,
        -8.39109253e-03,
        4.13999408e-02,
        4.80524451e-02,
        -1.30979285e-01,
        1.20896446e-02,
        3.29752080e-03,
        1.12912074e-01,
        -3.41235578e-01,
        -2.70350613e-02,
        9.74903032e-02,
        2.75945589e-02,
        -1.47139832e-01,
        1.13367677e-01,
        1.48642898e-01,
        8.99729282e-02,
        1.03949472e-01,
        -4.10821894e-03,
        -6.81165606e-02,
        4.55444194e-02,
        -4.38216142e-02,
        -1.13105468e-01,
        7.68103898e-02,
        -8.22182558e-03,
        8.40303749e-02,
        -3.69247437e-01,
        3.14305015e-02,
        -3.24385539e-02,
        9.71234366e-02,
        1.70932133e-02,
        4.65171225e-02,
        -7.62375146e-02,
        7.61831412e-03,
        -1.22916289e-02,
        2.76451521e-02,
        3.60075235e-02,
        6.63158670e-02,
        -3.63206208e-01,
        3.14066894e-02,
        -2.34397762e-02,
        2.22846735e-02,
        -1.01814337e-01,
        -7.84933940e-02,
        -1.44586191e-01,
        -5.03048189e-02,
        -4.58807461e-02,
        2.30968129e-02,
        2.35308846e-03,
        -2.12525968e-02,
        -1.71551630e-01,
        1.48138598e-01,
        2.63835248e-02,
        9.64394063e-02,
    ],
    [
        -1.67898759e-02,
        -3.09002638e-01,
        6.24420270e-02,
        -1.91659536e-02,
        7.35950598e-04,
        4.47909757e-02,
        -3.32014076e-02,
        3.89734283e-02,
        3.01569309e-02,
        -2.44012661e-02,
        4.24102731e-02,
        -1.69759430e-02,
        -9.62148458e-02,
        -2.38140121e-01,
        -8.33820999e-02,
        -1.04792975e-02,
        -1.32890806e-01,
        -2.39703268e-01,
        6.91384226e-02,
        -4.95907385e-03,
        -1.52721837e-01,
        1.68013740e-02,
        1.55500732e-02,
        2.88472474e-02,
        -5.19257858e-02,
        -2.72472519e-02,
        7.35650435e-02,
        1.97878871e-02,
        1.85016058e-02,
        -6.61803875e-03,
        8.08821246e-02,
        3.91553454e-02,
        7.29271735e-04,
        1.15637191e-01,
        2.27384642e-02,
        5.76240793e-02,
        -7.45915691e-04,
        -1.18825339e-01,
        9.53570306e-02,
        -9.95605290e-02,
        -8.80042464e-02,
        4.98772897e-02,
        1.17783565e-02,
        6.66556433e-02,
        1.81013532e-02,
        -3.87242474e-02,
        9.36998278e-02,
        -1.20642766e-01,
        -5.13998093e-03,
        -1.38676828e-02,
        4.47331518e-02,
        -8.06794241e-02,
        7.88304061e-02,
        -3.90641997e-03,
        -1.21334784e-01,
        8.85303468e-02,
        1.09108739e-01,
        7.04638055e-03,
        4.79584411e-02,
        -1.83605701e-01,
        -7.98741877e-02,
        -2.38853954e-02,
        6.09501377e-02,
        1.22433998e-01,
    ],
    [
        -1.13483176e-01,
        -1.10384256e-01,
        2.39038244e-02,
        4.04453725e-02,
        2.90463679e-03,
        8.01922679e-02,
        6.07851706e-02,
        8.00297484e-02,
        7.60999024e-02,
        -8.26593712e-02,
        4.88400124e-02,
        -6.24116212e-02,
        -8.47317129e-02,
        -7.56343082e-02,
        -7.80377388e-02,
        2.79724021e-02,
        -1.03019752e-01,
        -8.63686725e-02,
        -3.98166254e-02,
        -7.46712461e-03,
        -6.62613511e-02,
        -1.42679503e-02,
        5.92829809e-02,
        -2.66264006e-02,
        -8.55094269e-02,
        -2.18947474e-02,
        5.31763434e-02,
        2.61851195e-02,
        4.72601354e-02,
        2.60519497e-02,
        3.63290496e-02,
        8.22492838e-02,
        9.73894149e-02,
        1.61315314e-02,
        -3.44559364e-02,
        7.70454183e-02,
        6.69318140e-02,
        -8.05485621e-02,
        5.82763962e-02,
        -3.78504694e-02,
        -1.03890672e-01,
        -1.00192882e-01,
        7.68158138e-02,
        4.59690727e-02,
        -1.09639578e-01,
        -1.61075071e-01,
        -7.69732893e-02,
        -7.98836201e-02,
        1.06356822e-01,
        -1.03091978e-01,
        3.56685068e-03,
        -4.24428657e-02,
        8.86626095e-02,
        1.24048412e-01,
        5.64838015e-03,
        2.45280005e-02,
        2.77136937e-02,
        2.43407544e-02,
        -6.65972903e-02,
        -1.25160247e-01,
        4.13760357e-02,
        -4.90346923e-02,
        -4.98828329e-02,
        -3.61117758e-02,
    ],
    [
        -2.03069031e-01,
        -3.15421104e-01,
        1.61287263e-02,
        2.85195094e-02,
        -2.82174572e-02,
        -2.79215612e-02,
        6.07503206e-02,
        2.08048318e-02,
        6.48868158e-02,
        7.39950240e-02,
        2.68251840e-02,
        4.59880531e-02,
        -1.38521772e-02,
        -3.04903299e-01,
        -2.17645451e-01,
        6.40068650e-02,
        -2.61384010e-01,
        -3.79566163e-01,
        3.91868427e-02,
        1.84913278e-02,
        -2.73026168e-01,
        8.54892209e-02,
        -8.22160393e-03,
        9.34061557e-02,
        -7.84991402e-03,
        1.16759114e-01,
        7.33432397e-02,
        8.36910233e-02,
        2.47866418e-02,
        1.21865924e-02,
        2.95194667e-02,
        -2.11285427e-02,
        2.36585084e-02,
        3.06512676e-02,
        5.04944623e-02,
        1.52384201e-02,
        5.31880185e-03,
        -2.96935588e-01,
        -8.53770319e-03,
        -1.41188964e-01,
        -1.39492959e-01,
        1.02461427e-01,
        -3.91891338e-02,
        -1.50188087e-02,
        -2.11595640e-01,
        -1.06492497e-01,
        1.19839720e-01,
        -9.01617855e-02,
        5.21328393e-03,
        -7.16735870e-02,
        -2.63209967e-03,
        -2.42317557e-01,
        2.06124976e-01,
        8.46905820e-03,
        8.22970923e-03,
        8.05094652e-03,
        -8.19481686e-02,
        3.46890837e-02,
        -5.60326278e-02,
        -2.32709378e-01,
        2.37637218e-02,
        3.45566012e-02,
        1.86602827e-02,
        1.22348674e-01,
    ],
    [
        -2.41091952e-01,
        -2.41324291e-01,
        9.31731761e-02,
        -6.65100990e-03,
        -3.89923714e-02,
        4.37362343e-02,
        3.03928554e-02,
        6.78220913e-02,
        4.32277098e-02,
        -5.39880199e-03,
        2.01090481e-02,
        -4.26182002e-02,
        -6.67709634e-02,
        -3.12721044e-01,
        -1.77056700e-01,
        5.75321130e-02,
        -3.71997714e-01,
        -4.13898379e-01,
        2.86295731e-02,
        -1.37433037e-02,
        -3.16961348e-01,
        1.03462204e-01,
        4.06885780e-02,
        -1.05845332e-02,
        -4.07566540e-02,
        1.07835300e-01,
        7.55309835e-02,
        1.17415197e-01,
        1.38290962e-02,
        2.32846458e-02,
        -9.14232433e-02,
        -2.19394732e-02,
        6.39829189e-02,
        -1.39722386e-02,
        -1.21938828e-02,
        1.34625331e-01,
        -1.95811670e-02,
        -3.28336000e-01,
        -2.79233828e-02,
        -9.66819823e-02,
        -1.91011414e-01,
        7.79424682e-02,
        6.95045143e-02,
        8.18211660e-02,
        -2.57618070e-01,
        -8.08658302e-02,
        7.55033316e-03,
        -9.45378318e-02,
        -1.05527453e-02,
        -1.98361240e-02,
        6.23558164e-02,
        -3.38301986e-01,
        2.28883520e-01,
        -3.71244806e-03,
        -1.97833702e-02,
        -2.50535272e-03,
        -6.79965168e-02,
        1.32648125e-01,
        5.79010323e-03,
        -3.33037436e-01,
        3.80254313e-02,
        -2.67685410e-02,
        -1.53181469e-03,
        4.20562290e-02,
    ],
    [
        3.55371386e-02,
        -2.81984985e-01,
        7.90460706e-02,
        -2.68495064e-02,
        -2.88736192e-03,
        9.04638991e-02,
        -3.21329609e-02,
        1.14776202e-01,
        -1.38511676e-02,
        -1.18587501e-02,
        8.50469843e-02,
        2.23998278e-02,
        -8.53410438e-02,
        -2.40636081e-01,
        -7.31051415e-02,
        -2.00946424e-02,
        -1.17377043e-01,
        -1.81059659e-01,
        6.48375005e-02,
        -1.41980508e-02,
        -1.40281022e-01,
        6.71980307e-02,
        9.02630463e-02,
        7.48862848e-02,
        2.15899176e-03,
        6.42265603e-02,
        6.18903674e-02,
        1.23468615e-01,
        -3.91039439e-03,
        3.40245366e-02,
        1.02887399e-01,
        -2.69962940e-02,
        2.38852333e-02,
        9.01870355e-02,
        1.89127475e-02,
        -3.29539627e-02,
        -4.86953780e-02,
        -1.71459168e-01,
        5.72949648e-02,
        -7.65088350e-02,
        -5.85901514e-02,
        4.53765839e-02,
        5.94382435e-02,
        1.02214245e-02,
        9.93971899e-03,
        -7.80691653e-02,
        7.37577751e-02,
        -1.31925732e-01,
        4.84490357e-02,
        -3.73223354e-03,
        4.28102873e-02,
        -1.10083677e-01,
        2.68623885e-02,
        1.97272506e-02,
        -1.37115225e-01,
        6.37726933e-02,
        4.50559258e-02,
        -5.49413674e-02,
        6.44025654e-02,
        -2.06619322e-01,
        -8.11131001e-02,
        -5.98149188e-02,
        3.41375880e-02,
        1.23590440e-01,
    ],
    [
        -1.10697687e-01,
        -4.49953601e-02,
        -2.15553865e-02,
        -5.49325859e-03,
        8.20326060e-03,
        5.61660081e-02,
        3.06425970e-02,
        4.55734544e-02,
        9.28382576e-02,
        -1.69832427e-02,
        -2.93848682e-02,
        -1.97002478e-02,
        -2.99235974e-02,
        -6.66657686e-02,
        -4.65510637e-02,
        7.65807629e-02,
        -2.46955976e-02,
        -9.38003808e-02,
        1.99808814e-02,
        8.12733918e-02,
        -1.32438660e-01,
        8.67804214e-02,
        5.89078963e-02,
        1.89357568e-02,
        2.27057785e-02,
        1.26423210e-01,
        4.65457328e-02,
        5.86336618e-03,
        7.88594559e-02,
        7.67177939e-02,
        3.42082493e-02,
        2.62094624e-02,
        6.11195341e-02,
        9.39986110e-03,
        6.57332987e-02,
        -4.82380651e-02,
        3.67868543e-02,
        -8.53348374e-02,
        4.65629958e-02,
        -3.19274813e-02,
        -1.38992295e-01,
        3.46151292e-02,
        -1.27492305e-02,
        4.36757207e-02,
        -6.60606697e-02,
        -8.10534060e-02,
        -4.41590650e-03,
        -5.96801750e-02,
        2.78168060e-02,
        -1.75230429e-01,
        2.92139836e-02,
        -8.34049508e-02,
        2.70378478e-02,
        4.35546003e-02,
        -6.69664005e-03,
        5.04173227e-02,
        -2.90511306e-02,
        -5.38042886e-03,
        5.39138317e-02,
        -7.30924085e-02,
        1.16639398e-01,
        4.54418957e-02,
        6.09533489e-03,
        5.01597710e-02,
    ],
    [
        -1.09993853e-01,
        -2.62156546e-01,
        -1.58892591e-02,
        8.68471265e-02,
        -5.08518331e-02,
        -8.53581820e-03,
        3.99737135e-02,
        9.70191360e-02,
        -9.29325819e-03,
        3.57055962e-02,
        6.30258545e-02,
        8.56620446e-03,
        -4.96673435e-02,
        -2.07856819e-01,
        -9.84644890e-02,
        1.16024613e-02,
        -1.05694056e-01,
        -2.07288027e-01,
        8.79294351e-02,
        -4.13584560e-02,
        -1.11977488e-01,
        2.60733757e-02,
        8.82325321e-03,
        3.61665189e-02,
        -3.39539014e-02,
        -6.16076067e-02,
        -3.41868065e-02,
        4.79039066e-02,
        6.05918951e-02,
        1.26263008e-01,
        1.85152777e-02,
        2.67547537e-02,
        1.27881998e-02,
        7.11607412e-02,
        -7.59451091e-02,
        7.38770813e-02,
        -4.71140966e-02,
        -1.13324188e-01,
        2.67620720e-02,
        -8.02182481e-02,
        -1.40013039e-01,
        6.60087168e-03,
        -2.50200089e-02,
        6.06754329e-03,
        -6.07418008e-02,
        -7.13159889e-02,
        4.35816683e-02,
        -1.36217207e-01,
        2.45068371e-02,
        -1.05268985e-01,
        -1.83973275e-02,
        -1.17055386e-01,
        4.77781519e-02,
        8.97268802e-02,
        -8.63427371e-02,
        2.99123786e-02,
        1.20820314e-01,
        3.20794284e-02,
        8.83140892e-04,
        -1.68938890e-01,
        -1.71645768e-02,
        5.15040830e-02,
        6.68422654e-02,
        3.73716392e-02,
    ],
    [
        -1.42531991e-02,
        -2.77906246e-02,
        7.06056207e-02,
        -1.61233515e-01,
        -1.45370796e-01,
        -1.14369310e-01,
        -2.28046644e-02,
        -4.68438379e-02,
        -1.87375888e-01,
        1.00811973e-01,
        -1.72488555e-01,
        -4.05576490e-02,
        9.92707312e-02,
        5.90113886e-02,
        -3.78956720e-02,
        -1.26983389e-01,
        -3.05468570e-02,
        -2.10998245e-02,
        1.54306829e-01,
        -1.80586707e-02,
        -4.64520268e-02,
        4.79061604e-02,
        -1.29493609e-01,
        -7.34738633e-02,
        1.34209961e-01,
        3.10968962e-02,
        7.01052463e-03,
        1.14737675e-01,
        -6.42070919e-02,
        -3.55695598e-02,
        -6.30031433e-03,
        -1.15158394e-01,
        -1.56171560e-01,
        3.62021923e-02,
        -9.67574343e-02,
        -3.93853262e-02,
        -4.36586104e-02,
        -4.85311169e-03,
        -1.52486056e-01,
        1.24228168e-02,
        8.18976015e-02,
        3.74024659e-02,
        -8.55548903e-02,
        -4.70978916e-02,
        5.95873669e-02,
        5.40316850e-02,
        8.82601812e-02,
        -1.13661774e-02,
        -3.79391722e-02,
        3.89995836e-02,
        -9.80943218e-02,
        -5.14881425e-02,
        -4.59161215e-02,
        -8.77208635e-02,
        1.92699790e-01,
        -1.35917932e-01,
        -9.58463401e-02,
        1.26766399e-01,
        -7.48138055e-02,
        2.15260927e-02,
        -5.25520854e-02,
        8.08263570e-02,
        -9.44003984e-02,
        9.43915248e-02,
    ],
    [
        1.00632748e-02,
        1.29904089e-04,
        8.05735737e-02,
        -1.03589252e-01,
        -9.99004394e-02,
        -5.72346225e-02,
        -1.18872775e-02,
        -6.57348931e-02,
        -1.14497706e-01,
        1.89744323e-01,
        -1.17709875e-01,
        -5.93336523e-02,
        1.35933489e-01,
        3.95735502e-02,
        -1.49081247e-02,
        -5.41164316e-02,
        1.59178544e-02,
        -1.61504000e-02,
        8.89775008e-02,
        -8.95504840e-03,
        -3.58366705e-02,
        3.45073603e-02,
        -1.40864924e-02,
        -4.76275980e-02,
        1.56606242e-01,
        -9.71772056e-03,
        -3.33022811e-02,
        9.99961048e-02,
        -3.49998772e-02,
        -9.35078040e-02,
        -3.73440869e-02,
        -3.82256694e-02,
        -7.71062076e-02,
        2.39763670e-02,
        -5.33838160e-02,
        4.05355133e-02,
        -3.89403999e-02,
        3.23127396e-02,
        -4.08539101e-02,
        6.37236536e-02,
        1.50787756e-01,
        1.34128913e-01,
        -1.02468446e-01,
        -6.07008673e-02,
        4.64913175e-02,
        4.70178314e-02,
        6.14021793e-02,
        6.94728717e-02,
        -4.84625399e-02,
        1.42171860e-01,
        -2.23269630e-02,
        -9.33820941e-03,
        -7.99332708e-02,
        -1.56743273e-01,
        1.15378566e-01,
        -8.24324787e-02,
        -8.37017596e-03,
        1.16564088e-01,
        -6.37368187e-02,
        3.17414477e-02,
        -3.70477699e-02,
        1.31783366e-01,
        -2.08052155e-02,
        6.98697567e-02,
    ],
    [
        1.02872051e-01,
        1.32785384e-02,
        8.21001530e-02,
        -1.51896015e-01,
        -1.38045490e-01,
        -2.01835185e-01,
        -2.44427528e-02,
        -4.35568206e-02,
        -1.21776208e-01,
        7.80892298e-02,
        -2.28597865e-01,
        -1.41389042e-01,
        4.64457795e-02,
        -6.05289172e-03,
        -4.84273061e-02,
        -9.49390903e-02,
        -8.52845684e-02,
        -9.09007806e-03,
        5.87998629e-02,
        4.30703983e-02,
        -2.37360559e-02,
        4.67296392e-02,
        -1.63718760e-01,
        -2.26054311e-01,
        3.57274450e-02,
        3.65325250e-02,
        1.35901039e-02,
        9.88057777e-02,
        -1.36578649e-01,
        -3.50350291e-02,
        -4.23052274e-02,
        -7.79215395e-02,
        -9.91078988e-02,
        -7.12227747e-02,
        -8.25337470e-02,
        -7.80155212e-02,
        2.12520696e-02,
        -6.99077025e-02,
        -2.23371238e-01,
        1.48956235e-02,
        7.97234923e-02,
        2.29362752e-02,
        -1.36793271e-01,
        -3.51488292e-02,
        -1.58844739e-02,
        1.83102433e-02,
        2.55352873e-02,
        -3.15959156e-02,
        -6.55577034e-02,
        -2.62613166e-02,
        -1.12087645e-01,
        2.92799789e-02,
        -1.31048933e-01,
        -8.99967030e-02,
        -7.30824992e-02,
        -2.30021134e-01,
        -1.53381273e-01,
        9.71013010e-02,
        -1.00557506e-01,
        6.41355738e-02,
        -1.01377502e-01,
        6.92038164e-02,
        -1.68247938e-01,
        6.96052685e-02,
    ],
    [
        5.51775172e-02,
        -5.31905191e-03,
        5.73883802e-02,
        -1.11286215e-01,
        -1.04583599e-01,
        -1.04883745e-01,
        -7.43448827e-03,
        -1.84883773e-02,
        -1.08930692e-01,
        1.47736132e-01,
        -1.34336889e-01,
        -4.08447906e-02,
        8.79145414e-02,
        -1.32418489e-02,
        -1.23592354e-02,
        -6.16765767e-02,
        -1.12717394e-02,
        -1.05605135e-02,
        1.48980781e-01,
        -5.61833475e-03,
        1.42943664e-02,
        2.47145351e-02,
        -5.69003150e-02,
        -1.07540749e-01,
        1.46483004e-01,
        2.41136714e-03,
        -6.35234034e-03,
        5.98851070e-02,
        -5.19943908e-02,
        -3.98880392e-02,
        2.28314362e-02,
        -9.41294357e-02,
        -8.71831551e-02,
        9.21080075e-03,
        -9.46454406e-02,
        3.76418456e-02,
        1.00452611e-02,
        1.07338009e-02,
        -3.51551920e-02,
        9.35856476e-02,
        8.34542066e-02,
        1.50385991e-01,
        -9.67680514e-02,
        -8.47665146e-02,
        3.25879864e-02,
        4.23795767e-02,
        4.43721823e-02,
        6.59922734e-02,
        -3.24908085e-02,
        3.80322225e-02,
        -3.80483009e-02,
        3.20032127e-02,
        -7.73391277e-02,
        -1.58829927e-01,
        4.94750440e-02,
        -9.27149355e-02,
        -1.23208225e-01,
        1.62572905e-01,
        -4.49879244e-02,
        -1.61108784e-02,
        3.09917871e-02,
        9.94710997e-02,
        -1.34035826e-01,
        1.10121310e-01,
    ],
    [
        -1.98026195e-01,
        -2.57982790e-01,
        -2.94599100e-03,
        1.52680455e-02,
        5.61764017e-02,
        4.17719921e-03,
        3.95130850e-02,
        -1.82403512e-02,
        -2.40565874e-02,
        1.40448809e-02,
        3.97785008e-03,
        -3.86979543e-02,
        -3.77303325e-02,
        -2.64207721e-01,
        -1.71298146e-01,
        5.93581758e-02,
        -2.52641261e-01,
        -2.82627910e-01,
        6.58838898e-02,
        -1.10603850e-02,
        -2.27794424e-01,
        9.11589563e-02,
        -2.06893794e-02,
        5.57342311e-03,
        -9.78717208e-03,
        5.28551638e-02,
        8.69086236e-02,
        9.52244997e-02,
        4.06785905e-02,
        -3.82278375e-02,
        1.00018367e-01,
        -2.87848357e-02,
        2.20114030e-02,
        8.47869739e-02,
        7.48025551e-02,
        8.07945654e-02,
        -9.36138071e-03,
        -2.44723231e-01,
        -3.79802212e-02,
        -1.12809047e-01,
        -1.65826365e-01,
        7.06202909e-02,
        9.39986259e-02,
        7.38979597e-03,
        -1.40857443e-01,
        -7.27771595e-02,
        -1.13952216e-02,
        -1.12866491e-01,
        -4.20504697e-02,
        -1.34678036e-01,
        -3.45627740e-02,
        -1.71163067e-01,
        1.16923347e-01,
        1.43210724e-01,
        -1.61134705e-01,
        -1.23748081e-02,
        -8.39749351e-03,
        6.45545498e-02,
        8.51564575e-03,
        -2.41890669e-01,
        -1.19551510e-01,
        6.90742061e-02,
        -5.73034398e-03,
        6.49189129e-02,
    ],
    [
        -1.54726431e-01,
        -2.44779184e-01,
        -7.12661371e-02,
        1.35951098e-02,
        -1.87048055e-02,
        8.47495161e-03,
        7.32979178e-02,
        8.64338577e-02,
        -2.03875527e-02,
        5.19691482e-02,
        6.12810953e-04,
        3.23185585e-02,
        -3.24425958e-02,
        -3.38250190e-01,
        -2.08030343e-01,
        4.13600318e-02,
        -3.18453461e-01,
        -3.42000544e-01,
        4.90936786e-02,
        -2.76381951e-02,
        -3.15325826e-01,
        1.11825168e-01,
        2.54477151e-02,
        1.33201142e-03,
        -1.22392932e-02,
        6.84982613e-02,
        7.86296874e-02,
        1.30854025e-01,
        -3.53229754e-02,
        3.75049077e-02,
        -8.69492292e-02,
        -7.74130691e-03,
        3.18358317e-02,
        1.05301160e-02,
        3.24807912e-02,
        6.78080544e-02,
        -2.33390518e-02,
        -2.55792856e-01,
        1.65492855e-02,
        -1.57261103e-01,
        -1.65001303e-01,
        1.11661568e-01,
        1.08536340e-01,
        5.95076308e-02,
        -8.82097334e-02,
        -4.28813919e-02,
        9.44927111e-02,
        -7.85676837e-02,
        -6.13920391e-02,
        -1.26126602e-01,
        3.00196484e-02,
        -2.93289483e-01,
        1.76145703e-01,
        -3.55048664e-03,
        -3.59077044e-02,
        -1.62142999e-02,
        -7.08926991e-02,
        -3.02045438e-02,
        3.65487160e-03,
        -2.57743567e-01,
        4.97755557e-02,
        5.25658280e-02,
        2.57229377e-02,
        6.13989495e-02,
    ],
    [
        9.94171053e-02,
        3.92833911e-03,
        9.15521756e-02,
        -6.27493905e-03,
        -9.51394886e-02,
        -5.54949082e-02,
        -1.49948942e-02,
        -8.41384009e-03,
        -1.07912160e-01,
        8.97303447e-02,
        -1.63498949e-02,
        -2.64211930e-02,
        8.61371905e-02,
        2.31579412e-02,
        5.50262891e-02,
        -5.83847575e-02,
        2.14319080e-02,
        5.54918349e-02,
        1.41163230e-01,
        2.50835307e-02,
        -9.98072326e-03,
        5.27558737e-02,
        -3.06100165e-03,
        -2.79524736e-02,
        1.39586121e-01,
        -4.05080505e-02,
        -2.31755488e-02,
        9.35512409e-02,
        2.34271362e-02,
        -5.01149595e-02,
        -1.62271168e-02,
        -4.78683338e-02,
        -3.75369042e-02,
        -4.87001799e-03,
        -9.38689336e-03,
        -3.37131098e-02,
        9.85220727e-03,
        3.17812990e-03,
        -2.89148148e-02,
        6.24423213e-02,
        1.04079105e-01,
        8.19845647e-02,
        -4.06451151e-02,
        -8.43653679e-02,
        1.35060726e-02,
        8.06275308e-02,
        9.46500227e-02,
        3.05499546e-02,
        -9.91426967e-03,
        5.85458912e-02,
        -1.72822922e-02,
        -9.41669662e-03,
        -7.81662390e-03,
        -1.15810171e-01,
        9.00197700e-02,
        -5.97300865e-02,
        8.21439270e-03,
        8.26278776e-02,
        -6.67002201e-02,
        1.57676172e-02,
        -6.79398933e-03,
        8.45477059e-02,
        -1.53775606e-02,
        2.49076597e-02,
    ],
    [
        -4.54673320e-02,
        -1.76585302e-01,
        -7.13290647e-02,
        -9.45874117e-03,
        4.63851430e-02,
        8.79829526e-02,
        6.90801442e-02,
        1.34314168e-02,
        -2.05098260e-02,
        -7.76041448e-02,
        2.94997636e-03,
        5.09033203e-02,
        3.07358187e-02,
        -2.12209612e-01,
        -1.51664857e-02,
        8.82251933e-02,
        -5.25200032e-02,
        -1.43680990e-01,
        7.33036846e-02,
        -1.05109159e-02,
        -1.11146905e-01,
        9.77236927e-02,
        1.78835653e-02,
        -3.27524208e-02,
        2.33074334e-02,
        1.11249521e-01,
        7.53850564e-02,
        -2.30360385e-02,
        3.61987427e-02,
        2.87329610e-02,
        -8.58077127e-03,
        2.26049367e-02,
        3.75109538e-02,
        4.07284871e-02,
        8.25268999e-02,
        -7.12289615e-03,
        -1.12803262e-02,
        -1.28268734e-01,
        3.86206396e-02,
        -7.54300877e-02,
        -8.84282216e-02,
        1.96557250e-02,
        4.04828936e-02,
        7.55608082e-02,
        -1.74613092e-02,
        -7.72767290e-02,
        2.42600795e-02,
        -3.71166319e-02,
        -3.68837379e-02,
        -1.02589108e-01,
        4.02986221e-02,
        -7.31186848e-03,
        -3.63283698e-03,
        4.03292105e-02,
        -1.56353757e-01,
        6.22199290e-03,
        7.96859562e-02,
        -3.56137231e-02,
        1.03486523e-01,
        -1.19349405e-01,
        -1.21433623e-01,
        -4.63907234e-02,
        7.55643249e-02,
        1.44756675e-01,
    ],
    [
        2.60864142e-02,
        2.32028328e-02,
        6.25022426e-02,
        -9.23147351e-02,
        -1.10570289e-01,
        -6.58061877e-02,
        -2.87800562e-03,
        -7.75837675e-02,
        -1.06249496e-01,
        2.03667521e-01,
        -1.21130399e-01,
        -3.93527150e-02,
        1.03631914e-01,
        -3.27486806e-02,
        -8.74567777e-03,
        -7.63236359e-02,
        -2.45793574e-02,
        5.46582788e-02,
        1.14836790e-01,
        -4.81815599e-02,
        -4.74572815e-02,
        4.07058629e-04,
        -4.13977765e-02,
        -1.12963483e-01,
        1.22508839e-01,
        1.75614748e-02,
        -1.55793615e-02,
        4.98198345e-02,
        -1.11016169e-01,
        -8.31501111e-02,
        1.58913974e-02,
        -9.22497362e-02,
        -1.00435778e-01,
        -2.10495526e-03,
        -1.29631236e-01,
        -1.19388076e-02,
        -2.69307960e-02,
        -4.72018635e-03,
        -5.83903864e-02,
        4.87802774e-02,
        1.34368315e-01,
        1.48130000e-01,
        -1.04088895e-01,
        -9.84842628e-02,
        4.52812836e-02,
        3.16841677e-02,
        4.21350300e-02,
        -6.52033370e-04,
        -3.30269672e-02,
        6.91068694e-02,
        -5.75693250e-02,
        3.11606191e-02,
        -9.19109955e-02,
        -1.53701663e-01,
        6.80837482e-02,
        -9.08232629e-02,
        -1.06379747e-01,
        8.53275880e-02,
        -5.99492155e-02,
        4.22915854e-02,
        -6.69198576e-03,
        1.74763948e-01,
        -1.06127188e-01,
        1.06937759e-01,
    ],
    [
        -1.11563489e-01,
        -1.57397047e-01,
        -3.04652285e-02,
        -6.40698448e-02,
        -8.62877723e-03,
        5.28531522e-02,
        6.40891120e-02,
        1.97332799e-02,
        9.88688320e-02,
        -6.41877204e-02,
        3.04195774e-03,
        -3.86594012e-02,
        -5.98772131e-02,
        -1.02059729e-01,
        -6.88620657e-02,
        8.53553712e-02,
        -7.90970922e-02,
        -7.82612264e-02,
        -3.01155392e-02,
        2.47972608e-02,
        -1.05429694e-01,
        -2.80788802e-02,
        2.80503239e-02,
        2.52597425e-02,
        -9.90638286e-02,
        -2.31897994e-03,
        5.75865880e-02,
        1.20302371e-03,
        7.65476897e-02,
        1.58200879e-02,
        3.69267426e-02,
        3.96573320e-02,
        1.15290433e-02,
        4.06698100e-02,
        1.59834307e-02,
        6.14182651e-02,
        3.49351540e-02,
        -6.66532665e-02,
        5.02345636e-02,
        -1.01553462e-01,
        -1.44581035e-01,
        -1.11761346e-01,
        1.10843621e-01,
        7.54607841e-02,
        -6.93422258e-02,
        -1.24826729e-01,
        -4.08101864e-02,
        -1.19153693e-01,
        -4.56854980e-03,
        -1.38287842e-01,
        5.88991097e-04,
        -3.88849489e-02,
        -1.13279384e-05,
        9.59525406e-02,
        -1.12180613e-01,
        2.30911970e-02,
        -3.14666354e-03,
        -4.81213033e-02,
        8.42792094e-02,
        -1.45733356e-01,
        -5.51604666e-02,
        -2.68652430e-03,
        7.89420959e-03,
        -6.74528182e-02,
    ],
    [
        -1.20472401e-01,
        -1.24131843e-01,
        -7.49731064e-02,
        4.64416631e-02,
        -1.49512701e-02,
        -7.91174918e-03,
        6.63950816e-02,
        -1.29916491e-02,
        9.19132121e-03,
        -4.21298072e-02,
        6.99419342e-03,
        4.51057702e-02,
        -2.10059956e-02,
        -1.51169956e-01,
        -3.82805313e-03,
        9.44340825e-02,
        -3.21120545e-02,
        -9.57412869e-02,
        1.44434404e-02,
        1.24665899e-02,
        -1.16480999e-01,
        6.14425242e-02,
        7.69482031e-02,
        -1.58428624e-02,
        1.78530347e-02,
        8.61432999e-02,
        8.05497393e-02,
        5.29379072e-03,
        7.02051297e-02,
        6.95843473e-02,
        2.90395264e-02,
        3.96157131e-02,
        -5.78576978e-03,
        2.04588268e-02,
        6.92334399e-02,
        -6.96949437e-02,
        5.01208603e-02,
        -9.47497338e-02,
        7.49761332e-03,
        -8.30993503e-02,
        -1.68866426e-01,
        3.33565213e-02,
        8.17316994e-02,
        3.71453986e-02,
        -4.73564453e-02,
        -7.31577724e-02,
        -4.00509611e-02,
        -3.01436777e-03,
        -1.59031264e-02,
        -1.78334907e-01,
        3.36406976e-02,
        -4.05745208e-02,
        2.57976726e-02,
        4.52936301e-03,
        -1.12157665e-01,
        1.06645944e-02,
        1.72636695e-02,
        -9.49602015e-03,
        7.84293115e-02,
        -1.19147263e-01,
        -7.30495602e-02,
        3.42597365e-02,
        6.03385940e-02,
        3.68433148e-02,
    ],
    [
        -1.45041989e-02,
        1.21769896e-02,
        1.03698917e-01,
        -2.31640667e-01,
        -1.97795942e-01,
        -6.67227665e-03,
        -2.48290747e-02,
        5.78666553e-02,
        -1.01183072e-01,
        1.32239580e-01,
        -1.09833159e-01,
        -6.30768836e-02,
        1.23616427e-01,
        4.27899836e-03,
        3.00934333e-02,
        -4.41141687e-02,
        -1.29374210e-02,
        2.03789938e-02,
        9.65527967e-02,
        -1.81628913e-01,
        4.05183285e-02,
        4.03986052e-02,
        -3.25341299e-02,
        -1.19316224e-02,
        2.16954157e-01,
        4.53531221e-02,
        2.39175521e-02,
        1.14782028e-01,
        4.78614196e-02,
        1.56626124e-02,
        7.49933347e-02,
        -5.61863035e-02,
        -6.21228442e-02,
        8.37576389e-02,
        -8.44249353e-02,
        -1.80510730e-02,
        -1.41990036e-01,
        -1.70098599e-02,
        -3.60597596e-02,
        9.93101299e-02,
        4.02341187e-02,
        1.01979524e-01,
        1.32082580e-02,
        -3.46194096e-02,
        1.24458456e-02,
        2.49599442e-02,
        5.35363257e-02,
        5.33633865e-02,
        -1.44841880e-01,
        5.82402386e-02,
        -1.90611649e-02,
        1.91047378e-02,
        -1.70393549e-02,
        -1.05294913e-01,
        5.46368100e-02,
        -8.30162987e-02,
        1.27627961e-02,
        3.74235250e-02,
        -1.80053208e-02,
        -5.06431051e-03,
        5.52453585e-02,
        1.73931792e-01,
        -3.83921601e-02,
        8.02594200e-02,
    ],
    [
        -8.78286809e-02,
        -1.21081080e-02,
        -6.29253015e-02,
        -6.52642101e-02,
        -5.69139011e-02,
        1.08059570e-02,
        7.63666332e-02,
        -4.76090573e-02,
        7.21318200e-02,
        -1.82115391e-01,
        6.39931113e-02,
        -6.75803274e-02,
        -2.13321224e-01,
        -1.45953568e-03,
        -3.06811053e-02,
        9.45849642e-02,
        3.35670379e-03,
        2.95740906e-02,
        -4.47473489e-02,
        3.04925684e-02,
        -1.39496606e-02,
        -1.42935947e-01,
        5.40986471e-02,
        -3.30808088e-02,
        -1.38605237e-01,
        8.73954371e-02,
        4.87069637e-02,
        3.09284609e-02,
        1.34651763e-02,
        5.59658632e-02,
        3.51446271e-02,
        -5.03429491e-03,
        8.82052258e-02,
        -1.12894006e-01,
        1.22877054e-01,
        7.81930611e-02,
        -1.08867576e-02,
        -8.99209455e-03,
        5.48425764e-02,
        -7.13223666e-02,
        -1.34427577e-01,
        -1.72085240e-01,
        3.76596935e-02,
        6.75199628e-02,
        -5.02695292e-02,
        -1.55959785e-01,
        -1.56524464e-01,
        -5.88746369e-02,
        1.16135284e-01,
        -2.14903295e-01,
        1.24274738e-01,
        -8.65298975e-03,
        7.57549107e-02,
        1.47437990e-01,
        -2.18754876e-02,
        4.82661240e-02,
        2.16579493e-02,
        -6.89575225e-02,
        -1.53503111e-02,
        -2.43864246e-02,
        3.63070853e-02,
        -1.08838961e-01,
        -1.07051641e-01,
        -7.54686147e-02,
    ],
    [
        -8.13846663e-02,
        -1.75226688e-01,
        -7.81108364e-02,
        -5.29716630e-03,
        5.09312842e-03,
        6.28395611e-03,
        2.62829680e-02,
        -3.05613433e-03,
        2.17038337e-02,
        -3.58912759e-02,
        1.57570781e-03,
        4.17183600e-02,
        -1.98291671e-02,
        -1.82144642e-01,
        -5.14944568e-02,
        6.43617958e-02,
        -6.56751767e-02,
        -1.38764054e-01,
        4.63970117e-02,
        8.79764557e-03,
        -1.29010767e-01,
        7.97064006e-02,
        5.84886186e-02,
        -3.55617441e-02,
        -3.41531634e-02,
        7.00856373e-02,
        9.10492241e-02,
        -2.44072992e-02,
        5.78151597e-03,
        -5.58502926e-03,
        -1.16730565e-02,
        3.60360881e-03,
        3.81285660e-02,
        3.17037515e-02,
        5.98967299e-02,
        -3.61786969e-02,
        1.39848189e-02,
        -1.31316468e-01,
        4.21431176e-02,
        -9.48054716e-02,
        -1.09151654e-01,
        2.02746708e-02,
        4.47479524e-02,
        2.54142061e-02,
        -3.81651223e-02,
        -7.51934126e-02,
        2.45911982e-02,
        -2.59865988e-02,
        -6.83518723e-02,
        -1.05001986e-01,
        3.75285372e-02,
        -5.33043779e-02,
        1.05579831e-02,
        -1.38435150e-02,
        -1.08661868e-01,
        3.51954289e-02,
        3.08089796e-02,
        -4.24132310e-02,
        5.94939776e-02,
        -1.42910302e-01,
        -1.06064267e-01,
        -1.54918917e-02,
        7.47492015e-02,
        3.01125813e-02,
    ],
    [
        -5.49706630e-02,
        -2.56577551e-01,
        3.38046532e-03,
        -2.62109768e-02,
        3.46053182e-03,
        3.19958702e-02,
        -2.32977550e-02,
        1.06992699e-01,
        3.65048349e-02,
        -1.14938579e-02,
        2.41147596e-02,
        1.78722646e-02,
        -6.00520596e-02,
        -2.07836315e-01,
        -3.58841643e-02,
        9.42891557e-03,
        -1.20395064e-01,
        -2.24002242e-01,
        7.16918781e-02,
        6.13252958e-03,
        -1.52616411e-01,
        9.79270414e-02,
        9.51304380e-03,
        -7.80423079e-03,
        3.51843461e-02,
        -2.02087825e-03,
        5.79347350e-02,
        9.06216204e-02,
        6.34061322e-02,
        8.07974041e-02,
        8.79223943e-02,
        2.69386284e-02,
        4.15796563e-02,
        9.54356641e-02,
        -1.56367552e-02,
        4.95358072e-02,
        2.23936066e-02,
        -1.48161113e-01,
        4.77491207e-02,
        -1.16927132e-01,
        -1.22935094e-01,
        6.66454211e-02,
        6.65147742e-03,
        4.95793633e-02,
        -2.86594145e-02,
        -8.88371244e-02,
        8.47476870e-02,
        -1.12113103e-01,
        3.65968831e-02,
        -8.48843977e-02,
        4.63471785e-02,
        -1.50202066e-01,
        7.47649148e-02,
        -3.32901776e-02,
        -1.21002853e-01,
        6.82717701e-03,
        6.93299025e-02,
        2.30105110e-02,
        2.02812664e-02,
        -1.77011728e-01,
        -7.61344880e-02,
        4.19304147e-02,
        4.51964028e-02,
        5.89086004e-02,
    ],
    [
        -3.05589661e-03,
        -2.84141097e-02,
        3.48402485e-02,
        -3.49879591e-03,
        -4.21396531e-02,
        -9.35640261e-02,
        1.12172426e-03,
        -4.10121717e-02,
        -1.34995505e-01,
        1.89163268e-01,
        -1.05356418e-01,
        -3.45735513e-02,
        9.55296159e-02,
        3.77157703e-02,
        -2.66526756e-03,
        -3.41253467e-02,
        7.92238116e-02,
        5.11093810e-03,
        6.98746964e-02,
        1.39420489e-02,
        4.10269350e-02,
        2.41717007e-02,
        -6.60963207e-02,
        -7.00685084e-02,
        1.27076834e-01,
        -3.78503231e-04,
        -2.65631694e-02,
        8.84968117e-02,
        -7.30285943e-02,
        -2.42174938e-02,
        -6.28530281e-03,
        -7.84529001e-02,
        -6.54548928e-02,
        -3.98608893e-02,
        -7.68247321e-02,
        2.95749139e-02,
        4.75900769e-02,
        -6.26591966e-03,
        -2.29816642e-02,
        7.04155862e-02,
        1.65450156e-01,
        9.65936482e-02,
        -9.62450057e-02,
        -1.13451563e-01,
        2.49807462e-02,
        7.87288323e-02,
        2.54543517e-02,
        3.54297422e-02,
        1.82641875e-02,
        8.64531100e-02,
        -5.35610281e-02,
        -6.42673448e-02,
        -8.13233182e-02,
        -1.64394215e-01,
        4.41612164e-03,
        -9.53919888e-02,
        -1.23024188e-01,
        1.95124686e-01,
        -1.01020709e-01,
        -1.81028713e-02,
        -4.52763215e-03,
        1.90289974e-01,
        -1.03018679e-01,
        6.13378324e-02,
    ],
    [
        2.03164965e-02,
        -1.28783034e-02,
        1.56915709e-01,
        -2.28302881e-01,
        -2.18674019e-01,
        -1.69403628e-01,
        -3.43317464e-02,
        -5.71062453e-02,
        -2.00932786e-01,
        1.02634422e-01,
        -1.70675740e-01,
        -1.64242506e-01,
        3.28097418e-02,
        -1.26686720e-02,
        -2.47295331e-02,
        -1.04981817e-01,
        3.53527395e-03,
        -9.57583450e-03,
        -1.66328810e-02,
        -3.78162079e-02,
        -1.99117372e-03,
        3.62157300e-02,
        -1.00478917e-01,
        -2.93894261e-01,
        5.21967597e-02,
        9.12461802e-02,
        6.21955190e-03,
        8.46480653e-02,
        -2.01260075e-01,
        -7.45087415e-02,
        3.89660010e-03,
        -5.30884527e-02,
        -1.75959572e-01,
        -5.09413145e-02,
        -1.15776636e-01,
        -7.38875344e-02,
        -1.37222186e-01,
        9.70569439e-03,
        -2.52907336e-01,
        -6.26076236e-02,
        4.79415767e-02,
        4.88208905e-02,
        -1.78289250e-01,
        -7.75443092e-02,
        1.24942968e-02,
        -5.00954986e-02,
        6.58255592e-02,
        -2.54024863e-02,
        -9.36867222e-02,
        -6.28813822e-03,
        -1.18536338e-01,
        -4.36928794e-02,
        -9.75316465e-02,
        -1.53731182e-01,
        3.08020767e-02,
        -2.02372164e-01,
        -2.20694929e-01,
        4.17725779e-02,
        -1.04097016e-01,
        -3.02908253e-02,
        -1.08616874e-02,
        9.93502066e-02,
        -1.76457360e-01,
        7.29627088e-02,
    ],
    [
        3.74505706e-02,
        2.06064042e-02,
        3.13613117e-02,
        -1.63494185e-01,
        -1.42661393e-01,
        -7.02571124e-03,
        1.54772094e-02,
        5.61149567e-02,
        -1.97472170e-01,
        6.47392869e-02,
        -1.19512938e-02,
        -1.23415098e-01,
        1.00456312e-01,
        -2.05193777e-02,
        7.59304874e-03,
        -4.43696082e-02,
        -8.95011518e-03,
        -1.83844585e-02,
        9.09136161e-02,
        -2.18779474e-01,
        6.14824556e-02,
        9.01133474e-03,
        -4.83177826e-02,
        -5.78250401e-02,
        1.70139626e-01,
        6.94201216e-02,
        2.46116668e-02,
        8.02062452e-02,
        -3.37841064e-02,
        7.89781753e-03,
        -2.32485891e-03,
        -1.60796253e-03,
        -6.84506968e-02,
        1.24927893e-01,
        -6.60933107e-02,
        2.86057610e-02,
        -1.33403197e-01,
        5.19046141e-03,
        -2.76156552e-02,
        6.88039437e-02,
        4.79002707e-02,
        1.17501080e-01,
        -3.38362344e-02,
        -5.72209759e-03,
        -1.21275531e-02,
        -3.06993574e-02,
        7.35765696e-02,
        3.67758833e-02,
        -2.15935349e-01,
        1.62704825e-01,
        -2.74137836e-02,
        3.44188847e-02,
        -6.47438094e-02,
        -1.49784267e-01,
        -3.88899408e-02,
        -9.38574076e-02,
        -9.04287025e-03,
        8.64861757e-02,
        -2.11102962e-02,
        -1.24048227e-02,
        4.27484252e-02,
        1.05439171e-01,
        1.05365859e-02,
        7.42015895e-03,
    ],
    [
        -6.31849468e-02,
        -1.56509742e-01,
        -8.21522325e-02,
        -8.52734689e-03,
        2.95345727e-02,
        6.00988492e-02,
        1.54712312e-02,
        8.31505656e-03,
        6.60924837e-02,
        -5.76170906e-03,
        1.47231352e-02,
        -1.01925638e-02,
        3.66058317e-03,
        -1.85219362e-01,
        -1.32483942e-02,
        6.13335632e-02,
        -4.54801470e-02,
        -1.02686927e-01,
        3.83768305e-02,
        -2.84579732e-02,
        -7.50564635e-02,
        4.49024141e-02,
        9.21211690e-02,
        1.45208156e-02,
        2.22071614e-02,
        1.20215960e-01,
        7.87861496e-02,
        5.91264712e-03,
        3.37169133e-02,
        2.16817781e-02,
        2.30376963e-02,
        3.35937389e-03,
        2.90162191e-02,
        6.49815798e-02,
        6.15080148e-02,
        2.56314711e-03,
        2.14979574e-02,
        -1.05765641e-01,
        5.93824647e-02,
        -7.01505095e-02,
        -1.53150931e-01,
        2.68461462e-02,
        8.37167539e-03,
        2.01894753e-02,
        -6.59078956e-02,
        -8.66209343e-02,
        7.45034823e-03,
        1.73131041e-02,
        2.03568116e-02,
        -1.38529733e-01,
        -5.36815403e-03,
        -4.03273329e-02,
        -3.37691978e-03,
        2.27761734e-02,
        -1.48264125e-01,
        4.82314676e-02,
        3.31180282e-02,
        -3.61413099e-02,
        5.84475361e-02,
        -1.14852041e-01,
        -8.52765888e-02,
        3.07306778e-02,
        7.29237199e-02,
        4.22000438e-02,
    ],
    [
        1.53642958e-02,
        -2.41089743e-02,
        -4.93009724e-02,
        1.61280893e-02,
        2.72412132e-02,
        6.65555894e-02,
        -1.93271469e-02,
        5.35628349e-02,
        1.55531853e-01,
        -1.74027652e-01,
        4.00610603e-02,
        -3.59797239e-04,
        -1.53769642e-01,
        -5.96679631e-04,
        -1.14602752e-01,
        1.10126454e-02,
        3.70283518e-03,
        -1.10250637e-02,
        -1.32257074e-01,
        4.53796573e-02,
        -2.69607212e-02,
        -3.38317230e-02,
        7.16404915e-02,
        4.03943770e-02,
        -1.05319746e-01,
        8.29510763e-02,
        8.84458646e-02,
        -7.87435248e-02,
        1.42834842e-01,
        1.01443760e-01,
        1.00876242e-01,
        2.87177544e-02,
        1.14658155e-01,
        -2.94987150e-02,
        1.23262905e-01,
        -7.27738217e-02,
        1.17338086e-02,
        -1.79881807e-02,
        3.80449295e-02,
        -1.07670188e-01,
        -6.37622625e-02,
        -1.15682065e-01,
        1.36804223e-01,
        9.16699842e-02,
        -2.38733385e-02,
        -7.28865340e-02,
        -1.61411434e-01,
        -9.99373719e-02,
        4.81506661e-02,
        -1.48644939e-01,
        6.90225139e-02,
        -3.29841562e-02,
        6.53930902e-02,
        1.98159143e-01,
        -5.14883623e-02,
        4.70025577e-02,
        2.95508727e-02,
        -7.91254193e-02,
        2.96855029e-02,
        4.75367485e-03,
        2.04196554e-02,
        -1.30949587e-01,
        1.56547260e-02,
        2.37450898e-02,
    ],
];
pub const VALUE_NN_B2: [f64; 32] = [
    3.91655080e-02,
    1.05931468e-01,
    5.71090579e-02,
    1.14786319e-01,
    2.41840305e-03,
    2.29769964e-02,
    5.38810268e-02,
    2.08830256e-02,
    5.21369167e-02,
    2.04378441e-02,
    7.93729201e-02,
    1.12229742e-01,
    1.44745708e-01,
    1.37636185e-01,
    1.39365062e-01,
    1.14867292e-01,
    7.96700493e-02,
    5.85483275e-02,
    3.34358998e-02,
    1.54169858e-01,
    1.50650352e-01,
    1.04915775e-01,
    1.64260760e-01,
    2.89340541e-02,
    -1.11969362e-03,
    1.88519105e-01,
    7.26082847e-02,
    9.46465731e-02,
    1.44138277e-01,
    2.54268516e-02,
    1.62663162e-01,
    5.25796823e-02,
];
pub const VALUE_NN_W3: [f64; 32] = [
    5.07012010e-01,
    1.13147140e-01,
    -1.44054532e-01,
    1.32408068e-01,
    3.97599936e-01,
    -2.06826866e-01,
    -1.93159252e-01,
    -5.33634841e-01,
    -6.59498811e-01,
    -2.09066644e-01,
    -1.06786497e-01,
    -2.29835689e-01,
    2.24877685e-01,
    1.21013761e-01,
    3.10289919e-01,
    1.33584350e-01,
    -4.11075056e-01,
    -5.33284903e-01,
    5.80585971e-02,
    -1.62226841e-01,
    1.53342247e-01,
    -1.59307420e-01,
    -1.45167440e-01,
    1.67605653e-01,
    -1.94203049e-01,
    -1.26175001e-01,
    -2.41837159e-01,
    1.29331291e-01,
    3.91655266e-01,
    1.76822171e-01,
    -1.41138300e-01,
    -1.58493966e-01,
];
pub const VALUE_NN_B3: f64 = -3.74700166e-02;
// AUTO-GENERATED END

/// 学習時と同じ正規化 + Linear(22→64) → ReLU → Linear(64→32) → ReLU
/// → Linear(32→1) → Sigmoid のforward pass（Dropoutは推論時無効なので存在しない）。
/// 出力は [0,1]（手番側の勝率相当）
pub fn value_nn_forward(features: &[f64; 22]) -> f64 {
    let mut x = [0.0f64; 22];
    for i in 0..22 {
        x[i] = (features[i] - VALUE_NN_MEAN[i]) / VALUE_NN_STD[i];
    }
    let mut h1 = [0.0f64; 64];
    for j in 0..64 {
        let mut s = VALUE_NN_B1[j];
        for i in 0..22 {
            s += VALUE_NN_W1[j][i] * x[i];
        }
        h1[j] = s.max(0.0); // ReLU
    }
    let mut h2 = [0.0f64; 32];
    for j in 0..32 {
        let mut s = VALUE_NN_B2[j];
        for i in 0..64 {
            s += VALUE_NN_W2[j][i] * h1[i];
        }
        h2[j] = s.max(0.0); // ReLU
    }
    let mut z = VALUE_NN_B3;
    for j in 0..32 {
        z += VALUE_NN_W3[j] * h2[j];
    }
    1.0 / (1.0 + (-z).exp()) // Sigmoid
}

// ---------------------------------------------------------------------------
// 戦略（strategy.rs の estimator 戦略のコピー）
// ---------------------------------------------------------------------------

const EVAL_PARTICLES: usize = 192;

/// 1手の思考予算（ms）の既定値。TSUITATE_THINK_BUDGET_MS で上書きできる。
/// このリポジトリのアリーナは 1000秒+3秒 なので既定はやや厚めに使う。
/// 本番サイト（300秒+3秒）へのデプロイ時は環境変数で絞って
/// 思考時間（=強さ）を調整する（例: 900 で v5 相当の予算）
const DEFAULT_THINK_BUDGET_MS: u64 = 2000;
/// スケール1.0の基準予算。v5 までの暗黙の実測上限（p99 ≒ 900ms）
const REFERENCE_BUDGET_MS: f64 = 900.0;

/// 思考予算（ms）。環境変数 > 既定値
fn think_budget_ms() -> u64 {
    std::env::var("TSUITATE_THINK_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THINK_BUDGET_MS)
}

/// 思考予算に比例して各種の粒子数・読み幅を決める
#[derive(Debug, Clone, Copy)]
struct SearchBudget {
    /// 推定器へ渡すスケール（粒子数・リプレイ予算）
    scale: f64,
    /// 評価に使うユニーク粒子数の上限
    eval_particles: usize,
    /// 王周辺圧力を測る粒子数
    pressure_samples: usize,
    /// valueネット（value_nn.rs）を評価する粒子数
    nn_samples: usize,
    /// 2手読みする上位候補数
    depth2_top_k: usize,
    /// 2手読みに使う粒子数
    depth2_particles: usize,
}

impl SearchBudget {
    fn from_ms(ms: u64) -> Self {
        let scale = (ms as f64 / REFERENCE_BUDGET_MS).clamp(0.25, 8.0);
        let f = |base: usize, lo: usize, hi: usize| ((base as f64 * scale) as usize).clamp(lo, hi);
        SearchBudget {
            scale,
            eval_particles: f(EVAL_PARTICLES, 48, 2048),
            pressure_samples: f(PRESSURE_SAMPLES, 8, 64),
            nn_samples: f(NN_SAMPLES, 16, 256),
            depth2_top_k: f(DEPTH2_TOP_K, 4, 32),
            depth2_particles: f(DEPTH2_PARTICLES, 16, 384),
        }
    }
}

/// 王周辺圧力を測る粒子数の基準値（スケール1.0時）
const PRESSURE_SAMPLES: usize = 16;

/// valueネットを評価する粒子数の基準値（スケール1.0時）。forward pass自体は
/// 約0.6µs/回だが、transition特徴量の利き走査が粒子×候補ごとに掛かるため
/// 圧力項（PRESSURE_SAMPLES）と同様に粒子数を絞る
const NN_SAMPLES: usize = 48;

/// 2手読み（相手応手のサンプル再評価）を行う上位候補数の基準値（スケール1.0時）。
/// 1手読みの静的リスク項は近似なので、有望手だけ実際の応手分布で検算する
const DEPTH2_TOP_K: usize = 8;
/// 2手読みに使う粒子数の基準値（1候補あたり・スケール1.0時）
const DEPTH2_PARTICLES: usize = 48;
/// 応手で詰まされる場合のペナルティ（壊滅的なのでSPSA対象にしない）
const DEPTH2_MATE_PEN: f64 = 30.0;

/// 駒交換で動く価値: 盤上価値と持ち駒価値（基本駒種）の平均。
/// 素の駒は piece_value と一致し、成駒は取られても相手の持ち駒に入るのは
/// 基本駒種ぶんなので割り引かれる（と金を取り返された反動 = (6+1)/2 = 3.5）。
/// 逆に成駒を取る側の得も同じ理由で割り引く
pub(crate) fn exchange_value(role: Role) -> f64 {
    (piece_value(role) + piece_value(unpromote_role(role))) / 2.0
}

/// 着手後の自駒の利き被覆マス数（自分に見える盤面だけの近似）。
/// 相手の駒は見えないため飛び駒は自駒にだけ遮られる楽観値
fn coverage_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let mut pieces: Vec<VisiblePiece> = view.your_pieces.clone();
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let from_usi = make_usi_square(from);
            let Some(p) = pieces.iter_mut().find(|p| p.square == from_usi) else {
                return 0.0;
            };
            if promote {
                if let Some(r) = promote_role(p.role) {
                    p.role = r;
                }
            }
            p.square = make_usi_square(to);
        }
        ShogiMove::Drop { role, to } => pieces.push(VisiblePiece {
            square: make_usi_square(to),
            role,
        }),
    }
    let mut covered: HashSet<Coord> = HashSet::new();
    for p in &pieces {
        covered.extend(move_targets(&pieces, p, view.your_color));
    }
    covered.len() as f64
}

/// 持ち駒の歩を成れる圏内（敵陣＋一段手前）へ打つ手か（1.0/0.0）。
/// 打った直後の利きは1マスだが、次に成れば利きが6マスへ広がる索敵ユニットになり、
/// 取り返されても相手に渡るのは歩1枚で反動が最小。重みは params.tokin_probe_w
fn tokin_probe(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let ShogiMove::Drop {
        role: Role::Pawn,
        to,
    } = *mv
    else {
        return 0.0;
    };
    let depth_from_back = match view.your_color {
        Color::Sente => to.rank,
        Color::Gote => 10 - to.rank,
    };
    if depth_from_back <= 4 {
        1.0
    } else {
        0.0
    }
}

/// アンチドロー（終盤の寄せ）: 増幅を始める手数（plies）
const ANTI_DRAW_START: f64 = 60.0;
/// 増幅が最大になる手数。アリーナの手数上限200の手前で全開にする
const ANTI_DRAW_FULL: f64 = 160.0;
/// リードの正規化単位（歩換算。8 ≒ 飛車1枚のリードでほぼフル増幅）
const ANTI_DRAW_LEAD_UNIT: f64 = 8.0;

/// 終盤の攻め増幅係数。手数が進むほど・素材リードがあるほど大きくなる。
/// 互角でも弱く掛けて膠着を破りにいくが、負けているときは掛けない
/// （負けているときの引き分けは0.5勝ぶんの価値がある）
fn endgame_push(move_number: u32, lead: f64) -> f64 {
    let ramp = ((f64::from(move_number) - ANTI_DRAW_START) / (ANTI_DRAW_FULL - ANTI_DRAW_START))
        .clamp(0.0, 1.0);
    (ramp * (0.3 + (lead / ANTI_DRAW_LEAD_UNIT).clamp(-0.3, 1.2))).max(0.0)
}

/// 観測から確実に分かる素材リード（歩換算・相対値）。
/// 自分の駒の増減は取った駒（持ち駒に入る）と取られた駒を両方含み、
/// 相手側は鏡像（自分が+vなら相手は-v）なので、リード = 自分の変化×2。
/// 成りは基本駒種で数える（成駒を取った得は過小評価だが単調な信号としては十分）
fn material_lead(view: &PlayerView) -> f64 {
    let current: f64 = view
        .your_pieces
        .iter()
        .map(|p| piece_value(unpromote_role(p.role)))
        .sum::<f64>()
        + view
            .your_hand
            .iter()
            .map(|(r, n)| piece_value(*r) * f64::from(*n))
            .sum::<f64>();
    let initial: f64 = Position::initial()
        .pieces()
        .filter(|(_, p)| p.color == view.your_color)
        .map(|(_, p)| piece_value(p.role))
        .sum();
    2.0 * (current - initial)
}

/// evaluate() の結果。最終スコアでなく内訳を保持し、2手読みが
/// gain を組み替えた後に同じ最終式を適用し直せるようにする
/// （min形の非線形式に対して後から線形補正すると負のgainで壊れるため）
struct EvalOut {
    /// 期待値＋バイアス項（合法確率・反則コストを含まない）
    gain: f64,
    /// 静的な取られリスク項（mover/hidden の max）の粒子加重平均。
    /// 2手読みがこの分をサンプル実測で置き換える
    risk_mean: f64,
    p_legal: f64,
    foul_cost: f64,
    /// gain のうち王手駒の除去期待値（checker_removal_w × removal_term）分。
    /// 王手中の候補にだけ入る（内訳表示用。gain には加算済み）
    checker_removal: f64,
    /// gain から引かれた捕獲の賭け分散ペナルティ
    /// （capture_bet_var_w × p_hit(1−p_hit) × E[捕獲価値|hit]）。
    /// 正の値 = そのぶん gain が減っている（内訳表示用。gain には控除済み）
    capture_bet_penalty: f64,
}

impl EvalOut {
    fn score(&self) -> f64 {
        combine_score(self.gain, self.p_legal, self.foul_cost)
    }
}

/// 最終スコア: 期待値が負の手を p_legal で割り引かない（min の形）。
/// 割り引くと「合法確率が低いほどスコアが高い」= わざと反則に寄る手が
/// 選ばれてしまう。反則しても手番は残るので悪い局面からは逃げられず、
/// 反則の価値は「次善手の価値 − 反則コスト」でしかない
fn combine_score(gain: f64, p_legal: f64, foul_cost: f64) -> f64 {
    (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost
}

/// evaluate() まわりの調整可能パラメータ。Default が現行の手調整値。
/// bin/tune.rs の SPSA がこれを最適化する（凍結版は各自のコピーを持ち依存しない）
#[derive(Debug, Clone)]
pub struct EvalParams {
    /// 王手ボーナスの基本値
    pub check_bonus: f64,
    /// 王手ボーナスの相手反則数スケール
    pub check_foul_scale: f64,
    /// 着手駒の取られリスク重み（駒を取った直後 = 位置がバレている）
    pub mover_w_captured: f64,
    /// 着手駒の取られリスク重み（静かな手）
    pub mover_w_quiet: f64,
    /// 着手駒の取られリスク重みへの加算（王手をかけた手）。王手宣言は「王を攻撃
    /// できる（マス,駒種）」まで仮説を絞らせるので、相手は反則覚悟の探り取りで
    /// 王手駒を高確率で回収できる（対人実戦: 竜の王手→2反則で竜を取られた）。
    /// 旧 mover_w_check は quiet/captured との max で不感帯があった
    /// （SPSAで勾配が立たない）ため、非負の加算に変更
    pub mover_check_extra: f64,
    /// 捕獲後の残留露見リスク（自駒価値に掛ける割合）。取ったマスは相手に
    /// 通知されるため、粒子に守り駒が見えなくても取り返しの下限リスクを敷く。
    /// 等価な取りなら安い駒で取る、というタイブレークにもなる
    /// （対人実戦: 成桂でも取れる角を竜で取って竜を回収された）
    pub capture_reveal_risk: f64,
    /// 敵陣リスク下限の「静かな進入」係数（捕獲時は 1.0）
    pub camp_known_quiet: f64,
    /// 敵陣の守られ事前確率のスケール（1.0 で 0.25/0.2/0.15）
    pub camp_scale: f64,
    /// 露出リスクの基本重み
    pub exposed_base: f64,
    /// 露出リスクの既知度係数
    pub exposed_known: f64,
    /// 初期配置から動いていない駒の既知度
    pub home_knownness: f64,
    /// 紐つき割引（着手駒）
    pub recapture_defended: f64,
    /// 紐つき割引（露出駒）
    pub exposed_defended: f64,
    /// 相手玉周辺への攻め圧力の重み
    pub attack_w: f64,
    /// 自玉周辺への相手圧力の重み
    pub pressure_w: f64,
    /// 反則コストの基本値
    pub foul_cost_base: f64,
    /// 反則コストの急峻さ（残り反則数に対する冪）
    pub foul_cost_pow: f64,
    /// 前進バイアス
    pub advance_w: f64,
    /// 成りバイアス
    pub promote_bias: f64,
    /// 打ちバイアス
    pub drop_bias: f64,
    /// p(合法) 事前確率の擬似観測数
    pub prior_weight: f64,
    /// 粒子退化時に prior_weight へ加算する上限（ユニーク粒子が減るほど事前を信じる。
    /// 少数の複製・偏った粒子への過信 = 「自信過剰な間違い」を防ぐ）
    pub prior_weight_degen: f64,
    /// 着手後に自分が当たりを付けている敵駒の価値への重み（露出リスクの鏡像）。
    /// 1手読みでは見えない「次の駒得」（飛車頭への歩打ち等）を作る手に価値を与える
    pub threat_w: f64,
    /// 桂馬の高跳び歩の餌食: 敵桂馬への攻撃マス（桂馬の直前1マス）への歩の
    /// 接近を評価する重み。桂馬は後退できないので安い歩で追い詰めれば
    /// 駒得が確定しやすい（人間レビューでの指摘: 序盤に安全に桂馬を狙う
    /// 手段として大駒より歩が優先されるべき）。threat_w は着手直後に当たりが
    /// 「付いている」手しか拾えない（1手読み）ため、複数手かけて歩を寄せる
    /// 「狙いに行く」計画性は別項として持つ
    pub knight_bait_w: f64,
    /// 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど加点。
    /// 取れても空振りでも観測が推定を絞る（情報の価値）
    pub info_bonus: f64,
    /// 大駒（飛・角）が初期位置に残っていることへのペナルティ（1枚あたり）。
    /// 初期位置の大駒は位置が予測可能で、開いた筋の背後を歩・桂で狙われる
    /// （対人50局で頻発）。展開を促す勾配を作り、動かせば消える
    pub big_home_penalty: f64,
    /// 相手の持ち駒による「打ち込み王手の受け入れ面積」への重み。
    /// 相手の持ち駒は既知（=取られた自駒）で、飛を持たれたら玉への開いた直線、
    /// 金銀なら玉の隣接空きマスがすべて王手打ちの入口になる。
    /// 持ち駒が空なら居玉でもコストゼロ（一律の玉移動推奨はしない）
    pub hand_drop_w: f64,
    /// 手戻り減点
    pub backtrack_penalty: f64,
    /// 直前に動かした駒をまた動かす手の減点（雑なシャッフルの抑制。
    /// 駒得や王手が絡む手は期待値側が勝つので実質影響しない）
    pub shuffle_penalty: f64,
    /// 【C-7 P1 で未使用化】ソフト救済粒子の評価重み減衰。フィルタ側の
    /// EPS_INFO（estimator.rs）へ統合された。SPSAベクタのレイアウト互換のため
    /// フィールドは残す（調整しても無効）
    pub soft_decay: f64,
    /// 王探しの情報利得: 粒子間で王手判定が割れる手への p(1-p) 加点
    pub king_probe_bonus: f64,
    /// 利き被覆1マスあたりの加点（自駒のみ考慮の近似被覆）
    pub coverage_w: f64,
    /// 成れる圏内への歩打ちのと金ポテンシャル加点
    pub tokin_probe_w: f64,
    /// 2手読みで静的リスク項をサンプル実測に置き換える割合（0=従来、1=全面置換）
    pub depth2_replace: f64,
    /// 2手読みで応手に王手を掛けられた場合のペナルティ
    pub depth2_check_pen: f64,
    /// 2手読みの取り返し補償の割引（取り返し自体への反撃リスクの近似）
    pub depth2_recap_discount: f64,
    /// 反則コストの残数差項: ×(相手残数/10)^pow。相手が反則上限に近いほど
    /// 自分の反則は相対的に安い（反則レースの相対価値。0=従来）
    pub foul_diff_pow: f64,
    /// 王手の反則誘発価値の上限加速: check_foul_scale 項に ×(10/相手残数)^accel。
    /// 相手が反則負けに近づくほど1回の誘発の限界価値が跳ねる（0=従来）
    pub check_limit_accel: f64,
    /// 粒子上のvalueネット（value_nn.rs、NN段階③）の重み。粒子ごとに
    /// (state特徴量16 + transition特徴量6) → 勝率相当[0,1] を推論し、
    /// 重み付き平均の (avg − 0.5) をこの係数で歩価値スケールへ換算して
    /// gain に加算する。手作り項が横並びになる静かな局面の序列付けが狙い
    /// （54手目9二香: 意味を問わない advance_bias だけで手が決まる問題）。
    /// 0 = NN無効（従来と同一挙動）
    pub value_nn_w: f64,
    /// 王手中の仮説条件付き「王手駒の除去期待値」（CheckSolver::removal_term、
    /// 歩価値スケール）の重み。王手駒のマスを取る手には+交換価値、王手駒を
    /// 盤に残す解消手には−残存脅威を、受理を条件付けた仮説の事後分布で
    /// 平均して gain へ加算する。p_legal は合法性しか平均しないため、粒子が
    /// 真の王手駒を外している局面では捕獲の価値が評価のどこにも現れない
    /// （kakutori.kif）ことへの対応。旧 CHECK_CAPTURE_P_LEGAL_FLOOR
    /// （一律0.35のp_legal下限）の置き換え。0 = 無効（従来と同一挙動）
    pub checker_removal_w: f64,
    /// 捕獲の賭け分散ペナルティの重み。p_hit(1−p_hit) × E[捕獲価値|hit] を
    /// gain から引く（王手中は無効）。占有が五分に近いマスへの高額な捕獲賭けは
    /// 空振り分岐の認識悪化（信念の前提崩壊＋進出駒の孤立）を素の期待値が
    /// 数えないことへの補正。0 = 無効（従来と同一挙動）
    pub capture_bet_var_w: f64,
}

impl Default for EvalParams {
    fn default() -> Self {
        // SPSA第2ラウンドの収束点（2026-07-14、60反復×2×40局 vs estimator_v5、
        // 共通乱数法・tuning/tune-round2.jsonl、最終中心点の追加評価 score=0.675）。
        // 第1ラウンド（2026-07-11）からの主な動き: check_bonus 大幅減
        // （0.75→0.16。王手自体より check_foul_scale 側=相手の反則蓄積で加点）、
        // prior_weight_degen 増（4.7→8.0、退化時は事前をさらに信頼）、
        // threat_w 増（0.31→0.46）、coverage_w はほぼゼロへ
        // （利き被覆の一律加点は効かず、と金・王探しの個別項が残った）
        EvalParams {
            check_bonus: 0.1619,
            check_foul_scale: 0.0983,
            mover_w_captured: 0.8042,
            mover_w_quiet: 0.7312,
            mover_check_extra: 0.0622,
            capture_reveal_risk: 0.1313,
            camp_known_quiet: 0.4472,
            camp_scale: 0.1252,
            exposed_base: 0.4576,
            exposed_known: 0.1659,
            home_knownness: 0.0027,
            recapture_defended: 0.4692,
            exposed_defended: 0.3031,
            attack_w: 0.0434,
            pressure_w: 0.0918,
            foul_cost_base: 0.637,
            foul_cost_pow: 1.3331,
            advance_w: 0.0699,
            promote_bias: 0.1466,
            drop_bias: 0.2616,
            prior_weight: 4.9065,
            prior_weight_degen: 7.9515,
            threat_w: 0.4586,
            // 新項（2026-07-19、人間レビュー指摘を受けて追加）。0 = 従来と同一挙動。
            // 未調整のため控えめな初期値。次のSPSAラウンドの調整対象
            knight_bait_w: 0.15,
            info_bonus: 0.64,
            big_home_penalty: 0.3156,
            hand_drop_w: 0.0757,
            backtrack_penalty: 0.3685,
            shuffle_penalty: 0.2996,
            soft_decay: 0.6753,
            king_probe_bonus: 0.2451,
            coverage_w: 0.0013,
            tokin_probe_w: 0.2025,
            depth2_replace: 0.6205,
            depth2_check_pen: 0.178,
            depth2_recap_discount: 0.7612,
            // 反則経済の新項（2026-07-16、オラクル測定で36ptの伸びしろを確認後に追加）。
            // 0 = 従来と同一挙動。SPSA第4ラウンド（反則経済マスク）の調整対象
            foul_diff_pow: 0.0,
            check_limit_accel: 0.0,
            // valueネット統合（2026-07-22、NN段階③フェーズ2）。NNの候補間スコア差は
            // 0.1〜0.2程度（pairwise margin=0.1で学習）なので、6.0で0.6〜1.2歩相当。
            // w選定スイープ（w=3/6/10 × 5シナリオ）: w=3はgold-checkの悪手を
            // 変えられず（17/20）、w=6で2/20に反転。王手中の反則増（dragon-check-
            // drop）は you_in_check ゲートで遮断したうえでの採用値
            value_nn_w: 6.0,
            // 仮説条件付き除去期待値（2026-07-24、p_legalフロアの置き換え）。
            // w スイープ（kakutori 捕獲率: w=0.5で10/20, w=1で19/20, w=2で18/20 /
            // dragon-check-drop: w=1で玉逃げ20/20維持・反則18→28 /
            // keima: w=1で捕獲20/20維持）から採用。挙動は「捕獲プローブ→
            // 反則観測→仮説減衰→真の捕獲」の系列で、プローブ反則が少し増える
            // 対価はアリーナの反則経済で判定した
            checker_removal_w: 1.0,
            // 捕獲賭け分散（2026-07-24、play-estimator-20260724 16手目
            // 「8八と > 8八歩打」レビューを受けて追加）。0 = 従来と同一挙動。
            // w スイープ: tokin-bet で 8g8h(と金の五分賭け) は w=1で gain 6.4→4.5
            // (1位のまま)、w=2で2.5(P*8fと拮抗1位)、w=2.5で0.8(44位まで沈み
            // P*8f を選択 = 人間レビューの意図どおり)。アリーナ104局 vs v10:
            // w=1 59.2%±9.5 / w=2 62.5%±9.3。keima 20/20・kakutori 19/20
            // （王手中ゲートで不変）を確認して 2.5 を採用
            capture_bet_var_w: 2.5,
        }
    }
}

/// SPSA用のパラメータ仕様（名前と探索範囲）。to_vec/from_vec と同じ順序
pub struct ParamSpec {
    pub name: &'static str,
    pub lo: f64,
    pub hi: f64,
}

impl EvalParams {
    pub const SPECS: [ParamSpec; 41] = [
        ParamSpec {
            name: "check_bonus",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "check_foul_scale",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "mover_w_captured",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "mover_w_quiet",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "mover_check_extra",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "capture_reveal_risk",
            lo: 0.0,
            hi: 0.6,
        },
        ParamSpec {
            name: "camp_known_quiet",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "camp_scale",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "exposed_base",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "exposed_known",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "home_knownness",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "recapture_defended",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "exposed_defended",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "attack_w",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "pressure_w",
            lo: 0.0,
            hi: 0.6,
        },
        ParamSpec {
            name: "foul_cost_base",
            lo: 0.2,
            hi: 6.0,
        },
        ParamSpec {
            name: "foul_cost_pow",
            lo: 0.5,
            hi: 3.0,
        },
        ParamSpec {
            name: "advance_w",
            lo: -0.1,
            hi: 0.3,
        },
        ParamSpec {
            name: "promote_bias",
            lo: -0.2,
            hi: 0.6,
        },
        ParamSpec {
            name: "drop_bias",
            lo: -0.5,
            hi: 0.3,
        },
        ParamSpec {
            name: "prior_weight",
            lo: 0.5,
            hi: 16.0,
        },
        ParamSpec {
            name: "prior_weight_degen",
            lo: 0.0,
            hi: 32.0,
        },
        ParamSpec {
            name: "threat_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "knight_bait_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "info_bonus",
            lo: 0.0,
            hi: 2.0,
        },
        ParamSpec {
            name: "big_home_penalty",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "hand_drop_w",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "backtrack_penalty",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "shuffle_penalty",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "soft_decay",
            lo: 0.05,
            hi: 1.0,
        },
        ParamSpec {
            name: "king_probe_bonus",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "coverage_w",
            lo: 0.0,
            hi: 0.1,
        },
        ParamSpec {
            name: "tokin_probe_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "depth2_replace",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "depth2_check_pen",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "depth2_recap_discount",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "foul_diff_pow",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "check_limit_accel",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "value_nn_w",
            lo: 0.0,
            hi: 10.0,
        },
        ParamSpec {
            name: "checker_removal_w",
            lo: 0.0,
            hi: 2.0,
        },
        ParamSpec {
            name: "capture_bet_var_w",
            lo: 0.0,
            hi: 3.0,
        },
    ];

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.check_bonus,
            self.check_foul_scale,
            self.mover_w_captured,
            self.mover_w_quiet,
            self.mover_check_extra,
            self.capture_reveal_risk,
            self.camp_known_quiet,
            self.camp_scale,
            self.exposed_base,
            self.exposed_known,
            self.home_knownness,
            self.recapture_defended,
            self.exposed_defended,
            self.attack_w,
            self.pressure_w,
            self.foul_cost_base,
            self.foul_cost_pow,
            self.advance_w,
            self.promote_bias,
            self.drop_bias,
            self.prior_weight,
            self.prior_weight_degen,
            self.threat_w,
            self.knight_bait_w,
            self.info_bonus,
            self.big_home_penalty,
            self.hand_drop_w,
            self.backtrack_penalty,
            self.shuffle_penalty,
            self.soft_decay,
            self.king_probe_bonus,
            self.coverage_w,
            self.tokin_probe_w,
            self.depth2_replace,
            self.depth2_check_pen,
            self.depth2_recap_discount,
            self.foul_diff_pow,
            self.check_limit_accel,
            self.value_nn_w,
            self.checker_removal_w,
            self.capture_bet_var_w,
        ]
    }

    pub fn from_vec(v: &[f64]) -> EvalParams {
        assert_eq!(v.len(), Self::SPECS.len());
        EvalParams {
            check_bonus: v[0],
            check_foul_scale: v[1],
            mover_w_captured: v[2],
            mover_w_quiet: v[3],
            mover_check_extra: v[4],
            capture_reveal_risk: v[5],
            camp_known_quiet: v[6],
            camp_scale: v[7],
            exposed_base: v[8],
            exposed_known: v[9],
            home_knownness: v[10],
            recapture_defended: v[11],
            exposed_defended: v[12],
            attack_w: v[13],
            pressure_w: v[14],
            foul_cost_base: v[15],
            foul_cost_pow: v[16],
            advance_w: v[17],
            promote_bias: v[18],
            drop_bias: v[19],
            prior_weight: v[20],
            prior_weight_degen: v[21],
            threat_w: v[22],
            knight_bait_w: v[23],
            info_bonus: v[24],
            big_home_penalty: v[25],
            hand_drop_w: v[26],
            backtrack_penalty: v[27],
            shuffle_penalty: v[28],
            soft_decay: v[29],
            king_probe_bonus: v[30],
            coverage_w: v[31],
            tokin_probe_w: v[32],
            depth2_replace: v[33],
            depth2_check_pen: v[34],
            depth2_recap_discount: v[35],
            foul_diff_pow: v[36],
            check_limit_accel: v[37],
            value_nn_w: v[38],
            checker_removal_w: v[39],
            capture_bet_var_w: v[40],
        }
    }
}

/// 観測履歴から相手局面を推定して指す戦略。
///
/// 候補手（自分に見える範囲の疑似合法手）を、推定粒子の平均で評価する:
/// - 駒得の期待値（その粒子でそのマスに相手駒がいるか）
/// - 反則確率（粒子上で非合法な割合）× 反則コスト（残り反則数が減るほど高い）
/// - 指した直後に取り返されるリスク（粒子上での相手の即時駒取り）
/// - 王手・詰みボーナス
pub struct EstimatorV11 {
    est: Option<Estimator>,
    book: Option<OpeningBook>,
    /// Some なら定跡をこのラインに固定する（定跡特化チューニング用）
    book_line: Option<usize>,
    params: EvalParams,
    /// 思考予算に応じた粒子数・読み幅（TSUITATE_THINK_BUDGET_MS 由来）
    budget: SearchBudget,
    /// Some なら推定器・定跡選択・タイブレークの乱数をこのシードから導出する
    /// （SPSA の共通乱数法用。None は従来どおりエントロピー由来）
    seed: Option<u64>,
    /// 評価タイブレーク用の乱数（seed があれば決定論的）
    rng: StdRng,
    /// 直近の choose 時点の内部状態（記録用）
    last_debug: Option<serde_json::Value>,
    /// 直近の choose 時点の全候補評価（スコア降順、scenario-gui 用）
    last_ranking: Option<Vec<CandidateScore>>,
}

impl EstimatorV11 {
    pub fn new() -> Self {
        Self::with_params(EvalParams::default())
    }

    /// パラメータを差し替えて作る（bin/tune.rs のSPSA評価用）
    pub fn with_params(params: EvalParams) -> Self {
        Self::with_params_line_seed(params, None, None)
    }

    /// パラメータと定跡ライン固定を指定して作る（定跡特化チューニング用）
    pub fn with_params_and_line(params: EvalParams, book_line: Option<usize>) -> Self {
        Self::with_params_line_seed(params, book_line, None)
    }

    /// シードつきで作る（SPSA の f+/f− 評価で対局条件を揃える共通乱数法用）
    pub fn with_params_line_seed(
        params: EvalParams,
        book_line: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        // valueネット重みの運用ノブ（デプロイ時の切り戻し・アブレーション用）。
        // SPSA（with_params 経由）でも env が設定されていればそちらを優先する
        let params = match std::env::var("TSUITATE_VALUE_NN_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            Some(w) => EvalParams {
                value_nn_w: w,
                ..params
            },
            None => params,
        };
        // 除去期待値項の運用ノブ（w スイープ・切り戻し用）
        let params = match std::env::var("TSUITATE_CHECKER_REMOVAL_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            Some(w) => EvalParams {
                checker_removal_w: w,
                ..params
            },
            None => params,
        };
        // 捕獲賭け分散ペナルティの運用ノブ（w スイープ・切り戻し用）
        let params = match std::env::var("TSUITATE_CAPTURE_BET_VAR_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            Some(w) => EvalParams {
                capture_bet_var_w: w,
                ..params
            },
            None => params,
        };
        EstimatorV11 {
            est: None,
            book: None,
            book_line,
            params,
            budget: SearchBudget::from_ms(think_budget_ms()),
            seed,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s ^ 0xA5A5_5A5A_DEAD_BEEF),
                None => StdRng::seed_from_u64(rand::rng().random()),
            },
            last_debug: None,
            last_ranking: None,
        }
    }
}

impl Default for EstimatorV11 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorV11 {
    fn prewarm(&mut self, view: &PlayerView, log: &ObservationLog) {
        let budget = self.budget;
        let seed = self.seed;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.update(log);
    }

    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let budget = self.budget;
        let seed = self.seed;
        // 定跡・候補ゼロで早期 return したとき前の手番のランキングが残らないように
        self.last_ranking = None;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.update(log);

        // 序盤定跡（静かな間だけ）。ブック中も推定器の update は回して粒子を保つ
        let book_line = self.book_line;
        let book = self.book.get_or_insert_with(|| match (book_line, seed) {
            (Some(idx), _) => OpeningBook::with_line(view.your_color, idx),
            (None, Some(s)) => OpeningBook::with_seed(view.your_color, s),
            (None, None) => OpeningBook::new(view.your_color),
        });
        if let Some(usi) = book.next(view, log, foul_tried) {
            return Some(usi);
        }

        let mut candidates = candidate_moves(view, foul_tried);
        if view.you_in_check {
            // 王手中: 解消しえない手は（王手駒がどこにいても）王手放置で必ず反則に
            // なるので候補から外す。全滅したら元の候補に戻す（投了よりは反則のほうが
            // 手番を失わないぶんまし。真に詰みならサーバー側で終局している）
            let filtered: Vec<_> = candidates
                .iter()
                .filter(|(_, mv)| may_resolve_check(view, mv))
                .cloned()
                .collect();
            if !filtered.is_empty() {
                candidates = filtered;
            }
        }
        if candidates.is_empty() {
            return None;
        }

        // 同一指紋の粒子は質量を畳み込んでユニーク化して評価に使う
        // （ESSリサンプリング後は複製数が事後質量。ただし p(合法) ブレンドの
        // 実効 n はユニーク数で数える = 複製は独立な証拠ではない）。
        // ソフト救済の減衰はフィルタが logw へ課金済み（EPS_INFO）。
        // 粒子尤度モデル（likelihood.rs）で真の局面に近い粒子を厚くする。
        // 相手玉の位置で層化して抽出する（stratified_sample 参照）。
        // 粒子が完全に枯渇していても、事前確率だけで安全側の評価が成り立つ
        let particle_ctx = ParticleCtx {
            // 直近で自駒が取られたマス（相手の駒がそこに着地した）
            opp_landed_last: log.events().iter().rev().find_map(|e| match e {
                Observation::OpponentMoved {
                    captured_my_piece_at: Some(sq),
                    ..
                } => parse_usi_square(sq),
                _ => None,
            }),
        };
        let sample = stratified_sample(
            est.particles(),
            est.info_miss(),
            est.phys_taint(),
            est.log_weights(),
            view.your_color,
            &particle_ctx,
            budget.eval_particles,
            &mut self.rng,
        );

        // 相手の盤上駒数の概算（取った枚数ぶん減る。相手の打ちで戻る分は無視）
        let my_captures = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Observation::MyMove {
                        captured: Some(_),
                        ..
                    }
                )
            })
            .count();
        let opp_board_n = (20 - my_captures.min(19)) as f64;

        // 直前に受理された自分の手（手戻りシャッフルの抑制に使う）
        let last_my_move = log.events().iter().rev().find_map(|e| match e {
            Observation::MyMove { usi, .. } => parse_usi(usi),
            _ => None,
        });

        // クリーン粒子が全滅しているときだけ taint 粒子を取り出す（C-7 P3 / D4:
        // 嘘の盤面だが直近まで観測と整合していた歴史なので、用途を限定すれば
        // ブラインドの手探りより役立つ）。王手ソルバーの仮説投票・玉攻め・
        // ハング回避リスクで共有する（重複計算を避ける）。
        // **上限つき**（長手数の対局で持続したブラインドはユニーク taint 粒子が
        // 数百〜数千に膨らみうる。候補手ごとに O(particles×pieces) の被覆度
        // 走査があるため無制限だと思考予算を溶かす — 125te/132te シナリオの
        // 実測で検出。重み上位だけに絞る（自己正規化する関数群なので偏りは
        // 軽微、末尾は寄与が薄い）
        let taint_pool: Vec<(&Position, f64)> = if sample.is_empty() {
            let mut pool = taint_particles(est);
            if pool.len() > TAINT_POOL_CAP {
                pool.select_nth_unstable_by(TAINT_POOL_CAP, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                pool.truncate(TAINT_POOL_CAP);
            }
            pool
        } else {
            vec![]
        };
        let opp_color = view.your_color.other();

        // 王手中は粒子に依存しない制約推論で「王手を解消する確率」を出す
        // （粒子が枯渇する終盤の反則バースト対策。check.rs 参照）。
        // taint 投票は駒得・リスク・p(合法) には混ぜない
        let mut check_solver = if view.you_in_check {
            let fouls: Vec<ShogiMove> = foul_tried.iter().filter_map(|u| parse_usi(u)).collect();
            let votes = if sample.is_empty() {
                &taint_pool
            } else {
                &sample
            };
            CheckSolver::new(view, votes, &fouls, log)
        } else {
            None
        };

        // 相手が位置を知っている自駒（露出）の地図
        let known = knownness_map(view, log, self.params.home_knownness);

        // 2手読み用: 自分が駒を取ったマス（露見）と自分の手が触れたマス
        // （estimator の my_capture_sq / my_touched_sq と同じ定義）。
        // my_fouls_this_turn はこの手番でここまでに自分が試みた反則の回数
        // （反則リトライ中は >0）。相手は反則宣言の回数を観測しているので、
        // 応手予測の my_foul_count_last_turn 特徴量として渡す
        let mut my_capture_squares: Vec<Coord> = vec![];
        let mut my_touched_squares: Vec<Coord> = vec![];
        let mut my_fouls_this_turn: u32 = 0;
        for e in log.events() {
            match e {
                Observation::MyMove { usi, captured, .. } => {
                    my_fouls_this_turn = 0;
                    if let Some(mv) = parse_usi(usi) {
                        let to = match mv {
                            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                        };
                        if captured.is_some() {
                            my_capture_squares.push(to);
                        }
                        if let ShogiMove::Board { from, .. } = mv {
                            my_touched_squares.push(from);
                        }
                        my_touched_squares.push(to);
                    }
                }
                Observation::MyFoul { .. } => my_fouls_this_turn += 1,
                _ => {}
            }
        }

        // アンチドロー: 終盤にリードがあるほど攻め項を増幅して膠着を破る。
        // 手戻り/シャッフルの減点も同時に強めて「その場で回る」手を締め出す
        let push = endgame_push(view.move_number, material_lead(view));
        let params = {
            let mut p = self.params.clone();
            if push > 0.0 {
                p.check_bonus *= 1.0 + push;
                p.attack_w *= 1.0 + push;
                p.advance_w *= 1.0 + 0.5 * push;
                p.backtrack_penalty *= 1.0 + push;
                p.shuffle_penalty *= 1.0 + push;
            }
            p
        };

        // ブラインド時の玉攻め勾配（C-7 P3 追補）+ 局所被覆度ビリーフ（追補2）:
        // taint_pool の玉位置分布だけを抽出して攻めへ使う。個々の駒種・位置は
        // 特定しない「マスへの利き枚数密度」（ユーザーの実際の推論=
        // 「５七への相手利き≥2枚の確率が低い」に対応）は blind_hang_risk が
        // 受け（ハング回避）に使う
        let blind_king_dist: Vec<(Coord, f64)> = if taint_pool.is_empty() {
            vec![]
        } else {
            taint_king_distribution(&taint_pool, opp_color)
        };
        // 着地マスごとの被覆度をキャッシュ（成り/不成の同一着地マス等での
        // 重複走査を避ける）
        let mut coverage_cache: HashMap<Coord, f64> = HashMap::new();
        // ブラインドハング回避リスクは**既定で無効**（実験用オプトイン）。
        // codex レビュー: 5g（真実利き1枚）で期待値0.03（ほぼ0と誤信）、
        // 4h（真実1枚）で期待値1.48（過大評価）という較正不良は「ノイズの多い
        // 弱い特徴」ではなく「明確な誤誘導」水準で、blind_king_attack の
        // ボーナスを重み1.0の piece_value×coverage が簡単に相殺してしまう
        // （kakunari continue の指し継ぎが 2a1c 主体の無目的手へ逆戻りした
        // 実測とも整合）。局所被覆度は玉位置と違い複数駒の相対位置が同時に
        // 正しくないと当たらない複合情報で、taint の単純な force_apply では
        // 再現できない（ユーザーの実践知見どおり）。再設計するまでは無効
        let hang_risk_enabled = std::env::var("TSUITATE_ENABLE_HANG_RISK").is_ok();
        let debug_check_enabled = std::env::var("TSUITATE_DEBUG_CHECK").is_ok();

        let rng = &mut self.rng;
        // valueネットのstate特徴量キャッシュ（sample と同じ並び。候補間で共通なので
        // 手番ごとに1回だけ計算する）
        let mut nn_state_cache: Vec<Option<[f64; VALUE_FEATURES]>> = vec![None; sample.len()];
        // 1段目: 全候補を1手読み（静的リスク項つき）で評価する。
        // (usi, mv, 内訳, gain外の補正, 1段目スコア)
        let mut scored: Vec<(String, ShogiMove, EvalOut, f64, f64)> = vec![];
        for (usi, mv) in candidates {
            let mut prior = prior_legal(view, &mv, opp_board_n);
            if view.you_in_check {
                prior *= match check_solver.as_mut() {
                    Some(solver) => solver.resolve_probability(&mv).clamp(0.02, 1.0),
                    // ソルバーが作れないときは従来の粗い事前確率
                    // （玉移動 > 取り/合駒の順）に落とす
                    None => in_check_prior(view, &mv),
                };
            }
            let mut out = evaluate(
                view,
                &mv,
                &sample,
                prior,
                &known,
                &params,
                budget,
                &mut nn_state_cache,
            );
            // 王手中: 仮説条件付きの「王手駒の除去期待値」（check.rs::removal_term）。
            // 王手駒のマスを取る手は受理された未来で脅威ごと駒を排除し、玉逃げ等の
            // 解消手は王手駒を盤に残す。この差は粒子が真の王手駒を外している局面
            // （kakutori.kif）では gain に現れないため、CheckSolver の仮説分布で
            // 補正する。gain の内側（= combine_score の p_legal 割引の内側）に
            // 置くこと: 王手中の加点を外側に置くと反則確実な手が素通りする
            // （dragon-check-drop の教訓）
            if view.you_in_check && params.checker_removal_w != 0.0 {
                if let Some(term) = check_solver
                    .as_mut()
                    .and_then(|solver| solver.removal_term(&mv))
                {
                    out.checker_removal = params.checker_removal_w * term;
                    out.gain += out.checker_removal;
                }
            }
            if debug_check_enabled && view.you_in_check {
                eprintln!(
                    "DEBUG {usi}: prior={prior:.4} gain={:.3} p_legal={:.4} foul_cost={:.3} score={:.4}",
                    out.gain,
                    out.p_legal,
                    out.foul_cost,
                    out.score()
                );
            }
            // gain の外側の補正（タイブレーク乱数・手戻り/シャッフル減点）は
            // 2手読み後の再計算でも同じ値を使うので分離して持つ
            let mut adjust = rng.random_range(0.0..0.01);
            if !blind_king_dist.is_empty() {
                // 攻め加点は p(合法) で割り引く（加点が実現するのは手が受理された
                // ときだけ）。adjust は combine_score の外側に加算されるため、
                // 割引がないと反則確実な手の攻めボーナスが反則コストを素通りで
                // 上書きする。王手中が顕著（dragon-check-drop.kif: 解消確率ゼロの
                // G*5h が信念上の敵玉 5i/4h への利きで +1.7 を得て正解の玉逃げ
                // 5c4d を逆転）だが、平時のブラインドでも taint 粒子は反則の説明
                // （打ちマス占有など）を緩和しているため同じ穴が開く
                adjust += out.p_legal
                    * BLIND_KING_ATTACK_W
                    * blind_king_attack(view, &mv, &blind_king_dist);
            }
            if hang_risk_enabled && !taint_pool.is_empty() {
                adjust -= BLIND_HANG_RISK_W
                    * blind_hang_risk(view, &mv, &taint_pool, opp_color, &mut coverage_cache);
            }
            // 手戻り（直前の手をそのまま逆に戻す）は膠着の典型なので減点。
            // 直前に動かした駒をまた動かすだけの手も雑なシャッフルとして軽く減点
            if let (
                Some(ShogiMove::Board {
                    from: pf, to: pt, ..
                }),
                ShogiMove::Board { from, to, .. },
            ) = (last_my_move, mv)
            {
                if from == pt && to == pf {
                    adjust -= params.backtrack_penalty;
                } else if from == pt {
                    adjust -= params.shuffle_penalty;
                }
            }
            let score = out.score() + adjust;
            scored.push((usi, mv, out, adjust, score));
        }

        // 2段目: 上位候補だけ相手の応手をサンプルして再評価。
        // gain 内の静的リスク項の depth2_replace 分を実測の期待損失で
        // 置き換えて（一致するなら無変化）、最終式を適用し直す
        scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        // (usi, 選択手の p_legal, スコア)
        let mut best: Option<(String, f64, f64)> = None;
        let mut ranking: Vec<CandidateScore> = vec![];
        for (i, (usi, mv, out, adjust, score)) in scored.into_iter().enumerate() {
            let depth2 = i < budget.depth2_top_k;
            let (final_gain, final_score) = if depth2 {
                let delta = depth2_delta(
                    view,
                    &mv,
                    &sample,
                    &known,
                    &my_capture_squares,
                    &my_touched_squares,
                    my_fouls_this_turn,
                    &params,
                    budget,
                    &mut *rng,
                );
                let gain2 = out.gain + params.depth2_replace * (out.risk_mean + delta);
                (
                    gain2,
                    combine_score(gain2, out.p_legal, out.foul_cost) + adjust,
                )
            } else {
                (out.gain, score)
            };
            ranking.push(CandidateScore {
                usi: usi.clone(),
                static_score: score,
                static_gain: out.gain,
                score: final_score,
                gain: final_gain,
                p_legal: out.p_legal,
                foul_cost: out.foul_cost,
                adjust,
                depth2,
                checker_removal: out.checker_removal,
                capture_bet_penalty: out.capture_bet_penalty,
            });
            if best.as_ref().is_none_or(|(_, _, s)| final_score > *s) {
                best = Some((usi, out.p_legal, final_score));
            }
        }
        ranking.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.last_ranking = Some(ranking);

        let mut debug = debug_summary(est, &sample, push);
        // 選択手の p(合法) 予測を記録へ残す（C-7 P3 の前提整備: アリーナ真実の
        // 受理/反則と突き合わせて Brier/logloss を測る。bin/analyze 参照）
        if let (Some((_, p_legal, _)), Some(obj)) = (&best, debug.as_object_mut()) {
            obj.insert(
                "p_legal".into(),
                serde_json::json!(((p_legal * 1000.0).round()) / 1000.0),
            );
        }
        self.last_debug = Some(debug);
        best.map(|(usi, _, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last_debug.clone()
    }
}

/// 評価用の粒子サンプルを相手玉の位置で**層化抽出**する。
///
/// 従来は penalty 昇順の先頭から eval_particles 件を採っていたが、層内の並びは
/// 生存順で相関しており、少数の玉位置仮説群だけで候補を評価する偏りがあった。
/// 設計（2026-07-15 のレビュー指摘対応込み）:
/// - 採用数は**必ず eval_particles 以下**（カバレッジ枠→D'Hondt式の質量比例配分）
/// - 層内は決定的シャッフルで代表抽出（生存順バイアスを切る。rng は対局シード由来）
/// - 出力は層をまたぐ**ラウンドロビン順**: 先頭 k 件しか見ない評価
///   （王周辺圧力・2手読み）でも玉位置の分布が近似される
/// - 採らなかった質量は同層の採用粒子へ再配分（層合計の重みを保存）
/// - **multiplicity 畳み込み**（C-7 P1）: ESS リサンプリング後は複製数そのものが
///   事後質量なので、同一指紋の個体は捨てずに質量 Σexp(logw) を畳み込む
///   （旧「最良個体で代表」だとリサンプリングの結果が評価時に消える —
///   2026-07-17 codex レビュー最重要指摘）
/// - 重み和は較正アンカー legacy_mass へ正規化する: info_miss 昇順の先頭
///   min(eval, unique) 件の EPS_INFO^info_miss 和（= 旧方式の soft 重み和の後継。
///   複製は独立な証拠ではないので、p(合法) ブレンドの実効 n はユニーク数で数える）
/// - 粒子尤度モデル（likelihood.rs、アリーナ真実で教師あり学習）の exp(θ·φ) を
///   乗じる: 真の局面に近い粒子ほど評価に効く。相対的な再重み付けなので
///   合計質量（較正）は変えない
/// - 推定器の観測尤度の対数重み（Estimator::log_weights、SIR の重み更新）は
///   個体質量の側で効く: 観測を「相手が指しにくい手」でしか説明できない粒子
///   （幻の角の飛び込み王手等）を粒子間で相対的に軽くする。
///   ソフト減衰はフィルタが logw へ課金済み（EPS_INFO）なのでここでは掛けない
fn stratified_sample<'a>(
    particles: &'a [Position],
    info_miss: &[u8],
    phys_taint: &[u8],
    log_weights: &[f64],
    my_color: Color,
    ctx: &ParticleCtx,
    eval_particles: usize,
    rng: &mut StdRng,
) -> Vec<(&'a Position, f64)> {
    let opp = my_color.other();
    // ユニーク化: 同一指紋の質量 logΣexp(logw) と最小 info_miss を畳み込む。
    // 物理不整合（phys_taint>0）の粒子は**通常サンプルから除外**する
    // （C-7 P3 / D4: 嘘の盤面を駒得・リスク・p(合法) に混ぜない。
    // 必要な補助評価は別途作る taint_pool を直接使う）
    struct Unique<'a> {
        pos: &'a Position,
        mass_log: f64,
        min_miss: u8,
        logl: f64,
    }
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut uniques: Vec<Unique> = vec![];
    for (i, pos) in particles.iter().enumerate() {
        if phys_taint.get(i).copied().unwrap_or(0) > 0 {
            continue;
        }
        let lw = log_weights.get(i).copied().unwrap_or(0.0);
        let miss = info_miss.get(i).copied().unwrap_or(0);
        match seen.entry(pos.fingerprint()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                let logl =
                    particle_log_weight(&particle_features(pos, my_color, ctx), &FITTED_THETA);
                e.insert(uniques.len());
                uniques.push(Unique {
                    pos,
                    mass_log: lw,
                    min_miss: miss,
                    logl,
                });
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                let u = &mut uniques[*e.get()];
                u.mass_log = logaddexp(u.mass_log, lw);
                u.min_miss = u.min_miss.min(miss);
            }
        }
    }
    if uniques.is_empty() {
        return vec![];
    }
    // 較正アンカー: 旧方式（penalty昇順の先頭 min(eval, unique) 件の soft 重み和）の
    // 後継。ソフト減衰の較正はフィルタと同じ EPS_INFO^info_miss で数える。
    // **尤度・logw 適用前**のベース重みで計るのは従来どおり（p(合法) ブレンドの
    // 実効質量 n が尤度分布に引きずられて prior_weight の較正が崩れるため —
    // 2026-07-16 レビュー指摘）。ESS リサンプリングで複製が増えても n は
    // ユニーク数でしか増えない = リサンプリングは確信を偽装しない
    let mut miss_sorted: Vec<u8> = uniques.iter().map(|u| u.min_miss).collect();
    miss_sorted.sort_unstable();
    let legacy_mass: f64 = miss_sorted
        .iter()
        .take(eval_particles)
        .map(|&m| EPS_INFO.powi(i32::from(m)))
        .sum();
    // 分布重み: 個体質量 × 粒子尤度 = exp(mass_log + logl)（オーバーフロー対策で
    // max を引く。全体スケールは最後に legacy_mass へ正規化されるので相対値だけが
    // 意味を持つ）
    let max_logl = uniques
        .iter()
        .map(|u| u.mass_log + u.logl)
        .fold(f64::MIN, f64::max);
    let uniques: Vec<(&Position, f64)> = uniques
        .into_iter()
        .map(|u| (u.pos, (u.mass_log + u.logl - max_logl).exp()))
        .collect();

    // 玉位置で層化（質量降順）
    let mut index: HashMap<Option<Coord>, usize> = HashMap::new();
    let mut strata: Vec<(Vec<(&Position, f64)>, f64)> = vec![];
    for (pos, w) in uniques {
        let k = pos.king_square(opp);
        let i = *index.entry(k).or_insert_with(|| {
            strata.push((vec![], 0.0));
            strata.len() - 1
        });
        strata[i].0.push((pos, w));
        strata[i].1 += w;
    }
    strata.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 採用枠の配分（合計は eval_particles を超えない）:
    // まずカバレッジ枠（各層 MIN_STRATUM 件まで、質量降順のラウンドロビン）、
    // 残り予算は D'Hondt（mass/(quota+1) が最大の層へ1件ずつ）で質量比例に配る
    const MIN_STRATUM: usize = 4;
    let n = strata.len();
    let mut quotas = vec![0usize; n];
    let mut budget = eval_particles;
    'coverage: for _ in 0..MIN_STRATUM {
        for i in 0..n {
            if budget == 0 {
                break 'coverage;
            }
            if quotas[i] < strata[i].0.len() {
                quotas[i] += 1;
                budget -= 1;
            }
        }
    }
    while budget > 0 {
        let mut best: Option<(usize, f64)> = None;
        for i in 0..n {
            if quotas[i] >= strata[i].0.len() {
                continue;
            }
            let score = strata[i].1 / (quotas[i] as f64 + 1.0);
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((i, score));
            }
        }
        let Some((i, _)) = best else {
            break; // 全層が member 数まで採用済み
        };
        quotas[i] += 1;
        budget -= 1;
    }

    // 層内の採用: 重み付き systematic resampling。
    // 選択確率 ∝ 重みで quota 件を等間隔に引き、各出力へ**等重み**（層質量/quota）を
    // 割り当てる。「重み比例で選び、さらに元の重みも配る」と二重適用になり
    // 低重み粒子の期待寄与を過小評価する（2026-07-15 追加レビュー指摘）。
    // 等重み割当なら任意の quota で E[粒子iの寄与] = w_i の不偏性が成り立つ
    // （同一粒子が複数スロットに乗ることもあるが合計質量は固定）。
    // 出力後に層内を一様シャッフル（等重みなので不偏のまま）して、
    // prefix利用時の生存順相関を切る
    let resampled: Vec<Vec<(&Position, f64)>> = strata
        .iter()
        .zip(&quotas)
        .map(|((members, mass), &q)| {
            if q == 0 || *mass <= 0.0 {
                return vec![];
            }
            let unit = mass / q as f64;
            let offset: f64 = rng.random_range(0.0..unit);
            let mut out: Vec<(&Position, f64)> = Vec::with_capacity(q);
            let mut cum = 0.0;
            let mut idx = 0;
            for k in 0..q {
                let target = offset + k as f64 * unit;
                while idx + 1 < members.len() && cum + members[idx].1 <= target {
                    cum += members[idx].1;
                    idx += 1;
                }
                out.push((members[idx].0, unit));
            }
            for i in (1..out.len()).rev() {
                let j = rng.random_range(0..=i);
                out.swap(i, j);
            }
            out
        })
        .collect();
    // 層をまたぐラウンドロビン出力（prefixしか見ない評価でも層化が効く）
    let max_quota = quotas.iter().copied().max().unwrap_or(0);
    let mut sample: Vec<(&Position, f64)> = vec![];
    for round in 0..max_quota {
        for stratum in &resampled {
            if let Some(&entry) = stratum.get(round) {
                sample.push(entry);
            }
        }
    }

    // 旧方式の重み和へ正規化（較正の維持）
    let sample_mass: f64 = sample.iter().map(|(_, w)| w).sum();
    if sample_mass > 0.0 {
        let norm = legacy_mass / sample_mass;
        for (_, w) in sample.iter_mut() {
            *w *= norm;
        }
    }
    sample
}

/// taint 粒子を王手ソルバー投票に使う深さの上限（それ以上は嘘が深すぎる）
const TAINT_VOTE_MAX: u8 = 6;
/// ブラインド時の玉攻めボーナスの重み（クリーン粒子全滅時のみ。
/// taint 粒子から抽出した**玉位置分布だけ**を使い、盤面の嘘は評価に入れない。
/// kakunari 実測: 玉位置信念は 91.8% で真実に集中するのに、評価が使えず
/// 無目的手を選んでいた）
const BLIND_KING_ATTACK_W: f64 = 2.0;
/// ブラインド時のハング回避リスクの重み（クリーン粒子全滅時のみ。追補2）。
/// 個々の駒種・位置を特定しない「マスへの相手利き枚数の期待値」を使い、
/// 着地マスの被覆度が高いほど期待損失（駒の価値×密度）を引く。今までは
/// 全滅すると exposed_capture_risk 等が完全に働かず、ただ取られるリスクへの
/// 認識がゼロになっていた
const BLIND_HANG_RISK_W: f64 = 1.0;
/// taint_pool の上限（重み上位のみ使用。長手数対局での計算量爆発対策）
const TAINT_POOL_CAP: usize = 256;

/// taint 粒子を指紋でユニーク化し、深度減衰つきの重みで合算して返す
/// （taint_king_distribution・taint_square_coverage の共通部品。
/// 深い taint は信用が下がるので 0.5^(taint-1) で減衰し、
/// taint > TAINT_VOTE_MAX は除外する）
fn taint_particles(est: &Estimator) -> Vec<(&Position, f64)> {
    let max_lw = est
        .log_weights()
        .iter()
        .zip(est.phys_taint())
        .filter(|&(_, &t)| t > 0 && t <= TAINT_VOTE_MAX)
        .map(|(&lw, _)| lw)
        .fold(f64::MIN, f64::max);
    if max_lw == f64::MIN {
        return vec![];
    }
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut out: Vec<(&Position, f64)> = vec![];
    for ((pos, &t), &lw) in est
        .particles()
        .iter()
        .zip(est.phys_taint())
        .zip(est.log_weights())
    {
        if t == 0 || t > TAINT_VOTE_MAX {
            continue;
        }
        let w = (lw - max_lw).exp() * 0.5f64.powi(i32::from(t) - 1);
        match seen.entry(pos.fingerprint()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(out.len());
                out.push((pos, w));
            }
            std::collections::hash_map::Entry::Occupied(e) => out[*e.get()].1 += w,
        }
    }
    out
}

/// taint 粒子から相手玉の位置分布（正規化済み）だけを抽出する。
/// 深い taint は玉位置も信用が下がるので投票と同じ減衰・上限を適用
fn taint_king_distribution(particles: &[(&Position, f64)], opp: Color) -> Vec<(Coord, f64)> {
    let mut tally: HashMap<Coord, f64> = HashMap::new();
    let mut total = 0.0f64;
    for (pos, w) in particles {
        let Some(sq) = pos.king_square(opp) else {
            continue;
        };
        *tally.entry(sq).or_insert(0.0) += w;
        total += w;
    }
    if total <= 0.0 {
        return vec![];
    }
    tally.into_iter().map(|(sq, w)| (sq, w / total)).collect()
}

/// 指定マスへの相手利き枚数の期待値（taint 粒子由来）。個々の駒種・位置は
/// 特定せず**密度だけ**を見る — kakunari 分析でのユーザーの実際の推論
/// （「５七への相手利き≥2枚の確率が低い」）に対応する部品。
/// 攻め（信念マスへ利きを作る）だけでなく受け（信念被覆度が高いマスへの
/// 着地を避ける）にも使える
fn taint_square_coverage(particles: &[(&Position, f64)], sq: Coord, opp: Color) -> f64 {
    if particles.is_empty() {
        return 0.0;
    }
    let mut total_w = 0.0f64;
    let mut weighted_count = 0.0f64;
    for (pos, w) in particles {
        let n = pos
            .pieces()
            .filter(|(from, p)| p.color == opp && pos.attacks(*from, sq))
            .count();
        weighted_count += w * n as f64;
        total_w += w;
    }
    if total_w <= 0.0 {
        0.0
    } else {
        weighted_count / total_w
    }
}

/// ブラインド時の玉攻めボーナス: 候補手の着地駒が「信念上の玉マス」へ利きを
/// 作る度合い。自駒だけの盤（相手駒は不可視なので候補手生成と同じ仮定）で
/// 着地点からの利きを判定する — taint 粒子の盤面（嘘を含む）は使わない
fn blind_king_attack(view: &PlayerView, mv: &ShogiMove, dist: &[(Coord, f64)]) -> f64 {
    if dist.is_empty() {
        return 0.0;
    }
    // 自駒だけの盤面を作って候補手を適用する
    let mut pos = Position::empty(view.your_color);
    for p in &view.your_pieces {
        let (Some(sq), role) = (parse_usi_square(&p.square), p.role) else {
            continue;
        };
        pos.set(
            sq,
            Some(crate::shogi::Piece {
                color: view.your_color,
                role,
            }),
        );
    }
    for (role, n) in &view.your_hand {
        pos.set_hand(view.your_color, *role, *n as u8);
    }
    if !pos.is_pseudo_legal(mv) {
        return 0.0;
    }
    pos.play_unchecked(mv);
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    dist.iter()
        .map(|&(k, p)| if pos.attacks(to, k) { p } else { 0.0 })
        .sum()
}

/// ブラインド時のハング回避リスク: 着地マスの taint 由来の被覆度（期待利き
/// 枚数）× 着地する自駒の価値。相手駒は不可視なので着地駒の役割（成りを
/// 反映）だけで価値を決める。取り（着地に既に自駒がある＝取られる駒がない）
/// は対象外。cache は着地マスごとの被覆度の使い回し（成り/不成の同一着地マス
/// 等で同じスキャンを繰り返さない。長手数対局での計算量対策）
fn blind_hang_risk(
    view: &PlayerView,
    mv: &ShogiMove,
    taint_pool: &[(&Position, f64)],
    opp: Color,
    cache: &mut HashMap<Coord, f64>,
) -> f64 {
    let (to, role) = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let Some(p) = view
                .your_pieces
                .iter()
                .find(|p| p.square == make_usi_square(from))
            else {
                return 0.0;
            };
            let role = if promote {
                promote_role(p.role).unwrap_or(p.role)
            } else {
                p.role
            };
            (to, role)
        }
        ShogiMove::Drop { role, to } => (to, role),
    };
    let coverage = *cache
        .entry(to)
        .or_insert_with(|| taint_square_coverage(taint_pool, to, opp));
    piece_value(role) * coverage
}

/// log(exp(a) + exp(b))（オーバーフロー安全）
fn logaddexp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// 記録用の推定サマリ: 粒子の健全性・ユニーク数・相手玉の位置分布（上位）。
/// 事後分析で「推定が外れていたのか、評価が悪かったのか」を切り分けるために残す
fn debug_summary(est: &Estimator, sample: &[(&Position, f64)], push: f64) -> serde_json::Value {
    let opp = est.my_color().other();
    // 層化で少数派にも最低枠が付くため、件数でなく重みで集計する
    let mut king_votes: HashMap<Coord, f64> = HashMap::new();
    let mut total_w = 0.0f64;
    // systematic resampling は同じ粒子を複数スロットに乗せうるので、
    // ユニーク数はスロット数（sample.len()）と別に指紋で数える
    let mut fingerprints = HashSet::new();
    for (pos, w) in sample {
        total_w += w;
        fingerprints.insert(pos.fingerprint());
        if let Some(sq) = pos.king_square(opp) {
            *king_votes.entry(sq).or_default() += w;
        }
    }
    let mut top: Vec<(Coord, f64)> = king_votes.into_iter().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let n = total_w.max(1e-9);
    let opp_king_top: Vec<serde_json::Value> = top
        .iter()
        .take(3)
        .map(|(sq, votes)| {
            serde_json::json!({
                "sq": make_usi_square(*sq),
                "p": ((votes / n) * 1000.0).round() / 1000.0,
            })
        })
        .collect();
    serde_json::json!({
        "healthy": est.healthy(),
        "unique_particles": fingerprints.len(),
        "sample_slots": sample.len(),
        "soft_particles": est.info_miss().iter().filter(|&&p| p > 0).count(),
        "taint_particles": est.phys_taint().iter().filter(|&&t| t > 0).count(),
        "ess": (est.last_ess() * 10.0).round() / 10.0,
        "resamples": est.resamples(),
        "endgame_push": (push * 100.0).round() / 100.0,
        "opp_king_top": opp_king_top,
    })
}

/// 自分に見える範囲の候補手（foul_tried を除く）。bin/analyze の検証でも使う
pub fn candidate_moves(
    view: &PlayerView,
    foul_tried: &HashSet<String>,
) -> Vec<(String, ShogiMove)> {
    let color = view.your_color;
    let mut out = vec![];
    let push = |usi: String, out: &mut Vec<(String, ShogiMove)>| {
        if !foul_tried.contains(&usi) {
            if let Some(mv) = parse_usi(&usi) {
                out.push((usi, mv));
            }
        }
    };
    for piece in &view.your_pieces {
        let Some(from) = parse_usi_square(&piece.square) else {
            continue;
        };
        for to in move_targets(&view.your_pieces, piece, color) {
            match promotion_choice(piece.role, from, to, color) {
                Promotion::None => push(make_usi_move(from, to, false), &mut out),
                Promotion::Forced => push(make_usi_move(from, to, true), &mut out),
                Promotion::Optional => {
                    // 成れるなら成る（不成が有利な局面はまれなので候補を絞る）
                    push(make_usi_move(from, to, true), &mut out);
                }
            }
        }
    }
    for (&role, &count) in &view.your_hand {
        if count == 0 {
            continue;
        }
        for to in drop_targets(&view.your_pieces, role, color) {
            if let Some(usi) = make_usi_drop(role, to) {
                push(usi, &mut out);
            }
        }
    }
    out
}

/// 自玉のマス（PlayerView の自駒リストから引く）
fn king_square(view: &PlayerView) -> Option<Coord> {
    view.your_pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square))
}

/// 王手されているとき、この手が王手を解消しうるか（自分に見える情報だけで判定）。
/// 解消手段は (a) 玉を動かす (b) 王手駒を取る (c) 合駒。王手駒の位置は不明でも
/// (b) の着地点は自玉に利きが通るマス（クイーンライン上か桂の利き元）、
/// (c) は玉と王手駒の間（クイーンライン上）に限られる。
/// どれにも該当しない手は王手放置で必ず反則になる
fn may_resolve_check(view: &PlayerView, mv: &ShogiMove) -> bool {
    let Some(king) = king_square(view) else {
        return true; // 玉が見つからないなら判定不能（除外しない）
    };
    let on_ray = |to: Coord| {
        let df = to.file - king.file;
        let dr = to.rank - king.rank;
        (df != 0 || dr != 0) && (df == 0 || dr == 0 || df.abs() == dr.abs())
    };
    // 相手の桂が自玉に利くマス（桂の王手は取るしかなく、合駒では防げない）
    let knight_source = |to: Coord| {
        let dr = match view.your_color {
            Color::Sente => -2, // 相手（後手）の桂は rank+2 へ利く → 利き元は rank-2 側
            Color::Gote => 2,
        };
        (to.file - king.file).abs() == 1 && to.rank - king.rank == dr
    };
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            if from == king {
                return true; // 玉を動かす
            }
            on_ray(to) || knight_source(to)
        }
        // 打ちは駒を取れないので合駒（ライン上）のみ
        ShogiMove::Drop { to, .. } => on_ray(to),
    }
}

/// 王手中の p(合法) 補正係数。玉移動が最も解消しやすく、
/// 取り/合駒は王手駒の位置に当たっている必要があるので低め
fn in_check_prior(view: &PlayerView, mv: &ShogiMove) -> f64 {
    match *mv {
        ShogiMove::Board { from, .. } if Some(from) == king_square(view) => 0.5,
        _ => 0.25,
    }
}

/// 観測ゼロでも成り立つ p(合法) の事前確率。
/// 経路上の「中身の見えないマス」1つごとに空である確率 q を掛ける。
/// 打ちは着地点が空である確率 q（隠れた相手駒の上に打つのが典型的な反則源）
fn prior_legal(view: &PlayerView, mv: &ShogiMove, opp_board_n: f64) -> f64 {
    let my_n = view.your_pieces.len() as f64;
    let q = (1.0 - opp_board_n / (81.0 - my_n)).clamp(0.05, 1.0);
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            let df = to.file - from.file;
            let dr = to.rank - from.rank;
            let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
            // 候補手は自駒には塞がれていないので、中間マスはすべて未知マス
            let unknown = if aligned {
                (df.abs().max(dr.abs()) - 1).max(0)
            } else {
                0 // 桂・1マス移動
            };
            q.powi(unknown as i32)
        }
        ShogiMove::Drop { .. } => q,
    }
}

/// 相手が位置を知っている自駒の地図（マス → 既知度 0.0〜1.0）。
///
/// 対人対局の分析（records/ 2026-07-08）より: 相手は (a) 自駒が死んだマス =
/// こちらの駒がいるマス、(b) 初期配置から動いていない駒、に当たりを付けて
/// 一方的に駒を回収してくる。ついたて将棋で相手に漏れる自駒の位置情報は
/// この2種類が主なので、露出リスクの重み付けに使う
/// - 1.0: 駒を取って位置が暴露し、以降動いていない駒
/// - home_knownness: 初期配置から一度も動いていない駒（相手は初期配置を知っている）
fn knownness_map(
    view: &PlayerView,
    log: &ObservationLog,
    home_knownness: f64,
) -> HashMap<Coord, f64> {
    let mut revealed: HashSet<Coord> = HashSet::new();
    let mut touched: HashSet<Coord> = HashSet::new();
    for e in log.events() {
        match e {
            Observation::MyMove { usi, captured, .. } => match parse_usi(usi) {
                Some(ShogiMove::Board { from, to, .. }) => {
                    revealed.remove(&from);
                    if captured.is_some() {
                        revealed.insert(to);
                    } else {
                        revealed.remove(&to);
                    }
                    touched.insert(from);
                    touched.insert(to);
                }
                Some(ShogiMove::Drop { to, .. }) => {
                    // 打った駒の位置は相手から見えない
                    revealed.remove(&to);
                    touched.insert(to);
                }
                None => {}
            },
            Observation::OpponentMoved {
                captured_my_piece_at: Some(sq),
                ..
            } => {
                if let Some(c) = parse_usi_square(sq) {
                    revealed.remove(&c);
                }
            }
            _ => {}
        }
    }

    let initial = Position::initial();
    let mut map = HashMap::new();
    for piece in &view.your_pieces {
        let Some(sq) = parse_usi_square(&piece.square) else {
            continue;
        };
        let k = if revealed.contains(&sq) {
            1.0
        } else if !touched.contains(&sq)
            && initial
                .piece_at(sq)
                .is_some_and(|p| p.color == view.your_color && p.role == piece.role)
        {
            home_knownness
        } else {
            0.0
        };
        if k > 0.0 {
            map.insert(sq, k);
        }
    }
    map
}

/// 敵陣のマスが（見えない駒に）守られている事前確率。
/// 粒子が枯渇・偏っていて守り駒を見落としていても、敵陣への単騎突入
/// （対人5局で歩→高価な駒の損な交換が9回）を抑えるための下限に使う
fn camp_defended_prior(to: Coord, me: Color, camp_scale: f64) -> f64 {
    let depth_from_back = match me {
        Color::Sente => to.rank,     // 相手（後手）の陣は rank 1..=3
        Color::Gote => 10 - to.rank, // 相手（先手）の陣は rank 7..=9
    };
    camp_scale
        * match depth_from_back {
            1 => 0.25,
            2 => 0.2,
            3 => 0.15,
            _ => 0.0,
        }
}

/// 候補手をユニーク粒子の加重平均で評価する（重み = ソフト救済の減衰）
fn evaluate(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    prior: f64,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
    budget: SearchBudget,
    // valueネットのstate特徴量キャッシュ（particles と同じ並び。候補間で共通なので
    // choose() が1手番ぶん保持し、最初に使う候補の評価時に遅延計算する）
    nn_state_cache: &mut [Option<[f64; VALUE_FEATURES]>],
) -> EvalOut {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0.0f64;
    let mut value_sum = 0.0;
    let mut risk_sum = 0.0;
    // 着地マスに敵駒がいた（=駒を取れた）粒子の重み。探索ボーナスの不一致度に使う
    let mut capture_hits = 0.0f64;
    // 捕獲価値の重み付き和（賭け分散ペナルティの stake = E[捕獲価値|hit] 用）
    let mut capture_value_sum = 0.0f64;
    // 王手になった粒子の重み。王探しの情報利得（判定が割れるほど価値）に使う
    let mut check_hits = 0.0f64;
    // 王周辺の圧力は粒子間の分散が小さいわりに計算が重い（9マス×利き走査）ので
    // 少数の粒子でだけ測って平均する（数は思考予算に比例）
    let pressure_samples = budget.pressure_samples;
    let mut pressure_sum = 0.0;
    let mut attack_sum = 0.0;
    let mut danger_sum = 0.0;
    let mut pressure_n = 0usize;
    // 圧力項もソフト粒子の重みで加重する（他の項と同じ扱い）
    let mut pressure_w_sum = 0.0f64;
    // valueネット（粒子=真の局面仮説ごとの勝率相当を重み付き平均）。
    // 圧力項と同じく少数の粒子でだけ測る（transition特徴量の利き走査が重い）
    let mut nn_sum = 0.0f64;
    let mut nn_w_sum = 0.0f64;
    let mut nn_n = 0usize;

    for (pi, &(pos, w)) in particles.iter().enumerate() {
        if !pos.is_legal(mv) {
            continue;
        }
        legal += w;
        let mut v = 0.0;

        // 駒得（盤上価値で数える。成駒を取れば大きい）
        let mut captured_value = 0.0;
        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    captured_value = exchange_value(p.role);
                }
            }
        }
        v += captured_value;
        if captured_value > 0.0 {
            capture_hits += w;
            capture_value_sum += w * captured_value;
        }

        let mut next = pos.clone();
        next.play_unchecked(mv);

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がり、上限（反則負け）に
        // 近づくほど1回の誘発の限界価値が跳ねるので check_limit_accel で加速する
        // （オラクル測定 2026-07-16: 王手中反則の完全知識だけで vs v6 +9.5pt）
        let gives_check = next.in_check(opp);
        if gives_check {
            let opp_fouls_left = f64::from(10u32.saturating_sub(view.fouls.opponent).max(1));
            let accel = (10.0 / opp_fouls_left).powf(params.check_limit_accel);
            v += params.check_bonus
                + params.check_foul_scale * f64::from(view.fouls.opponent) * accel;
            check_hits += w;
            if next.legal_moves().is_empty() {
                v += 1000.0; // 詰み（真の局面がこの粒子なら勝ち）
            }
        }

        // 取られリスクは「相手がこの駒の位置を知っているか」で重みを分ける。
        // 駒を取った直後は取られたマスが相手に通知される → 着手駒の位置は確実にバレて
        // いて、取り返しはほぼ実行される。それ以外の駒への当たりは相手から見えない
        // （推定はされうる）ので薄く見積もる
        let to = match *mv {
            ShogiMove::Board { to, .. } => to,
            ShogiMove::Drop { to, .. } => to,
        };
        // 相手が取れるのは1手で1枚なので、重み付きリスクの最大値だけを引く。
        // 敵陣への着手は「粒子には見えない守り駒がいる」事前確率を下限に敷く
        // （駒を取った直後は位置が確実にバレているので下限をフルに、静かな
        // 進入は相手からまだ見えないので薄く適用する）
        // 王手をかけた手は王手宣言で位置の仮説が絞られ、相手は反則覚悟の
        // 探り取りで回収に来る（人間の実証済み戦術）ので、露見扱いにする
        let mut mover_w = if captured_value > 0.0 {
            params.mover_w_captured
        } else {
            params.mover_w_quiet
        };
        if gives_check {
            mover_w += params.mover_check_extra;
        }
        let own_after = next
            .piece_at(to)
            .map(|p| exchange_value(p.role))
            .unwrap_or(0.0);
        let known_factor = if captured_value > 0.0 {
            1.0
        } else {
            params.camp_known_quiet
        };
        let mut floor = own_after * camp_defended_prior(to, me, params.camp_scale) * known_factor;
        if captured_value > 0.0 {
            // 取ったマスは相手に通知される。粒子に守りが見えなくても
            // 取り返しの残留リスクを敷く（= 等価な取りは安い駒で取る）
            floor = floor.max(own_after * params.capture_reveal_risk);
        }
        let mover_risk =
            mover_w * recapture_risk(&next, me, to, params.recapture_defended).max(floor);
        let hidden_risk = exposed_capture_risk(&next, me, Some(to), known, params);
        let risk = mover_risk.max(hidden_risk);
        v -= risk;
        risk_sum += w * risk;

        // 自分が敵駒に当たりを付けている価値（露出リスクの鏡像）。
        // 1手読みでは見えない「次の駒得」を作る手（大駒の頭への歩打ち等）に価値を与える
        v += params.threat_w * threat_value(&next, me);
        // 桂馬の高跳び歩の餌食: 歩が敵桂馬の攻撃マスへ近づくほど加点
        v += params.knight_bait_w * knight_bait_value(&next, me, mv);

        // 王の安全度と攻撃圧力（利き走査が重いので少数の粒子でだけ測って平均する）
        if pressure_n < pressure_samples {
            // 自玉の周囲に当たっている相手の利き（守り）
            pressure_sum += w * king_zone_pressure(&next, me, opp);
            // 相手玉の周囲に当たっている自分の利き（攻め）。王手にならない攻め駒の
            // 集結にも報酬を与える（王手/詰みボーナスだけだと攻めを組み立てない）
            attack_sum += w * king_zone_pressure(&next, opp, me);
            // 相手の持ち駒による王手打ちの受け入れ面積（対局実験の教訓:
            // 飛車を持たれた瞬間、玉への開いた直線はすべて即王手の入口になる）
            danger_sum += w * drop_check_danger(&next, me);
            pressure_w_sum += w;
            pressure_n += 1;
        }

        // valueネット: 学習時の規約（state=指す前の局面・指す側視点、transition=
        // その一手。docs/nn-value-phase1.md）どおり、粒子=真の局面仮説として推論する。
        // state特徴量は候補間で共通なので粒子単位にキャッシュする。
        // **自分が王手されている間は無効**: 王手回避は CheckSolver（制約推論）の
        // 領分で、NNの加点が回避プローブの反則試行を増やす実測があった
        // （dragon-check-drop で w=6 時に反則負け2/20が発生。w選定スイープ
        // 2026-07-22）。王手中の候補序列は p_legal（解消確率）が支配すべき
        if params.value_nn_w != 0.0 && !view.you_in_check && nn_n < budget.nn_samples {
            let state = nn_state_cache[pi].get_or_insert_with(|| value_features(pos, me));
            let trans = transition_features(pos, mv, &next, me);
            let mut f = [0.0f64; VALUE_FEATURES + TRANSITION_FEATURES];
            f[..VALUE_FEATURES].copy_from_slice(state);
            f[VALUE_FEATURES..].copy_from_slice(&trans);
            nn_sum += w * value_nn_forward(&f);
            nn_w_sum += w;
            nn_n += 1;
        }

        value_sum += w * v;
    }

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）。
    // 粒子が退化している（実効重みが評価上限に届かない）ほど事前の重みを
    // 増やし、少数の偏った粒子への過信を防ぐ。ソフト粒子は重みぶんしか
    // 数えないので、退化度にも自然に反映される
    let n: f64 = particles.iter().map(|(_, w)| w).sum();
    let degen = 1.0 - (n / budget.eval_particles as f64).min(1.0);
    let w = params.prior_weight + params.prior_weight_degen * degen;
    let p_legal = (legal + prior * w) / (n + w);
    // 賭け分散ペナルティの内訳（ランキング表示用に expected の外へ持ち出す）
    let mut capture_bet_penalty = 0.0;
    let expected = if legal > 0.0 {
        // 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど、
        // 指せば（取れても空でも）推定が絞れる。捕獲の期待値とは別の情報の価値
        let p_hit = capture_hits / legal;
        // 王探し: 王手判定が粒子間で割れる手は、指せば王手宣言の有無で
        // 玉位置仮説が絞れる（互角膠着で「玉が見つからない」を崩す勾配）
        let p_chk = check_hits / legal;
        // 攻め圧力は粒子の健全度でゲートする。退化した粒子は間違った玉位置に
        // 固まりやすく、「誰もいない場所への攻め」が加点され続ける
        // （対人実戦: 終盤の成桂の徘徊）。健全度が低いときは確実な項だけ残す
        let confidence = (n / budget.eval_particles as f64).min(1.0);
        // valueネット項: 勝率相当[0,1]の重み付き平均を中心化して歩価値スケールへ。
        // gain の内側（= combine_score の p_legal 割引を受ける側）に置くことで、
        // 反則確実な手への加点素通り（dragon-check-drop の教訓）を構造的に防ぐ
        let nn_term = if nn_w_sum > 0.0 {
            params.value_nn_w * (nn_sum / nn_w_sum - 0.5)
        } else {
            0.0
        };
        // 捕獲の賭け分散ペナルティ: 期待駒得が「占有が割れているマスへの
        // 大きな捕獲」1本に集中している手を凹に割り引く。素の期待値
        // p×stake は空振り分岐（賭けの前提が崩れ、進出駒だけが未知領域に
        // 残る側）の質の悪さを見ないため、信念が五分に近いほど・賭け金
        // （stake = E[捕獲価値|hit]）が大きいほど p(1−p)×stake で課金する。
        // 同じ1ビット（マスの占有）を買うなら安い駒のプローブが相対的に
        // 浮く設計（play-estimator-20260724 16手目: 8八と>8八歩打の逆転が発端）。
        // 王手中は無効: 王手駒捕獲の序列は CheckSolver（removal_term・p_legal）の
        // 領分で、五分の信念での捕獲プローブはむしろ推奨挙動（kakutori）
        if capture_hits > 0.0 && !view.you_in_check {
            capture_bet_penalty = params.capture_bet_var_w
                * p_hit
                * (1.0 - p_hit)
                * (capture_value_sum / capture_hits);
        }
        value_sum / legal - capture_bet_penalty
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + params.king_probe_bonus * p_chk * (1.0 - p_chk)
            + nn_term
            + (params.attack_w * confidence * attack_sum
                - params.pressure_w * pressure_sum
                - params.hand_drop_w * danger_sum)
                / pressure_w_sum.max(1e-9)
    } else {
        0.0
    };

    // 反則コスト: 手番は失わないが反則数を消費する。残りが少ないほど急激に高価。
    // 序盤の「安い反則で情報を得る」は低コスト側で自然に許容される。
    // 勝敗は反則レース（先に10回）なので、コストは絶対値でなく**残数差の相対価値**:
    // 相手が上限間際（残数小）なら自分の1反則は相対的に安い（foul_diff_pow で調整。
    // 0 = 従来どおり自分の残数のみ。tune-round3 の分析でスコアと反則差の相関0.75）
    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let opp_fouls_left = (10u32.saturating_sub(view.fouls.opponent)).max(1) as f64;
    let foul_cost = params.foul_cost_base
        * (10.0 / fouls_left).powf(params.foul_cost_pow)
        * (opp_fouls_left / 10.0).powf(params.foul_diff_pow);

    // 前進の弱い事前バイアス（推定が薄い序盤に駒をぶつけに行くため）
    let advance_bias = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let adv = match me {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            params.advance_w * adv + if promote { params.promote_bias } else { 0.0 }
        }
        ShogiMove::Drop { .. } => params.drop_bias,
    };

    // 大駒を初期位置に置き続けるペナルティ（この手の後に残る枚数分）。
    // 動かす手だけペナルティが軽くなるので、展開への勾配になる
    let development = -params.big_home_penalty * big_home_after(view, mv);

    // 利き被覆（広い索敵網）と、成れる圏内への歩打ち（と金ポテンシャル）。
    // どちらも粒子に依存しない自明な情報だけで計算できる
    let coverage = params.coverage_w * coverage_after(view, mv);
    let probe = params.tokin_probe_w * tokin_probe(view, mv);

    let gain = expected + advance_bias + development + coverage + probe;
    EvalOut {
        gain,
        risk_mean: if legal > 0.0 { risk_sum / legal } else { 0.0 },
        p_legal,
        foul_cost,
        checker_removal: 0.0,
        capture_bet_penalty,
    }
}

/// 2手読み: 候補手の後の相手応手の損失を方策加重の**期待値**で評価する。
/// （露見度で割引した駒損 − 取り返し補償、被王手/被詰みペナルティ）。
/// 静的リスク項（EvalOut::risk_mean）の置き換え先。値は「加点」方向（通常は負）。
///
/// 旧実装は応手を1手サンプルしていたため、低確率の大損失を引いたかどうかで
/// 候補順位が揺れた（モンテカルロノイズ）。応手の列挙と重みは既に計算している
/// ので、駒損が出る応手（自駒を取る手）は全て厳密に評価して重み平均し、
/// 静かな応手は駒損ゼロ・王手ペナルティのみを少数サンプルで近似する
#[allow(clippy::too_many_arguments)]
fn depth2_delta(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    known: &HashMap<Coord, f64>,
    my_captures: &[Coord],
    my_touched: &[Coord],
    my_fouls_this_turn: u32,
    params: &EvalParams,
    budget: SearchBudget,
    rng: &mut impl rand::Rng,
) -> f64 {
    let me = view.your_color;
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    // 被王手/被詰みの評価（clone+play が要るのでここに集約）
    let check_pen = |next2: &mut Position| -> f64 {
        if next2.in_check(me) {
            let mut p = params.depth2_check_pen;
            if next2.legal_moves().is_empty() {
                p += DEPTH2_MATE_PEN;
            }
            p
        } else {
            0.0
        }
    };
    let mut sum = 0.0;
    let mut n = 0.0;
    for (pos, w) in particles.iter().take(budget.depth2_particles) {
        if !pos.is_legal(mv) {
            continue;
        }
        let mut next = (*pos).clone();
        let my_capture = next.play_unchecked(mv);
        let gives_check = next.in_check(me.other());
        n += w;
        // この候補手で駒を取った場合、捕獲通知でそのマスは相手に露見する。
        // 応手予測の既知地点に加えないと、最有力の応手である「即時の取り返し」に
        // PREDICT_RECAPTURE_BOOST が掛からず、捕獲手を過度に楽観視してしまう
        let extended;
        let known_for_reply: &[Coord] = if my_capture.is_some() {
            extended = [my_captures, &[to]].concat();
            &extended
        } else {
            my_captures
        };
        let replies = opp_reply_weights(&next, me, known_for_reply, my_touched, my_fouls_this_turn);
        let total_rw: f64 = replies.iter().map(|(_, rw)| rw).sum();
        if replies.is_empty() || total_rw <= 0.0 {
            continue; // 応手なし（詰み/ステイルメイト）は stage1 のボーナス側で評価済み
        }
        let mut exp_delta = 0.0;
        // 静かな応手（駒損なし）: 重みを溜めて王手ペナルティだけ後でサンプル近似
        let mut quiet: Vec<(ShogiMove, f64)> = vec![];
        let mut quiet_w = 0.0;
        for (reply, rw) in &replies {
            let reply_to = match *reply {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let lost = next
                .piece_at(reply_to)
                .filter(|p| p.color == me)
                .map(|p| exchange_value(p.role))
                .unwrap_or(0.0);
            if lost <= 0.0 {
                quiet_w += rw;
                quiet.push((*reply, *rw));
                continue;
            }
            let mut next2 = next.clone();
            next2.play_unchecked(reply);
            // 露見度スケール: 着手駒は stage1 の mover_w と同じ規則、
            // それ以外の駒は exposed_capture_risk と同じ knownness 重み。
            // 粒子上の応手はこちらの駒が全部見えてしまうので、実戦で相手が
            // その取りを狙える確率で割り引く（情報非対称の担保）
            let scale = if reply_to == to {
                let mut s = if my_capture.is_some() {
                    params.mover_w_captured
                } else {
                    params.mover_w_quiet
                };
                if gives_check {
                    s += params.mover_check_extra;
                }
                s
            } else {
                let knownness = known.get(&reply_to).copied().unwrap_or(0.0);
                params.exposed_base + params.exposed_known * knownness
            };
            // 取り返し補償: 応手の駒に自分の利きが残っていれば取り返せる
            let comp = if !next2.in_check(me) && next2.is_attacked(reply_to, me) {
                params.depth2_recap_discount
                    * next2
                        .piece_at(reply_to)
                        .map(|p| exchange_value(p.role))
                        .unwrap_or(0.0)
            } else {
                0.0
            };
            let d = -scale * (lost - comp).max(0.0) - check_pen(&mut next2);
            exp_delta += rw * d;
        }
        if quiet_w > 0.0 {
            // 静かな応手の被王手率は低頻度なので2サンプルで近似する
            let samples = quiet.len().min(2);
            let mut pen = 0.0;
            for _ in 0..samples {
                let mut t = rng.random_range(0.0..quiet_w);
                let mut chosen = &quiet[quiet.len() - 1].0;
                for (r, rw) in &quiet {
                    t -= rw;
                    if t <= 0.0 {
                        chosen = r;
                        break;
                    }
                }
                let mut next2 = next.clone();
                next2.play_unchecked(chosen);
                pen += check_pen(&mut next2);
            }
            exp_delta -= quiet_w * pen / samples as f64;
        }
        sum += w * (exp_delta / total_rw);
    }
    if n > 0.0 {
        sum / n
    } else {
        0.0
    }
}

/// この手の後も初期位置に残っている自分の大駒（飛・角）の枚数
fn big_home_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let (rook_home, bishop_home) = match view.your_color {
        Color::Sente => (Coord { file: 2, rank: 8 }, Coord { file: 8, rank: 8 }),
        Color::Gote => (Coord { file: 8, rank: 2 }, Coord { file: 2, rank: 2 }),
    };
    let from = match *mv {
        ShogiMove::Board { from, .. } => Some(from),
        ShogiMove::Drop { .. } => None,
    };
    let mut n = 0.0;
    for piece in &view.your_pieces {
        let Some(sq) = parse_usi_square(&piece.square) else {
            continue;
        };
        let home = (piece.role == Role::Rook && sq == rook_home)
            || (piece.role == Role::Bishop && sq == bishop_home);
        if home && from != Some(sq) {
            n += 1.0;
        }
    }
    n
}

/// 自分が当たりを付けている敵駒の最大価値（露出リスクの鏡像）。
/// 紐つき（相手が守っている）なら取ったときに取り返されるぶん割り引く。
/// 玉への当たりは王手であり合法性・王手ボーナス側で扱うので除く
fn threat_value(pos: &Position, me: Color) -> f64 {
    let opp = me.other();
    let mut best = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != opp || piece.role == Role::King {
            continue;
        }
        if !pos.is_attacked(sq, me) {
            continue;
        }
        let defended = pos.is_attacked(sq, opp);
        let gain = exchange_value(piece.role) * if defended { 0.45 } else { 1.0 };
        best = best.max(gain);
    }
    best
}

/// 桂馬の高跳び歩の餌食: 敵桂馬への攻撃マス（桂馬の直前1マス。歩がそこに
/// いれば次に桂馬を取れる）へ、着手した歩がどれだけ近づいたかを評価する。
/// 桂馬は後退できないので、安い歩で追い詰められれば駒得がほぼ確定する
/// （人間レビューでの指摘: 序盤の桂馬狙いは大駒より歩を優先すべき）。
/// BFS距離（deduce、多段ガイドと同じ空盤近似の下限）が縮むほど指数的に
/// 加点し、攻撃マスに直接着地した手（距離0）が最大。
/// `min_moves_empty_board(..., want_promoted=false)` は「成っても不成でも
/// 良いなら最短」であり成り駒（金型移動）経由で筋を跨げてしまうため、
/// ここでは `distance_empty_board` から不成状態の距離だけを直接引く
/// （歩が本当に同じ筋を歩数だけ進む距離。筋違いの桂馬には自然に届かない）
fn knight_bait_value(next: &Position, me: Color, mv: &ShogiMove) -> f64 {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    // 着手後にそのマスにいる駒が歩でなければ関係ない（成った歩=と金も除外）
    if !next.piece_at(to).is_some_and(|p| p.role == Role::Pawn) {
        return 0.0;
    }
    let opp = me.other();
    let mut best = 0.0f64;
    for (sq, piece) in next.pieces() {
        if piece.color != opp || piece.role != Role::Knight {
            continue;
        }
        let attack_rank = match me {
            Color::Sente => sq.rank + 1,
            Color::Gote => sq.rank - 1,
        };
        if !(1..=9).contains(&attack_rank) {
            continue;
        }
        let attack_sq = Coord {
            file: sq.file,
            rank: attack_rank,
        };
        let Some(dist) = crate::deduce::distance_empty_board(Role::Pawn, me, to, attack_sq, false)
        else {
            continue;
        };
        let decay = 0.6f64.powi(dist as i32);
        best = best.max(exchange_value(Role::Knight) * decay);
    }
    best
}

/// 着手駒（マス to にいる自駒）が次の相手番で取られるリスク。
/// 紐つきなら取り返せるぶん割り引く（相手のどの駒で取るかは不明なので近似）
fn recapture_risk(pos: &Position, me: Color, to: Coord, defended_discount: f64) -> f64 {
    let opp = me.other();
    let Some(piece) = pos.piece_at(to).filter(|p| p.color == me) else {
        return 0.0;
    };
    if piece.role == Role::King || !pos.is_attacked(to, opp) {
        return 0.0;
    }
    let defended = pos.is_attacked(to, me);
    exchange_value(piece.role) * if defended { defended_discount } else { 1.0 }
}

/// 次の相手番で失いうる駒の概算: 相手の利きが当たっている自駒の最大重み付き価値。
/// 自分の利きも当たっている（紐つき）なら取り返せるぶん割り引く。
/// 相手がその駒の位置を知っているほど（knownness_map）実際に取られやすいので
/// 重みを引き上げる。位置が漏れていない駒は従来通り薄く見積もる。
/// exclude（着手駒のマス）は recapture_risk 側で別の重みで数えるので除外する。
/// 合法手の完全列挙（ピン考慮など）はコストに見合わないので利きベースの近似
fn exposed_capture_risk(
    pos: &Position,
    me: Color,
    exclude: Option<Coord>,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
) -> f64 {
    let opp = me.other();
    let mut worst = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != me || piece.role == Role::King {
            continue; // 玉が当たっているなら王手なので合法性の側で処理される
        }
        if exclude == Some(sq) {
            continue;
        }
        if !pos.is_attacked(sq, opp) {
            continue;
        }
        let defended = pos.is_attacked(sq, me);
        let knownness = known.get(&sq).copied().unwrap_or(0.0);
        let weight = params.exposed_base + params.exposed_known * knownness;
        let loss = exchange_value(piece.role)
            * if defended {
                params.exposed_defended
            } else {
                1.0
            }
            * weight;
        worst = worst.max(loss);
    }
    worst
}

/// 相手の持ち駒による「王手打ちの受け入れ面積」。
/// 相手の持ち駒はこの粒子上で正確に分かる（取られた自駒 − 打たれた駒）。
/// - 飛: 玉からの縦横の空き直線の長さ（その各マスが王手打ちの入口）
/// - 角: 斜めの空き直線の長さ
/// - 香: 相手の香が王手できる側の1直線
/// - 金/銀: 玉の隣接空きマス（打てば即王手）
/// - 歩: 玉頭の1マス
/// 持ち駒が空ならゼロ = 居玉そのものは咎めない
pub(crate) fn drop_check_danger(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me) else {
        return 0.0;
    };
    let opp = me.other();
    let on_board = |c: &Coord| (1..=9).contains(&c.file) && (1..=9).contains(&c.rank);
    let ray_len = |df: i8, dr: i8| -> f64 {
        let mut n = 0;
        let mut c = Coord {
            file: king.file + df,
            rank: king.rank + dr,
        };
        while on_board(&c) && pos.piece_at(c).is_none() {
            n += 1;
            c = Coord {
                file: c.file + df,
                rank: c.rank + dr,
            };
        }
        n as f64
    };

    let mut danger = 0.0;
    if pos.hand_count(opp, Role::Rook) > 0 {
        danger += ray_len(1, 0) + ray_len(-1, 0) + ray_len(0, 1) + ray_len(0, -1);
    }
    if pos.hand_count(opp, Role::Bishop) > 0 {
        danger += ray_len(1, 1) + ray_len(1, -1) + ray_len(-1, 1) + ray_len(-1, -1);
    }
    // 相手の香・歩は「相手から見て前へ」利くので、自玉側から見ると
    // 自分の陣の奥方向の直線・玉頭が入口になる
    let toward = if me == Color::Sente { -1 } else { 1 };
    if pos.hand_count(opp, Role::Lance) > 0 {
        danger += ray_len(0, toward);
    }
    if pos.hand_count(opp, Role::Pawn) > 0 {
        let head = Coord {
            file: king.file,
            rank: king.rank + toward,
        };
        if on_board(&head) && pos.piece_at(head).is_none() {
            danger += 1.0;
        }
    }
    let generals = pos.hand_count(opp, Role::Gold) > 0 || pos.hand_count(opp, Role::Silver) > 0;
    if generals {
        let mut air = 0.0;
        for df in -1..=1i8 {
            for dr in -1..=1i8 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let c = Coord {
                    file: king.file + df,
                    rank: king.rank + dr,
                };
                if on_board(&c) && pos.piece_at(c).is_none() {
                    air += 0.5;
                }
            }
        }
        danger += air;
    }
    danger
}

/// owner 玉の周囲8マス（と玉のマス）に当たっている by 側の利きの数
pub(crate) fn king_zone_pressure(pos: &Position, owner: Color, by: Color) -> f64 {
    let Some(king) = pos.king_square(owner) else {
        return 0.0;
    };
    let mut pressure = 0;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let c = crate::board::Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if (1..=9).contains(&c.file) && (1..=9).contains(&c.rank) && pos.is_attacked(c, by) {
                pressure += 1;
            }
        }
    }
    pressure as f64
}

impl EstimatorV11 {
    /// アリーナの共通乱数法用（凍結時に追加。挙動は with_params_line_seed と同じ）
    pub fn with_seed(seed: u64) -> Self {
        EstimatorV11::with_params_line_seed(EvalParams::default(), None, Some(seed))
    }
}
