//! estimator の凍結版 v10（2026-07-23 凍結）。
//!
//! v9 からの主な差分（NN段階③フェーズ2、ブランチ nn-value-integration）:
//! - **粒子上のvalueネットを evaluate() へ統合**: 粒子=真の局面仮説ごとに
//!   (state特徴量16 + transition特徴量6) → 勝率相当[0,1] を手書きforward pass
//!   （22→64→32→1、約0.6µs/回）で推論し、重み付き平均の中心化値を
//!   value_nn_w=6.0 で歩価値スケール化して gain に加算。手作り項が横並びに
//!   なる静かな局面の序列付けが狙い（gold-check.kif の悪手 17/20→1/20）。
//!   学習は tsuitate-nn/train.py（勝敗回帰 + pairwise補助loss w=20 m=0.1、
//!   新定石1536局 run 29918060369、4シード中 gold-check/kakudo 両正解の seed1）
//! - **王手中（you_in_check）はNN項を無効化**: NNの加点が王手回避プローブの
//!   反則試行を増やした実測（dragon-check-drop で w=6 時に反則負け2/20）への
//!   対応。王手回避は CheckSolver（制約推論）の領分
//! - NN項は combine_score の内側（p_legal 割引を受ける側）に置き、反則確実な
//!   手への加点素通り（dragon-check-drop の教訓）を構造的に防ぐ
//! - v9 凍結後の main 修正を含む: blind_king_attack 加点の p_legal 素通り修正・
//!   コードレビュー一括修正（CheckSolver決定化ほか）
//! - 凍結版は TSUITATE_VALUE_NN_W 環境変数に反応しない（挙動を固定）
//!
//! 凍結時の成績（GitHub Actions、2026-07-22〜23）:
//! vs v6 77.0%±5.8（200局）/ vs v7 69.8%±6.4（200局）/
//! vs v8 56.9%±3.4（800局合算 454-344-2）/ vs v9 57.3%±6.9（200局）
//! — 全凍結版に有意勝ち越し。scenario suite 回帰なし
//!
//! 凍結後は編集しない（シード注入等の挙動を変えない追加のみ許容）。

use std::collections::{HashMap, HashSet, VecDeque};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::board::{
    Coord, Promotion, dead_end_rank, drop_targets, make_usi_drop, make_usi_move, make_usi_square,
    move_targets, parse_usi_square, promotion_choice,
};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::{
    Piece, Position, ShogiMove, parse_usi, piece_value, promote_role, unpromote_role,
};
use crate::strategy::Strategy;


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
    /// 観測できる。opp_move_weight の特徴量として使う）
    OppMove {
        captured_at: Option<Coord>,
        gives_check: bool,
        foul_count: u32,
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
                Constraint::MyMove { .. } => self.in_check = false,
                Constraint::MyFoul { .. } => {}
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
                Constraint::MyFoul { mv } => {
                    foul_consistent(&pos, my_color, mv).then_some(0.0)
                }
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                    foul_count,
                } => sample_opp_move(
                    &mut pos,
                    my_color,
                    *captured_at,
                    Some(*gives_check),
                    *foul_count,
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
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(budget_ms);
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
        let epoch_shift = if epoch_shift == f64::MIN { 0.0 } else { epoch_shift };
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
                Constraint::MyMove { captured, gives_check, .. } => {
                    format!("MyMove(cap={captured:?},chk={gives_check})")
                }
                Constraint::MyFoul { .. } => "MyFoul".into(),
                Constraint::OppMove { captured_at, gives_check, .. } => {
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
                ..
            } => sample_opp_move(
                pos,
                self.my_color,
                *captured_at,
                None,
                *foul_count,
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
        let mut new_hist: VecDeque<Snap> =
            hist.iter().filter(|s| s.cidx <= snap.cidx).cloned().collect();
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
                hist.back().is_some_and(|s| n - s.cidx <= GRAVEYARD_MAX_SEGMENT)
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
        self.last_ess = if sum2 > 0.0 { total * total / sum2 } else { 0.0 };
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
                    if let Some(mut p) =
                        pos.piece_at(from).filter(|p| p.color == my_color)
                    {
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
fn sample_opp_move(
    pos: &mut Position,
    my_color: Color,
    captured_at: Option<Coord>,
    gives_check: Option<bool>,
    foul_count_this_turn: u32,
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
        let consistent =
            capture_ok && gives_check.is_none_or(|gc| next.in_check(my_color) == gc);
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
fn guide_boost_factor(pos: &Position, next: &Position, mv: &ShogiMove, guide: &Guide, mover: Color) -> f64 {
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
    if guide.attacks.iter().any(|&sq| sq != to && next.attacks(to, sq)) {
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
                .chain(crate::deduce::distance_empty_board(role, mover, to, target, true))
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
    rng: &mut R,
) -> Option<ShogiMove> {
    weighted_choice(
        &opp_reply_weights(pos, my_color, known_squares, my_touched),
        rng,
    )
}

/// 相手の全合法応手と方策重み（事前分布モデル＋露見マスの取り返しブースト）。
/// 2手読みの期待値評価用: サンプルせず重み付き平均を取れる
pub fn opp_reply_weights(
    pos: &Position,
    my_color: Color,
    known_squares: &[Coord],
    my_touched: &[Coord],
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
        // 2手読み予測はまだ起きていない相手の応手を当てるので、この手番の
        // 反則回数は未知（観測なし）。既定値0（実データの最頻値）を使う
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
            let actually_in_check = pos
                .king_square(my_color)
                .is_some_and(|k| pos.pieces().any(|(sq, p)| p.color == opp && pos.attacks(sq, k)));
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

/// 駒種特化ブロック（末尾10特徴量）。one-hotは成る前の駒種（unpromote）で
/// 立て、玉は全ゼロ（既存のis_king_moveが担う）。打ちはone-hotのみ
/// （dist=0, from_home=false, promoted=false）
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
    let base = crate::shogi::unpromote_role(role_raw);
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

// AUTO-GENERATED BEGIN (export_opp_move_weights.py)
// 学習: seed=0 hidden=16 val_nll=2.6029 val_top1=0.326 (7124決定点)
// 再生成: tsuitate-nn/export_opp_move_weights.py --data data/opp_move_data_piece_v2.csv --out ../tsuitate-bot/src/opp_move_nn.rs
pub const OPP_MOVE_NN_MEAN: [f64; 23] = [2.35620961e-01, 2.96360929e-03, 3.73274530e-03, 6.66909873e-01, 2.37797350e-01, 2.33220547e-01, 3.45509611e-02, 9.08421911e-03, 2.97663733e-02, 7.64817894e-02, 1.10504240e-01, 1.53451845e-01, 1.21668890e-01, 1.38100132e-01, 1.42608285e-01, 1.45957202e-01, 1.57402709e-01, 1.37834400e-01, 1.08427867e-01, 1.35118440e-01, 3.03539280e-02, 4.55895007e-01, 1.77477017e-01];
pub const OPP_MOVE_NN_STD: [f64; 23] = [7.13434398e-01, 5.42815216e-02, 6.08693138e-02, 4.74458277e-01, 4.30749178e-01, 4.25494701e-01, 1.80915147e-01, 9.44889858e-02, 1.67984039e-01, 2.63105720e-01, 3.16384524e-01, 3.66432369e-01, 4.06966567e-01, 3.45470637e-01, 3.51286083e-01, 3.52810502e-01, 3.68679315e-01, 3.45605552e-01, 3.14655155e-01, 3.42999905e-01, 1.69433326e-01, 8.62318218e-01, 3.82403940e-01];
pub const OPP_MOVE_NN_W1: [[f64; 23]; 16] = [
    [1.68057752e+00, -2.06585735e-01, -3.46413314e-01, -6.75095618e-02, -3.61814946e-01, 4.89170760e-01, 9.97593850e-02, -1.72861964e-02, -4.67493653e-01, 5.40614910e-02, -6.23640001e-01, 9.09997746e-02, -1.13709915e+00, 1.64650500e-01, 6.59997523e-01, -1.56713828e-01, -1.63836032e-01, 7.36852586e-01, -9.41342890e-01, -7.24782586e-01, 1.33954614e-01, -5.00219464e-01, -1.13039744e+00],
    [6.29862189e-01, 1.18328311e-01, 7.84092322e-02, -6.44954592e-02, -8.14876676e-01, -1.59662831e+00, -2.90393442e-01, 2.17357706e-02, 3.23224455e-01, -3.78575444e-01, -2.47903258e-01, -3.46646100e-01, -1.28789041e-02, -4.70404804e-01, 1.86269268e-01, 8.09303343e-01, -2.85762221e-01, 1.11681364e-01, -2.25908905e-01, 4.24010128e-01, -3.17685485e-01, -1.55349863e+00, -1.59390718e-01],
    [-3.32267806e-02, -2.34113768e-01, -6.11816011e-02, -6.47719383e-01, 1.22894786e-01, 1.18594038e+00, -5.31974852e-01, 1.35534778e-02, 2.68981811e-02, -8.38981986e-01, -6.15009904e-01, -7.83992052e-01, -1.03753197e+00, -1.65791973e-01, -8.44526589e-01, 6.06956601e-01, 6.04917370e-02, -3.78165752e-01, 3.14076304e-01, 6.15237653e-01, 1.77716106e-01, -9.44229245e-01, 1.04542387e+00],
    [7.06437409e-01, -4.29021567e-01, -3.04436743e-01, -6.15958691e-01, -2.01275479e-02, 1.19325817e+00, 3.23976129e-01, 3.02250870e-02, -2.15725377e-01, -1.35047510e-01, -1.84587702e-01, -1.29711151e-01, 1.39636648e+00, 5.25133967e-01, -3.77416462e-01, -5.60170710e-01, -7.53012598e-01, 1.20027270e-02, 1.03113495e-01, 7.90625632e-01, -1.54382959e-01, -2.11276293e+00, 2.62570381e-01],
    [-5.08185327e-01, -1.41633734e-01, -1.98958173e-01, -5.54285944e-01, -1.52215087e+00, -6.37395501e-01, -3.95261437e-01, -6.35767132e-02, 3.27346712e-01, 2.13020563e-01, -3.39687765e-02, -1.13912213e+00, -8.22148561e-01, -3.93080443e-01, -2.05484539e-01, 3.90591770e-01, 6.38490319e-02, -3.14000368e-01, 6.43224776e-01, 6.96602702e-01, -6.57378554e-01, -2.72943705e-01, -6.20091915e-01],
    [-5.73727310e-01, 2.01309383e-01, 5.57462931e-01, 8.33901986e-02, 2.68193245e-01, -8.06501150e-01, -3.23563069e-01, -5.14080115e-02, -2.23224491e-01, 2.04858899e-01, 8.07631910e-02, -5.71982384e-01, -3.81116569e-01, 9.75823104e-01, 1.02973238e-01, -7.56408051e-02, -3.81176680e-01, -8.80528510e-01, -2.44503945e-01, 1.91061616e-01, -7.03276038e-01, -2.04822469e+00, -6.68230355e-01],
    [-2.65224367e-01, 4.42031741e-01, -3.55208933e-01, 3.51181567e-01, 6.46746993e-01, 1.20643973e+00, -8.19212258e-01, -6.14217967e-02, 3.93599331e-01, -6.31018400e-01, 1.49573505e-01, -5.66939592e-01, 8.04115310e-02, 3.82874638e-01, -5.32039046e-01, 1.76755279e-01, -1.14762716e-01, -1.14339340e+00, 5.18626809e-01, 4.63964105e-01, -8.07530522e-01, 2.28149537e-02, 1.58136582e+00],
    [1.45075545e-01, -7.77152777e-01, -8.01855266e-01, -1.01556087e+00, -6.18919253e-01, -4.96962257e-02, -8.69018078e-01, -1.10999867e-02, 9.60666239e-01, 1.01375926e+00, -3.41467828e-01, -2.29828745e-01, -4.57022637e-02, -3.72168005e-01, 4.20006424e-01, -6.92065895e-01, 1.33217052e-01, 5.55895686e-01, 3.44091535e-01, 2.87306637e-01, -9.02838051e-01, 1.01732790e+00, -4.51661617e-01],
    [-7.65623450e-01, -2.56863594e-01, -1.65158302e-01, 2.44294792e-01, -1.01555014e+00, -1.15425766e+00, 1.88002676e-01, -5.16149215e-02, -6.97560847e-01, -2.38817394e-01, -3.78098160e-01, -1.61763501e+00, -1.79419830e-01, 5.18760383e-01, 2.73100846e-02, -3.93246680e-01, 2.24292889e-01, 2.34904617e-01, -1.90763330e+00, 4.22535807e-01, -8.57798830e-02, -1.06088674e+00, 1.67985451e+00],
    [5.85857451e-01, 1.66604549e-01, -5.29329717e-01, 1.56356847e+00, -4.07826424e-01, 1.17839420e+00, 1.16616324e-01, -2.41938293e-01, -9.20288712e-02, 6.44699693e-01, -1.81972444e-01, 9.48504150e-01, -7.24387467e-02, -6.65267527e-01, 8.15561950e-01, -5.50967129e-03, -1.53956562e-01, -1.03905320e+00, 7.42074788e-01, 5.11751175e-01, -1.53542781e+00, 7.64246106e-01, -1.42341733e+00],
    [-1.02178550e+00, -8.57861340e-02, -6.56494647e-02, 1.08028662e+00, -1.12196815e+00, -2.59085774e-01, -4.69954401e-01, -3.45728993e-02, -2.54289448e-01, -1.88605949e-01, 1.58328846e-01, 4.38516319e-01, -2.54281163e-01, -5.19412398e-01, 1.22398674e+00, 8.92853916e-01, 1.14189573e-01, 1.98473409e-01, -1.27332592e+00, -5.38068175e-01, -6.23185754e-01, -1.38415694e+00, 1.75417447e+00],
    [-6.94822848e-01, -2.32244551e-01, -3.08014899e-01, -1.96431065e+00, -1.18299775e-01, -2.95082271e-01, -1.82647407e-01, -1.74774304e-01, -5.46782196e-01, 1.89809322e-01, 8.11303973e-01, 8.36393118e-01, 1.78416267e-01, -9.62692678e-01, 3.21680337e-01, 6.79881155e-01, 3.51936109e-02, 2.45334297e-01, 1.20598853e-01, -1.17647022e-01, -7.24349022e-01, -4.52761799e-01, -2.30041802e-01],
    [6.75100088e-02, -6.78946674e-02, -2.11640149e-02, 2.53145546e-01, 9.51514915e-02, -3.35648805e-02, 1.31565645e-01, 1.69422671e-01, -1.01674646e-01, -2.83582687e-01, 4.01078016e-02, 1.47631288e-01, 1.24952450e-01, -9.11583230e-02, 4.15415950e-02, 4.55528460e-02, 4.27446067e-02, 6.16029724e-02, 1.49583265e-01, -2.34145541e-02, 1.28749877e-01, -1.05240121e-01, 8.07036906e-02],
    [-7.70846665e-01, -1.44316062e-01, -4.18785065e-01, 2.34986112e-01, 4.46781926e-02, 7.67889380e-01, -2.78781682e-01, 4.70077604e-01, -3.71531248e-01, -1.19481254e+00, 6.43863022e-01, -8.24862838e-01, -9.03639793e-02, -5.98260224e-01, -4.72538948e-01, 1.94113418e-01, 6.23870969e-01, -1.18709016e+00, 5.11245608e-01, 1.02666986e+00, -1.04374945e+00, -5.75109482e-01, -1.24971986e+00],
    [-2.03173113e+00, -1.97387516e-01, 9.32397991e-02, 4.12949681e-01, 2.35389605e-01, 3.78958620e-02, 2.64287025e-01, -1.01920938e+00, -5.17498851e-01, -4.92789179e-01, 5.53183891e-02, 4.30663794e-01, 2.45564535e-01, -2.57324249e-01, -5.72921693e-01, 6.10081792e-01, -2.57070184e-01, 7.50587106e-01, -1.34533435e-01, 1.10380328e-03, 2.70882696e-01, 1.88223556e-01, -1.22586489e+00],
    [-1.18295264e+00, 4.55853999e-01, -2.89416593e-02, -2.27135077e-01, -1.11392319e+00, 1.04249442e+00, 1.09171614e-01, -5.07883847e-01, -1.00122845e+00, 8.51731360e-01, 4.01641756e-01, 1.66902959e-01, -1.07441738e-01, -6.65358841e-01, 1.02567720e+00, 7.84739256e-01, -5.56902923e-02, 1.33724166e-02, 1.02069578e-03, -1.05423570e+00, -4.46548522e-01, 5.59315324e-01, 7.89598882e-01],
];
pub const OPP_MOVE_NN_B1: [f64; 16] = [-2.03199649e+00, -4.74606931e-01, -7.00383663e-01, -6.86869383e-01, -1.40031195e+00, -7.84394324e-01, -3.84654373e-01, -2.05938935e+00, -4.81621295e-01, -3.64736557e-01, -7.69063309e-02, -1.73976672e+00, -5.19845307e-01, -2.56025016e-01, 1.40186083e+00, -7.20858634e-01];
pub const OPP_MOVE_NN_W2: [f64; 16] = [2.54354656e-01, 1.38985917e-01, 2.43878514e-01, 2.49244422e-01, -3.27984959e-01, 2.60426015e-01, 2.17521861e-01, -3.41697901e-01, -3.73856157e-01, -2.88700551e-01, -4.90161419e-01, 3.54575515e-01, 2.85066701e-02, -3.09361517e-01, -2.48380899e-01, 2.32507974e-01];
pub const OPP_MOVE_NN_B2: f64 = -1.19723612e-03;
// AUTO-GENERATED END

/// 学習時と同じ正規化 + Linear(23→16) → ReLU → Linear(16→1) のforward pass。
/// 出力はlogit（Sigmoidではない）。呼び出し側は `clamp(-15, 15)` してから
/// `exp(logit)` として使う
pub fn opp_move_nn_forward(features: &[f64; 23]) -> f64 {
    let mut x = [0.0f64; 23];
    for i in 0..23 {
        x[i] = (features[i] - OPP_MOVE_NN_MEAN[i]) / OPP_MOVE_NN_STD[i];
    }
    let mut h = [0.0f64; 16];
    for j in 0..16 {
        let mut s = OPP_MOVE_NN_B1[j];
        for i in 0..23 {
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
    "king_moved",    // 相手玉が初期位置から動いた
    "king_advance",  // 相手玉の前進量（段。負=後退はない）
    "king_shift",    // 相手玉の横ずれ量（筋）
    "pawn_advance",  // 相手の歩（と金含む）の平均前進量
    "pieces_home",   // 初期位置に残っている相手駒の割合（0..1）
    "at_my_death",   // 直近で自駒が死んだマスに相手駒がいる（取った駒は残留しがち）
    "in_my_camp",    // 自陣（3段）内の相手駒数
    "past_mid",      // 中央線を越えて自分側にいる相手駒数（歩・玉以外）
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
pub fn particle_log_weight(features: &[f64; PARTICLE_FEATURES], theta: &[f64; PARTICLE_FEATURES]) -> f64 {
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
            let role = particle_majority_role(particles, my_color.other(), sq)
                .unwrap_or(Role::Tokin);
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
        };
        solver.enumerate(&opponent_role_counts(view, log));
        if solver.hypotheses.is_empty() {
            return None;
        }
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
        let king = self.base.king_square(self.my_color).expect("new で確認済み");
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
                    self.base.set(
                        sq,
                        Some(crate::shogi::Piece { color: opp, role }),
                    );
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
                if pos.piece_at(h.square)
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
    /// `resolve_probability`は仮説ごとの重みで平均するため、生存仮説が
    /// 多いと正しい捕獲でも確率が薄まってしまう（王手駒の粒子ビリーフが
    /// 誤っている局面では特に顕著。kakutori.kif参照）。捕獲そのものは
    /// 「当たれば王手駒を排除できる、外れても反則1回ぶんの探索コストで
    /// 済む」性質を持つ数少ない手なので、combine_score側でp_legalの
    /// フロアとして特別扱いする（strategy.rsのchoose参照）
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
    let (role, n) = counts
        .into_iter()
        .max_by(|(ra, a), (rb, b)| {
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
            (1..=BUILTIN_LINES.len()).map(|i| format!("組み込み{i}")).collect(),
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
    "material_diff",     // 自分の駒価値合計（盤上+持ち駒） − 相手の同値
    "my_hand_value",      // 自分の持ち駒価値合計
    "opp_hand_value",      // 相手の持ち駒価値合計
    "king_pressure_on_me", // 自玉周囲8マスへの相手の利き数
    "king_pressure_on_opp", // 相手玉周囲8マスへの自分の利き数
    "drop_check_danger_me", // 自玉への打ち込み王手の受け入れ面積（相手持ち駒基準）
    "drop_check_danger_opp", // 相手玉への同（自分の持ち駒基準）
    "my_in_check",        // 自分が王手されている
    "opp_in_check",        // 相手が王手されている
    "my_pieces_in_opp_camp", // 敵陣3段にいる自分の駒数（歩・と金・玉除く）
    "opp_pieces_in_my_camp", // 自陣3段にいる相手の駒数（歩・と金・玉除く）
    "my_max_hanging",      // 相手の利きが当たり自分の紐が無い自分の駒の最大価値
    "opp_max_hanging",      // 同、相手側（=自分が取れる駒の最大価値）
    "my_max_exchange_loss", // 相手に取られた場合の最悪交換損失（取り返しの補償を差し引いた後）
    "opp_max_exchange_loss", // 同、相手側（=自分が仕掛けられる最悪の交換損失）
    "ply_progress",        // 手数を100で割った進行度（局面フェーズの粗い指標）
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
        .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.min(v))))
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
    "moved_piece_value",           // 直前に着手された駒（動いた/打たれた駒）の価値
    "moved_piece_hanging_value",   // 同、紐なしで即取られる状態なら価値、そうでなければ0
    "moved_piece_exchange_loss",   // 同、紐つきでも駒種の交換で損する額（取り返しの補償を差し引いた後）
    "captured_value",              // その着手で取った相手駒の価値（打つ手・非取りなら0）
    "net_capture_then_recapture",  // captured_value − moved_piece_exchange_loss（この一手の実質損得）
    "gives_check",                 // その着手で相手に王手をかけたか
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
// value NN（value_nn.rs のコピー、新定石1536局 seed1 で学習）
// ---------------------------------------------------------------------------

// AUTO-GENERATED BEGIN (export_value_weights.py)
// 学習メタ: data=value_data_1536_v2.csv(新定石+estimator/v8/v9教師, run 29918060369) pairwise w=20 m=0.1 seed=1 (4シード中gold-check/kakudo両正解の唯一のシード)
// 再生成: tsuitate-nn/export_value_weights.py --model-dir out/ --out ../tsuitate-bot/src/value_nn.rs
pub const VALUE_NN_MEAN: [f64; 22] = [-1.70364046e+00, 9.65206909e+00, 9.86211586e+00, 1.21764040e+00, 9.45620835e-01, 2.39332724e+00, 2.41132283e+00, 1.09563582e-01, 0.00000000e+00, 5.93359292e-01, 6.92237020e-01, 3.12112188e+00, 3.15917802e+00, 3.13197279e+00, 3.13786817e+00, 4.21031326e-01, 4.96969748e+00, 1.44204342e+00, 1.53196728e+00, 1.58191979e+00, 4.99525070e-02, 1.19485430e-01];
pub const VALUE_NN_STD: [f64; 22] = [2.41784801e+01, 1.06453466e+01, 1.07022438e+01, 1.61010003e+00, 1.38687766e+00, 3.33732533e+00, 3.38806367e+00, 3.12409043e-01, 1.00000000e+00, 8.99865508e-01, 9.32459235e-01, 3.47648382e+00, 3.55359721e+00, 3.37954402e+00, 3.43225908e+00, 2.90218264e-01, 3.46273875e+00, 2.92448211e+00, 2.78401256e+00, 2.79292488e+00, 3.87853909e+00, 3.24377447e-01];
pub const VALUE_NN_W1: [[f64; 22]; 64] = [
    [-1.00923687e-01, 1.66729875e-02, -1.46716923e-01, 1.02222055e-01, -1.84369311e-01, 1.02118971e-02, -2.77199298e-02, 7.96692967e-02, -4.66431863e-39, -1.25938252e-01, 4.24873233e-02, 1.43028507e-02, 4.73585352e-02, -5.70171922e-02, -3.41898762e-02, 1.64520860e-01, 1.16580509e-01, -1.29684418e-01, 1.59333110e-01, 2.21648049e-02, -6.98912814e-02, -1.37566701e-01],
    [-1.04670905e-01, -2.37763319e-02, 9.97240376e-03, -1.14242900e-02, -1.54408589e-01, 8.72408375e-02, -1.50524080e-01, 1.45473192e-02, 5.98107956e-39, 8.63551721e-02, 2.27517281e-02, 5.14352228e-03, -4.99425791e-02, -1.29328489e-01, 4.62171948e-03, -2.61109620e-01, 4.87100072e-02, -3.46032791e-02, 8.54380876e-02, -5.96432760e-02, 9.20213833e-02, -1.74276624e-02],
    [-3.26699346e-01, 2.13599145e-01, -4.86635463e-03, 1.04778089e-01, -2.33788937e-01, 2.96498369e-02, -1.22678436e-01, 2.77540162e-02, -4.70753888e-39, 1.52637810e-01, 3.67981847e-03, -3.56192291e-02, -4.83527184e-02, 6.21935390e-02, 8.47133249e-02, -2.53621489e-01, 3.34672146e-02, -4.00072411e-02, 6.12564944e-02, 1.19650453e-01, -5.84803931e-02, -9.34275612e-02],
    [-8.12453479e-02, -3.07236742e-02, -4.28547449e-02, -9.69450362e-03, -5.76758161e-02, 3.97941098e-02, -5.40527552e-02, 1.60848781e-01, -5.81597157e-39, 5.00914827e-02, 7.81800002e-02, 3.56837958e-02, -6.53472394e-02, -1.37671471e-01, -3.59134525e-02, -8.78390446e-02, -9.33302473e-03, -1.65235363e-02, 1.15974277e-01, -3.07752937e-01, -2.71967292e-01, -3.75266075e-02],
    [2.85661608e-01, -1.10068128e-01, 1.67150259e-01, 2.79542208e-02, 4.05964267e-04, 7.83532262e-02, 2.65130249e-04, -2.70696253e-01, -1.26630157e-39, -1.05338602e-03, 4.52820025e-02, 1.39984384e-01, 9.95759070e-02, -7.88808018e-02, 1.05294257e-01, 2.23132335e-02, -2.00979579e-02, -2.35597372e-01, 1.48429751e-01, -9.23062023e-03, 7.90761709e-02, 1.36790201e-01],
    [2.44258299e-01, 1.02843270e-01, 1.32485390e-01, -5.05638160e-02, 1.43476307e-01, -1.73095748e-01, 1.22547122e-02, -6.96946606e-02, -2.26040513e-39, -9.51011106e-02, 9.94031429e-02, -7.24154105e-03, 5.33211678e-02, -3.44270878e-02, 1.37841702e-01, -3.77944916e-01, -5.93797043e-02, -6.43333793e-02, 3.77662405e-02, 1.17350429e-01, -4.80742604e-02, -5.87917641e-02],
    [2.88015723e-01, -1.15601584e-01, 1.22057207e-01, -1.57976404e-01, 2.05545738e-01, -1.12534299e-01, 1.08565599e-01, -4.65364695e-01, -4.53356487e-40, -1.51486635e-01, 9.55448970e-02, 9.74728633e-03, 1.67251248e-02, -1.65777858e-02, 8.01092684e-02, 2.56697256e-02, 6.62567019e-02, -7.37951137e-03, -1.31471213e-02, 9.11736712e-02, -5.63395023e-02, 7.35253701e-03],
    [-2.13121980e-01, 1.90755203e-02, -6.86800256e-02, -1.78726893e-02, 4.49380018e-02, -1.52256014e-02, -1.44900882e-03, 1.43638169e-02, 5.96803908e-39, -6.87478809e-03, 2.78720690e-04, 1.13694761e-02, -8.73775780e-02, 4.77627676e-04, 1.87442433e-02, -6.78463876e-02, 3.89125384e-02, 5.81186637e-02, -1.09500863e-01, 2.43766874e-01, 2.53029853e-01, 9.24005210e-02],
    [-1.31904706e-02, -6.02162071e-02, 3.60837393e-02, 7.20342039e-04, -5.57637215e-02, 4.52166656e-03, -2.10547537e-01, 1.89309523e-01, -6.36133451e-40, -8.49919021e-03, -1.63355358e-02, -9.36275199e-02, -1.84894539e-02, -2.76918411e-02, 3.22648808e-02, -6.45679906e-02, -3.16974819e-02, 3.11366525e-02, 2.11700395e-01, -3.25660139e-01, -2.26731211e-01, -5.51886819e-02],
    [8.21408480e-02, -1.26118943e-01, -6.02097325e-02, -1.55026421e-01, -2.53550783e-02, -1.32425074e-02, -4.39962884e-03, -1.74005315e-01, 3.65173896e-39, -4.37920764e-02, 4.81684208e-02, -1.21759055e-02, 7.17476830e-02, -7.46302456e-02, -8.59338325e-04, 1.79726109e-01, 7.21374601e-02, 6.96465224e-02, 1.09396294e-01, -6.85764253e-02, -5.93217798e-02, 3.37634459e-02],
    [-2.11169243e-01, 1.45334378e-01, -1.36042476e-01, 9.40664336e-02, -1.01068936e-01, 2.35036928e-02, 1.44161647e-02, 9.87614691e-02, 2.56563596e-39, -1.76162105e-02, 7.40538612e-02, 1.31162899e-02, 2.14938447e-02, 7.49301985e-02, -1.09646097e-01, 1.93189718e-02, 3.17710601e-02, -2.85715275e-02, 8.60167369e-02, -9.87937674e-02, -1.23878859e-01, -7.47198462e-02],
    [-4.02185135e-02, -3.44351828e-02, -2.09460229e-01, 4.42483537e-02, -1.73164114e-01, 4.10226993e-02, -6.32549673e-02, 1.66554615e-01, 2.87131100e-39, -1.10429853e-01, -2.04560962e-02, 4.53110784e-02, -1.32578705e-02, -1.03957206e-01, 2.73088999e-02, 4.47877906e-02, -3.19452351e-03, -6.04365654e-02, 2.00353086e-01, -4.75006290e-02, -1.94138065e-01, 3.98716889e-02],
    [-1.71612874e-02, -2.27662370e-01, 1.04558337e-02, 3.52025665e-02, -8.00299346e-02, 1.34206275e-02, -8.85038823e-02, 2.06029281e-01, 3.91380279e-39, -4.43683974e-02, 8.81930441e-03, -8.28481019e-02, 3.31104510e-02, -3.12090330e-02, -3.38148028e-02, -3.75658460e-02, -5.42828394e-03, 4.15641256e-02, 1.85144559e-01, -2.62223631e-01, -2.57919580e-01, -4.82602380e-02],
    [-5.16481176e-02, -2.95583140e-02, -3.27899978e-02, 8.44074320e-03, -3.89778167e-02, 2.37240847e-02, -3.64369340e-02, 1.70005485e-01, 2.55157953e-39, 3.69676612e-02, -6.11823574e-02, -5.84250083e-03, 2.37468295e-02, -1.53162986e-01, -8.05065781e-02, -4.49626856e-02, 7.37886038e-03, 3.82932760e-02, 4.72179540e-02, -3.97794068e-01, -3.46899271e-01, -6.49453551e-02],
    [2.07847670e-01, 2.14422524e-01, 1.76534548e-01, 6.21492341e-02, -3.08003444e-02, -1.11745887e-01, 3.38878669e-02, -2.18551323e-01, -1.90879412e-39, 3.13164331e-02, 9.92107093e-02, 2.46258639e-02, -1.07798921e-02, -1.22707356e-02, -4.07860754e-03, -3.17590356e-01, -9.93933305e-02, -4.19613048e-02, -3.27335000e-02, 1.25852719e-01, -2.16554124e-02, 3.31020467e-02],
    [1.14793256e-01, -1.84910074e-02, 8.15565065e-02, 1.62536018e-02, 1.11013465e-01, -3.37298959e-02, 2.66206563e-02, -9.30463001e-02, -6.67127651e-39, 1.46014045e-03, -2.03461740e-02, -5.59170507e-02, -3.09340447e-01, 7.07906112e-02, -1.69477716e-01, 5.26452288e-02, -8.23254660e-02, -8.86958465e-02, -1.61519319e-01, 1.19195804e-01, 2.18749508e-01, 6.87287282e-03],
    [1.71032652e-01, -9.29938629e-02, 8.28976259e-02, -5.71345631e-03, 2.32915282e-01, -1.95586368e-01, 6.93952441e-02, 5.35194837e-02, 6.75469861e-39, -1.61035266e-02, 8.93239491e-03, -6.18100353e-03, -4.72235233e-02, 1.82654690e-02, -2.69730985e-01, -2.55260058e-02, -1.10349525e-02, 7.81788230e-02, -2.73148775e-01, 1.70308933e-01, 1.92327797e-01, -3.54976989e-02],
    [2.46679425e-01, 1.09598346e-01, 1.05627038e-01, -1.59024168e-02, 8.18080753e-02, -6.40223408e-03, 6.58127144e-02, -4.03822213e-01, 1.71170990e-39, 3.08757415e-03, -7.45590404e-02, 5.08171134e-03, 5.27894571e-02, -8.00664909e-03, -5.11362925e-02, -1.49302781e-01, 3.98612432e-02, 1.47037357e-01, -4.10520956e-02, -4.13234308e-02, 1.71211779e-01, -3.46940905e-02],
    [2.31632590e-01, -5.54657727e-02, 1.09133452e-01, -2.18017679e-02, 1.90986190e-02, -8.61873850e-02, 1.32306799e-01, -4.90529329e-01, 1.53711371e-39, 2.70700529e-02, 4.97720279e-02, -4.24086601e-02, 1.13210902e-01, 1.25423707e-02, -7.01555312e-02, 5.30060232e-02, -6.59596128e-03, -2.73234937e-02, 1.06565900e-01, 9.19215605e-02, -1.21307023e-01, -1.64445583e-02],
    [-4.18255180e-02, -2.79543474e-02, -2.31605838e-03, -3.11663803e-02, -7.28172390e-03, -2.26535387e-02, 1.93771981e-02, 4.67207506e-02, -6.74308044e-39, -5.73010044e-03, 2.23932955e-02, 1.27568888e-02, 1.11479880e-02, -2.41517723e-02, -4.27768230e-02, 2.51244698e-02, 6.93250168e-03, -4.93048057e-02, -2.83682555e-01, 3.63162726e-01, 7.15847313e-01, -6.34577274e-02],
    [-1.87143460e-01, 1.20887563e-01, -3.23724687e-01, 1.20394334e-01, -3.33430395e-02, 1.75129790e-02, 7.43883476e-02, 7.02937171e-02, -5.94118039e-39, 5.52637465e-02, -4.26148623e-02, 8.30206424e-02, 1.49284273e-01, 2.60623619e-02, -1.12880118e-01, 2.10723400e-01, 2.52315644e-02, -2.38424893e-02, 1.34338439e-01, -5.98098598e-02, 3.70393228e-03, -4.16418277e-02],
    [-3.38873804e-01, 2.23661974e-01, -9.40182656e-02, 1.59203276e-01, -9.56066418e-03, 5.58342226e-02, -4.55110036e-02, -4.64807600e-02, -2.15792256e-39, 7.04928711e-02, 3.40866856e-02, 2.21612696e-02, -9.67149448e-04, 5.32086045e-02, 5.61882630e-02, -3.99965316e-01, 3.93370874e-02, -9.86221619e-03, 6.93449154e-02, -2.10091993e-02, 2.08013970e-02, -4.14094403e-02],
    [-4.17620651e-02, 1.90678462e-02, -4.25436705e-01, 1.38016582e-01, -1.06006436e-01, -3.96590587e-03, 5.24313860e-02, -3.60603593e-02, 3.59818553e-39, 6.30714074e-02, 1.58918020e-03, 7.76199028e-02, 1.43382847e-01, -5.45107294e-03, -1.98106885e-01, 2.10132524e-01, -6.59057498e-03, 3.49093713e-02, -7.36597255e-02, -9.48067233e-02, -1.06273033e-01, 1.52876100e-03],
    [-4.36633080e-02, -4.79119010e-02, 2.22364329e-02, 3.68457511e-02, -1.25229478e-01, 4.73040454e-02, 2.39994936e-02, 1.74745739e-01, 1.55481071e-40, -4.74780202e-02, -1.43093318e-02, -2.53268722e-02, -2.68386733e-02, -7.42230043e-02, 2.58825123e-02, -2.37266526e-01, 4.13651355e-02, -1.41146183e-02, 4.63267118e-02, -7.92117715e-02, -5.46741337e-02, 6.32868856e-02],
    [9.46325213e-02, -3.18863958e-01, -9.50450310e-04, -1.37978569e-01, 7.50753433e-02, 4.72222678e-02, 3.45296711e-02, -8.27984661e-02, -2.58946924e-39, 1.34602487e-01, -1.11824632e-01, 1.22959800e-02, 2.53946055e-02, -3.79468650e-02, -8.23761299e-02, 1.19994760e-01, -1.82794437e-01, 2.71599442e-02, -9.52397808e-02, 9.11377296e-02, -1.39100775e-02, 1.64786484e-02],
    [8.18433147e-03, -9.20062046e-03, 7.08013251e-02, 2.67409459e-02, 7.74563057e-03, -3.81215778e-03, 2.35458445e-02, 3.46544869e-02, -5.10900668e-39, -1.22238253e-03, -6.34793565e-03, 2.33355127e-02, -4.56429739e-03, -4.12158817e-02, -1.21232541e-02, -1.22555979e-02, 4.33060573e-03, -2.98734009e-02, -3.46587032e-01, 3.85126770e-01, 7.03713894e-01, -5.31158112e-02],
    [-2.82304399e-02, 6.94620907e-02, 1.22741638e-02, 3.11542694e-02, -2.72445716e-02, -3.82519066e-02, 7.96111226e-02, -7.63384625e-02, -3.60750277e-40, 2.39248965e-02, -9.38332155e-02, 6.42117187e-02, -1.43360868e-01, -3.17166001e-02, -6.26868084e-02, 3.81587334e-02, 3.82156037e-02, 1.07247364e-02, -1.82373092e-01, 2.23475188e-01, 2.74423003e-01, 1.39026381e-02],
    [1.80924490e-01, -4.37860727e-01, 3.79896797e-02, -6.90902993e-02, 1.15495279e-01, 9.06361863e-02, 5.55414371e-02, -1.05332434e-01, 6.25580693e-39, -4.25186642e-02, 3.82780842e-02, 6.08992800e-02, 1.47164643e-01, -1.11689135e-01, -6.62742034e-02, 2.14265764e-01, 4.42411453e-02, 2.78113652e-02, -1.39497388e-02, 1.62522886e-02, 2.45802738e-02, 4.91256751e-02],
    [3.82917047e-01, -8.93102586e-02, 2.84506500e-01, -3.97498310e-02, 1.50494203e-01, -6.16203472e-02, 1.72494426e-02, -3.28314826e-02, 6.56565784e-40, 6.33410364e-02, 8.12374279e-02, -1.58963613e-02, -9.33985226e-03, -1.76415853e-02, 1.21724419e-01, -2.82089770e-01, -7.30435178e-02, 1.88580470e-03, 3.85534577e-02, 1.27082288e-01, 1.76562965e-02, 1.76760089e-02],
    [-3.20192501e-02, -4.71395813e-03, -6.20362125e-02, 1.08912401e-02, 4.67202216e-02, -3.82500119e-03, 2.64050998e-03, -1.28166564e-02, -2.31913354e-39, 2.86384113e-02, -1.19516794e-02, 2.38392665e-03, -4.52414840e-01, 2.17541605e-02, -6.99008778e-02, 7.33244494e-02, 1.36147821e-02, -1.10214494e-01, -2.64455646e-01, 3.42307210e-01, 3.30720663e-01, -6.99970126e-02],
    [-3.34945731e-02, 7.80679211e-02, 7.73361389e-05, -2.25911766e-01, 2.06239194e-01, -1.18468665e-01, 4.41687144e-02, -2.92469054e-01, 3.86534449e-39, -1.01103932e-01, -2.02355720e-03, -4.08910699e-02, -1.52910978e-01, 6.13903850e-02, 1.37248620e-01, -1.31819420e-03, 5.96607029e-02, -1.55873924e-01, 2.03775853e-01, 2.12266505e-01, 1.37838647e-01, 3.63887325e-02],
    [-1.36533871e-01, 7.78940022e-02, 5.97492047e-02, 2.82135028e-02, -3.53755429e-02, -2.19094437e-02, -1.70350373e-01, 1.13215126e-01, 3.29471754e-39, 1.90993934e-03, 2.17079092e-02, 4.01502103e-02, -5.08067831e-02, -6.77411705e-02, 5.49842976e-02, -2.62748986e-01, -3.79982330e-02, 5.62457182e-03, 1.04739843e-03, -1.47225946e-01, -1.51991904e-01, 5.07658720e-02],
    [-2.47521326e-01, 1.85972854e-01, 1.19068682e-01, 5.18009067e-02, -1.01699509e-01, -3.19762714e-03, -1.16098307e-01, -4.46745977e-02, 1.31466038e-39, 1.91302702e-01, -3.87618728e-02, -6.50382740e-03, -6.50147200e-02, 3.44300158e-02, 7.40376189e-02, -3.84246320e-01, 6.06890544e-02, 4.43611555e-02, 1.95172280e-02, -5.59767857e-02, 8.71476829e-02, 6.40239939e-02],
    [-8.51994008e-03, -1.97662022e-02, 9.90443770e-03, -4.27032746e-02, -1.05357347e-02, -5.96962012e-02, -9.85456631e-03, 2.12799534e-02, -3.83435477e-39, -6.65674685e-03, 2.65233982e-02, 1.30216517e-02, -3.45356413e-03, -3.41454782e-02, -2.57047787e-02, -3.13644595e-02, 2.16024257e-02, 1.93234216e-02, -3.89360875e-01, 4.41351026e-01, 6.43849492e-01, 3.74279581e-02],
    [-6.86222762e-02, 2.30395645e-02, 4.99405004e-02, 1.40150845e-01, -4.44769077e-02, -1.75310839e-02, -1.25492394e-01, 2.52659991e-03, -9.49456781e-40, 1.27292663e-01, -5.01134470e-02, 9.17086005e-02, -8.37246403e-02, 3.86259868e-03, 3.72194238e-02, -2.20505357e-01, -1.62240133e-01, 6.63700253e-02, 1.72057264e-02, 3.80960368e-02, -7.24316463e-02, 5.23943491e-02],
    [-2.12425411e-01, 1.60589546e-01, -3.49357054e-02, 1.16594777e-01, 2.70268619e-02, 1.87873933e-02, 1.81502644e-02, -5.35660796e-02, 2.32191372e-39, 9.11900774e-02, -1.15363039e-02, 3.31475139e-02, 4.34209928e-02, 4.35292423e-02, 1.10619636e-02, -1.95845008e-01, -2.55516112e-01, 3.59606594e-02, 9.81382802e-02, -1.03047956e-03, -1.12782083e-01, -1.02006450e-01],
    [2.86265373e-01, -2.74554379e-02, 2.23183185e-01, 7.67817423e-02, 7.02909231e-02, 3.72503139e-02, 1.15856156e-02, -6.25951663e-02, 1.23488446e-39, -2.43693963e-02, 5.81417941e-02, 7.74898473e-03, -7.55142272e-02, 7.25522712e-02, 3.30231227e-02, -8.90976787e-02, -4.94440794e-02, -1.61364958e-01, -1.01975597e-01, -9.50183836e-04, 1.56716287e-01, 2.51922131e-01],
    [-9.56544727e-02, -3.08554340e-02, 3.47941741e-02, 1.17225766e-01, -1.02507293e-01, 5.01968190e-02, -3.35190631e-02, 1.37331352e-01, -1.52534981e-39, -7.01251347e-03, -5.78272194e-02, -4.68977988e-02, 2.06701849e-02, -1.58500299e-02, -9.15501416e-02, -9.53995809e-02, 3.12583484e-02, 1.84588693e-02, 5.24622798e-02, -4.33659315e-01, -1.56334326e-01, 2.97321286e-02],
    [-1.95112601e-02, -6.43190090e-03, 9.51408297e-02, 8.12787563e-02, -7.65385404e-02, -1.20338472e-02, -7.61647224e-02, 1.44258827e-01, -3.63348284e-40, -4.44531962e-02, -4.94554229e-02, -2.98047848e-02, 9.59063619e-02, 2.29699723e-02, -3.78781818e-02, -8.00266340e-02, -1.62791952e-01, 4.00092825e-03, 7.73675889e-02, -1.96329579e-01, -3.12629312e-01, -7.71348253e-02],
    [7.39652663e-02, -7.34126195e-02, 3.81863713e-02, -1.21294826e-01, -3.22428271e-02, -3.49163520e-03, -3.58543806e-02, 2.73058861e-01, 1.95962622e-39, 1.73385125e-02, 1.37053337e-02, 1.44984126e-02, -1.37491167e-01, -2.03310251e-02, -6.30932674e-02, -6.43914239e-03, 1.36621132e-01, -1.69843696e-02, -2.39488259e-02, 4.22108471e-02, -8.76154751e-02, -1.22304752e-01],
    [-1.80040807e-01, 1.67657986e-01, -1.91813782e-02, 2.26152074e-02, 5.00283875e-02, 1.81111433e-02, 7.88536482e-03, -1.67641963e-03, 4.08940707e-38, 4.25244272e-02, -1.69827454e-02, 8.14687647e-03, 1.02920391e-01, 5.68767115e-02, -9.10006985e-02, 2.19915509e-02, -1.15251377e-01, 8.07258263e-02, -8.25611949e-02, -2.28529051e-01, -1.73556402e-01, -1.38860121e-01],
    [-2.15507954e-01, 2.28705049e-01, 9.78312567e-02, 7.63564697e-03, -2.14688718e-01, 1.56157874e-02, -1.67109504e-01, -1.06752077e-02, -3.24769416e-39, 1.17738053e-01, -5.50482608e-02, -2.03965176e-02, 4.18111831e-02, 9.88037437e-02, 1.99235766e-03, -2.82874674e-01, 9.50485542e-02, -4.82246615e-02, -2.88276351e-03, 6.43481389e-02, -6.34773374e-02, 1.62512474e-02],
    [2.95823097e-01, -2.68498123e-01, 1.54971644e-01, -4.90535656e-03, 4.98985052e-02, 8.19644108e-02, 1.07555259e-02, -2.19190642e-01, 1.38084511e-38, 3.73912901e-02, 5.16471528e-02, 1.22935891e-01, 7.78620243e-02, -1.26638517e-01, 8.47073644e-02, 4.54453006e-02, 5.41199856e-02, 7.16911182e-02, 7.53198490e-02, 1.03901257e-03, 1.24598607e-01, 3.84047367e-02],
    [7.18750581e-02, -7.50491545e-02, 1.62401631e-01, -1.76637262e-01, 6.59533441e-02, -2.25364454e-02, -8.78156722e-03, 2.34206125e-01, -1.62745963e-39, 1.06967539e-01, -3.58067788e-02, 6.71938807e-02, -4.77991663e-02, -6.91697001e-02, 4.93007712e-02, 5.62444888e-02, 1.93372983e-02, -3.40918377e-02, 6.24204464e-02, 9.35607255e-02, 6.46235123e-02, -3.04469597e-02],
    [2.21512660e-01, -1.99060380e-01, 1.81485862e-01, -1.42392829e-01, 1.36484370e-01, -6.43528402e-02, 3.38986441e-02, 1.13474421e-01, -7.01355767e-39, -2.66302265e-02, 1.63172204e-02, 8.19971040e-03, -1.47155747e-01, -1.60293072e-04, -8.07289407e-02, 1.05075411e-01, -1.36241198e-01, -1.39806494e-01, -6.61518350e-02, 2.10617855e-01, 3.71382385e-02, 4.76539880e-02],
    [8.40189233e-02, -1.12667188e-01, 5.84894493e-02, -3.43735851e-02, 1.03878886e-01, -1.51497591e-02, 1.70180965e-02, -7.63044283e-02, 8.70520237e-40, 3.27778943e-02, -2.69893780e-02, -6.14207797e-02, -1.70348659e-01, 1.64912234e-03, -1.77492917e-01, 6.45158663e-02, -8.60568807e-02, -3.00472140e-01, -6.35979474e-02, 1.36426628e-01, 2.30335057e-01, 1.10283084e-01],
    [3.23159009e-01, 4.92995158e-02, 1.77151591e-01, -1.28860408e-02, 2.09562071e-02, -7.58555681e-02, 5.23386039e-02, -2.57939752e-02, -1.73392748e-39, 3.88985910e-02, 1.37424111e-01, 1.79835279e-02, -1.30007276e-02, -2.34458279e-02, -9.23940763e-02, -4.93087620e-01, -4.59047668e-02, 8.33575651e-02, -3.24014388e-02, 2.98286695e-02, 1.16699912e-01, 3.23449038e-02],
    [-8.36489126e-02, 1.40370000e-02, -4.45288010e-02, 7.44935274e-02, -2.34061647e-02, 6.35435283e-02, -1.79183744e-02, 1.98855877e-01, 6.11009711e-39, 3.23607288e-02, 3.79871652e-02, 2.37323791e-02, -5.02508059e-02, -1.58036849e-03, 6.52093366e-02, 1.49090737e-02, 1.28282428e-01, -3.97055782e-02, 2.32941121e-01, -1.11813344e-01, -1.21468633e-01, -3.56732458e-02],
    [9.76503044e-02, -2.82167315e-01, 1.39568448e-02, -1.34530425e-01, 1.17489792e-01, 3.12582999e-02, 5.56513742e-02, -4.72440012e-02, 2.59600069e-39, 5.52399196e-02, -4.14876593e-03, -1.41630480e-02, -5.64810149e-02, -1.28942989e-02, -9.88708884e-02, 1.29979998e-01, -1.11729622e-01, 1.84782799e-02, -1.27937868e-01, 7.55686089e-02, 1.18632987e-01, -1.73691306e-02],
    [2.74034917e-01, -3.38174067e-02, 1.77682459e-01, -2.89013218e-02, 8.84463638e-02, -1.88136529e-02, 3.43527980e-02, 1.24298766e-01, 1.39555734e-39, 2.85545457e-03, 8.65737572e-02, 6.91443868e-03, -5.68780862e-02, 2.97230426e-02, -1.65147498e-01, -8.29757452e-02, 1.27778426e-01, 1.36101231e-01, -1.54147133e-01, 1.77406266e-01, 1.47023380e-01, -3.97972502e-02],
    [-7.86267128e-03, -1.79782212e-02, -6.10929821e-03, -1.29334740e-02, 4.64806333e-03, -1.92224719e-02, 5.72821777e-03, 1.73807796e-02, -3.30970022e-39, 3.36952657e-02, 2.86182035e-02, 3.93750612e-03, -2.96477042e-02, -5.89258336e-02, -3.62600014e-02, -2.47203782e-02, 1.21959578e-03, -2.51723021e-01, -4.07183617e-01, 4.00230348e-01, 6.31745517e-01, -1.01952024e-01],
    [-9.99623444e-03, 1.48563785e-02, -4.57502455e-02, -2.84171030e-02, -7.30935531e-03, 2.78360844e-02, -4.18519005e-02, 1.05472974e-01, 8.01175581e-40, -1.34963868e-02, -8.87648435e-04, -2.47315411e-03, 2.32965015e-02, -6.89788139e-04, -2.80173868e-03, -6.18330762e-02, 7.65459761e-02, 3.60882767e-02, -3.82818311e-01, 4.22810197e-01, 7.08294153e-01, -6.08864706e-03],
    [-9.33044627e-02, 8.23919550e-02, 2.05058083e-02, 3.90788503e-02, -6.89416453e-02, 9.64916591e-03, -6.65077865e-02, 1.07130766e-01, -3.24702714e-39, 2.83249933e-02, -2.91028004e-02, 4.25966531e-02, -7.77011365e-02, 1.22212609e-02, 3.06042600e-02, -9.02132019e-02, -2.72644252e-01, 3.99453864e-02, 1.45687193e-01, -9.48276520e-02, -1.09239094e-01, -3.07844058e-02],
    [1.15673244e-01, 6.31593019e-02, -8.66137519e-02, -2.19334900e-01, 5.04072234e-02, -7.79298395e-02, 9.11136493e-02, 1.54078767e-01, 1.19722995e-37, -1.05188601e-02, 1.16243824e-01, 1.91265196e-02, 2.77753081e-02, -5.84933646e-02, -2.32013449e-01, -2.11357862e-01, 8.39622095e-02, 1.00509390e-01, -1.14064433e-01, 7.01129362e-02, 1.50164366e-01, -4.49294597e-02],
    [5.58077451e-03, -1.42100438e-01, 6.70911446e-02, -1.43598095e-01, 6.20174669e-02, -6.94348514e-02, 9.83615518e-02, 2.17966273e-01, -5.12804753e-39, -3.17573920e-02, 1.28284963e-02, 1.08263483e-02, 3.94084752e-02, -4.88703884e-02, -3.12821902e-02, 3.39271836e-02, 2.65333056e-01, -2.68232469e-02, -8.95296596e-03, 1.27086103e-01, -5.33853238e-03, -1.03213869e-01],
    [2.07285076e-01, 3.50197144e-02, 7.09720030e-02, -1.06857426e-01, 1.46837041e-01, -1.04908124e-01, 1.11794084e-01, 2.25787275e-02, -7.80020178e-40, -1.00525662e-01, 4.27178629e-02, -6.03070073e-02, -7.02104568e-02, 6.06411472e-02, -5.67343123e-02, -1.68244988e-01, 2.42353734e-02, -1.61343649e-01, -1.12876117e-01, 2.71011554e-02, 1.85512632e-01, 2.77152091e-01],
    [3.45431571e-03, -8.82266182e-03, 3.66307311e-02, 2.21792646e-02, -1.04120886e-02, 2.31956434e-03, 1.27445441e-02, 7.04137050e-03, -5.32554373e-39, -2.28570141e-02, 1.70231201e-02, 4.29323837e-02, -5.32243922e-02, -3.07291485e-02, -6.90630497e-03, -3.26588415e-02, 9.60635114e-03, -8.95677805e-02, -3.12943280e-01, 3.39423478e-01, 7.46013343e-01, -1.88606799e-01],
    [-2.70637423e-01, 1.76584139e-01, -2.02511907e-01, 1.11370496e-02, -3.45139466e-02, -2.57299952e-02, 1.08234882e-01, -1.91869196e-02, -2.91049831e-39, 4.09036055e-02, 3.36820595e-02, 4.55790050e-02, 9.00270045e-02, 1.27013192e-01, -5.26729301e-02, 4.26426232e-02, -1.54019877e-01, -1.86065491e-02, 3.13204639e-02, -2.00863499e-02, -6.13912642e-02, -1.69273898e-01],
    [-3.05462718e-01, 2.00056568e-01, 9.46264789e-02, 2.62949783e-02, -2.07780488e-02, 2.73437183e-02, 3.70927155e-02, 6.22466356e-02, 1.73138833e-40, 5.45769408e-02, -2.17249189e-02, -2.71243602e-02, -9.07458551e-03, 2.21933456e-04, 3.40965278e-02, -3.49219233e-01, 1.73019275e-01, -2.19678488e-02, -5.38689606e-02, 7.06997290e-02, -8.47352594e-02, 4.02734056e-02],
    [-4.09446210e-02, 6.59236079e-03, -1.94011535e-02, 2.18419125e-03, -1.69736557e-02, -6.50427723e-03, 1.66261010e-02, -3.16731036e-02, 2.88494844e-39, 2.50640623e-02, -9.94806061e-04, 6.98735239e-03, -1.75703824e-01, -2.75820419e-02, -4.09963951e-02, -1.02057438e-02, 2.43694857e-02, -1.02664605e-01, -2.49633327e-01, 4.05164182e-01, 5.96848369e-01, -1.16623648e-01],
    [-2.68736899e-01, 2.76186883e-01, -1.22561000e-01, 1.49314165e-01, -6.05120584e-02, 5.56119867e-02, -4.07551639e-02, 1.11414725e-02, -1.47353400e-39, 3.98594253e-02, 6.83838129e-02, -9.27823503e-03, 4.78026737e-03, 9.57226157e-02, 6.15928657e-02, -2.26769641e-01, 6.79689795e-02, -5.10515831e-02, -4.00011130e-02, 6.22778907e-02, -8.43854174e-02, -3.05019859e-02],
    [-3.15484583e-01, 1.01047963e-01, -2.21296996e-02, -8.91686301e-04, -3.20963651e-01, 4.62254174e-02, 3.29537317e-02, 3.84379923e-02, -5.58824095e-39, 1.43265659e-02, -1.40587380e-02, -9.77343321e-03, -1.37304831e-02, 6.15678728e-02, -1.68659743e-02, -1.27978966e-01, 6.34884387e-02, -9.26641226e-02, 5.91245703e-02, 2.38484249e-01, -1.69403404e-02, 1.04460688e-02],
    [4.06811804e-01, -1.92923576e-01, 2.32721865e-01, -3.45018134e-02, 6.42972291e-02, -8.09305310e-02, 4.81856056e-02, -4.46679950e-01, 4.59843238e-39, 5.12912460e-02, 1.03524156e-01, -9.77360550e-03, 7.43437931e-03, -7.71237072e-03, 4.35408484e-03, -3.36005315e-02, 2.43217032e-03, -4.55424711e-02, 5.29026687e-02, 6.82908744e-02, -7.16390610e-02, -5.78509411e-03],
    [-2.49038860e-02, -2.98063699e-02, 1.96601767e-02, -5.63372159e-03, 1.61849819e-02, 1.37571327e-03, -1.34619372e-02, 2.65082940e-02, -9.65877196e-40, 3.66976038e-02, 2.76671983e-02, -2.03776751e-02, -1.37692122e-02, -2.90586501e-02, 1.97854899e-02, -9.99242589e-02, 4.08249907e-03, -2.68345594e-01, -3.22537392e-01, 1.87935963e-01, 4.55568194e-01, -6.13106694e-03],
];
pub const VALUE_NN_B1: [f64; 64] = [-2.67529696e-01, -3.50863487e-01, -1.38424873e-01, -2.36687452e-01, -3.57881457e-01, -2.15555802e-01, 3.89877073e-02, -1.23733662e-01, -2.52620637e-01, -2.39896238e-01, -2.59915829e-01, -3.03121030e-01, -2.48111755e-01, -2.63800085e-01, -1.02420613e-01, -2.50965953e-01, -2.30197519e-01, -3.33864093e-02, -1.94750745e-02, 5.97352423e-02, -4.42952424e-01, -1.90008193e-01, -4.22882289e-01, -2.31749013e-01, -2.49479294e-01, -9.16759484e-04, 6.60473853e-02, -3.08354020e-01, -2.04426691e-01, -2.79230237e-01, 4.77695391e-02, -2.96576381e-01, -1.91082254e-01, 7.99353719e-02, -2.44193077e-01, -3.83399993e-01, -4.02225614e-01, -2.29547203e-01, -2.77202159e-01, -2.33453482e-01, -3.61899108e-01, -2.63240159e-01, -3.19529057e-01, -1.36243120e-01, -1.41797319e-01, -2.67409682e-01, -2.84683585e-01, -4.70955312e-01, -2.80595839e-01, -4.06146377e-01, -2.16758661e-02, -1.76266536e-01, -2.74865329e-01, -3.55949819e-01, -1.75758108e-01, -1.45659134e-01, 3.56870554e-02, -4.10181969e-01, -2.04827577e-01, -3.45303640e-02, -2.14938447e-01, -2.16752157e-01, -1.46107495e-01, -7.91763514e-02];
pub const VALUE_NN_W2: [[f64; 64]; 32] = [
    [-1.06613591e-01, 3.71013805e-02, -1.64615750e-01, -4.38288320e-03, 1.93668827e-02, 7.14255124e-02, 2.42553763e-02, -3.75388414e-02, -9.53534544e-02, -5.12650423e-02, -4.84394729e-02, -1.15087144e-01, -3.22491489e-02, -2.41299886e-02, 2.27273405e-02, 7.80564025e-02, 1.51240170e-01, 4.46436293e-02, 5.13123460e-02, 1.00555625e-02, -1.60039380e-01, -7.14341998e-02, -1.40632361e-01, -5.43713346e-02, 6.19599521e-02, 3.23187234e-03, 6.38357014e-04, 8.43023136e-02, 1.13389581e-01, 6.13693893e-03, 2.20970833e-03, -4.24531884e-02, -1.31737605e-01, 1.61443148e-02, 6.60431338e-04, -1.21022444e-02, 1.52756408e-01, -4.93328972e-03, -2.97315773e-02, 9.80203450e-02, -6.63518757e-02, -1.09677359e-01, 5.96720129e-02, 7.53002241e-03, 1.02969527e-01, 8.74320343e-02, 1.31478280e-01, -1.06329307e-01, 1.08831376e-01, 1.04526743e-01, 2.90256739e-03, 2.37246584e-02, -2.33008191e-02, 9.65444520e-02, 2.18319315e-02, 9.15892720e-02, 1.74276177e-02, -1.33626759e-01, -1.62637204e-01, 9.41349473e-03, -7.14997947e-02, -8.13188404e-02, 1.24571770e-01, 3.77382562e-02],
    [-1.47428941e-02, 3.85630317e-02, 9.86648723e-03, 1.93667673e-02, -5.88991866e-02, -3.92555371e-02, -1.65050998e-02, -5.07735722e-02, 1.59866922e-02, 4.30664867e-02, 4.64824662e-02, 1.89239532e-02, 1.55378282e-02, 5.98280318e-02, -1.17240280e-01, 2.91622840e-02, 5.06141074e-02, -9.24360156e-02, -2.67044194e-02, -1.08578064e-01, 8.27618502e-03, 3.18455845e-02, 4.42205407e-02, 5.18800840e-02, 2.79766638e-02, -1.03581443e-01, -1.93696860e-02, -1.25198618e-01, -3.63734551e-02, -8.26974735e-02, 1.88795626e-02, 6.38215616e-02, 2.66050715e-02, -1.18784651e-01, 5.48296832e-02, 6.29734099e-02, -2.78827511e-02, 3.89101207e-02, 2.08377969e-02, 3.53559703e-02, 7.35828429e-02, 1.20125618e-02, -6.32806420e-02, -1.25603750e-02, 1.61757823e-02, 4.36506420e-02, 4.94407024e-03, 5.50684659e-03, -5.39224176e-03, -2.07850933e-02, -7.00762719e-02, -1.95249990e-01, 7.43383244e-02, 6.77518873e-03, -2.41716001e-02, 2.40343735e-02, -9.29666758e-02, 8.93578306e-02, -4.47189994e-03, -6.77176714e-02, 1.42278001e-02, -4.70676012e-02, -4.14360352e-02, -2.58364789e-02],
    [-1.11332409e-01, 9.41888615e-02, -7.65944421e-02, -5.48586696e-02, 5.24237528e-02, 6.87015504e-02, 5.84118254e-03, -2.46951226e-02, -1.42220750e-01, -3.71096209e-02, -1.16902351e-01, -1.67713150e-01, -1.30208150e-01, -6.67819753e-02, 6.82330728e-02, 5.87631129e-02, 1.08428903e-01, 5.06686755e-02, 3.96854766e-02, -4.70249802e-02, -5.85674569e-02, -7.30145499e-02, -1.55298874e-01, -5.35849556e-02, 5.27984090e-02, 2.76114023e-03, -1.49293616e-02, 9.39850956e-02, 1.23211332e-01, 2.86970697e-02, 2.73728892e-02, -4.06906791e-02, -9.22955126e-02, -7.66468048e-03, 3.15957926e-02, -7.52452835e-02, 1.77592769e-01, -5.82448468e-02, -8.97793472e-02, 5.35606891e-02, -8.08509141e-02, -9.08350199e-02, 8.06187093e-02, 1.47577487e-02, 6.34831041e-02, 7.47959241e-02, 1.36056721e-01, -1.12805583e-01, 9.00162309e-02, 1.22563072e-01, -1.84728205e-03, 2.86680758e-02, -5.25852963e-02, 1.49462879e-01, 2.49289162e-02, 1.75506435e-02, -2.58653574e-02, -1.09686673e-01, -1.21002294e-01, -6.35453686e-03, -2.32790820e-02, -7.67427683e-02, 8.10036957e-02, 2.50281896e-02],
    [-8.81116390e-02, 7.57149756e-02, -1.10324875e-01, -5.23735695e-02, 1.19025661e-02, 1.10015683e-01, 1.20745134e-03, -1.36928800e-02, -6.45291507e-02, 3.99724953e-02, -9.67347622e-02, -1.34728551e-01, -9.06948596e-02, -4.88416106e-02, 6.49390370e-02, 6.33610338e-02, 7.92626143e-02, 1.39222974e-02, 4.67346124e-02, 2.16120947e-02, -6.55643567e-02, -1.01737447e-01, -1.07778929e-01, -7.20038861e-02, 7.43450746e-02, 1.59332319e-03, -2.35901456e-02, 8.32559764e-02, 1.27001360e-01, 1.69034190e-02, 6.33644611e-02, -6.93986341e-02, -1.05596095e-01, 1.63905807e-02, -7.66239166e-02, -4.16982509e-02, 1.19924888e-01, -6.90462217e-02, -8.81285220e-02, 3.20332758e-02, -9.18316767e-02, -1.36569262e-01, 4.49294858e-02, 3.48241478e-02, 8.64621028e-02, 1.12859003e-01, 1.35629207e-01, -1.30429149e-01, 1.20713688e-01, 1.89551696e-01, 3.53276879e-02, 4.97152796e-03, -7.73911625e-02, 1.50569513e-01, 6.31446624e-03, 7.56258219e-02, 5.50659467e-03, -9.54449177e-02, -1.40433684e-01, -2.15238091e-02, -4.03909460e-02, -7.90071860e-02, 6.79798648e-02, -2.37372499e-02],
    [-5.96007779e-02, 6.37767911e-02, -1.07473478e-01, -7.52559351e-03, 4.07652445e-02, 1.28337309e-01, 3.84066030e-02, -1.01356748e-02, -2.04404406e-02, -3.02430876e-02, -4.02368940e-02, -1.12210445e-01, -3.58262807e-02, -2.45325640e-02, 6.29061684e-02, 6.72869608e-02, 7.10486621e-02, -6.96240645e-03, 6.02626279e-02, 4.02555652e-02, -7.87724778e-02, -5.42163923e-02, -1.05741628e-01, -2.16030255e-02, 5.24521731e-02, 1.39961339e-05, -9.75697208e-03, 1.26818120e-01, 1.57400265e-01, 2.71431897e-02, -2.24627145e-02, -8.94495919e-02, -8.12812001e-02, 1.55906864e-02, 2.23267991e-02, -5.97636849e-02, 1.19320787e-01, -8.26721936e-02, -5.31310067e-02, 9.04251486e-02, -3.77256647e-02, -1.25306159e-01, 9.41676274e-02, 4.45784628e-02, 3.14725712e-02, 6.78390414e-02, 1.72974318e-01, -7.23920092e-02, 1.39758870e-01, 9.06491131e-02, -5.00537921e-03, -1.27000092e-02, -8.00909624e-02, 1.30283594e-01, 4.70890738e-02, 8.45802873e-02, 6.82973117e-03, -1.25466064e-01, -1.36325881e-01, -8.88564740e-04, 4.54062643e-03, -1.00727640e-01, 9.44146439e-02, 1.61498524e-02],
    [-3.80666107e-02, -1.33737214e-02, -1.00682870e-01, -1.40382782e-01, 7.17219412e-02, 8.86671096e-02, 7.63134584e-02, -1.07627464e-02, -1.78196952e-01, -1.89858377e-02, -9.84745920e-02, -1.39671177e-01, -1.62967756e-01, -1.20758310e-01, 8.49062800e-02, 5.96009195e-02, 1.06577218e-01, 9.84075107e-03, 6.28877729e-02, -1.48797706e-02, -4.27648388e-02, -5.68478741e-02, -9.93112922e-02, -8.47389400e-02, 1.22228429e-01, 1.01427790e-02, 1.54451104e-02, 1.38047457e-01, 4.98685092e-02, -3.93020883e-02, 2.50988379e-02, -1.04419343e-01, -9.41890180e-02, -8.44395719e-03, 3.23132016e-02, -8.01912695e-02, 1.52459949e-01, -1.66378528e-01, -8.83661956e-02, 4.57526892e-02, -4.76544574e-02, -7.97467157e-02, 4.09391113e-02, 4.44853157e-02, 8.75932351e-02, 6.97116777e-02, 1.21078588e-01, -1.46843269e-01, 1.47592783e-01, 9.07979533e-02, -1.03761964e-02, 1.15047414e-02, -7.08143786e-02, 1.42896146e-01, 2.31784396e-02, 9.00780708e-02, 8.77587125e-03, -8.84232149e-02, -9.06417891e-02, 6.03725610e-04, -1.76173206e-02, 2.71979161e-03, 7.70646483e-02, 1.96538605e-02],
    [-1.87873747e-02, 4.79145050e-02, 7.54185989e-02, 4.53425460e-02, -7.96016231e-02, -3.87796760e-03, -1.21143878e-01, -7.69280940e-02, 5.06973825e-02, -1.30737647e-01, -1.22914640e-02, -4.43893000e-02, 4.81075421e-03, 1.28169553e-02, -5.16265072e-02, 3.41014750e-02, 5.45857213e-02, -9.93210822e-02, -1.38591900e-01, -1.00896567e-01, 1.99892037e-02, 5.90718128e-02, -9.57978982e-03, 3.20696570e-02, 4.70337793e-02, -1.32960454e-01, -1.52709773e-02, -8.38387981e-02, 2.35640723e-02, -7.84474164e-02, -7.99404979e-02, 3.00295576e-02, 4.08119373e-02, -1.60176218e-01, 5.52133806e-02, 5.45872785e-02, 2.85659377e-02, 4.03178819e-02, 4.27453220e-02, 4.90403622e-02, 1.90705620e-02, 1.86300986e-02, -3.96744721e-02, 6.52005970e-02, 3.06852777e-02, 4.89843078e-02, 4.97826971e-02, 2.97853500e-02, 2.00161599e-02, 5.94832860e-02, -7.53339902e-02, -2.05204844e-01, 3.23200710e-02, 4.23641764e-02, -3.39558572e-02, 7.04162642e-02, -9.94355381e-02, -4.66677174e-02, -1.60450358e-02, -6.53470159e-02, 9.94890369e-03, 7.31506525e-03, -1.25669748e-01, -3.38322744e-02],
    [4.59088795e-02, 8.09703246e-02, 3.42613459e-02, 8.14374257e-03, -9.07752588e-02, 2.33260589e-03, -4.39170860e-02, -5.78023866e-02, 2.40714028e-02, -3.28141712e-02, 5.90177290e-02, 1.07050547e-02, -6.75986521e-04, 3.16014774e-02, -1.21934563e-01, 1.99463014e-02, 3.17681730e-02, -8.81219357e-02, -7.79150203e-02, -1.04627781e-01, 1.29536008e-02, 1.80957075e-02, -3.34310494e-02, 7.77132437e-02, 8.40995163e-02, -9.91048887e-02, -5.86450025e-02, -1.59656867e-01, 2.89069419e-03, -7.23058209e-02, -8.84167776e-02, 4.04340364e-02, 4.05961536e-02, -9.54673365e-02, 2.82810330e-02, 3.97971347e-02, 4.19239216e-02, 2.52948646e-02, 1.53480079e-02, 3.28898728e-02, 7.52496794e-02, 3.54451425e-02, -1.15377925e-01, 5.01725413e-02, 4.37434539e-02, 7.16066658e-02, 2.59938985e-02, 2.77734976e-02, 3.71277742e-02, -1.39752338e-02, -4.65625860e-02, -2.18507320e-01, 4.90233153e-02, 5.61546907e-03, -4.08562683e-02, 7.48446733e-02, -8.93956721e-02, -1.00379223e-02, -1.56762637e-03, -7.29752630e-02, -1.13757234e-02, 1.87405162e-02, -1.15521662e-01, 1.18376641e-02],
    [4.19789739e-02, 1.39378935e-01, -1.76926665e-02, 2.66725011e-02, -1.13839298e-01, 1.30008524e-02, -1.84391946e-01, -1.08415730e-01, 3.27865928e-02, -1.87307388e-01, -3.20543088e-02, 6.31131185e-03, -3.08944788e-02, 3.90087627e-02, 3.17707285e-02, -3.53914755e-03, -3.48258615e-02, -8.71020034e-02, -1.76814094e-01, -2.37069383e-01, 4.27455120e-02, 4.03767824e-02, -3.76355164e-02, 6.21389337e-02, 1.42157197e-01, -3.21860611e-01, -7.97262192e-02, -1.13761127e-01, 1.02851316e-01, -1.52137652e-01, -1.71110928e-01, 6.65599331e-02, -1.71770453e-02, -2.77505219e-01, 2.85333078e-02, 2.31050272e-02, 6.33467436e-02, 1.68805115e-03, -9.53517202e-03, 9.03758481e-02, 1.64705627e-02, 3.66138597e-03, -8.14387128e-02, 3.70962285e-02, -6.35752901e-02, 4.87030856e-02, 1.34644866e-01, 4.69773300e-02, 7.28474334e-02, 1.49381325e-01, -3.78006428e-01, -2.74085760e-01, 4.66397442e-02, 1.49096236e-01, 2.18813151e-01, -3.08751054e-02, -3.09373379e-01, -6.23682328e-02, 6.23803139e-02, -1.83548674e-01, 1.58974938e-02, 3.84374708e-02, -1.00927614e-01, -2.59178400e-01],
    [2.93784924e-02, 5.77072576e-02, 5.80336303e-02, 3.57407928e-02, -7.22251981e-02, -1.13346733e-01, -7.85808824e-03, 4.25118767e-03, 7.74910534e-03, 1.16815343e-02, 4.92645390e-02, 6.08177930e-02, 2.47812853e-03, 1.54342381e-02, -1.69147730e-01, 2.32073897e-03, -3.20778415e-02, -4.47320119e-02, 4.21864679e-03, -6.79775849e-02, 5.42867072e-02, 4.79585640e-02, 3.38755101e-02, 6.49340674e-02, -6.85429946e-03, -7.32811317e-02, -2.18472425e-02, -2.41666399e-02, -1.44376040e-01, -7.22450987e-02, 1.34575097e-02, 7.25854039e-02, 1.77936796e-02, -6.90760165e-02, 3.41790579e-02, -1.60158949e-03, -1.50900539e-02, 6.81800097e-02, 2.77131312e-02, -7.48641044e-03, 2.50797961e-02, 2.12494787e-02, -5.85095286e-02, -1.02799572e-01, -7.60791451e-02, 1.51886540e-02, -9.22455937e-02, 5.19402474e-02, -2.95209866e-02, -3.03780828e-02, -4.46812809e-02, -1.50800601e-01, 4.39970046e-02, -1.49627188e-02, 2.05509458e-02, 1.70343220e-02, -7.85224959e-02, 5.13012558e-02, 4.46405560e-02, -8.83556455e-02, 6.01253696e-02, 4.54196939e-03, -5.60949072e-02, -3.43799181e-02],
    [6.91148564e-02, 1.66085400e-02, 8.95545259e-03, -3.21200266e-02, -9.78288800e-02, 1.09460251e-02, -1.00565456e-01, 1.60862599e-02, 4.86524357e-03, 1.09174650e-03, 7.65625909e-02, 4.29095849e-02, 2.20895242e-02, -2.33474141e-03, -8.18630978e-02, -1.81682389e-02, -1.15065584e-02, -1.31298408e-01, -5.43388091e-02, -1.65794343e-01, 8.19977466e-03, 4.69196290e-02, 8.87538493e-02, 3.63743268e-02, -2.08773594e-02, -7.04464689e-02, -1.34456724e-01, -5.50008118e-02, -2.48450171e-02, -1.83174625e-01, -3.46791781e-02, 6.91492483e-02, -4.63766679e-02, -1.90149665e-01, 7.03355148e-02, 1.32579571e-02, 6.11680653e-03, 1.93952248e-02, 3.98549438e-02, 6.36389107e-02, 5.80103137e-02, 6.08024560e-02, -3.64703201e-02, 1.23143658e-01, 6.27166778e-02, -1.30397510e-02, 2.66245138e-02, 2.61513074e-03, 6.56055426e-03, 3.39336097e-02, -8.53869841e-02, -1.87307537e-01, 6.08499534e-02, 4.92892563e-02, 3.20128947e-02, 1.75342597e-02, -2.02897996e-01, 3.54067199e-02, 6.19505346e-02, -2.58976728e-01, 1.93140805e-02, 8.61144159e-03, -1.23811036e-01, -4.77659740e-02],
    [8.37985873e-02, 8.99297148e-02, 5.49374186e-02, 2.41903253e-02, -1.04596436e-01, 5.99396192e-02, -1.46922424e-01, -5.04450873e-02, 5.04415371e-02, -2.15918735e-01, -2.94608139e-02, 8.31712317e-03, -3.46513912e-02, -2.25664824e-02, 5.79570048e-02, 6.39033169e-02, 2.52339374e-02, -1.28239781e-01, -2.01091900e-01, -2.37137958e-01, 4.53583710e-02, 2.07005143e-02, 5.54820299e-02, 5.23257293e-02, 1.24919333e-01, -3.59087974e-01, -5.26564382e-02, -3.00025418e-02, 9.98997837e-02, -1.13984019e-01, -1.60798058e-01, 3.22762094e-02, -7.34873291e-04, -2.09482402e-01, 5.87657392e-02, 1.79471113e-02, 6.99630305e-02, 4.34816331e-02, -4.88736294e-03, 3.92907113e-02, -3.58432345e-02, 6.27649436e-03, -1.06174678e-01, 6.34422377e-02, 2.17062179e-02, 2.13212110e-02, 1.77543581e-01, 1.14423074e-01, 7.37747923e-02, 1.21434748e-01, -3.30640286e-01, -2.25682542e-01, -1.91101711e-02, 9.59612951e-02, 1.72534838e-01, 1.97999571e-02, -2.89834410e-01, -4.04987969e-02, 6.59321547e-02, -2.09436938e-01, -1.69699695e-02, 9.54943299e-02, -1.03671350e-01, -2.64760226e-01],
    [5.28536849e-02, 3.32268029e-02, 4.10522223e-02, -5.15525043e-03, -7.63841122e-02, -5.61837628e-02, -2.06651781e-02, -7.37026110e-02, 1.83197688e-02, 8.95886030e-03, 1.04179069e-01, 6.23119511e-02, -4.26345086e-03, 1.16324006e-02, -7.26481974e-02, 5.29110692e-02, 3.80529277e-02, 9.65516083e-04, -8.40239599e-03, -9.54603702e-02, 5.23076132e-02, 7.18529448e-02, 8.53281617e-02, 5.99260591e-02, 4.10724021e-02, -5.97362556e-02, -2.66308542e-02, -7.03838021e-02, -9.99792963e-02, -1.11237317e-01, -6.36876449e-02, 7.36485422e-02, 3.94348353e-02, -9.40300226e-02, 1.45316301e-02, 6.22005016e-02, -5.68829360e-04, 6.85904920e-02, 6.38225749e-02, -4.54567838e-03, 1.02938317e-01, 1.23967398e-02, -6.01776056e-02, -5.72223440e-02, -2.10384955e-03, 5.96606219e-03, 2.36940440e-02, 3.56041342e-02, 9.74139478e-03, 1.61545649e-02, -2.43195649e-02, -1.66044220e-01, 7.30345100e-02, 5.50465249e-02, 1.45129133e-02, 3.29150772e-03, -8.13278928e-02, 8.19378793e-02, 5.39284237e-02, -9.25875977e-02, 6.37478083e-02, 1.26600815e-02, -7.62846395e-02, -6.60749013e-03],
    [-9.94201005e-02, 9.41053182e-02, -8.12156424e-02, -1.85062692e-01, 5.89387454e-02, 4.68713604e-02, 7.35814869e-02, -1.01704463e-01, -2.59993047e-01, -7.96243176e-02, -3.93869653e-02, -8.78106877e-02, -2.36092627e-01, -1.88785270e-01, 7.67276883e-02, 3.48450914e-02, 6.04503639e-02, 1.63206384e-02, 2.54163076e-03, 1.36754867e-02, -9.97687504e-02, -1.07273914e-01, -8.89382511e-02, -3.84608060e-02, 9.36313942e-02, -9.38372407e-03, 2.52864836e-03, 3.62716801e-02, 2.69225463e-02, 3.87276039e-02, -4.93876683e-03, -1.15224898e-01, -3.09918728e-02, -1.47100016e-02, -4.32403050e-02, -1.12961896e-01, 1.38458267e-01, -1.54428124e-01, -1.43447563e-01, 1.12153307e-01, -6.69125766e-02, -4.82989810e-02, 5.85438646e-02, 8.12088922e-02, 7.23249093e-02, 5.03577776e-02, 9.40977857e-02, -1.62476823e-01, 1.05375484e-01, 1.71981141e-01, -3.77895050e-02, -6.55278265e-02, -5.76341748e-02, 1.09485008e-01, 2.68625710e-02, 9.53614712e-02, 9.76110436e-03, -1.06082357e-01, -4.11860384e-02, 9.72268451e-03, -1.74575839e-02, -9.62123275e-02, 3.91594060e-02, 1.70569364e-02],
    [-9.87643078e-02, 3.11880782e-02, -1.52030334e-01, -3.48041132e-02, 9.56210271e-02, 1.17206596e-01, 3.59789357e-02, -3.55476849e-02, -6.48856759e-02, -1.19656064e-02, -5.87045103e-02, -1.02939107e-01, -8.64203721e-02, -6.22021593e-02, 7.44137838e-02, 5.74355833e-02, 8.17078575e-02, 2.89644318e-04, 2.21621636e-02, -2.13632006e-02, -5.28563298e-02, -9.33077186e-02, -8.12070295e-02, -5.37371784e-02, 8.81393403e-02, -2.22980063e-02, -1.10789081e-02, 1.24821357e-01, 9.65739116e-02, 2.01767869e-02, -1.28271040e-02, -6.34915158e-02, -8.69061872e-02, -4.61734179e-03, 4.32506902e-03, -6.54179752e-02, 8.84474441e-02, -7.92418271e-02, -5.19723222e-02, 7.42424130e-02, -5.26464880e-02, -1.20604463e-01, 9.55888629e-02, 9.85722244e-02, 5.31851836e-02, 6.13531731e-02, 1.34454116e-01, -1.16870806e-01, 1.47001639e-01, 1.17155313e-01, -1.52556226e-02, 5.58481878e-03, -6.72593713e-02, 1.06364690e-01, 3.14939879e-02, 7.68498331e-02, -1.97824021e-03, -1.11180633e-01, -1.20959841e-01, 3.21000256e-03, -4.74949889e-02, -9.84637961e-02, 5.87733425e-02, -2.08584517e-02],
    [5.51814660e-02, 4.79008630e-02, -4.63162661e-02, 8.50186031e-03, -6.11916743e-02, -8.48945230e-02, -1.48885082e-02, 3.33442688e-02, -1.50497034e-02, -1.24388440e-02, 2.63393093e-02, -3.14646307e-03, -8.70352797e-03, 2.92432718e-02, -5.10528982e-02, -1.56450450e-01, -8.59820396e-02, -1.04894251e-01, 1.05967391e-02, -1.74276650e-01, 2.57666372e-02, 1.49544533e-02, 1.34695873e-01, 6.81259260e-02, 4.84717712e-02, -8.93596653e-03, -1.39401987e-01, -1.12803303e-01, -9.63197425e-02, -2.45217130e-01, 3.70751880e-02, 6.54837713e-02, 6.69068769e-02, -1.46397576e-01, 5.46751022e-02, 7.92144332e-03, -3.11714765e-02, 3.17523591e-02, -2.55375355e-02, 2.26119366e-02, 1.05922662e-01, -1.02622407e-02, -1.68466911e-01, 4.43119518e-02, 1.59079712e-02, -8.40819180e-02, -9.41564608e-03, 1.30665544e-02, 2.81206165e-02, 6.43368345e-03, -2.24855337e-02, -9.29804221e-02, 5.89368902e-02, -6.49581179e-02, 2.20108945e-02, -1.76200181e-01, -1.52432412e-01, 1.66986272e-01, 8.98556039e-03, -2.43155494e-01, 5.33912331e-02, -2.91060880e-02, -6.39770254e-02, -1.32150054e-02],
    [3.01610958e-02, 8.33290890e-02, 8.64472985e-02, 3.71162705e-02, 2.84718983e-02, -1.57872617e-01, -3.58473742e-03, -6.22157864e-02, -2.68225744e-02, 3.76950987e-02, 9.83981648e-04, 4.13147211e-02, -2.53003892e-02, -2.33261026e-02, -2.64677182e-02, -1.78856269e-01, -1.40003085e-01, -6.47282749e-02, 4.46040705e-02, 2.43347529e-02, 7.85665140e-02, 1.91372618e-01, 1.43068358e-01, 4.51030321e-02, -2.80720945e-02, 3.24073583e-02, -8.33310634e-02, -1.05709083e-01, -9.51621905e-02, -1.34809166e-01, -5.28099807e-03, 2.34349705e-02, 1.80985451e-01, 7.53278425e-03, 7.24407285e-02, 1.40636384e-01, -1.00059085e-01, -2.80699693e-02, -4.91716317e-04, -7.37485662e-02, 1.22435139e-02, 1.25681698e-01, -1.16839357e-01, -1.85960323e-01, -1.20810725e-01, -1.89827248e-01, -5.97143769e-02, -3.93867381e-02, -5.71255311e-02, -9.14452970e-02, 2.24861745e-02, -4.82035475e-03, 1.84670500e-02, -1.43135235e-01, -1.55489132e-01, -1.50434494e-01, 1.08890692e-02, 1.67957410e-01, 6.52331263e-02, -2.96468027e-02, 1.36599764e-01, -7.80438492e-03, -4.82737366e-03, 4.38002199e-02],
    [4.10894528e-02, -1.96796171e-02, 9.23687313e-03, 2.12556738e-02, -1.01857431e-01, 1.66434497e-02, -1.88478142e-01, -9.75698605e-02, 5.20766200e-03, -7.89015442e-02, 4.45098355e-02, 4.29371595e-02, -1.16613628e-02, 7.35634845e-03, 1.47374514e-02, 4.36693728e-02, 4.44388390e-02, -1.10398211e-01, -1.03974439e-01, -1.73721984e-01, 1.95738599e-02, 6.32001758e-02, 2.98962034e-02, 1.49290375e-02, 6.94114482e-03, -7.80619606e-02, -1.23841166e-01, -8.94738212e-02, 3.70487235e-02, -1.51236802e-01, -1.41423002e-01, 4.95574325e-02, 6.53852522e-02, -1.99163675e-01, 8.24706778e-02, 6.95634261e-02, -5.44948457e-03, 3.49666248e-03, 4.51706257e-03, 9.54194590e-02, 6.44669235e-02, 6.46175304e-03, -9.19409916e-02, 1.10842608e-01, 7.23985508e-02, -3.36007588e-02, 4.41309623e-02, 5.20034954e-02, -2.63626128e-02, 5.23419864e-02, -1.17522962e-01, -1.71460733e-01, 4.47458886e-02, 6.68221936e-02, 1.62313655e-02, -1.76314358e-02, -2.19533861e-01, -5.30902063e-03, 1.34073319e-02, -2.14063182e-01, 7.87633657e-02, -2.05810368e-02, -1.35341436e-01, -2.84624323e-02],
    [5.94357103e-02, -3.68104428e-02, 3.19035128e-02, 4.32823263e-02, -5.95979802e-02, -1.06945345e-02, -1.39560357e-01, -5.50019816e-02, 3.66671123e-02, -6.47957204e-03, -8.05689115e-03, 3.47604007e-02, -1.98311992e-02, 3.17684151e-02, -1.41344285e-02, -7.61233922e-03, 6.54048100e-03, -6.66428655e-02, -6.12517186e-02, -1.28569514e-01, 6.32271692e-02, 8.04292969e-04, 4.13569547e-02, 4.38156910e-02, -3.54982093e-02, -9.77836475e-02, -1.42542705e-01, -1.08852416e-01, 3.34365331e-02, -1.57987297e-01, -1.34545416e-01, 2.87102163e-02, -1.88120790e-02, -2.00286791e-01, 2.06643548e-02, 1.51425386e-02, 6.55902401e-02, 2.11765319e-02, 1.80720575e-02, 1.21118106e-01, 5.95812388e-02, 1.73180699e-02, -1.67843893e-01, 8.70385244e-02, 1.05428293e-01, -4.76142392e-02, 4.46390659e-02, 3.33201513e-02, -3.91636863e-02, 3.43732312e-02, -8.34898278e-02, -1.50800198e-01, 1.07405648e-01, 3.53225097e-02, 8.13671388e-03, -1.91369618e-03, -1.91325516e-01, 8.35228786e-02, 2.94739399e-02, -2.21987829e-01, 6.30578920e-02, 6.09561475e-03, -1.29342884e-01, 1.18473871e-02],
    [9.06684063e-03, 4.52906564e-02, 6.69392794e-02, 1.34015437e-02, -3.73836197e-02, -9.95217934e-02, -3.69850136e-02, -8.69185328e-02, 1.05478000e-02, 1.25510124e-02, 9.68532041e-02, 2.94925980e-02, -1.81790255e-02, 3.23350681e-03, -3.73212732e-02, 4.34822813e-02, 9.46286134e-03, 4.00484120e-03, 3.63690965e-02, -9.98998657e-02, 4.98906150e-02, 5.02257496e-02, 8.82388502e-02, 7.92689323e-02, 1.42705459e-02, -4.44944985e-02, -4.90606874e-02, -7.40480348e-02, -1.23595759e-01, -1.02994263e-01, -7.86346048e-02, 6.34519011e-02, 5.27902618e-02, -1.16846710e-01, 2.76850164e-02, 5.78577332e-02, -1.46208666e-02, 2.35373899e-02, 2.31780261e-02, 1.06650346e-04, 1.09361410e-01, 4.42409627e-02, -2.14820914e-02, -9.69384089e-02, 1.72324553e-02, 2.47185528e-02, -2.42808741e-02, 6.87066466e-02, 1.86711941e-02, -3.52799520e-02, -4.78665382e-02, -1.52276471e-01, 8.55011269e-02, 2.41040695e-03, -3.17699537e-02, -5.25342999e-03, -7.22939521e-02, 5.47018610e-02, 3.50734442e-02, -9.70597491e-02, 7.32398406e-02, 3.96896787e-02, -4.73085828e-02, -8.73775315e-03],
    [-1.54038861e-01, 1.40481621e-01, -1.69166699e-01, -2.86711752e-01, 1.72690406e-01, 9.81648266e-02, -3.68258893e-03, -1.96810998e-02, -3.04579735e-01, -8.53878707e-02, -1.07621267e-01, -5.10524213e-02, -3.19851696e-01, -3.41019690e-01, 8.21463838e-02, -5.37962019e-02, -4.63584401e-02, 1.59816556e-02, 5.20245284e-02, -1.71989016e-02, -2.99268931e-01, -5.18876575e-02, -2.94203639e-01, 6.86848955e-03, 9.29704681e-02, -1.93065517e-02, -1.04085309e-02, 1.08851932e-01, -9.75316949e-03, 2.28960700e-02, -3.24721001e-02, -8.01562518e-02, -2.72250678e-02, 1.48438178e-02, -9.17021036e-02, -9.02687088e-02, 1.02436110e-01, -2.98144102e-01, -1.98319539e-01, -5.02422976e-04, -1.78275630e-01, -7.84760043e-02, -4.18551117e-02, -6.77007600e-04, 4.32391614e-02, 7.17792213e-02, 1.72469169e-01, -1.69715777e-01, 1.09026819e-01, 1.06495596e-01, 1.23469587e-02, -5.36461687e-03, -8.65784436e-02, 5.94451204e-02, 1.03911432e-02, 3.13400477e-02, -6.83085574e-03, -1.66383505e-01, -1.01318486e-01, -7.26729911e-03, -1.70162693e-02, -2.04879612e-01, 2.67703086e-02, -1.51780210e-02],
    [1.39694288e-02, 2.86296252e-02, 2.76769437e-02, 3.45160440e-02, -5.74722029e-02, -3.40898223e-02, -2.24912930e-02, -7.38528371e-02, 1.23464670e-02, -9.02289758e-04, 8.42850581e-02, 5.87036572e-02, -1.57368165e-02, -6.01956109e-03, -6.93464652e-02, 5.03118485e-02, 5.52530177e-02, -3.30494978e-02, -4.48994385e-03, -9.24910307e-02, 2.01222915e-02, 5.93210049e-02, 5.85296601e-02, 8.16121399e-02, 2.05185656e-02, -1.26076803e-01, -5.07049002e-02, -9.02408510e-02, -9.11634639e-02, -1.06799766e-01, -6.48664609e-02, 6.70529753e-02, 3.99541557e-02, -1.23168007e-01, 3.50929983e-02, 6.01678602e-02, -1.30831916e-02, 4.64724414e-02, 1.25859659e-02, -9.87720676e-03, 5.69157340e-02, 5.70730157e-02, -4.58790921e-02, -5.16522080e-02, -2.71325372e-02, 5.84969111e-02, 2.31046267e-02, 8.11001435e-02, 7.61842448e-03, 1.32746017e-03, -4.90944274e-02, -1.62663937e-01, 4.38925624e-02, -1.36774182e-02, -4.04414944e-02, 5.85034816e-03, -9.60787758e-02, 7.00812042e-02, 1.01796277e-02, -9.12030116e-02, 6.61587417e-02, -2.44978331e-02, -4.83068675e-02, -2.42353287e-02],
    [-1.40686989e-01, 8.30771402e-02, -1.82484195e-01, -4.45894077e-02, 7.05313608e-02, 9.59355012e-02, -1.09093972e-02, -5.91785312e-02, -1.08378798e-01, -1.04199275e-02, -2.75928173e-02, -1.42278388e-01, -1.15611039e-01, -5.07786870e-02, 6.53941855e-02, 5.26018925e-02, 7.34547898e-02, 5.12821376e-02, 5.56879602e-02, -3.66362482e-02, -6.89360201e-02, -8.80618393e-02, -1.13567933e-01, 3.59067484e-03, 1.19480386e-01, 1.03850206e-02, -2.36998070e-02, 8.47213790e-02, 1.51509866e-01, -2.72253156e-03, 2.04292629e-02, -3.54194567e-02, -8.37382004e-02, -6.07596524e-03, 4.67812084e-03, -5.07371537e-02, 1.42950773e-01, -7.93161020e-02, -1.13502599e-01, 7.23253787e-02, -2.89946981e-02, -7.46004954e-02, 7.77363405e-02, 3.06257643e-02, 7.48053119e-02, 1.01652421e-01, 1.50720030e-01, -1.27739847e-01, 1.01489633e-01, 1.67605311e-01, 5.65735344e-03, -1.23943025e-02, -1.04354508e-02, 1.02048896e-01, -2.77494127e-03, 4.48762216e-02, 1.39380898e-02, -9.86877382e-02, -1.18371509e-01, 3.47804874e-02, -1.59888603e-02, -6.19741715e-02, 6.33383617e-02, -6.46631327e-03],
    [8.38802978e-02, 9.89262909e-02, 3.28566879e-03, 2.59266719e-02, -1.15239099e-01, 5.25624678e-02, -2.22567335e-01, -9.47808176e-02, -1.19658792e-03, -1.77444205e-01, 1.98606551e-02, -9.80059244e-03, -3.31665011e-04, -8.91961064e-03, 4.91852760e-02, 1.27310865e-02, -4.90015605e-03, -1.11175142e-01, -1.99243650e-01, -3.51524055e-01, 4.81238253e-02, -4.34994102e-02, -1.06619997e-02, 5.11193760e-02, 1.13645129e-01, -3.29188526e-01, -8.53962544e-03, -8.08664411e-02, 8.03514123e-02, -6.06536232e-02, -1.98222443e-01, 4.69283275e-02, -3.70565057e-02, -2.40928665e-01, 5.16091324e-02, 4.62995283e-03, 4.81867045e-02, -2.96139568e-02, -1.06675019e-02, 2.64446121e-02, -3.14728804e-02, 3.31936404e-02, -1.09102882e-01, 6.14620633e-02, -6.09716550e-02, 4.00564112e-02, 1.57366991e-01, 9.65379626e-02, 2.99354773e-02, 1.19637653e-01, -3.59225839e-01, -2.96405226e-01, 3.91771942e-02, 1.60540164e-01, 1.67174697e-01, 1.78810190e-02, -3.28780293e-01, 1.79415736e-02, 7.04532787e-02, -1.30752802e-01, 3.06324530e-02, 3.72316092e-02, -1.49685994e-01, -2.99790442e-01],
    [4.21361141e-02, 2.05800477e-02, 6.21541291e-02, -8.52696970e-03, -4.87003550e-02, -1.38382256e-01, -5.05290255e-02, -1.60225593e-02, 1.04405815e-02, 1.99680403e-02, 7.92192146e-02, 2.97961589e-02, 5.40216863e-02, 7.07148865e-05, 1.09759597e-02, -1.48355260e-01, -1.85109168e-01, 7.98495859e-03, 4.19868268e-02, -1.88636873e-02, 7.20239729e-02, 1.89591825e-01, 9.42614526e-02, 7.18168216e-03, -1.57430656e-02, -5.35848550e-02, -3.28068957e-02, -5.10663204e-02, -1.30767301e-01, -7.94224814e-02, -1.21885259e-02, 1.27291325e-02, 1.36651948e-01, -1.60708576e-02, 2.01144032e-02, 1.36857033e-01, -7.13112876e-02, -4.78528254e-03, 4.96181063e-02, 6.00873493e-04, 2.25911252e-02, 1.06111526e-01, -8.61487836e-02, -1.27492219e-01, -1.42315328e-01, -1.37097374e-01, -5.87314274e-03, 7.81918764e-02, -5.28043211e-02, -3.37059163e-02, -2.47816499e-02, 1.39459455e-02, 4.19048518e-02, -7.96633735e-02, -6.81567863e-02, -1.53705642e-01, -3.98120582e-02, 9.42994207e-02, 2.46553924e-02, 3.83998756e-03, 1.46837756e-01, 6.40109628e-02, -4.37073261e-02, -1.49425957e-02],
    [3.16080153e-02, 8.51269513e-02, -2.99338382e-02, -1.46513255e-02, -1.08890936e-01, -9.43822935e-02, 7.11489003e-03, -3.17290165e-02, 1.55123314e-02, -1.97541453e-02, 2.81693302e-02, 3.39587368e-02, -2.12429892e-02, 4.36673164e-02, -8.09123740e-02, -1.22936256e-01, -7.11195320e-02, -4.60126996e-02, -1.68063194e-02, -1.80740818e-01, 1.02836952e-01, 1.84723418e-02, 1.43444419e-01, 6.79034144e-02, 5.45333736e-02, -5.11489250e-02, -1.37556091e-01, -9.08000581e-03, -1.20210238e-01, -1.96388140e-01, -1.06413814e-03, 7.23529458e-02, -5.98840695e-03, -1.66876659e-01, 4.18526642e-02, 4.44235429e-02, -8.02091882e-02, 4.81956415e-02, 3.88616719e-03, 4.92471531e-02, 6.37853965e-02, 1.68214943e-02, -5.64529561e-02, 1.78699736e-02, 1.92761639e-04, -9.12485570e-02, -5.03485948e-02, -5.54416627e-02, 2.43079662e-03, -3.05324458e-02, -1.07194958e-02, -1.12693489e-01, 4.78141382e-02, -9.33313444e-02, 3.77654806e-02, -1.69387504e-01, -1.50572777e-01, 1.10855669e-01, 2.98630558e-02, -1.85490370e-01, 3.43930312e-02, -2.56144311e-02, -9.74888206e-02, -7.95641076e-03],
    [8.36863741e-02, 7.64229968e-02, 7.97317773e-02, 1.98083441e-03, -9.02844444e-02, -1.07274212e-01, -1.07946403e-01, 4.64603727e-06, 3.07430904e-02, 1.05770901e-02, 7.21230209e-02, -1.71812978e-02, 3.80791537e-02, 3.55394185e-02, 7.31270388e-02, -1.47862494e-01, -2.54397273e-01, 9.19625629e-03, -3.65420952e-02, -1.87021624e-02, 5.52026480e-02, 1.96102396e-01, 8.73186737e-02, -3.26054506e-02, 4.19992693e-02, -2.28030235e-03, 5.54545149e-02, -1.04258865e-01, -1.11590788e-01, -2.31045615e-02, 3.41887139e-02, 5.23504056e-02, 1.91808358e-01, 3.53246718e-03, 6.27870113e-02, 9.43134204e-02, -9.59256142e-02, -4.10171263e-02, 4.63013276e-02, -1.22214034e-02, -4.35499996e-02, 1.43785566e-01, -1.15351036e-01, -8.53352025e-02, -1.80022821e-01, -1.83206856e-01, -3.57796848e-02, 1.96754374e-02, -6.62064180e-02, 1.26212901e-02, 2.97800489e-02, -5.82940551e-03, 3.70701030e-02, -1.34520188e-01, -1.55088641e-02, -1.91208482e-01, 2.23588366e-02, 1.13931216e-01, 6.41315207e-02, -2.09683720e-02, 9.93809849e-02, -3.45778093e-02, -6.67071715e-02, -1.61263552e-02],
    [3.57583724e-02, 1.24811545e-01, 1.00245528e-01, -3.92606035e-02, -1.01705663e-01, -1.47722989e-01, -1.15281127e-01, -1.44918459e-02, 2.93792109e-04, 6.60082847e-02, -5.71901631e-03, 4.69033420e-03, 8.80888023e-04, -1.35538550e-02, -1.37394248e-02, -2.31780171e-01, -2.76881993e-01, -5.50930984e-02, 5.82674053e-03, -4.93110996e-03, 9.15694162e-02, 1.70899466e-01, 7.64413849e-02, 1.80168878e-02, 5.84784299e-02, 1.81755871e-02, -9.15183499e-03, -1.24464750e-01, -1.38538986e-01, -9.81390253e-02, -5.50108068e-02, 1.99011136e-02, 1.66708946e-01, 5.53650607e-04, 8.29000175e-02, 1.01370372e-01, -1.10895827e-01, -2.68209223e-02, -8.58430564e-03, -2.40356661e-02, -3.18814367e-02, 1.52967289e-01, -1.57808676e-01, -1.21994749e-01, -2.40326852e-01, -1.85893640e-01, -2.95513421e-02, -8.42812844e-03, -1.60063338e-02, -5.27688488e-03, 8.88793450e-03, 1.77389057e-03, 6.46731956e-03, -1.55631050e-01, -2.18193997e-02, -1.85290948e-01, 3.12276725e-02, 1.01668559e-01, 4.22334485e-02, 6.88392855e-03, 1.16180234e-01, 3.76540534e-02, -7.77051598e-02, -7.43069407e-03],
    [3.89488898e-02, 2.49745399e-02, -1.12434449e-02, -1.43005070e-03, -7.32343793e-02, 2.22767312e-02, -1.84682518e-01, -9.97809321e-02, -1.98427197e-02, -1.45033836e-01, -2.58059613e-03, -1.42450137e-02, 7.76274204e-02, -3.36155109e-02, -4.66592312e-02, -1.41397053e-02, 1.92764997e-02, -1.66103959e-01, -1.66265234e-01, -2.49547467e-01, 3.46333832e-02, 6.04098327e-02, 2.02454161e-02, 1.21359993e-02, 1.01446554e-01, -1.54450774e-01, -9.88868028e-02, -8.90386254e-02, 3.13367769e-02, -1.54848993e-01, -1.83657899e-01, 5.90099357e-02, 4.06699590e-02, -2.43689343e-01, 3.50246467e-02, 4.61084582e-03, 3.46854478e-02, 2.40358734e-03, -9.05283540e-02, 1.12569302e-01, 4.95021716e-02, -1.33237252e-02, -8.45575556e-02, 1.04529619e-01, 9.27556008e-02, 1.08423606e-02, 7.89429396e-02, 7.17320666e-02, 1.81747470e-02, 5.12852482e-02, -1.84559926e-01, -1.81197494e-01, 1.42792361e-02, 1.29249647e-01, 1.55836735e-02, -6.13433821e-03, -2.41619378e-01, -1.00683337e-02, 4.23325337e-02, -2.50167280e-01, -1.62444748e-02, -1.07996883e-02, -1.28714725e-01, -8.83137584e-02],
    [-8.61139446e-02, 2.01258272e-01, -1.40565023e-01, -2.57851064e-01, 2.00773686e-01, 4.38094549e-02, -1.73650030e-02, -6.70711994e-02, -2.79590726e-01, -7.88239241e-02, -8.95312652e-02, -5.30745611e-02, -2.88611829e-01, -3.27198446e-01, 7.89466947e-02, 4.78841970e-03, -2.02821679e-02, -3.35502140e-02, 4.78991345e-02, -6.38693920e-04, -2.65624553e-01, -7.01246336e-02, -2.90545315e-01, 1.22060124e-02, 1.16418004e-01, -1.13214329e-02, -2.81907991e-02, 8.56275409e-02, 2.97488403e-02, 2.26909295e-02, -3.45200263e-02, -8.77175853e-02, -8.47494155e-02, 8.50955956e-03, -3.61992642e-02, -8.83409381e-02, 1.25859261e-01, -2.69887805e-01, -2.03090861e-01, 2.34488621e-02, -1.47359028e-01, -1.02350518e-01, 2.39801481e-02, -2.05443725e-02, 2.88537368e-02, 2.69540939e-02, 1.35643333e-01, -2.42718428e-01, 1.05339684e-01, 1.28387913e-01, -8.64750985e-03, 5.90054505e-03, -8.80784392e-02, 7.92673677e-02, 4.73614633e-02, 4.79003973e-03, 1.08649386e-02, -1.19942449e-01, -1.89931229e-01, -1.51648503e-02, -6.39716722e-03, -2.00655937e-01, 3.61804552e-02, -2.85419207e-02],
    [5.02605848e-02, 2.84708347e-02, 3.03370575e-03, 2.85869166e-02, -8.12390000e-02, -5.66510074e-02, -5.66119663e-02, -7.66429454e-02, 6.04787935e-03, -2.25920584e-02, 1.52193671e-02, 8.69366433e-03, 1.40789617e-02, 5.39367041e-03, -1.33284926e-01, 3.37003060e-02, 4.51429486e-02, -1.31864294e-01, -7.40694702e-02, -7.25233629e-02, 7.00242759e-04, 3.39607820e-02, 1.58024784e-02, 4.94622588e-02, 9.18940082e-02, -9.08884108e-02, -7.70499557e-02, -3.65051404e-02, -4.68425862e-02, -6.77350312e-02, -1.42913768e-02, 6.11958392e-02, 3.71953510e-02, -9.62165594e-02, 1.24402354e-02, 3.25600989e-03, -5.27362041e-02, 4.48009521e-02, 2.80609317e-02, -1.14858327e-02, -4.62830765e-03, 6.39877468e-02, -9.32314247e-02, -1.90759078e-02, 1.99650042e-02, 2.81103272e-02, 6.42698677e-03, 5.09257987e-02, 2.75502969e-02, 1.28826713e-02, -2.94118244e-02, -1.68192357e-01, 1.92956943e-02, 4.44611497e-02, -5.46804741e-02, 3.49680521e-02, -7.98554048e-02, 2.17648149e-02, 4.67031170e-03, -9.42629352e-02, 3.22872028e-02, -7.74896331e-03, -1.38164699e-01, -3.32976766e-02],
    [-1.19826081e-03, 2.45993156e-02, 3.69625464e-02, 5.07614389e-02, -8.38244036e-02, 4.18531336e-03, -1.37720332e-01, -5.17296493e-02, 3.03867273e-02, -1.60586499e-02, 4.97064590e-02, 3.59509550e-02, 2.86081433e-02, 3.44281867e-02, -9.94067267e-02, 3.56576666e-02, 8.60969815e-03, -1.12868220e-01, -9.62487683e-02, -8.07990283e-02, 8.57795589e-03, 7.95853361e-02, 2.49347985e-02, 4.70706411e-02, 5.88361062e-02, -1.05258547e-01, -7.42299706e-02, -1.12141922e-01, 6.01855479e-03, -7.39874318e-02, -1.06368393e-01, 5.04444838e-02, 6.06912337e-02, -1.27903700e-01, 2.76184399e-02, 4.32051346e-02, 1.35270599e-02, 3.65583487e-02, 1.55502688e-02, -1.58438012e-02, 4.16890346e-02, 3.18980850e-02, -9.88672599e-02, 5.18895611e-02, 1.37360981e-02, 2.95952484e-02, 2.60358267e-02, 5.23814857e-02, 3.04020997e-02, 4.60703345e-03, -6.37615621e-02, -1.87987953e-01, -1.21292621e-02, 1.92744788e-02, -4.24584001e-02, 5.48072010e-02, -8.86026993e-02, -1.63335633e-02, 2.44837664e-02, -9.11622122e-02, -5.69329271e-03, 3.66863571e-02, -1.35398090e-01, -6.07073260e-03],
];
pub const VALUE_NN_B2: [f64; 32] = [7.20551834e-02, 1.38920024e-01, 9.91835222e-02, 8.45336989e-02, 3.81567962e-02, 8.14359486e-02, 1.28174067e-01, 1.32096246e-01, 7.87561294e-03, 1.58777371e-01, 1.02174886e-01, 2.68068686e-02, 1.63956642e-01, 6.25853986e-02, 8.46397281e-02, 8.57980028e-02, 7.61764795e-02, 8.52917284e-02, 7.33302236e-02, 1.52361304e-01, 4.85449359e-02, 1.68006331e-01, 1.05136707e-01, 1.33972773e-02, 1.25156060e-01, 8.94609541e-02, 9.54694301e-02, 1.07286423e-01, 8.17798376e-02, 5.63090667e-02, 1.68117851e-01, 1.41683578e-01];
pub const VALUE_NN_W3: [f64; 32] = [1.75348803e-01, -1.59364611e-01, 1.80605933e-01, 1.77651823e-01, 1.60254478e-01, 2.21021220e-01, -2.11365253e-01, -2.10898921e-01, -7.67833471e-01, -1.45975709e-01, -3.03590328e-01, -6.99237525e-01, -1.61955565e-01, 2.81240761e-01, 1.53442234e-01, -3.24690670e-01, -2.47974366e-01, -3.44587713e-01, -3.26809227e-01, -1.56209797e-01, 5.29791415e-01, -1.45128265e-01, 1.97603151e-01, -8.07426035e-01, -1.72563061e-01, -3.16901147e-01, -2.19528541e-01, -2.43732139e-01, -4.78055656e-01, 5.40676951e-01, -1.64292768e-01, -2.01771796e-01];
pub const VALUE_NN_B3: f64 = 5.48093207e-02;
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

/// 評価に使う粒子数の基準値（スケール1.0時）。実際の値は思考予算に比例する
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
        let f = |base: usize, lo: usize, hi: usize| {
            ((base as f64 * scale) as usize).clamp(lo, hi)
        };
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

/// 王手駒捕獲候補（CheckSolver::captures_checker）に敷くp_legalの下限。
/// CheckSolverの仮説平均化は生存仮説が多いと正しい捕獲でも確率を薄める
/// （王手駒の粒子ビリーフが誤っている局面では特に顕著。kakutori.kif:
/// 真の捕獲p_legal=0.061で王移動(p_legal=0.99)に完敗していた）。
/// 捕獲は「当たれば王手駒を排除、外れても反則1回ぶんの探索コストで済む」
/// 数少ない手なので、粒子由来のlegal/n項が外していても最低限試す価値を
/// 保証する（2026-07-20、codexレビュー: 原因1単体では届かない数値だったため
/// p_legalフロアで対応）
const CHECK_CAPTURE_P_LEGAL_FLOOR: f64 = 0.35;

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
    if depth_from_back <= 4 { 1.0 } else { 0.0 }
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
    pub const SPECS: [ParamSpec; 39] = [
        ParamSpec { name: "check_bonus", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "check_foul_scale", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "mover_w_captured", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "mover_w_quiet", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "mover_check_extra", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "capture_reveal_risk", lo: 0.0, hi: 0.6 },
        ParamSpec { name: "camp_known_quiet", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "camp_scale", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "exposed_base", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "exposed_known", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "home_knownness", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "recapture_defended", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "exposed_defended", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "attack_w", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "pressure_w", lo: 0.0, hi: 0.6 },
        ParamSpec { name: "foul_cost_base", lo: 0.2, hi: 6.0 },
        ParamSpec { name: "foul_cost_pow", lo: 0.5, hi: 3.0 },
        ParamSpec { name: "advance_w", lo: -0.1, hi: 0.3 },
        ParamSpec { name: "promote_bias", lo: -0.2, hi: 0.6 },
        ParamSpec { name: "drop_bias", lo: -0.5, hi: 0.3 },
        ParamSpec { name: "prior_weight", lo: 0.5, hi: 16.0 },
        ParamSpec { name: "prior_weight_degen", lo: 0.0, hi: 32.0 },
        ParamSpec { name: "threat_w", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "knight_bait_w", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "info_bonus", lo: 0.0, hi: 2.0 },
        ParamSpec { name: "big_home_penalty", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "hand_drop_w", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "backtrack_penalty", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "shuffle_penalty", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "soft_decay", lo: 0.05, hi: 1.0 },
        ParamSpec { name: "king_probe_bonus", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "coverage_w", lo: 0.0, hi: 0.1 },
        ParamSpec { name: "tokin_probe_w", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "depth2_replace", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "depth2_check_pen", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "depth2_recap_discount", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "foul_diff_pow", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "check_limit_accel", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "value_nn_w", lo: 0.0, hi: 10.0 },
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
pub struct EstimatorV10 {
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
}

impl EstimatorV10 {
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
        EstimatorV10 {
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
        }
    }
}

impl Default for EstimatorV10 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorV10 {
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
            .filter(|e| matches!(e, Observation::MyMove { captured: Some(_), .. }))
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
            let fouls: Vec<ShogiMove> =
                foul_tried.iter().filter_map(|u| parse_usi(u)).collect();
            let votes = if sample.is_empty() { &taint_pool } else { &sample };
            CheckSolver::new(view, votes, &fouls, log)
        } else {
            None
        };

        // 相手が位置を知っている自駒（露出）の地図
        let known = knownness_map(view, log, self.params.home_knownness);

        // 2手読み用: 自分が駒を取ったマス（露見）と自分の手が触れたマス
        // （estimator の my_capture_sq / my_touched_sq と同じ定義）
        let mut my_capture_squares: Vec<Coord> = vec![];
        let mut my_touched_squares: Vec<Coord> = vec![];
        for e in log.events() {
            if let Observation::MyMove { usi, captured, .. } = e {
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
        let mut nn_state_cache: Vec<Option<[f64; VALUE_FEATURES]>> =
            vec![None; sample.len()];
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
            if view.you_in_check
                && out.gain > 0.0
                && check_solver
                    .as_mut()
                    .is_some_and(|solver| solver.captures_checker(&mv))
            {
                out.p_legal = out.p_legal.max(CHECK_CAPTURE_P_LEGAL_FLOOR);
            }
            if debug_check_enabled && view.you_in_check {
                eprintln!(
                    "DEBUG {usi}: prior={prior:.4} gain={:.3} p_legal={:.4} foul_cost={:.3} score={:.4}",
                    out.gain, out.p_legal, out.foul_cost, out.score()
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
                Some(ShogiMove::Board { from: pf, to: pt, .. }),
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
        for (i, (usi, mv, out, adjust, score)) in scored.into_iter().enumerate() {
            let final_score = if i < budget.depth2_top_k {
                let delta = depth2_delta(
                    view,
                    &mv,
                    &sample,
                    &known,
                    &my_capture_squares,
                    &my_touched_squares,
                    &params,
                    budget,
                    &mut *rng,
                );
                let gain2 = out.gain + params.depth2_replace * (out.risk_mean + delta);
                combine_score(gain2, out.p_legal, out.foul_cost) + adjust
            } else {
                score
            };
            if best.as_ref().is_none_or(|(_, _, s)| final_score > *s) {
                best = Some((usi, out.p_legal, final_score));
            }
        }

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
        "estimator_v10"
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
    if total_w <= 0.0 { 0.0 } else { weighted_count / total_w }
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
            let Some(p) = view.your_pieces.iter().find(|p| p.square == make_usi_square(from))
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
pub fn candidate_moves(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<(String, ShogiMove)> {
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
            let state = nn_state_cache[pi]
                .get_or_insert_with(|| value_features(pos, me));
            let trans = transition_features(pos, mv, &next, me);
            let mut f =
                [0.0f64; VALUE_FEATURES + TRANSITION_FEATURES];
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
        value_sum / legal
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
        let replies = opp_reply_weights(&next, me, known_for_reply, my_touched);
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
    if n > 0.0 { sum / n } else { 0.0 }
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
        let attack_sq = Coord { file: sq.file, rank: attack_rank };
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
            * if defended { params.exposed_defended } else { 1.0 }
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
        let mut c = Coord { file: king.file + df, rank: king.rank + dr };
        while on_board(&c) && pos.piece_at(c).is_none() {
            n += 1;
            c = Coord { file: c.file + df, rank: c.rank + dr };
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
        let head = Coord { file: king.file, rank: king.rank + toward };
        if on_board(&head) && pos.piece_at(head).is_none() {
            danger += 1.0;
        }
    }
    let generals =
        pos.hand_count(opp, Role::Gold) > 0 || pos.hand_count(opp, Role::Silver) > 0;
    if generals {
        let mut air = 0.0;
        for df in -1..=1i8 {
            for dr in -1..=1i8 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let c = Coord { file: king.file + df, rank: king.rank + dr };
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
            if (1..=9).contains(&c.file)
                && (1..=9).contains(&c.rank)
                && pos.is_attacked(c, by)
            {
                pressure += 1;
            }
        }
    }
    pressure as f64
}

impl EstimatorV10 {
    /// アリーナの共通乱数法用（凍結時に追加。挙動は with_params_line_seed と同じ）
    pub fn with_seed(seed: u64) -> Self {
        EstimatorV10::with_params_line_seed(EvalParams::default(), None, Some(seed))
    }
}
