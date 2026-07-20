//! estimator の凍結版 v8（2026-07-21 凍結）。
//!
//! v7 からの主な差分（v7 凍結コミット 01e0db4 以降、順に）:
//! - **C-7 P3**（v7 の P1+P2 に続く追補。docs/c7-continuous-filter.md）:
//!   ε_phys の「最後の砦」（phys_taint。若返り・ソフト救済後も完全全滅の
//!   ときだけ棄却粒子を force_apply で強制適用し logw += ln(EPS_PHYS) で残す。
//!   評価側は taint>0 を通常サンプルから除外）、エポック正規化（wipe をまたぐ
//!   若返り修復・墓場復活・新規リプレイ粒子のスケールを「スナップショット値
//!   から再出発」の一本の規約に統一）、ブラインド玉攻め勾配（クリーン粒子
//!   全滅時のみ taint 粒子から玉位置信念を抽出し評価へ接続。kakunari 指し継ぎ
//!   2/20→13/20 に改善）。局所被覆度ビリーフ（blind_hang_risk）は実測で有害と
//!   確定し既定無効（TSUITATE_ENABLE_HANG_RISK でオプトインの実験フラグに
//!   格下げ）
//! - 提案分布ガイドの拡張実験: 多段ガイド（Guide.approach）・C-8合成MVP
//!   （synth_particle）・defend 検出を試したが、多段ガイドの
//!   GUIDE_HORIZON 8→24 拡張は効果未確認のままコスト増（kakunari continue
//!   30分→48分）だけ実測されたため 8 へ巻き戻し（BFS結果はメモ化して再発
//!   防止）。defend 検出（自玉移動反則→attacks ブースト）はコストほぼゼロと
//!   確認できたため維持。synth_particle は主経路に未統合のまま実験的に温存
//! - occupies ガイド: 打ちマス反則（王手中でない）から相手駒の占有先を
//!   ガイド化（歩打ちは打ち歩詰めとの理由混同があるため対象外）
//! - 相手手サンプリング事前分布の駒種特化を拡張: home_lance_move（未動の
//!   香車が動く手を強く割引。反則 45%→10%）、knight_bait_w（桂馬の高跳び
//!   歩の餌食。安い歩で敵桂馬を追い詰める計画性を評価に追加）
//! - **王手中の駒捕獲候補への p_legal 下限**（CHECK_CAPTURE_P_LEGAL_FLOOR=0.35。
//!   v8 での主要な変更）: CheckSolver::captures_checker を新設し、王手駒仮説
//!   のマスへ移動してその仮説下で王手が解消する手には combine_score の
//!   p_legal に下限を敷く。CheckSolver の仮説平均化は生存仮説が多いと
//!   正しい捕獲でも確率が薄まる（scenarios/kakutori.kif: 真の捕獲
//!   p_legal=0.061 が玉移動 p_legal=0.99 に完敗）ため、粒子由来の legal/n 項が
//!   外れていても最低限は試す価値を保証する
//! - NN方向フェーズ1のツール追加（value_features.rs・bin/export_value_data 等。
//!   exchange_value/king_zone_pressure を pub(crate) 化しただけで戦略の
//!   挙動自体は不変。推論統合はまだ行っていない。docs/nn-value-phase1.md）
//!
//! 凍結時の成績（100局・match_seedなし、GitHub Actions
//! check-capture-floor ブランチ、2026-07-20実測）:
//! vs v6 71.3%±8.8%（72-29-3）反則/局(A) 5.4 /
//! vs v7 62.5%±9.3%（65-39-0）反則/局(A) 5.78
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
/// 固定深さだと「原因が窓の少し前」を拾えず、常に深いとコスト過剰）
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
    /// 相手の着手（captured_at: 自駒が取られたマス、gives_check: 自玉への王手宣言）
    OppMove {
        captured_at: Option<Coord>,
        gives_check: bool,
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
                Constraint::OppMove { gives_check, .. } => self.in_check = *gives_check,
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
                    }),
                    consumed,
                )
            }
            // 相手の反則は「相手が何か非合法手を試みた」ことしか分からないので使わない。
            // 単独で現れた Check（手と紐づかない）は情報としては手側で消化済みのはず
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
                } => sample_opp_move(
                    &mut pos,
                    my_color,
                    *captured_at,
                    Some(*gives_check),
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
                Constraint::OppMove { captured_at, gives_check } => {
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
            Constraint::OppMove { captured_at, .. } => sample_opp_move(
                pos,
                self.my_color,
                *captured_at,
                None,
                &self.my_capture_sq,
                &self.my_touched_sq,
                &Guide::default(),
                &mut self.rng,
            ),
        }
    }

    /// 若返り: 棄却粒子を直近の相手決定点へ巻き戻し、区間を引き直して修復する。
    /// 深さは REJUV_DEPTHS の順で adaptive に広げる（近い分岐から試す）。
    /// 巻き戻し先が今回の制約自身（= 同じ決定の再試行）は、整合クラス空が
    /// 決定的なので飛ばす。logw の規約は常に「スナップショット値から再出発」
    /// （wipe をまたぐスナップショットはエポック正規化済み — apply_constraint 参照）
    fn rejuvenate_one(
        &mut self,
        hist: &VecDeque<Snap>,
        upto: usize,
        current: Option<&Constraint>,
        deadline: std::time::Instant,
    ) -> Option<Repaired> {
        for &depth in &REJUV_DEPTHS {
            if depth > hist.len() {
                break;
            }
            let snap = &hist[hist.len() - depth];
            if snap.cidx == upto {
                continue;
            }
            if upto - snap.cidx > GRAVEYARD_MAX_SEGMENT {
                break;
            }
            for _ in 0..REJUV_TRIES {
                if std::time::Instant::now() > deadline {
                    return None;
                }
                if let Some(out) = self.replay_segment(snap, hist, upto, current) {
                    return Some(out);
                }
            }
        }
        None
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
                    let _ = f.4;
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
            opp,
            &mv,
            threat_known,
            threat_home,
            is_king,
            flee,
            moved_is_minor(pos, &mv),
            deep_unsupported(&next, &mv, opp),
            hangs_on_landing(pos, &next, &mv, opp),
            home_lance_move(pos, &mv, opp),
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
            let before = crate::deduce::min_moves_empty_board(role, mover, from, target, false);
            let after = crate::deduce::min_moves_empty_board(role, mover, to, target, false);
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
        let mut w = opp_move_weight(
            opp,
            &mv,
            threat_known,
            threat_home,
            is_king,
            flee,
            moved_is_minor(pos, &mv),
            deep_unsupported(&next, &mv, opp),
            hangs_on_landing(pos, &next, &mv, opp),
            home_lance_move(pos, &mv, opp),
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

/// 初期配置の香車マス（自分から動かしたことがない）からの移動か。
/// 香車は序盤ほぼ動かない駒で、これが立たない限り「初期配置のまま」を
/// 強く信じてよい（人間レビューでの指摘: 未観測の駒は初期配置のまま
/// とみなすべきで、隅への探り打ちを繰り返すのは反則を浪費するだけ）。
/// **定義は bin/fit_opp の home_lance_move と一致させること**
fn home_lance_move(pos: &Position, mv: &ShogiMove, opp: Color) -> bool {
    let ShogiMove::Board { from, .. } = *mv else {
        return false;
    };
    if !pos.piece_at(from).is_some_and(|p| p.color == opp && p.role == Role::Lance) {
        return false;
    }
    let home_rank = match opp {
        Color::Sente => 9,
        Color::Gote => 1,
    };
    from.rank == home_rank && (from.file == 1 || from.file == 9)
}

/// 相手の利きがあるマスへの紐なし着地か（取りは除く = 交換ではなく差し出し）。
/// 利き・紐とも着地後の盤面（next）で判定する（開き駒の利きを含む）。
/// 相手の玉の利きも数える（紐がなければ玉に取られる）。銀以上の駒での該当は
/// 実質タダの駒捨てで人間はほぼ指さない（馬@62 のような幻の飛び込み王手の
/// 過大評価を抑える）。**定義は bin/fit_opp の hangs_on_landing と一致させること**
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
/// **定義は bin/fit_opp の deep_unsupported と一致させること**
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
/// 遠ざかる手か。**定義は bin/fit_opp の flees_danger と一致させること**
fn flees_danger(from: Coord, to: Coord, danger: &[Coord]) -> bool {
    let near = |sq: Coord| danger.iter().map(|&d| dist(sq, d)).min();
    match (near(from), near(to)) {
        (Some(a), Some(b)) => b > a,
        _ => false,
    }
}

/// 相手の手の尤度づけ。対人59局の条件付き最尤推定（bin/fit_opp, 2026-07-17、
/// 成り・敵陣深入り・ハングの駒種分割）: パープレキシティ 28.2（旧手調整）→ 24.2。
/// 駒取り・王手の有無は観測との整合ですでに絞り込まれているため、
/// 事前分布には「観測クラス内で判別できる特徴量」だけが現れる。
/// king_flee がわずかに負なのは実測（守りを剥がされても玉は特に逃げない）。
/// 成り・深入り・ハングは小駒（歩香桂）と銀以上で分割: 垂れ歩・と金作りは
/// 好んで指されるが、大駒を相手の利きに紐なしで差し出す手（hang_major）は
/// 実質駒捨てで明確に避けられる（候補10.3%に対し選択3.6%）
fn opp_move_weight(
    opp: Color,
    mv: &ShogiMove,
    threat_known: bool,
    threat_home: bool,
    is_king_move: bool,
    king_flee: bool,
    moved_minor: bool,
    deep_unsup: bool,
    hang: bool,
    home_lance: bool,
) -> f64 {
    let mut s = 0.0;
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let advance = match opp {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            s += 0.162 * advance;
            if promote {
                s += if moved_minor { 1.647 } else { 0.659 };
            }
        }
        ShogiMove::Drop { .. } => s += -1.451,
    }
    if threat_known {
        s += 0.561;
    }
    if threat_home {
        s += 0.670;
    }
    if is_king_move {
        s += 0.131;
    }
    if king_flee {
        s += -0.161;
    }
    if deep_unsup {
        s += if moved_minor { 0.320 } else { 0.026 };
    }
    if hang {
        s += if moved_minor { 0.433 } else { -0.839 };
    }
    // 香車は序盤ほぼ動かない。threat_known/threat_home/promote 等の
    // 具体的な理由があれば上の加点で相殺されるので、無目的な初手だけを狙って割り引く
    if home_lance {
        s += -1.3;
    }
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
    map.into_iter()
        .filter(|(_, mn)| *mn >= since_move)
        .map(|(c, _)| c)
        .collect()
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
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    if n * 2.0 > total {
        Some(role)
    } else {
        None
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
            depth2_top_k: f(DEPTH2_TOP_K, 4, 32),
            depth2_particles: f(DEPTH2_PARTICLES, 16, 384),
        }
    }
}

/// 王周辺圧力を測る粒子数の基準値（スケール1.0時）
const PRESSURE_SAMPLES: usize = 16;

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
    pub const SPECS: [ParamSpec; 38] = [
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
pub struct EstimatorV8 {
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

impl EstimatorV8 {
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
        EstimatorV8 {
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

impl Default for EstimatorV8 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorV8 {
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

        let rng = &mut self.rng;
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
            let mut out = evaluate(view, &mv, &sample, prior, &known, &params, budget);
            if view.you_in_check
                && out.gain > 0.0
                && check_solver
                    .as_mut()
                    .is_some_and(|solver| solver.captures_checker(&mv))
            {
                out.p_legal = out.p_legal.max(CHECK_CAPTURE_P_LEGAL_FLOOR);
            }
            if std::env::var("TSUITATE_DEBUG_CHECK").is_ok() && view.you_in_check {
                eprintln!(
                    "DEBUG {usi}: prior={prior:.4} gain={:.3} p_legal={:.4} foul_cost={:.3} score={:.4}",
                    out.gain, out.p_legal, out.foul_cost, out.score()
                );
            }
            // gain の外側の補正（タイブレーク乱数・手戻り/シャッフル減点）は
            // 2手読み後の再計算でも同じ値を使うので分離して持つ
            let mut adjust = rng.random_range(0.0..0.01);
            if !blind_king_dist.is_empty() {
                adjust += BLIND_KING_ATTACK_W * blind_king_attack(view, &mv, &blind_king_dist);
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
        "estimator_v8"
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
    // 用途は王手ソルバーの投票フォールバック（taint_check_sample）に限定）
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
/// （taint_check_sample・taint_king_distribution・taint_square_coverage の
/// 共通部品。深い taint は信用が下がるので 0.5^(taint-1) で減衰し、
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

/// ε_phys の taint 粒子から王手ソルバー投票用のサンプルを作る（C-7 P3）。
/// クリーン粒子が全滅しているときのフォールバック専用。重みは指紋ごとの
/// 正規化 Σexp(logw)（taint の EPS_PHYS 課金は logw に済み、相対値だけ使う）に
/// 深度減衰 0.5^(taint-1) を掛ける（codex 指摘: ESS リセット後は logw の
/// ε 累積が複製数に実現されて消えるため、深い嘘の投票を別途薄める）。
/// taint > TAINT_VOTE_MAX は投票から除外
fn taint_check_sample(est: &Estimator) -> Vec<(&Position, f64)> {
    taint_particles(est)
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

    for (pos, w) in particles {
        let w = *w;
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

        let mut next = (*pos).clone();
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
        value_sum / legal
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + params.king_probe_bonus * p_chk * (1.0 - p_chk)
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
/// ここでは `all_distances_empty_board` から不成状態の距離だけを直接引く
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
        let dist_map = crate::deduce::all_distances_empty_board(Role::Pawn, me, to);
        let Some(&dist) = dist_map.get(&(attack_sq, false)) else {
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

impl EstimatorV8 {
    /// アリーナの共通乱数法用（凍結時に追加。挙動は with_params_line_seed と同じ）
    pub fn with_seed(seed: u64) -> Self {
        EstimatorV8::with_params_line_seed(EvalParams::default(), None, Some(seed))
    }
}
