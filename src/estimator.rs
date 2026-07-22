//! 相手局面の推定（determinization / パーティクルフィルタ）。
//!
//! ついたて将棋では相手の初期配置は既知（平手の初期局面）なので、
//! 「あり得る相手局面」= 観測と整合する相手の指し手列。厳密な情報集合は
//! 指数的に爆発するため、粒子（具体的なフル局面）の集合でモンテカルロ近似する。
//!
//! 使う観測（公平性: observation.rs にあるものだけ）:
//! - 自分の受理された手 … 粒子上でも合法で、取った駒種が一致しなければ棄却
//! - 自分の反則手 … 粒子上で合法だったら棄却（真の局面では非合法だったので）
//! - 相手の着手 … 粒子上の相手合法手から「取られたマス・王手宣言の有無」と
//!   整合する手をサンプルして進める。整合手がなければ棄却
//! - 王手宣言（の有無）… 手の直後の王手状態と一致しない粒子を棄却
//!
//! 粒子が枯渇したら、制約列を最初からリプレイして再生成する（回数上限つき）。
//!
//! 観測尤度の重み（SIR の重み更新）:
//! 相手手のサンプルは観測と整合するクラスへ絞ってから事前分布で正規化するため、
//! そのままでは「観測を相手が指しにくい手でしか説明できない粒子」も確率1で
//! 生き残ってしまう（例: 桂がいない粒子では角の飛び込み王手が強制される）。
//! そこで制約適用のたびに r = 整合クラスの事前質量 / 全合法手の事前質量 を
//! 対数重み logw へ累積し、評価側（strategy.rs の stratified_sample）が
//! 粒子間で正規化して乗じる。リプレイ生成粒子も全制約ぶん累積するので比較可能。
//!
//! ソフト粒子（POMCP の particle reinvigoration の変種）:
//! 厳密整合の生存粒子が target/4 を下回ったときは、棄却された粒子を
//! 「情報系の制約だけを緩和した」判定で救済し、info_miss を加算して残す。
//! 緩和するのは王手宣言の一致と自分の反則の説明のみで、物理的な制約
//! （自手の合法性・取った駒種・取られたマス）は緩和しない。救済時は
//! 観測尤度 EPS_INFO を logw へ課金する（C-7 P1: 従来の評価側 0.5^penalty
//! 減衰を連続重みへ統合）。info_miss は課金回数の別勘定カウンタで、
//! 上限管理（INFO_MISS_CAP）と評価側の較正（証拠数の勘定）にだけ使う。
//!
//! ESS リサンプリング（C-7 P1 / D2）:
//! logw の退化を ESS = (Σw)²/Σw² で監視し、閾値を割ったら systematic
//! resampling で質量を複製数へ実現して logw をリセットする。退化していないが
//! 頭数が目標に足りないときは、質量保存の分割複製（コピーと元で exp(logw) を
//! 等分）で埋める — 分布を変えずに、次の相手手サンプルで分岐する多様性の種を
//! 蒔く（従来の複製埋めの後継。評価側が multiplicity を畳み込むようになったため
//! 質量保存でない複製は事後分布を偏らせる）。info_miss はリサンプリングでも
//! リセットしない（嘘の昇格防止・較正はカウンタが担う）。
//!
//! 窓付き若返り（C-7 P2 / D3）:
//! 実戦の戦術連鎖（例: 角打の王手→同飛→同桂成で取得駒が飛車ちょうど）は
//! 前向きフィルタでは通せず全滅する（kakunari の中盤死）。厳密生存が薄い
//! ターンでは、棄却された粒子を即死させず**直近の相手決定点へ巻き戻して
//! 引き直す**: 各粒子は直近 REJUV_SNAPSHOTS 決定点のスナップショット
//! (Position, logw, info_miss) をリングで持ち、巻き戻しは近い決定点から
//! adaptive に広げる（REJUV_DEPTHS）。1回の修復コストは総手数に依存しない。
//!
//! 制約後読みガイド: 巻き戻し区間の再サンプルでは、後続の「自分の駒取り」
//! 制約（MyMove(to=X, captured=R)）から状態条件「X に相手の R が立つ」を集め、
//! それを満たす手を GUIDE_BOOST 倍する。ガイドは観測済みイベントのみ参照する
//! （未来情報ではない）。マスクはしない（成功しうる素の経路を提案の台から
//! 消すと重み補正が定義できなくなる）。
//!
//! 重みの会計（レビューで確定した規約）: 巻き戻しは logw をスナップショット値へ
//! 戻し（旧セグメントの累積 r を捨てる = 二重計上なし）、引き直した各決定点で
//! Δlogw = ln(r) + ln(p_class(選択)/g_class(選択)) を累積する。r は素の事前分布
//! での整合クラス質量比（従来どおり）、p/g はクラス内での素/ガイド付き提案の
//! 選択確率。これで修復粒子と生存粒子の logw が同じ生成モデルの規約に載る。
//! リサンプリング・分割複製で logw を動かすときはスナップショットの logw も
//! 同じ量シフトする（相対会計の保存）。

use std::collections::{HashMap, VecDeque};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::board::{Coord, dead_end_rank, parse_usi_square};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, Role};
use crate::shogi::{Piece, Position, ShogiMove, parse_usi, promote_role, unpromote_role};

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
    let pt = crate::opp_move_features::piece_type_features(pos, mv, opp);
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
    let s = crate::opp_move_nn::opp_move_nn_forward(&features).clamp(-15.0, 15.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Role;

    fn record_my_move(log: &mut ObservationLog, usi: &str, captured: Option<Role>) {
        log.record(Observation::MyMove {
            move_number: 0,
            usi: usi.into(),
            captured,
        });
    }

    fn record_opp_move(log: &mut ObservationLog, captured_at: Option<&str>) {
        log.record(Observation::OpponentMoved {
            move_number: 0,
            captured_my_piece_at: captured_at.map(String::from),
        });
    }

    #[test]
    fn particles_track_own_moves_exactly() {
        let mut est = Estimator::with_seed(Color::Sente, 42);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        est.update(&log);
        assert!(est.healthy());
        assert_eq!(est.particles().len(), TARGET_PARTICLES);
        for pos in est.particles() {
            // 自分側は全粒子で正確
            assert_eq!(
                pos.piece_at(Coord { file: 7, rank: 6 }).map(|p| p.role),
                Some(Role::Pawn)
            );
            // 相手は20枚のまま（駒は取られていない）
            assert_eq!(pos.pieces_of(Color::Gote).len(), 20);
            assert_eq!(pos.turn(), Color::Sente);
        }
    }

    #[test]
    fn foul_reveals_blocking_piece() {
        // 初手 8h2b+（角道が開いていない）はどの粒子でも非合法…ではなく
        // 実戦なら反則観測により「経路に何かある」情報が得られる形をテストする。
        // 7g7f / 相手手 / 8h2b+ が反則 → 相手の角道（7c〜3g のどこか）に駒がある粒子だけが残る
        let mut est = Estimator::with_seed(Color::Sente, 7);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        log.record(Observation::MyFoul {
            move_number: 0,
            usi: "8h2b+".into(),
        });
        est.update(&log);
        assert!(est.healthy());
        for pos in est.particles() {
            // 8h から 2b への斜線上（7g〜3c）のどこかに駒がある（=非合法の理由）。
            // 経路が通っていれば 2b への移動/駒取りは合法なので、その粒子は棄却されている
            let blocked = (3..=7).any(|i| {
                pos.piece_at(Coord { file: i, rank: i }).is_some()
            });
            assert!(blocked, "反則の説明がつかない粒子が残っている");
        }
    }

    #[test]
    fn capture_observation_pins_down_opponent_piece() {
        // 7g7f → 相手手 → 8h2b+ が受理され bishop を取った
        // → どの粒子でも「2b に角がいた」ことになり、相手の持ち駒推定も一致する
        let mut est = Estimator::with_seed(Color::Sente, 11);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        record_my_move(&mut log, "8h2b+", Some(Role::Bishop));
        est.update(&log);
        assert!(est.healthy());
        assert!(!est.particles().is_empty());
        for pos in est.particles() {
            assert_eq!(
                pos.piece_at(Coord { file: 2, rank: 2 }).map(|p| p.role),
                Some(Role::Horse), // 自分の馬がいる
            );
            // 相手の盤上駒は19枚（角を取られた）
            assert_eq!(pos.pieces_of(Color::Gote).len(), 19);
        }
    }

    #[test]
    fn check_declaration_filters_particles() {
        // 7g7f → 相手手 → 8h3c+（3cの歩を取って馬に）。馬が 4b 越しに 5a の玉を
        // 睨むため、王手宣言があった場合は「4b が空いている」粒子だけが残る
        let mut est = Estimator::with_seed(Color::Sente, 13);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        record_my_move(&mut log, "8h3c+", Some(Role::Pawn));
        log.record(Observation::Check {
            in_check: Color::Gote,
        });
        est.update(&log);
        assert!(est.healthy(), "王手と整合する粒子が残るはず");
        for pos in est.particles() {
            assert!(pos.in_check(Color::Gote));
        }
    }

    #[test]
    fn replay_backtracking_still_satisfies_all_constraints() {
        // 王手宣言つきの長め制約列でも、バックトラックで作った粒子が
        // 全制約と整合していること（check_declaration と同じ設定で枯渇→再生成）
        let mut est = Estimator::with_seed(Color::Sente, 23);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        record_my_move(&mut log, "8h3c+", Some(Role::Pawn));
        log.record(Observation::Check {
            in_check: Color::Gote,
        });
        record_opp_move(&mut log, None);
        // 相手が金の合駒（4a4b）をした粒子だけが次の制約と整合する。
        // 4b の金を取ると馬が 5a の玉に再度王手になる
        record_my_move(&mut log, "3c4b", Some(Role::Gold));
        log.record(Observation::Check {
            in_check: Color::Gote,
        });
        est.update(&log);
        est.particles.clear();
        est.info_miss.clear();
        est.logw.clear();
        est.hist.clear();
        est.phys_taint.clear();
        est.replenish();
        assert!(est.healthy(), "バックトラック付きリプレイで再生成できるはず");
        for pos in est.particles() {
            // 最終制約まで適用済み: 4b に自分の馬、相手の盤上駒は18枚（歩と金を取った）
            assert_eq!(
                pos.piece_at(Coord { file: 4, rank: 2 }).map(|p| p.role),
                Some(crate::protocol::Role::Horse)
            );
            assert_eq!(pos.pieces_of(Color::Gote).len(), 18);
        }
    }

    #[test]
    fn depleted_particles_regenerate_by_replay() {
        let mut est = Estimator::with_seed(Color::Sente, 17);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        est.update(&log);
        // 人工的に枯渇させる
        est.particles.clear();
        est.info_miss.clear();
        est.logw.clear();
        est.hist.clear();
        est.phys_taint.clear();
        est.replenish();
        assert!(est.healthy(), "リプレイで再生成できるはず");
        assert_eq!(est.particles().len(), TARGET_PARTICLES);
    }

    #[test]
    fn predict_opp_reply_returns_legal_move() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut pos = Position::initial();
        pos.play_unchecked(&parse_usi("7g7f").unwrap());
        let reply = predict_opp_reply(&pos, Color::Sente, &[], &[], &mut rng)
            .expect("初期局面の相手に応手がないはずはない");
        assert!(pos.is_legal(&reply));
        // 手番が自分側の局面では予測しない
        let initial = Position::initial();
        assert!(predict_opp_reply(&initial, Color::Sente, &[], &[], &mut rng).is_none());
    }

    #[test]
    fn reply_weights_apply_recapture_boost_deterministically() {
        // 7g7f / 3c3d / 8h2b+ の後、3a2b（取り返し）の重みは
        // 2b が既知地点のときだけ PREDICT_RECAPTURE_BOOST 倍される
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d", "8h2b+"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let recapture = parse_usi("3a2b").unwrap();
        let weight_of = |known: &[Coord]| -> f64 {
            opp_reply_weights(&pos, Color::Sente, known, &[])
                .iter()
                .find(|(mv, _)| *mv == recapture)
                .map(|(_, w)| *w)
                .expect("取り返しは合法応手のはず")
        };
        let with_boost = weight_of(&[Coord { file: 2, rank: 2 }]);
        let without = weight_of(&[]);
        assert!(
            (with_boost / without - PREDICT_RECAPTURE_BOOST).abs() < 1e-6,
            "with={with_boost} without={without}"
        );
    }

    #[test]
    fn recapture_boost_requires_known_square() {
        // 7g7f / 3c3d / 8h2b+（角で2bの角を取って馬に）。手番は後手で、
        // 3a銀による 2b の取り返しが合法。2b が既知地点なら取り返しが
        // 強くブーストされ、既知でなければ他の手と同程度の頻度に留まる
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d", "8h2b+"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let recapture = parse_usi("3a2b").unwrap();
        let freq = |known: &[Coord]| -> f64 {
            let mut rng = StdRng::seed_from_u64(99);
            let n = 400;
            let mut hits = 0;
            for _ in 0..n {
                if predict_opp_reply(&pos, Color::Sente, known, &[], &mut rng)
                    == Some(recapture)
                {
                    hits += 1;
                }
            }
            f64::from(hits) / f64::from(n)
        };
        let with_boost = freq(&[Coord { file: 2, rank: 2 }]);
        let without = freq(&[]);
        assert!(
            with_boost > without * 3.0,
            "既知地点の取り返しはブーストされるはず（with={with_boost:.3} without={without:.3}）"
        );
        // 絶対頻度はNNの事前分布の再学習で多少動く（①-b新定石データで0.083を実測。
        // 一様なら1/30≈0.033なのでブースト後2.5倍相当）。主検査は上の比率で、
        // ここは「ブーストしても埋もれて選ばれない」水準への劣化だけを見張る
        assert!(with_boost > 0.05, "with={with_boost:.3}");
    }

    #[test]
    fn strict_survivors_keep_zero_penalty() {
        let mut est = Estimator::with_seed(Color::Sente, 42);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        est.update(&log);
        assert!(est.info_miss().iter().all(|&p| p == 0));
        assert_eq!(est.particles().len(), est.info_miss().len());
        assert_eq!(est.particles().len(), est.log_weights().len());
    }

    #[test]
    fn soft_pass_rescues_check_declaration_mismatch() {
        // 初手 7g7f が王手になる粒子は存在しない → 厳密整合は全滅するが、
        // ソフト救済が王手宣言の一致を緩和して penalty=1 で全粒子を生かす
        let mut est = Estimator::with_seed(Color::Sente, 5);
        let c = Constraint::MyMove {
            mv: parse_usi("7g7f").unwrap(),
            captured: None,
            gives_check: true,
        };
        est.apply_constraint(&c);
        assert_eq!(est.particles.len(), TARGET_PARTICLES);
        assert!(est.info_miss.iter().all(|&p| p == 1));
        // 情報系ソフトの尤度 EPS_INFO が logw へ課金されている
        // （厳密生存者ゼロなので中央値課金はなく ln(EPS_INFO) のみ）
        assert!(
            est.logw.iter().all(|&lw| (lw - EPS_INFO.ln()).abs() < 1e-9),
            "ソフト救済の課金が logw に乗っていない"
        );
        // 物理的な適用（着手そのもの）は行われている
        for pos in est.particles() {
            assert_eq!(
                pos.piece_at(Coord { file: 7, rank: 6 }).map(|p| p.role),
                Some(Role::Pawn)
            );
        }
    }

    #[test]
    fn soft_pass_does_not_relax_physical_constraints() {
        // 初手で 5e の駒を取ることはどの粒子でも物理的に不可能
        // （5e への合法手自体がない）→ ソフト救済でも救えず、ε_phys 無効なら全滅
        let mut est = Estimator::with_seed(Color::Sente, 5);
        est.eps_phys = 0.0;
        let c = Constraint::MyMove {
            mv: parse_usi("5g5e").unwrap(),
            captured: Some(Role::Pawn),
            gives_check: false,
        };
        est.apply_constraint(&c);
        assert!(est.particles.is_empty());
    }

    #[test]
    fn eps_phys_keeps_tainted_particles_on_complete_wipe() {
        // 同じ完全全滅でも ε_phys 有効なら taint=1 の粒子として残り、
        // 自駒側の状態（5g5e の強制適用）は真実と同期する。厳密カウントは 0
        let mut est = Estimator::with_seed(Color::Sente, 5);
        est.eps_phys = 0.01;
        let c = Constraint::MyMove {
            mv: parse_usi("5g5e").unwrap(),
            captured: Some(Role::Pawn),
            gives_check: false,
        };
        est.apply_constraint(&c);
        assert!(!est.particles.is_empty(), "ε_phys の最後の砦で生存するはず");
        assert!(est.phys_taint.iter().all(|&t| t == 1));
        assert!(
            est.logw.iter().all(|&lw| (lw - 0.01f64.ln()).abs() < 1e-9),
            "ε_phys の課金が logw に乗る"
        );
        for pos in est.particles() {
            // 強制適用: 自分の歩が 5e に立ち、手番は相手へ
            assert_eq!(
                pos.piece_at(Coord { file: 5, rank: 5 }).map(|p| (p.color, p.role)),
                Some((Color::Sente, Role::Pawn))
            );
            assert_eq!(pos.turn(), Color::Gote);
        }
    }

    #[test]
    fn tainted_particles_are_excluded_from_strict_and_repairable() {
        // taint>0 は厳密カウントから除外される（リプレイ目標・ゲート判定）
        let mut est = Estimator::with_seed(Color::Sente, 9);
        let n = est.target;
        two_kind_particles(&mut est, n / 2, n - n / 2);
        for t in est.phys_taint.iter_mut().take(n / 2) {
            *t = 1;
        }
        let strict = est
            .info_miss
            .iter()
            .zip(&est.phys_taint)
            .filter(|&(&m, &t)| m == 0 && t == 0)
            .count();
        assert_eq!(strict, n - n / 2);
    }

    #[test]
    fn penalty_cap_culls_repeated_violators() {
        let mut est = Estimator::with_seed(Color::Sente, 5);
        for p in est.info_miss.iter_mut() {
            *p = INFO_MISS_CAP;
        }
        let c = Constraint::MyMove {
            mv: parse_usi("7g7f").unwrap(),
            captured: None,
            gives_check: true,
        };
        est.apply_constraint(&c);
        assert!(est.particles.is_empty(), "上限到達の粒子は救済されない");
    }

    /// 2種の局面を粒子に仕込むヘルパ（初期局面と 3c3d 後の局面は別指紋）
    fn two_kind_particles(est: &mut Estimator, n_a: usize, n_b: usize) -> (Position, Position) {
        let a = Position::initial();
        let mut b = Position::initial();
        b.play_unchecked(&parse_usi("7g7f").unwrap());
        b.play_unchecked(&parse_usi("3c3d").unwrap());
        est.particles.clear();
        est.info_miss.clear();
        est.logw.clear();
        est.hist.clear();
        est.phys_taint.clear();
        for _ in 0..n_a {
            est.particles.push(a.clone());
            est.info_miss.push(0);
            est.logw.push(0.0);
            est.hist.push(VecDeque::new());
            est.phys_taint.push(0);
        }
        for _ in 0..n_b {
            est.particles.push(b.clone());
            est.info_miss.push(0);
            est.logw.push(0.0);
            est.hist.push(VecDeque::new());
            est.phys_taint.push(0);
        }
        (a, b)
    }

    #[test]
    fn ess_degeneracy_triggers_systematic_resample() {
        // 重みが1粒子へ退化した集合: ESS ≈ 1 → リサンプリングが発動し、
        // logw リセット・target への複製・質量は複製数へ実現される
        let mut est = Estimator::with_seed(Color::Sente, 31);
        let n = est.target;
        let (a, b) = two_kind_particles(&mut est, 1, n - 1);
        est.logw[0] = 0.0; // a 粒子が支配的
        for lw in est.logw.iter_mut().skip(1) {
            *lw = -20.0;
        }
        est.replenish();
        assert_eq!(est.resamples(), 1, "ESS退化でリサンプリングされるはず");
        assert_eq!(est.particles().len(), est.target());
        assert!(est.logw.iter().all(|&lw| lw == 0.0), "リサンプリング後は logw=0");
        let n_a = est
            .particles()
            .iter()
            .filter(|p| p.fingerprint() == a.fingerprint())
            .count();
        let n_b = est
            .particles()
            .iter()
            .filter(|p| p.fingerprint() == b.fingerprint())
            .count();
        assert!(
            n_a > est.target() * 9 / 10,
            "支配的粒子の質量が複製数に実現されていない: a={n_a} b={n_b}"
        );
    }

    #[test]
    fn split_fill_preserves_mass_per_fingerprint() {
        // 退化していない不足（2個体、重み比 1 : e^-1）は分割複製で埋まり、
        // 指紋ごとの合計質量 exp(logw) が保存される（multiplicity 畳み込みと
        // 二重に効かないための不変条件）。replenish のリプレイ充填と切り離すため
        // split_fill を直接呼ぶ
        let mut est = Estimator::with_seed(Color::Sente, 37);
        let (a, b) = two_kind_particles(&mut est, 1, 1);
        est.logw[1] = -1.0;
        let ws: Vec<f64> = est.logw.iter().map(|&lw| lw.exp()).collect();
        let total: f64 = ws.iter().sum();
        est.split_fill(&ws, total);
        assert_eq!(est.particles.len(), est.target());
        let mass_of = |est: &Estimator, fp: u64| -> f64 {
            est.particles
                .iter()
                .zip(&est.logw)
                .filter(|(p, _)| p.fingerprint() == fp)
                .map(|(_, &lw)| lw.exp())
                .sum()
        };
        let mass_a = mass_of(&est, a.fingerprint());
        let mass_b = mass_of(&est, b.fingerprint());
        assert!((mass_a - 1.0).abs() < 1e-9, "a の質量が保存されていない: {mass_a}");
        assert!(
            (mass_b - (-1.0f64).exp()).abs() < 1e-9,
            "b の質量が保存されていない: {mass_b}"
        );
    }

    #[test]
    fn rejuvenation_rewinds_and_repairs_capture_constraint() {
        // 若返りウォークスルー（1粒子）:
        // 制約列 [▲2g2f, △?, ▲2f2e, △?] ののち、次の制約が
        // ▲2e2d（歩を取る）= 「相手のどちらかの手が 2c2d だった」ことの証明。
        // 粒子は相手が 3c3d / 8c8d を指した歴史を持ち物理的に満たせない。
        // 直近の相手決定点へ巻き戻し、ガイド (2d, 歩) 付きで引き直すと修復できる。
        // logw はスナップショット値から再出発する（旧セグメントは捨てる）
        let mut est = Estimator::with_seed(Color::Sente, 53);
        est.constraints.push(Constraint::MyMove {
            mv: parse_usi("2g2f").unwrap(),
            captured: None,
            gives_check: false,
        });
        est.constraints.push(Constraint::OppMove {
            captured_at: None,
            gives_check: false,
            foul_count: 0,
        });
        est.constraints.push(Constraint::MyMove {
            mv: parse_usi("2f2e").unwrap(),
            captured: None,
            gives_check: false,
        });
        est.constraints.push(Constraint::OppMove {
            captured_at: None,
            gives_check: false,
            foul_count: 0,
        });
        let current = Constraint::MyMove {
            mv: parse_usi("2e2d").unwrap(),
            captured: Some(Role::Pawn),
            gives_check: false,
        };
        // 粒子の歴史: △3c3d / △8c8d（2d に歩が来ない）
        let mut pre1 = Position::initial();
        pre1.play_unchecked(&parse_usi("2g2f").unwrap());
        let mut pre2 = pre1.clone();
        pre2.play_unchecked(&parse_usi("3c3d").unwrap());
        pre2.play_unchecked(&parse_usi("2f2e").unwrap());
        let snap_lw = -0.25; // 巻き戻しで復元されるべき基準値
        let mut hist = VecDeque::new();
        hist.push_back(Snap {
            cidx: 1,
            pos: pre1.clone(),
            logw: 0.0,
            miss: 0,
            taint: 0,
        });
        hist.push_back(Snap {
            cidx: 3,
            pos: pre2.clone(),
            logw: snap_lw,
            miss: 0,
            taint: 0,
        });
        // 確率的な修復なので、失敗したら rng を進めて繰り返す（20回以内に成功）
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5000);
        let mut out = None;
        for _ in 0..20 {
            let failed = vec![(pre2.clone(), 0, snap_lw, hist.clone(), 0)];
            let (mut repaired, _) =
                est.rejuvenate_batch(failed, 4, Some(&current), 1, deadline);
            out = repaired.pop();
            if out.is_some() {
                break;
            }
        }
        let (pos, miss, lw, new_hist, taint) = out.expect("巻き戻し＋ガイドで修復できるはず");
        assert_eq!(miss, 0);
        assert_eq!(taint, 0);
        // 修復後は ▲2e2d まで適用済み: 2d に自分の歩、相手の歩を1枚取った
        assert_eq!(
            pos.piece_at(Coord { file: 2, rank: 4 })
                .map(|p| (p.color, p.role)),
            Some((Color::Sente, Role::Pawn)),
            "修復後は 2e2d が適用済みのはず"
        );
        assert_eq!(pos.pieces_of(Color::Gote).len(), 19, "歩を1枚取っている");
        assert!(lw.is_finite());
        // 巻き戻し先が cidx=3 なら窓は cidx≤3、cidx=1 なら cidx≤1 のみ残る
        assert!(new_hist.iter().all(|s| s.cidx <= 3));
    }

    #[test]
    fn guided_sampling_importance_correction_is_unbiased() {
        // ガイドあり/なしで「補正済み重み付き分布」が一致すること（P2 の
        // 不偏性テスト）。初期局面の相手手番で 3d への歩（3c3d）をガイドすると
        // 提案頻度は上がるが、exp(Δlogw) の重みを掛けた正規化頻度は
        // ガイドなしの選択確率に一致する
        let mut pos0 = Position::initial();
        pos0.play_unchecked(&parse_usi("7g7f").unwrap());
        let target_mv = parse_usi("3c3d").unwrap();
        let wanted = Guide {
            lands: vec![(Coord { file: 3, rank: 4 }, Role::Pawn)],
            attacks: vec![],
            approach: vec![],
            occupies: vec![],
        };
        let n = 4000;
        // ガイドなしの素の選択頻度（大数近似で真の p_class）
        let mut rng = StdRng::seed_from_u64(7);
        let mut base_hits = 0.0f64;
        for _ in 0..n {
            let mut pos = pos0.clone();
            let dlw = sample_opp_move(
                &mut pos,
                Color::Sente,
                None,
                Some(false),
                0,
                &[],
                &[],
                &Guide::default(),
                &mut rng,
            )
            .unwrap();
            assert!(dlw.abs() < 1e-9, "ガイドなし・全手整合クラスなら Δ=ln(1)=0");
            if pos.piece_at(Coord { file: 3, rank: 4 }).is_some()
                && pos.piece_at(Coord { file: 3, rank: 3 }).is_none()
            {
                base_hits += 1.0;
            }
        }
        let base_freq = base_hits / n as f64;
        // ガイドあり: 重み exp(Δlogw - baseline) を掛けて正規化した頻度
        // （baseline = ガイドなしの Δ = ln r は全手共通なので比では消える）
        let mut rng = StdRng::seed_from_u64(8);
        let mut w_total = 0.0f64;
        let mut w_hits = 0.0f64;
        let mut guided_raw_hits = 0usize;
        for _ in 0..n {
            let mut pos = pos0.clone();
            let dlw = sample_opp_move(
                &mut pos,
                Color::Sente,
                None,
                Some(false),
                0,
                &[],
                &[],
                &wanted,
                &mut rng,
            )
            .unwrap();
            let w = dlw.exp();
            w_total += w;
            let hit = pos.piece_at(Coord { file: 3, rank: 4 }).is_some()
                && pos.piece_at(Coord { file: 3, rank: 3 }).is_none();
            if hit {
                w_hits += w;
                guided_raw_hits += 1;
            }
        }
        let corrected_freq = w_hits / w_total;
        let guided_freq = guided_raw_hits as f64 / n as f64;
        assert!(
            pos0.is_legal(&target_mv) && guided_freq > base_freq * 2.0,
            "ガイドが提案頻度を上げているはず: guided={guided_freq:.4} base={base_freq:.4}"
        );
        assert!(
            (corrected_freq - base_freq).abs() < 0.03,
            "補正済み分布がガイドなしと一致しない: corrected={corrected_freq:.4} base={base_freq:.4}"
        );
    }

    #[test]
    fn resample_keeps_info_miss_counter() {
        // リサンプリングは logw をリセットするが info_miss は引き継ぐ
        // （嘘の昇格防止。較正・上限管理はカウンタが担う）
        let mut est = Estimator::with_seed(Color::Sente, 41);
        let n = est.target;
        two_kind_particles(&mut est, 1, n - 1);
        est.info_miss[0] = 2;
        est.logw[0] = 0.0;
        for lw in est.logw.iter_mut().skip(1) {
            *lw = -20.0;
        }
        let max_lw = est.logw.iter().copied().fold(f64::MIN, f64::max);
        let ws: Vec<f64> = est.logw.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let total: f64 = ws.iter().sum();
        est.systematic_resample(&ws, total);
        assert_eq!(est.particles.len(), est.target());
        assert!(est.logw.iter().all(|&lw| lw == 0.0));
        assert!(
            est.info_miss.iter().filter(|&&m| m == 2).count() > est.target() * 9 / 10,
            "リサンプリングのコピーが info_miss を引き継いでいない"
        );
    }

    /// opp_move_weight が組み立てる23特徴量と opp_move_features::opp_move_features
    /// が同じ局面・同じ手に対して一致することを固定する（codexレビュー指摘、
    /// 2026-07-21: estimator.rs 側は hot path 用に private helper を複製したまま
    /// なので、将来どちらか一方だけ変更されてズレるのを検出する）
    #[test]
    fn opp_move_weight_features_match_shared_module() {
        use crate::opp_move_features;
        use std::collections::HashSet;

        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "3g3f", "3d3e"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let mover = pos.turn();
        let known_squares = vec![Coord { file: 5, rank: 3 }];
        let known_set: HashSet<Coord> = known_squares.iter().copied().collect();
        let touched: HashSet<Coord> = HashSet::new();
        let homes_set = opp_move_features::home_squares(&pos, mover.other(), &touched);
        let homes_vec: Vec<Coord> = homes_set.iter().copied().collect();

        let mut checked = 0;
        for mv in pos.legal_moves() {
            let mut next = pos.clone();
            next.play_unchecked(&mv);

            let threat_known = newly_threatens(&pos, &next, &mv, &known_squares);
            let threat_home = newly_threatens(&pos, &next, &mv, &homes_vec);
            let (is_king, flee) = match mv {
                ShogiMove::Board { from, to, .. } => {
                    let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
                    (is_king, is_king && flees_danger(from, to, &known_squares))
                }
                ShogiMove::Drop { .. } => (false, false),
            };
            let minor = moved_is_minor(&pos, &mv);
            let promotes = matches!(mv, ShogiMove::Board { promote: true, .. });
            let is_drop = matches!(mv, ShogiMove::Drop { .. });
            let advance = match mv {
                ShogiMove::Board { from, to, .. } => match mover {
                    Color::Sente => (from.rank - to.rank) as f64,
                    Color::Gote => (to.rank - from.rank) as f64,
                },
                ShogiMove::Drop { .. } => 0.0,
            };
            let deep_unsup = deep_unsupported(&next, &mv, mover);
            let hang = hangs_on_landing(&pos, &next, &mv, mover);
            // opp_foul_count_this_turn（13番目）はConstraint::OppMoveから素通しされる
            // だけの値なので、非ゼロ値（3）で両辺の一致も確認する
            let foul_count_this_turn = 3u32;
            let pt = opp_move_features::piece_type_features(&pos, &mv, mover);
            let est_features = [
                advance,
                (promotes && minor) as u8 as f64,
                (promotes && !minor) as u8 as f64,
                is_drop as u8 as f64,
                threat_known as u8 as f64,
                threat_home as u8 as f64,
                is_king as u8 as f64,
                flee as u8 as f64,
                (deep_unsup && minor) as u8 as f64,
                (deep_unsup && !minor) as u8 as f64,
                (hang && minor) as u8 as f64,
                (hang && !minor) as u8 as f64,
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

            let shared_features = opp_move_features::opp_move_features(
                &pos,
                &next,
                &mv,
                mover,
                &known_set,
                &homes_set,
                foul_count_this_turn,
            );
            assert_eq!(
                est_features, shared_features,
                "特徴量が一致しない: mv={mv:?}"
            );
            checked += 1;
        }
        assert!(checked > 10, "候補手が少なすぎてテストとして機能していない");
    }
}
