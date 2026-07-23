//! webhookペイロード（`webhook_protocol::BotTurnRequest`）から観測ログ・
//! PlayerViewを組み立て、ゲームごとの戦略インスタンスをキャッシュする。
//!
//! ソケット接続常駐の client.rs と違い、webhookは「今回のリクエストに至るまでの
//! 全ply履歴」を毎回受け取るステートレスなHTTP呼び出し。ここでは gameId ごとに
//! Strategy + 観測ログをメモリ上にキャッシュし、継続対局では新しいplyぶんだけ
//! 増分で読み進める。キャッシュを失った場合（プロセス再起動・老朽化した
//! セッションの掃除後など）は0手目から全件を読み直す。
//!
//! sfen は使わない: 各plyの `lastMove`(CSA)/`lastInfo`/`lastCapture`/
//! `wasPromotion` から直接 `Observation` イベントを組み立てられるため、
//! 既存の `GameModel::from_log` 相当の増分適用（`GameModel::apply`）だけで
//! 自分の可視局面が再構成できる（詳細はプロジェクトのplan参照）。

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::board::{make_usi_drop, make_usi_move, make_usi_square};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView};
use crate::shogi::{ShogiMove, promote_role};
use crate::strategy::{self, Strategy};
use crate::webhook_csa::{CsaMoveKind, parse_capture_letter, parse_csa_move, usi_move_to_csa};
use crate::webhook_protocol::{
    BotTurnRequest, PositionEntry, is_check_info, is_foul_info, parse_bw_color,
};

pub const SUPPORTED_GAME_TYPE: &str = "ついたて";

/// 古いゲームのセッションを掃除するまでの猶予（本番の対局時計 300秒+3秒 は
/// もちろん、アリーナ検証用の 1000秒+3秒 よりも十分長く取っておく）
const SESSION_TTL: Duration = Duration::from_secs(2 * 60 * 60);
/// コールドスタート時の逐次prewarmに使う時間上限。残りの履歴は choose 内の
/// 通常updateへ渡し、HTTP deadlineを無制限の復元処理で消費しない。
const DEFAULT_COLD_START_PREWARM_MS: u64 = 2_500;

#[derive(Debug)]
pub enum SessionError {
    UnsupportedRequestType(String),
    UnsupportedGameType(String),
    UnsupportedPlayers,
    UnknownStrategy(String),
    InvalidColor(String),
    MissingPosition(u32),
    MissingLastMove(u32),
    InvalidLastMove { ply: u32, raw: String },
    UnsupportedGameParameters,
    InconsistentHistory,
    StaleRequest,
    NoLegalMove,
    ResponseEncodingFailed,
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::UnsupportedRequestType(t) => write!(f, "unsupported_request_type: {t}"),
            SessionError::UnsupportedGameType(t) => write!(f, "unsupported_game_type: {t}"),
            SessionError::UnsupportedPlayers => write!(f, "unsupported_players"),
            SessionError::UnsupportedGameParameters => write!(f, "unsupported_game_parameters"),
            SessionError::InconsistentHistory => write!(f, "inconsistent_history"),
            SessionError::StaleRequest => write!(f, "stale_request"),
            SessionError::UnknownStrategy(name) => write!(f, "unknown_strategy: {name}"),
            SessionError::InvalidColor(c) => write!(f, "invalid_color: {c}"),
            SessionError::MissingPosition(ply) => write!(f, "missing_position: {ply}"),
            SessionError::MissingLastMove(ply) => write!(f, "missing_last_move: {ply}"),
            SessionError::InvalidLastMove { ply, raw } => {
                write!(f, "invalid_last_move: ply={ply} raw={raw}")
            }
            SessionError::NoLegalMove => write!(f, "no_legal_move"),
            SessionError::ResponseEncodingFailed => write!(f, "response_encoding_failed"),
        }
    }
}

impl SessionError {
    pub fn status_code(&self) -> u16 {
        match self {
            SessionError::UnsupportedRequestType(_)
            | SessionError::UnsupportedGameType(_)
            | SessionError::UnsupportedPlayers
            | SessionError::UnsupportedGameParameters
            | SessionError::InconsistentHistory
            | SessionError::InvalidColor(_)
            | SessionError::MissingPosition(_)
            | SessionError::MissingLastMove(_)
            | SessionError::InvalidLastMove { .. } => 400,
            SessionError::StaleRequest => 409,
            SessionError::UnknownStrategy(_) | SessionError::ResponseEncodingFailed => 500,
            SessionError::NoLegalMove => 422,
        }
    }
}

struct GameSession {
    my_color: Color,
    strategy: Box<dyn Strategy + Send>,
    model: GameModel,
    log: ObservationLog,
    /// 次に指される（＝いま決めようとしている）手の番号。Position::move_number()
    /// と同じ規約（初期局面で1、着手のたびに+1、反則は数えない）
    next_move_number: u32,
    /// 現在どちらかが王手されている場合、その色（直近の受理された手が
    /// 王手/詰みを宣言していれば Some、それ以外の受理手で解消済みなら None）
    check_holder: Option<Color>,
    /// このセッションが処理済みの ply（次に処理すべきは +1 から）
    processed_ply: u32,
    /// dispatcherの再送に対して同じ応答を返すためのrequestIdキャッシュ。
    /// 古い再送でセッションを過去へ巻き戻さないことも目的とする。
    request_cache: HashMap<String, String>,
    request_cache_order: VecDeque<String>,
}

impl GameSession {
    fn new(strategy_name: &str, my_color: Color) -> Option<Self> {
        Some(GameSession {
            my_color,
            strategy: strategy::make(strategy_name)?,
            model: GameModel::new(my_color),
            log: ObservationLog::default(),
            next_move_number: 1,
            check_holder: None,
            processed_ply: 0,
            request_cache: HashMap::new(),
            request_cache_order: VecDeque::new(),
        })
    }
}

type SessionEntry = (Instant, Arc<Mutex<GameSession>>);

pub struct SessionStore {
    strategy_name: String,
    games: Mutex<HashMap<String, SessionEntry>>,
}

impl SessionStore {
    pub fn new(strategy_name: String) -> Self {
        SessionStore {
            strategy_name,
            games: Mutex::new(HashMap::new()),
        }
    }

    fn session_for(
        &self,
        game_id: &str,
        my_color: Color,
    ) -> Result<(Arc<Mutex<GameSession>>, bool), SessionError> {
        let mut games = self.games.lock().expect("games mutex poisoned");
        let now = Instant::now();
        games.retain(|_, (last_seen, _)| now.duration_since(*last_seen) < SESSION_TTL);

        if let Some((last_seen, session)) = games.get_mut(game_id) {
            let same_color = session.lock().expect("session mutex poisoned").my_color == my_color;
            if same_color {
                *last_seen = now;
                return Ok((session.clone(), false));
            }
        }
        let fresh = GameSession::new(&self.strategy_name, my_color)
            .ok_or_else(|| SessionError::UnknownStrategy(self.strategy_name.clone()))?;
        let arc = Arc::new(Mutex::new(fresh));
        games.insert(game_id.to_string(), (now, arc.clone()));
        Ok((arc, true))
    }

    #[cfg(test)]
    fn session_count(&self) -> usize {
        self.games.lock().unwrap().len()
    }
}

/// リクエストを検証し、キャッシュされた（または新規の）ゲームセッションを
/// 現在のplyまで進めたうえで着手を選び、CSA文字列で返す。
pub fn choose_move(store: &SessionStore, req: &BotTurnRequest) -> Result<String, SessionError> {
    if req.kind != "your_turn" {
        return Err(SessionError::UnsupportedRequestType(req.kind.clone()));
    }
    if req.game.kind != SUPPORTED_GAME_TYPE {
        return Err(SessionError::UnsupportedGameType(req.game.kind.clone()));
    }
    if req.game.required_players.b != 1 || req.game.required_players.w != 1 {
        return Err(SessionError::UnsupportedPlayers);
    }
    validate_game_parameters(&req.game)?;
    let my_color =
        parse_bw_color(&req.color).ok_or_else(|| SessionError::InvalidColor(req.color.clone()))?;

    let (arc, new_session) = store.session_for(&req.game_id, my_color)?;
    let mut session = arc.lock().expect("session mutex poisoned");

    if !req.request_id.is_empty() {
        if let Some(cached) = session.request_cache.get(&req.request_id) {
            return Ok(cached.clone());
        }
    }

    // requestIdキャッシュから外れた遅延再送で、進行済みセッションを過去へ
    // 巻き戻すと、その後の履歴を二重適用するため拒否する。
    if session.processed_ply > req.ply {
        return Err(SessionError::StaleRequest);
    }

    let mut cold_start = new_session;
    if advance(&mut session, &req.positions, req.ply).is_err() {
        // 想定外の食い違い（プロセス再起動直後でキャッシュが空、ply欠落等）は
        // セッションを作り直して0手目からやり直す
        *session = GameSession::new(&store.strategy_name, my_color)
            .ok_or_else(|| SessionError::UnknownStrategy(store.strategy_name.clone()))?;
        advance(&mut session, &req.positions, req.ply)?;
        cold_start = true;
    }

    if !session.model.consistent() {
        return Err(SessionError::InconsistentHistory);
    }
    let view = build_player_view(&session, req)?;
    let mut foul_tried = collect_foul_tried(&session.log, session.next_move_number);
    let mut deduced_illegal = HashSet::new();
    exclude_moves_on_known_opponent(&session.log, &view, &mut deduced_illegal);

    let GameSession { strategy, log, .. } = &mut *session;
    if cold_start {
        // 一括 update だと長い履歴で粒子が完全枯渇するため、自分の手番ごとに
        // 逐次 prewarm してから choose する（bin/scenario.rs::prewarm_strategy
        // と同じ手当て。通常の増分パスは choose 自体の内部 update で十分）
        let budget_ms = std::env::var("TSUITATE_COLD_START_PREWARM_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_COLD_START_PREWARM_MS);
        strategy::prewarm_strategy_with_budget(
            &mut **strategy,
            &view,
            log,
            Some(Duration::from_millis(budget_ms)),
        );
    }
    let chosen = choose_avoiding_deduced_illegal(
        &mut **strategy,
        &view,
        log,
        &mut foul_tried,
        &deduced_illegal,
    );
    let usi = chosen.ok_or(SessionError::NoLegalMove)?;

    let model = &session.model;
    let csa = usi_move_to_csa(my_color, &usi, |c| {
        model
            .my_pieces()
            .into_iter()
            .find(|p| p.square == make_usi_square(c))
            .map(|p| p.role)
    })
    .ok_or(SessionError::ResponseEncodingFailed)?;

    if !req.request_id.is_empty() {
        const REQUEST_CACHE_LIMIT: usize = 128;
        session
            .request_cache
            .insert(req.request_id.clone(), csa.clone());
        session
            .request_cache_order
            .push_back(req.request_id.clone());
        while session.request_cache_order.len() > REQUEST_CACHE_LIMIT {
            if let Some(old) = session.request_cache_order.pop_front() {
                session.request_cache.remove(&old);
            }
        }
    }
    Ok(csa)
}

/// `Strategy::choose` に渡す `foul_tried` は、王手中は `CheckSolver` へも
/// 「実際に試みて反則になった手」という証拠としてそのまま流れ込む（凍結版
/// 含め全戦略共通の規約で、trait signature を変えない限り経路を分けられない）。
/// `exclude_moves_on_known_opponent` の演繹的除外（占有マスへの打ち・
/// そこを飛び越える長距離移動）を無条件に `foul_tried` へ混ぜると、実際には
/// 一度も試していない手が「反則だった」という誤った証拠として扱われ、
/// 無関係な王手駒仮説を誤って減衰させかねない
/// （占有マスの駒が王手駒仮説と紛らわしく `check.rs` の `base` から一時的に
/// 取り除かれるケースで顕在化する）。
///
/// そのため演繹的除外は事前に `foul_tried` へ混ぜず、戦略が実際にそれを
/// 選んだ場合だけ事後的に足して選び直させる。選ばれなかった除外候補は
/// 一切証拠として扱わないため、汚染は「戦略が現に選ぼうとした手」だけに
/// 限定される（サーバーへ反則を1回無駄撃ちする代わりに、ローカルで
/// 選び直す点は元の意図のまま）
fn choose_avoiding_deduced_illegal(
    strategy: &mut dyn Strategy,
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &mut HashSet<String>,
    deduced_illegal: &HashSet<String>,
) -> Option<String> {
    // deduced_illegal に含まれる手は candidate_moves から1つずつ確実に
    // 除外されていくため、高々 deduced_illegal.len() 回で終端する
    for _ in 0..=deduced_illegal.len() {
        let chosen = strategy.choose(view, log, foul_tried)?;
        if !deduced_illegal.contains(&chosen) {
            return Some(chosen);
        }
        foul_tried.insert(chosen);
    }
    None
}

/// 直近まで消化済みのログの末尾から、今回の手番で自分が試みた反則（まだ
/// move_number が進んでいないもの）を集める。client.rs の
/// `state.foul_tried`（move_number が変わったらクリア）と同じ規約
fn collect_foul_tried(log: &ObservationLog, current_move_number: u32) -> HashSet<String> {
    log.events()
        .iter()
        .filter_map(|e| match e {
            Observation::MyFoul { move_number, usi } if *move_number == current_move_number => {
                Some(usi.clone())
            }
            _ => None,
        })
        .collect()
}

/// 相手が直前の正規手で自駒を取った升には、相手の着手駒が確実に存在する。
/// 打ちは駒を取れないため、その升への駒打ちは粒子推定によらず必ず反則になる。
/// 呼び出し側（`choose_avoiding_deduced_illegal`）が `foul_tried` とは別に
/// 保持し、戦略が実際にそれを選んだ場合だけ事後的に反則として扱う
/// （`foul_tried` へ直接混ぜない理由は同関数のコメント参照）。
///
/// それより古い捕獲升は相手駒が既に動いた可能性があるので使わない。自分が
/// 王手回避を反則にされた場合も盤面は変わらず、直前の捕獲升は引き続き有効。
fn exclude_moves_on_known_opponent(
    log: &ObservationLog,
    view: &PlayerView,
    excluded: &mut HashSet<String>,
) {
    let mut occupied = HashSet::new();
    if let Some(square) = log
        .events()
        .iter()
        .rev()
        .find_map(|event| match event {
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => Some(
                captured_my_piece_at
                    .as_deref()
                    .and_then(crate::board::parse_usi_square),
            ),
            _ => None,
        })
        .flatten()
    {
        occupied.insert(square);
    }

    // 王手中でない非歩の駒打ちが反則になった場合、候補生成が既知の
    // 二歩・行き所のない駒を除外しているため、着地点は相手駒で占有されている。
    // 同じ手番中は反則で盤面が変わらないので、これも確定情報として使う。
    if !view.you_in_check {
        for event in log.events() {
            let Observation::MyFoul { move_number, usi } = event else {
                continue;
            };
            if *move_number != view.move_number {
                continue;
            }
            if let Some(ShogiMove::Drop { role, to }) = crate::shogi::parse_usi(usi) {
                if role != crate::protocol::Role::Pawn {
                    occupied.insert(to);
                }
            }
        }
    }

    if occupied.is_empty() {
        return;
    }

    // 駒打ちは着地点そのもの、盤上の長距離移動は確定占有升を
    // 飛び越える場合が必ず反則になる。着地点への盤上移動（捕獲）は許可する。
    let candidates = strategy::candidate_moves(view, &HashSet::new());
    for square in occupied {
        for (usi, mv) in &candidates {
            let blocked = match mv {
                ShogiMove::Drop { to, .. } => *to == square,
                ShogiMove::Board { from, to, .. } => crosses_square(*from, *to, square),
            };
            if blocked {
                excluded.insert(usi.clone());
            }
        }
    }
}

fn crosses_square(
    from: crate::board::Coord,
    to: crate::board::Coord,
    square: crate::board::Coord,
) -> bool {
    let df = to.file - from.file;
    let dr = to.rank - from.rank;
    let aligned = (df == 0 || dr == 0 || df.abs() == dr.abs()) && (df != 0 || dr != 0);
    if !aligned {
        return false;
    }
    let step_file = df.signum();
    let step_rank = dr.signum();
    let mut current = crate::board::Coord {
        file: from.file + step_file,
        rank: from.rank + step_rank,
    };
    while current != to {
        if current == square {
            return true;
        }
        current.file += step_file;
        current.rank += step_rank;
    }
    false
}

/// エンジンが標準ついたて用に固定実装されているため、ゲーム設定も標準値だけを
/// 受理する。`param` がない旧形式のテストpayloadは標準値として扱う。
fn validate_game_parameters(game: &crate::webhook_protocol::GameInfo) -> Result<(), SessionError> {
    let Some(param) = game.param.as_deref() else {
        return Ok(());
    };
    let mut values = HashMap::new();
    for pair in param.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        values.insert(key, value.replace("%2F", "/").replace("%2f", "/"));
    }
    let standard_board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL";
    let standard = [
        ("initial_board", standard_board),
        ("promotion_rank", "3"),
        ("foul_limits", "9.9"),
        ("draw_move_count", "150"),
        ("enable_try_rule", "false"),
    ];
    if standard
        .iter()
        .any(|(key, expected)| values.get(key).is_some_and(|actual| actual != expected))
    {
        return Err(SessionError::UnsupportedGameParameters);
    }
    Ok(())
}

fn build_player_view(
    session: &GameSession,
    req: &BotTurnRequest,
) -> Result<PlayerView, SessionError> {
    let fouls_present = req
        .positions
        .get("0")
        .and_then(|entry| entry.fouls)
        .is_some()
        && req
            .positions
            .get(&req.ply.to_string())
            .and_then(|entry| entry.fouls)
            .is_some();
    let (you_fouls, opponent_fouls) = if fouls_present {
        let current_fouls = fouls_at(&req.positions, req.ply)?;
        let start_fouls = fouls_at(&req.positions, 0)?;
        let (you_start, opp_start) = split_by_color(session.my_color, start_fouls);
        let (you_remaining, opp_remaining) = split_by_color(session.my_color, current_fouls);
        (
            you_start.saturating_sub(you_remaining),
            opp_start.saturating_sub(opp_remaining),
        )
    } else {
        (session.model.my_fouls(), session.model.opponent_fouls())
    };

    Ok(PlayerView {
        game_id: req.game_id.clone(),
        your_color: session.my_color,
        your_pieces: session.model.my_pieces(),
        your_hand: session.model.my_hand(),
        turn: session.my_color,
        move_number: session.next_move_number,
        // このサイトの clocks は戦略の意思決定に使われない（TSUITATE_THINK_BUDGET_MS
        // による固定の思考予算で足りている）ため、プレースホルダで埋める
        clocks: ClockState {
            sente_ms: 0,
            gote_ms: 0,
            running: None,
            server_time: 0,
        },
        fouls: FoulCounts {
            you: you_fouls,
            opponent: opponent_fouls,
        },
        you_in_check: session.check_holder == Some(session.my_color),
        opponent_in_check: false,
        status: GameStatus::Playing,
    })
}

fn fouls_at(
    positions: &HashMap<String, PositionEntry>,
    ply: u32,
) -> Result<(u32, u32), SessionError> {
    let entry = positions
        .get(&ply.to_string())
        .ok_or(SessionError::MissingPosition(ply))?;
    // 標準ついたての既定値。dispatcher契約上 fouls は任意フィールドなので、
    // 欠落していても標準ルールでは初期残数9として扱う。
    Ok(entry.fouls.map(|f| (f.b, f.w)).unwrap_or((9, 9)))
}

fn split_by_color(color: Color, (b, w): (u32, u32)) -> (u32, u32) {
    match color {
        Color::Sente => (b, w),
        Color::Gote => (w, b),
    }
}

/// セッションを `session.processed_ply + 1` から `target_ply` まで進める。
/// 各plyの `lastMove`(CSA) の符号から手番側を判定し（自分の手は常に開示、
/// 相手の手は捕獲時のみ移動先マスが判明）、Observation を組み立てて記録する。
fn advance(
    session: &mut GameSession,
    positions: &HashMap<String, PositionEntry>,
    target_ply: u32,
) -> Result<(), SessionError> {
    if session.processed_ply > target_ply {
        return Err(SessionError::MissingPosition(target_ply));
    }
    for ply in (session.processed_ply + 1)..=target_ply {
        let entry = positions
            .get(&ply.to_string())
            .ok_or(SessionError::MissingPosition(ply))?;
        let raw = entry
            .last_move
            .as_deref()
            .ok_or(SessionError::MissingLastMove(ply))?;
        let parsed = parse_csa_move(raw).ok_or_else(|| SessionError::InvalidLastMove {
            ply,
            raw: raw.to_string(),
        })?;
        let info = entry.last_info.unwrap_or(0);
        let is_foul = is_foul_info(info);
        let mover = parsed.mover;

        let event = if mover == session.my_color {
            let usi = match parsed.kind {
                CsaMoveKind::Board {
                    from,
                    to,
                    role_after,
                } => {
                    // wasPromotion が欠落した場合（反則エントリで観測済み）は、
                    // 着手前に from にあった自駒の成り先と着手後の駒種2文字を
                    // 比較して成りを復元する。役に立つ比較ができない場合
                    // （from に自駒が見つからない、成れない駒種等。存在しない駒を
                    // 動かす反則など role_after が pre-role と無関係な場合を含む）
                    // は不成扱い（従来どおり）
                    let promoted = entry.was_promotion.unwrap_or_else(|| {
                        session
                            .model
                            .my_pieces()
                            .into_iter()
                            .find(|p| p.square == make_usi_square(from))
                            .and_then(|p| promote_role(p.role))
                            == Some(role_after)
                    });
                    make_usi_move(from, to, promoted)
                }
                CsaMoveKind::Drop { role, to } => {
                    make_usi_drop(role, to).ok_or_else(|| SessionError::InvalidLastMove {
                        ply,
                        raw: raw.to_string(),
                    })?
                }
                CsaMoveKind::Masked { .. } => {
                    return Err(SessionError::InvalidLastMove {
                        ply,
                        raw: raw.to_string(),
                    });
                }
            };
            if is_foul {
                Observation::MyFoul {
                    move_number: session.next_move_number,
                    usi,
                }
            } else {
                let captured = entry.last_capture.as_deref().and_then(parse_capture_letter);
                Observation::MyMove {
                    move_number: session.next_move_number,
                    usi,
                    captured,
                }
            }
        } else if is_foul {
            let count = opponent_foul_count(session, positions, ply, mover)?;
            Observation::OpponentFoul { count }
        } else {
            let captured_my_piece_at = match parsed.kind {
                CsaMoveKind::Masked { to } => to.map(make_usi_square),
                _ => {
                    return Err(SessionError::InvalidLastMove {
                        ply,
                        raw: raw.to_string(),
                    });
                }
            };
            Observation::OpponentMoved {
                move_number: session.next_move_number,
                captured_my_piece_at,
            }
        };

        session.log.record(event.clone());
        session.model.apply(&event);

        if !is_foul {
            session.next_move_number += 1;
            session.check_holder = if is_check_info(info) {
                Some(mover.other())
            } else {
                None
            };
            if is_check_info(info) {
                let in_check = mover.other();
                session.log.record(Observation::Check { in_check });
                session.model.apply(&Observation::Check { in_check });
            }
        }
        session.processed_ply = ply;
    }
    Ok(())
}

/// 相手の残り反則数（`fouls`）から累計反則回数を逆算する。開始値は0手目
/// （初期局面）の残り数から読む
fn opponent_foul_count(
    session: &GameSession,
    positions: &HashMap<String, PositionEntry>,
    ply: u32,
    mover: Color,
) -> Result<u32, SessionError> {
    let start = positions.get("0").and_then(|entry| entry.fouls);
    let current = positions
        .get(&ply.to_string())
        .and_then(|entry| entry.fouls);
    let Some((start_b, start_w)) = start.map(|f| (f.b, f.w)) else {
        return Ok(session
            .log
            .events()
            .iter()
            .filter(|event| matches!(event, Observation::OpponentFoul { .. }))
            .count() as u32
            + 1);
    };
    let Some((cur_b, cur_w)) = current.map(|f| (f.b, f.w)) else {
        return Ok(session
            .log
            .events()
            .iter()
            .filter(|event| matches!(event, Observation::OpponentFoul { .. }))
            .count() as u32
            + 1);
    };
    let (start, cur) = match mover {
        Color::Sente => (start_b, cur_b),
        Color::Gote => (start_w, cur_w),
    };
    Ok(start.saturating_sub(cur))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Role;
    use crate::shogi::{Position, ShogiMove, promote_role};
    use crate::webhook_csa::{role_to_csa2, to_csa_square};
    use crate::webhook_protocol::{
        FoulsField, GameInfo, INFO_CHECK, INFO_FOUL, INFO_NONE, RequiredPlayers,
    };

    fn initial_entry() -> PositionEntry {
        PositionEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".into(),
            fouls: Some(FoulsField { b: 9, w: 9 }),
            last_move: None,
            last_info: None,
            last_capture: None,
            was_promotion: None,
        }
    }

    fn game_info(kind: &str) -> GameInfo {
        GameInfo {
            kind: kind.into(),
            required_players: RequiredPlayers { b: 1, w: 1 },
            param: None,
        }
    }

    fn request(
        game_id: &str,
        color: &str,
        ply: u32,
        positions: HashMap<String, PositionEntry>,
    ) -> BotTurnRequest {
        BotTurnRequest {
            kind: "your_turn".into(),
            request_id: "r1".into(),
            game_id: game_id.into(),
            color: color.into(),
            number: 0,
            ply,
            deadline_ms: 5000,
            positions,
            game: game_info(SUPPORTED_GAME_TYPE),
        }
    }

    #[test]
    fn rejects_unsupported_game_type() {
        let store = SessionStore::new("heuristic".into());
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        let mut req = request("g1", "b", 0, positions);
        req.game = game_info("カスタム");
        let err = choose_move(&store, &req).unwrap_err();
        assert!(matches!(err, SessionError::UnsupportedGameType(_)));
        assert_eq!(err.status_code(), 400);
    }

    #[test]
    fn rejects_relay_format() {
        let store = SessionStore::new("heuristic".into());
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        let mut req = request("g1", "b", 0, positions);
        req.game.required_players = RequiredPlayers { b: 2, w: 1 };
        let err = choose_move(&store, &req).unwrap_err();
        assert!(matches!(err, SessionError::UnsupportedPlayers));
    }

    #[test]
    fn first_move_from_initial_position_returns_legal_looking_move() {
        let store = SessionStore::new("heuristic".into());
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        let req = request("g1", "b", 0, positions);
        let mv = choose_move(&store, &req).unwrap();
        // 7文字固定・先手番の符号
        assert_eq!(mv.len(), 7);
        assert!(mv.starts_with('+'));
        assert_eq!(store.session_count(), 1);

        // 同じrequestIdの再送は戦略を再実行せず、同じ応答を返す
        assert_eq!(choose_move(&store, &req).unwrap(), mv);
    }

    #[test]
    fn missing_fouls_use_standard_default() {
        let store = SessionStore::new("heuristic".into());
        let mut entry = initial_entry();
        entry.fouls = None;
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), entry);
        let req = request("g-default-fouls", "b", 0, positions);
        assert!(choose_move(&store, &req).is_ok());
    }

    #[test]
    fn reuses_cached_session_across_incremental_requests() {
        let store = SessionStore::new("heuristic".into());
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        let req0 = request("g1", "b", 0, positions.clone());
        choose_move(&store, &req0).unwrap();
        assert_eq!(store.session_count(), 1);

        // 黒が7六歩、白が3四歩と進んだあと、再び黒番
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2".into(),
                fouls: Some(FoulsField { b: 9, w: 9 }),
                last_move: Some("+7776FU".into()),
                last_info: Some(INFO_NONE),
                last_capture: None,
                was_promotion: Some(false),
            },
        );
        positions.insert(
            "2".to_string(),
            PositionEntry {
                sfen: "lnsgkgsnl/1r5b1/pp1ppppp/9/2p6/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3".into(),
                fouls: Some(FoulsField { b: 9, w: 9 }),
                last_move: Some("-0000ZZ".into()),
                last_info: Some(INFO_NONE),
                last_capture: None,
                was_promotion: Some(false),
            },
        );
        let req2 = request("g1", "b", 2, positions);
        let mv = choose_move(&store, &req2).unwrap();
        assert_eq!(mv.len(), 7);
        assert!(mv.starts_with('+'));
        // 同じ gameId なのでセッションは1件のまま（作り直されていない）
        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn masked_opponent_check_and_capture_updates_view() {
        let mut session = GameSession::new("heuristic", Color::Gote).unwrap();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        // 先手が後手の2bの角(初期配置)を取りつつ王手をかけた体で合成
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "ignored".into(),
                fouls: Some(FoulsField { b: 9, w: 9 }),
                last_move: Some("+0022ZZ".into()),
                last_info: Some(INFO_CHECK),
                last_capture: Some("B".into()),
                was_promotion: None,
            },
        );
        advance(&mut session, &positions, 1).unwrap();

        assert_eq!(session.check_holder, Some(Color::Gote));
        assert!(session.model.consistent());
        assert_eq!(session.model.lost_pieces().len(), 1);
        assert_eq!(session.model.lost_pieces()[0].1, Role::Bishop);
        assert_eq!(session.next_move_number, 2);

        let req = request("g2", "w", 1, positions);
        let view = build_player_view(&session, &req).unwrap();
        assert!(view.you_in_check);
        assert!(!view.opponent_in_check);
    }

    #[test]
    fn foul_retry_is_tracked_and_does_not_advance_move_number() {
        let mut session = GameSession::new("heuristic", Color::Sente).unwrap();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "ignored".into(),
                fouls: Some(FoulsField { b: 8, w: 9 }),
                last_move: Some("+9998FU".into()), // 存在しない歩を動かす反則
                last_info: Some(INFO_FOUL),
                last_capture: None,
                was_promotion: None,
            },
        );
        advance(&mut session, &positions, 1).unwrap();

        assert_eq!(session.next_move_number, 1); // 反則では手数は進まない
        assert_eq!(session.model.my_fouls(), 1);
        let foul_tried = collect_foul_tried(&session.log, session.next_move_number);
        // "+9998FU" = 99(9i)から98(9h)への移動。USIは筋+段(段はa〜iの文字)表記
        assert!(foul_tried.contains("9i9h"));
    }

    #[test]
    fn missing_was_promotion_is_recovered_from_pre_move_role() {
        let mut session = GameSession::new("heuristic", Color::Sente).unwrap();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "ignored".into(),
                fouls: Some(FoulsField { b: 9, w: 9 }),
                // 初期配置の角(8h)が2bへ成り、実戦の反則エントリ等で
                // wasPromotion が欠落したケースを模する
                last_move: Some("+8822UM".into()),
                last_info: Some(INFO_NONE),
                last_capture: None,
                was_promotion: None,
            },
        );
        advance(&mut session, &positions, 1).unwrap();

        let my_move = session.log.events().iter().find_map(|e| match e {
            Observation::MyMove { usi, .. } => Some(usi.clone()),
            _ => None,
        });
        assert_eq!(my_move.as_deref(), Some("8h2b+"));
    }

    #[test]
    fn missing_was_promotion_defaults_to_unpromoted_when_pre_role_unknown() {
        // "+9998FU" は存在しない歩を動かす反則で、from(9i)には実際には香車がいる。
        // role_after(歩)がpromote_role(香車)と一致しないため、成りとは誤認しない
        // （foul_retry_is_tracked_and_does_not_advance_move_number と同じ入力で、
        // 成り判定の観点から明示的に確認する回帰テスト）
        let mut session = GameSession::new("heuristic", Color::Sente).unwrap();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "ignored".into(),
                fouls: Some(FoulsField { b: 8, w: 9 }),
                last_move: Some("+9998FU".into()),
                last_info: Some(INFO_FOUL),
                last_capture: None,
                was_promotion: None,
            },
        );
        advance(&mut session, &positions, 1).unwrap();

        let foul_usi = session.log.events().iter().find_map(|e| match e {
            Observation::MyFoul { usi, .. } => Some(usi.clone()),
            _ => None,
        });
        assert_eq!(foul_usi.as_deref(), Some("9i9h"));
    }

    /// choose_avoiding_deduced_illegal のテスト用スタブ: 呼ばれるたびに
    /// 事前に用意した候補列から、foul_tried に含まれない最初の1手を返す
    /// （実戦略の候補生成と同じ「除外されたら次点を返す」挙動だけを模す）。
    /// 実際に choose() へ渡された foul_tried の内容も記録し、除外された
    /// 手ぶんだけ余計な情報が渡っていないかを検証できるようにする
    struct StubStrategy {
        candidates: Vec<&'static str>,
        seen_foul_tried: Vec<HashSet<String>>,
    }

    impl Strategy for StubStrategy {
        fn choose(
            &mut self,
            _view: &PlayerView,
            _log: &ObservationLog,
            foul_tried: &HashSet<String>,
        ) -> Option<String> {
            self.seen_foul_tried.push(foul_tried.clone());
            self.candidates
                .iter()
                .find(|c| !foul_tried.contains(**c))
                .map(|c| c.to_string())
        }

        fn name(&self) -> &'static str {
            "stub"
        }
    }

    fn stub_view() -> PlayerView {
        PlayerView {
            game_id: "stub".into(),
            your_color: Color::Sente,
            your_pieces: vec![],
            your_hand: HashMap::new(),
            turn: Color::Sente,
            move_number: 1,
            clocks: ClockState {
                sente_ms: 0,
                gote_ms: 0,
                running: None,
                server_time: 0,
            },
            fouls: FoulCounts {
                you: 0,
                opponent: 0,
            },
            you_in_check: false,
            opponent_in_check: false,
            status: GameStatus::Playing,
        }
    }

    #[test]
    fn choose_avoiding_deduced_illegal_returns_first_pick_when_not_excluded() {
        let mut strategy = StubStrategy {
            candidates: vec!["7g7f", "2g2f"],
            seen_foul_tried: vec![],
        };
        let view = stub_view();
        let log = ObservationLog::default();
        let mut foul_tried = HashSet::new();
        let deduced_illegal = HashSet::new();

        let chosen = choose_avoiding_deduced_illegal(
            &mut strategy,
            &view,
            &log,
            &mut foul_tried,
            &deduced_illegal,
        );

        assert_eq!(chosen.as_deref(), Some("7g7f"));
        assert_eq!(
            strategy.seen_foul_tried.len(),
            1,
            "除外がなければ1回で決まる"
        );
        assert!(
            foul_tried.is_empty(),
            "選ばれなかった除外候補まで foul_tried を汚してはいけない"
        );
    }

    #[test]
    fn choose_avoiding_deduced_illegal_retries_locally_without_polluting_foul_tried() {
        // 戦略の一番手("L*5g")が演繹的除外候補と衝突するケース。
        // ローカルで選び直し、最終的に採用したのは次点("2g2f")だけであるべきで、
        // 除外候補のうち実際に選ばれなかったもの（"B*5g"）は foul_tried に
        // 残らない（=王手ソルバーへの偽の反則証拠として混入しない）ことを確認する
        let mut strategy = StubStrategy {
            candidates: vec!["L*5g", "2g2f"],
            seen_foul_tried: vec![],
        };
        let view = stub_view();
        let log = ObservationLog::default();
        let mut foul_tried = HashSet::new();
        let mut deduced_illegal = HashSet::new();
        deduced_illegal.insert("L*5g".to_string());
        deduced_illegal.insert("B*5g".to_string());

        let chosen = choose_avoiding_deduced_illegal(
            &mut strategy,
            &view,
            &log,
            &mut foul_tried,
            &deduced_illegal,
        );

        assert_eq!(chosen.as_deref(), Some("2g2f"));
        assert_eq!(strategy.seen_foul_tried.len(), 2, "1回除外されて選び直す");
        assert!(
            foul_tried.contains("L*5g"),
            "実際に選ばれた除外候補は反則として扱う"
        );
        assert!(
            !foul_tried.contains("B*5g"),
            "選ばれなかった除外候補まで反則扱いしてはいけない"
        );
    }

    #[test]
    fn excludes_drops_on_square_where_opponent_just_captured() {
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 25,
            captured_my_piece_at: Some("5g".into()),
        });
        log.record(Observation::Check {
            in_check: Color::Sente,
        });
        // 王手回避の失敗後も相手駒は5gに残っている。
        log.record(Observation::MyFoul {
            move_number: 26,
            usi: "5h7g".into(),
        });

        let mut hand = HashMap::new();
        hand.insert(Role::Lance, 2);
        hand.insert(Role::Bishop, 1);
        let view = PlayerView {
            game_id: "known-occupied-drop".into(),
            your_color: Color::Sente,
            your_pieces: vec![crate::protocol::VisiblePiece {
                square: "5i".into(),
                role: Role::Rook,
            }],
            your_hand: hand,
            turn: Color::Sente,
            move_number: 26,
            clocks: ClockState {
                sente_ms: 0,
                gote_ms: 0,
                running: None,
                server_time: 0,
            },
            fouls: FoulCounts {
                you: 1,
                opponent: 0,
            },
            you_in_check: true,
            opponent_in_check: false,
            status: GameStatus::Playing,
        };
        let mut excluded = HashSet::new();
        exclude_moves_on_known_opponent(&log, &view, &mut excluded);

        assert!(excluded.contains("L*5g"));
        assert!(excluded.contains("B*5g"));
        assert!(excluded.contains("5i5f"));
        assert!(!excluded.contains("5i5g"));
        assert!(!excluded.contains("L*5f"));
    }

    #[test]
    fn missing_fouls_after_a_foul_are_recovered_from_history() {
        let mut session = GameSession::new("heuristic", Color::Sente).unwrap();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "ignored".into(),
                fouls: None,
                last_move: Some("+9998FU".into()),
                last_info: Some(INFO_FOUL),
                last_capture: None,
                was_promotion: None,
            },
        );
        advance(&mut session, &positions, 1).unwrap();
        let req = request("g-missing-foul-after", "b", 1, positions);
        let view = build_player_view(&session, &req).unwrap();
        assert_eq!(view.fouls.you, 1);
    }

    #[test]
    fn rejects_nonstandard_game_parameters() {
        let store = SessionStore::new("heuristic".into());
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        let mut req = request("g-custom-param", "b", 0, positions);
        req.game.param = Some("promotion_rank=4".into());
        let err = choose_move(&store, &req).unwrap_err();
        assert!(matches!(err, SessionError::UnsupportedGameParameters));
        assert_eq!(err.status_code(), 400);
    }

    #[test]
    fn accepts_standard_game_parameters() {
        let store = SessionStore::new("heuristic".into());
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        let mut req = request("g-standard-param", "b", 0, positions);
        req.game.param = Some("initial_board=lnsgkgsnl%2F1r5b1%2Fppppppppp%2F9%2F9%2F9%2FPPPPPPPPP%2F1B5R1%2FLNSGKGSNL&promotion_rank=3&draw_move_count=150&enable_try_rule=false&foul_limits=9.9".into());
        assert!(choose_move(&store, &req).is_ok());
    }

    #[test]
    fn does_not_treat_an_older_capture_square_as_still_occupied() {
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("5g".into()),
        });
        log.record(Observation::OpponentMoved {
            move_number: 11,
            captured_my_piece_at: None,
        });
        let mut hand = HashMap::new();
        hand.insert(Role::Lance, 1);
        let view = PlayerView {
            game_id: "stale-capture-square".into(),
            your_color: Color::Sente,
            your_pieces: vec![],
            your_hand: hand,
            turn: Color::Sente,
            move_number: 12,
            clocks: ClockState {
                sente_ms: 0,
                gote_ms: 0,
                running: None,
                server_time: 0,
            },
            fouls: FoulCounts {
                you: 0,
                opponent: 0,
            },
            you_in_check: false,
            opponent_in_check: false,
            status: GameStatus::Playing,
        };
        let mut excluded = HashSet::new();
        exclude_moves_on_known_opponent(&log, &view, &mut excluded);

        assert!(!excluded.contains("L*5g"));
    }

    #[test]
    fn opponent_foul_count_is_derived_from_remaining_fouls() {
        let mut session = GameSession::new("heuristic", Color::Sente).unwrap();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        positions.insert(
            "1".to_string(),
            PositionEntry {
                sfen: "ignored".into(),
                fouls: Some(FoulsField { b: 9, w: 7 }), // 後手が反則2回済み
                last_move: Some("-9998FU".into()),
                last_info: Some(INFO_FOUL),
                last_capture: None,
                was_promotion: None,
            },
        );
        advance(&mut session, &positions, 1).unwrap();
        assert_eq!(session.model.opponent_fouls(), 2);
    }

    /// 実際に合法手を連続再生して長い履歴を合成し、Kifu/scenario.rs には
    /// 頼らずパイプライン全体（履歴組み立て→戦略呼び出し→CSAエンコード）を
    /// 一気通貫で検証する。相手側の手は運営側の書式でマスクする
    fn synth_positions(plies: usize, my_color: Color) -> HashMap<String, PositionEntry> {
        let mut pos = Position::initial();
        let mut positions = HashMap::new();
        positions.insert("0".to_string(), initial_entry());
        for ply in 1..=plies {
            let mover = pos.turn();
            let legal = pos.legal_moves();
            if legal.is_empty() {
                break;
            }
            let mv = legal[ply % legal.len()];
            let pre_role = match mv {
                ShogiMove::Board { from, .. } => pos.piece_at(from).map(|p| p.role),
                ShogiMove::Drop { role, .. } => Some(role),
            };
            let captured = pos.play_unchecked(&mv);
            let gives_check = pos.in_check(pos.turn());
            let last_info = if gives_check { INFO_CHECK } else { INFO_NONE };
            let was_promotion = matches!(mv, ShogiMove::Board { promote: true, .. });

            let (last_move, last_capture) = if mover == my_color {
                let role_after = if was_promotion {
                    promote_role(pre_role.unwrap()).unwrap_or(pre_role.unwrap())
                } else {
                    pre_role.unwrap()
                };
                let sign = if mover == Color::Sente { '+' } else { '-' };
                let body = match mv {
                    ShogiMove::Board { from, to, .. } => format!(
                        "{}{}{}",
                        to_csa_square(from),
                        to_csa_square(to),
                        role_to_csa2(role_after)
                    ),
                    ShogiMove::Drop { role, to } => {
                        format!("00{}{}", to_csa_square(to), role_to_csa2(role))
                    }
                };
                (format!("{sign}{body}"), captured.map(|r| role_letter(r)))
            } else {
                let sign = if mover == Color::Sente { '+' } else { '-' };
                let masked = match (captured, mv) {
                    (Some(_), ShogiMove::Board { to, .. }) => {
                        format!("00{}ZZ", to_csa_square(to))
                    }
                    _ => "0000ZZ".to_string(),
                };
                (format!("{sign}{masked}"), captured.map(|r| role_letter(r)))
            };

            positions.insert(
                ply.to_string(),
                PositionEntry {
                    sfen: "ignored".into(),
                    fouls: Some(FoulsField { b: 9, w: 9 }),
                    last_move: Some(last_move),
                    last_info: Some(last_info),
                    last_capture,
                    was_promotion: Some(was_promotion),
                },
            );
        }
        positions
    }

    fn role_letter(role: Role) -> String {
        // lastCapture は1文字USI駒コード（常に不成の基本形）
        match crate::shogi::unpromote_role(role) {
            Role::Pawn => "P",
            Role::Lance => "L",
            Role::Knight => "N",
            Role::Silver => "S",
            Role::Gold => "G",
            Role::Bishop => "B",
            Role::Rook => "R",
            Role::King => "K",
            _ => unreachable!("unpromote_role は基本形を返す"),
        }
        .to_string()
    }

    #[test]
    fn long_synthetic_history_replays_cold_start_with_heuristic() {
        let store = SessionStore::new("heuristic".into());
        let positions = synth_positions(60, Color::Sente);
        let last_ply = positions
            .keys()
            .filter_map(|k| k.parse::<u32>().ok())
            .max()
            .unwrap();
        let req = request("g-long", "b", last_ply, positions);
        let mv = choose_move(&store, &req).unwrap();
        assert_eq!(mv.len(), 7);
    }

    /// estimator_v10 での実測（手動実行用）。particle filter を含むため遅く、
    /// 通常の `cargo test` では走らせない: `cargo test -- --ignored` で確認する
    #[test]
    #[ignore]
    fn long_synthetic_history_replays_cold_start_with_estimator_v10_within_deadline() {
        let store = SessionStore::new("estimator_v10".into());
        let positions = synth_positions(80, Color::Sente);
        let last_ply = positions
            .keys()
            .filter_map(|k| k.parse::<u32>().ok())
            .max()
            .unwrap();
        let req = request("g-long-v10", "b", last_ply, positions);
        let start = std::time::Instant::now();
        let mv = choose_move(&store, &req).unwrap();
        let elapsed = start.elapsed();
        println!("estimator_v10 cold-start replay ({last_ply} plies) took {elapsed:?} -> {mv}");
        assert_eq!(mv.len(), 7);
        assert!(
            elapsed < Duration::from_secs(10),
            "cold start exceeded webhook budget"
        );
    }
}
