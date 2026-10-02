//! Python から n 局を並列に回す RL 環境（`tsuitate_rl.VecEnv`）。
//!
//! 1局ぶんの進行（相手戦略の自動着手・マスク・終局）は本体の `tsuitate_bot::rl::env::EnvGame`。
//! ここは並列化（rayon、GIL を外して回す）と numpy 変換だけを持つ薄いラッパー。
//! 使い方と罠（報酬の色は step の前に控える等）は rl-env/README.md。
//!
//! 終局した局は `step` の中で自動的に新しい局へ差し替わる（次の `observe` は新しい局の初手）。
//! `auto_reset=False` なら終局した局はそのまま止まり（行動は -1 を渡す）、`reset()` で
//! まとめて新しい局にする（R-NaD のように1局を丸ごと集める学習用）。

use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use tsuitate_bot::protocol::Color;
use tsuitate_bot::rl::action::NUM_ACTIONS;
use tsuitate_bot::rl::encode::{NUM_PLANES, OBS_LEN};
use tsuitate_bot::rl::env::{EnvGame, Outcome};
use tsuitate_bot::selfplay::{GameResult, mix};
use tsuitate_bot::strategy::{self, Strategy};

type ObserveOut<'py> = (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray1<i8>>,
);
type StepOut<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<bool>>);

/// 終局した1局の記録
struct Finished {
    game_no: u32,
    learner: Option<Color>,
    outcome: Outcome,
}

/// 新しい局を作るのに要る設定（`VecEnv` 本体は Sync でないので、並列に作るときはこれだけ渡す）
#[derive(Clone)]
struct GameSpec {
    /// 評価モードの相手（`strategy::make` の名前）。None なら自己対局
    opponent: Option<String>,
    seed: u64,
}

impl GameSpec {
    fn learner(&self, game_no: u32) -> Option<Color> {
        self.opponent.as_ref().map(|_| learner_of(game_no))
    }

    fn build(&self, game_no: u32) -> EnvGame {
        match &self.opponent {
            None => EnvGame::selfplay(None, game_no),
            Some(name) => {
                let opp = make_opponent(name, mix(self.seed ^ mix(u64::from(game_no))))
                    .expect("相手の名前は new で検査済み");
                EnvGame::versus(None, learner_of(game_no), opp, game_no)
            }
        }
    }
}

#[pyclass(unsendable)]
struct VecEnv {
    games: Vec<EnvGame>,
    spec: GameSpec,
    next_game_no: u32,
    finished: Vec<Finished>,
    /// 終局した局を step の中で新しい局へ差し替えるか
    auto_reset: bool,
}

fn color_code(c: Color) -> i8 {
    match c {
        Color::Sente => 0,
        Color::Gote => 1,
    }
}

/// 評価モードの学習側の色（偶数局で先手）
fn learner_of(game_no: u32) -> Color {
    if game_no % 2 == 0 {
        Color::Sente
    } else {
        Color::Gote
    }
}

fn make_opponent(name: &str, seed: u64) -> Option<Box<dyn Strategy + Send>> {
    strategy::make_seeded(name, seed).or_else(|| strategy::make(name))
}

/// `count` 局を新しく作る（GIL の外で呼ぶ）。局番号は連番で割り当て、構築（評価モードでは
/// 相手の初手まで）は rayon で並列に行う。作った直後に終局した局（相手の即投了など）は
/// `finished` に記録して作り直す。返す局は局番号の昇順
fn build_fresh(
    spec: &GameSpec,
    next_game_no: &mut u32,
    finished: &mut Vec<Finished>,
    count: usize,
) -> Vec<EnvGame> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let need = count - out.len();
        let first = *next_game_no;
        *next_game_no += need as u32;
        let built: Vec<EnvGame> = (first..first + need as u32)
            .into_par_iter()
            .map(|g| spec.build(g))
            .collect();
        for game in built {
            match game.outcome() {
                None => out.push(game),
                Some(outcome) => finished.push(Finished {
                    game_no: game.game_no(),
                    learner: spec.learner(game.game_no()),
                    outcome,
                }),
            }
        }
    }
    out
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, opponent=None, seed=0, auto_reset=true))]
    fn new(
        py: Python<'_>,
        n: usize,
        opponent: Option<String>,
        seed: u64,
        auto_reset: bool,
    ) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("n は 1 以上"));
        }
        if let Some(name) = &opponent
            && make_opponent(name, 0).is_none()
        {
            return Err(PyValueError::new_err(format!("未知の戦略: {name}")));
        }
        let spec = GameSpec { opponent, seed };
        let mut next_game_no = 0;
        let mut finished = vec![];
        let games = py.detach(|| build_fresh(&spec, &mut next_game_no, &mut finished, n));
        Ok(VecEnv {
            games,
            spec,
            next_game_no,
            finished,
            auto_reset,
        })
    }

    /// 全局を新しい局にする（途中の局は記録せずに捨てる）
    fn reset(&mut self, py: Python<'_>) {
        let n = self.games.len();
        let VecEnv {
            games,
            spec,
            next_game_no,
            finished,
            ..
        } = self;
        *games = py.detach(|| build_fresh(spec, next_game_no, finished, n));
    }

    /// 各局がまだ終局していないか（`auto_reset=False` で止まった局は False）
    fn alive<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let v: Vec<bool> = self.games.iter().map(|g| g.outcome().is_none()).collect();
        PyArray1::from_vec(py, v)
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.games.len()
    }

    /// 各局の今の手番の (観測, マスク, 手番の色)。終局して止まっている局のマスクは空
    // &mut なのは rayon で局を分けるため（相手戦略は Send だが Sync ではない）
    fn observe<'py>(&mut self, py: Python<'py>) -> PyResult<ObserveOut<'py>> {
        let n = self.games.len();
        let games = &mut self.games;
        let (obs, mask) = py.detach(|| {
            let mut obs = vec![0.0f32; n * OBS_LEN];
            let mut mask = vec![false; n * NUM_ACTIONS];
            obs.par_chunks_mut(OBS_LEN)
                .zip(mask.par_chunks_mut(NUM_ACTIONS))
                .zip(games.par_iter_mut())
                .for_each(|((o, m), g): ((&mut [f32], &mut [bool]), &mut EnvGame)| {
                    g.observe_into(o);
                    // 終局して止まっている局（auto_reset=False）はマスクが空 = 全部 false のまま
                    if !g.mask().is_empty() {
                        m.copy_from_slice(g.mask());
                    }
                });
            (obs, mask)
        });
        let player: Vec<i8> = self.games.iter().map(|g| color_code(g.to_move())).collect();
        Ok((
            PyArray1::from_vec(py, obs).reshape([n, NUM_PLANES, 9, 9])?,
            PyArray1::from_vec(py, mask).reshape([n, NUM_ACTIONS])?,
            PyArray1::from_vec(py, player),
        ))
    }

    /// 各局に行動を1つずつ適用する。終局した局は [先手,後手] の報酬と done=True を返し、
    /// 新しい局へ差し替える。
    ///
    /// **行動はバッチ全体を先に検査する**: 1つでも範囲外・マスク外があれば、どの局も
    /// 進めずに `ValueError` を返す（一部の局だけ進むと、終局した局の報酬が失われる）。
    /// `auto_reset=False` で終局して止まっている局には -1 を渡す（それ以外の値は拒否）
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<StepOut<'py>> {
        let n = self.games.len();
        let raw = actions.as_slice()?;
        if raw.len() != n {
            return Err(PyValueError::new_err(format!(
                "actions の長さ {} が局数 {n} と違う",
                raw.len()
            )));
        }
        let mut actions: Vec<Option<usize>> = Vec::with_capacity(n);
        let mut errors = vec![];
        for (i, (&a, g)) in raw.iter().zip(&self.games).enumerate() {
            if g.outcome().is_some() {
                // auto_reset=False で止まっている局（auto_reset=True では起きない）
                if a == -1 {
                    actions.push(None);
                } else {
                    errors.push(format!("env {i}: 終局済みの局には -1 を渡す（{a}）"));
                }
                continue;
            }
            match usize::try_from(a) {
                Ok(a) => match g.check_action(a) {
                    Ok(()) => actions.push(Some(a)),
                    Err(e) => errors.push(format!("env {i}: {e:?}")),
                },
                Err(_) => errors.push(format!("env {i}: 負の行動 {a}")),
            }
        }
        if !errors.is_empty() {
            return Err(PyValueError::new_err(errors.join(" / ")));
        }

        let VecEnv {
            games,
            spec,
            next_game_no,
            finished,
            auto_reset,
        } = self;
        let auto_reset = *auto_reset;
        let (rewards, done, internal) = py.detach(|| {
            let results: Vec<_> = games
                .par_iter_mut()
                .zip(actions.par_iter())
                .map(|(g, &a)| match a {
                    Some(a) => g.step(a),
                    None => Ok(None),
                })
                .collect();
            let mut rewards = vec![0.0f32; n * 2];
            let mut done = vec![false; n];
            let mut internal = vec![];
            for (i, r) in results.into_iter().enumerate() {
                match r {
                    Ok(None) => {}
                    Ok(Some(outcome)) => {
                        rewards[2 * i..2 * i + 2].copy_from_slice(&outcome.rewards());
                        done[i] = true;
                    }
                    // 検査済みなので起きない（起きたら環境の不具合）
                    Err(e) => internal.push(format!("env {i}: {e:?}")),
                }
            }
            // 終局した局を記録し、auto_reset なら新しい局へ差し替える（新しい局の構築も並列）
            let ended: Vec<usize> = (0..n).filter(|&i| done[i]).collect();
            for &i in &ended {
                let old = &games[i];
                finished.push(Finished {
                    game_no: old.game_no(),
                    learner: spec.learner(old.game_no()),
                    outcome: old.outcome().expect("done の局は終局している"),
                });
            }
            if auto_reset {
                let fresh = build_fresh(spec, next_game_no, finished, ended.len());
                for (i, game) in ended.into_iter().zip(fresh) {
                    games[i] = game;
                }
            }
            (rewards, done, internal)
        });
        if !internal.is_empty() {
            return Err(PyRuntimeError::new_err(internal.join(" / ")));
        }
        Ok((
            PyArray1::from_vec(py, rewards).reshape([n, 2])?,
            PyArray1::from_vec(py, done),
        ))
    }

    /// 評価モードで各局の学習側の色（0=先手 1=後手）。自己対局では -1。
    /// **今の局の色**なので、`step` が返す報酬の列を選ぶには step の前に取っておくこと
    fn learner_colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        let v: Vec<i8> = self
            .games
            .iter()
            .map(|g| self.spec.learner(g.game_no()).map_or(-1, color_code))
            .collect();
        PyArray1::from_vec(py, v)
    }

    /// 前回の呼び出し以降に終局した局の記録を取り出す
    fn pop_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = vec![];
        for f in self.finished.drain(..) {
            let d = PyDict::new(py);
            d.set_item("game_no", f.game_no)?;
            let winner = match f.outcome.result {
                GameResult::Win(c) => Some(color_code(c)),
                GameResult::Draw => None,
            };
            d.set_item("winner", winner)?;
            d.set_item("reason", f.outcome.reason)?;
            d.set_item("plies", f.outcome.plies)?;
            d.set_item("fouls", f.outcome.fouls.to_vec())?;
            d.set_item("learner", f.learner.map(color_code))?;
            out.push(d);
        }
        Ok(out)
    }
}

/// 対局記録（JSONL）を教師にする模倣学習のデータセット（M2a）。
///
/// 読み込み時に真実を `Referee` で再生し、**記録した側の観測が記録と一致し、記録の勝敗が
/// 再生の裁定と一致する局だけ**を残す（食い違う局は `skipped` に数える。勝敗は価値の教師になるので、
/// 勝者の取り違え・読めない結果・途中で切れた記録を通さない）。エンコードは局単位で `encode_games` を呼ぶ
/// （試行ごとに観測を保持すると重いので、呼ぶたびに再生し直す。局単位で並列）。
#[pyclass(unsendable)]
struct RecordDataset {
    games: Vec<LoadedGame>,
    skipped: Vec<(String, String)>,
}

struct LoadedGame {
    path: String,
    end: tsuitate_bot::protocol::GameEndPayload,
    /// 検査済みの勝敗（価値の教師の元）
    result: GameResult,
    /// 試行数
    attempts: usize,
    /// 着手列と反則試行列の署名（同じ棋譜を学習/検証の両方へ入れないための分割キー）
    signature: String,
}

fn game_signature(end: &tsuitate_bot::protocol::GameEndPayload) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for m in &end.moves {
        m.usi.hash(&mut h);
    }
    for f in &end.foul_attempts {
        (f.move_number, &f.usi).hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

/// 1局の全試行をエンコードした結果
#[derive(Default)]
struct Encoded {
    obs: Vec<f32>,
    mask: Vec<bool>,
    action: Vec<i64>,
    side: Vec<i8>,
    value: Vec<f32>,
    foul: Vec<bool>,
    game: Vec<i32>,
    /// マスクの外にあった教師の手。観測から反則が確定する手（`action::legal_mask` の除外）を
    /// 教師が試みたぶんだけ出る（それ以外で出ればエンコードかマスクの不具合）
    outside_mask: usize,
}

fn encode_game(g: &LoadedGame, game_idx: usize) -> Encoded {
    use tsuitate_bot::rl::action::{encode_usi, legal_mask};
    use tsuitate_bot::rl::encode::encode_into;
    use tsuitate_bot::rl::records::replay;
    let (end, result) = (&g.end, g.result);
    let mut e = Encoded::default();
    e.obs.reserve(g.attempts * OBS_LEN);
    e.mask.reserve(g.attempts * NUM_ACTIONS);
    let _ = replay(end, |r, a| {
        let view = r.view(a.side, [0, 0], game_idx as u32);
        let mask = legal_mask(&view, r.log(a.side), r.foul_tried(a.side));
        let Some(action) = encode_usi(&a.usi, a.side).filter(|&x| mask[x]) else {
            e.outside_mask += 1;
            return;
        };
        let start = e.obs.len();
        e.obs.resize(start + OBS_LEN, 0.0);
        encode_into(&view, r.log(a.side), r.foul_tried(a.side), &mut e.obs[start..]);
        e.mask.extend_from_slice(&mask);
        e.action.push(action as i64);
        e.side.push(color_code(a.side));
        e.value.push(match result {
            GameResult::Win(c) if c == a.side => 1.0,
            GameResult::Win(_) => -1.0,
            GameResult::Draw => 0.0,
        });
        e.foul.push(a.foul);
        e.game.push(game_idx as i32);
    });
    e
}

type BatchOut<'py> = (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<i32>>,
);

#[pymethods]
impl RecordDataset {
    #[new]
    fn new(py: Python<'_>, paths: Vec<String>) -> Self {
        use tsuitate_bot::rl::records::{
            recorded_observations, replay, same_observations, verify_outcome,
        };
        let loaded: Vec<_> = py.detach(|| {
            paths
                .par_iter()
                .map(|p| {
                    let content = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
                    let (color, end) = tsuitate_bot::truth_replay::parse_bot_and_end(&content)
                        .ok_or("match/end 行が無い")?;
                    let mut n = 0usize;
                    let r = replay(&end, |_, _| n += 1).map_err(|e| format!("{e:?}"))?;
                    if !same_observations(r.log(color).events(), &recorded_observations(&content)) {
                        return Err("再生した観測が記録と食い違う".to_string());
                    }
                    let result = verify_outcome(&end, &r).map_err(|e| format!("{e:?}"))?;
                    Ok(LoadedGame {
                        path: p.clone(),
                        signature: game_signature(&end),
                        end,
                        result,
                        attempts: n,
                    })
                })
                .collect()
        });
        let mut ds = RecordDataset {
            games: vec![],
            skipped: vec![],
        };
        for (p, r) in paths.into_iter().zip(loaded) {
            match r {
                Ok(g) => ds.games.push(g),
                Err(e) => ds.skipped.push((p, e)),
            }
        }
        ds
    }

    #[getter]
    fn num_games(&self) -> usize {
        self.games.len()
    }

    /// 全局の試行数の合計（= 標本数の上限。マスク外の手があればその分少なくなる）
    #[getter]
    fn num_attempts(&self) -> usize {
        self.games.iter().map(|g| g.attempts).sum()
    }

    /// 各局の試行数（学習率スケジュールの総更新数の計算用）
    #[getter]
    fn game_attempts(&self) -> Vec<usize> {
        self.games.iter().map(|g| g.attempts).collect()
    }

    /// 各局の棋譜の署名（着手列＋反則試行列）。同じ署名の局は学習/検証の同じ側へ入れる
    #[getter]
    fn signatures(&self) -> Vec<String> {
        self.games.iter().map(|g| g.signature.clone()).collect()
    }

    /// 読み込めなかった局の (パス, 理由)
    #[getter]
    fn skipped(&self) -> Vec<(String, String)> {
        self.skipped.clone()
    }

    fn path(&self, game: usize) -> Option<String> {
        self.games.get(game).map(|g| g.path.clone())
    }

    /// 指定した局の全試行を (観測, マスク, 教師の行動, 手番の色, 価値の教師, 反則だったか, 局番号)
    /// で返す。価値の教師は手番側から見た終局の結果（勝ち +1 / 負け −1 / 引き分け 0）。
    /// 教師の手がマスクの外にあった試行は落とす（数は `outside_mask` の戻り値で分かる）
    fn encode_games<'py>(
        &self,
        py: Python<'py>,
        games: Vec<usize>,
    ) -> PyResult<(BatchOut<'py>, usize)> {
        if let Some(&g) = games.iter().find(|&&g| g >= self.games.len()) {
            return Err(PyValueError::new_err(format!("局番号 {g} は範囲外")));
        }
        let all = &self.games;
        let parts: Vec<Encoded> = py.detach(|| {
            games
                .par_iter()
                .map(|&g| encode_game(&all[g], g))
                .collect()
        });
        let total: usize = parts.iter().map(|p| p.action.len()).sum();
        let mut e = Encoded::default();
        e.obs.reserve_exact(total * OBS_LEN);
        e.mask.reserve_exact(total * NUM_ACTIONS);
        for p in parts {
            e.obs.extend(p.obs);
            e.mask.extend(p.mask);
            e.action.extend(p.action);
            e.side.extend(p.side);
            e.value.extend(p.value);
            e.foul.extend(p.foul);
            e.game.extend(p.game);
            e.outside_mask += p.outside_mask;
        }
        let n = e.action.len();
        Ok((
            (
                PyArray1::from_vec(py, e.obs).reshape([n, NUM_PLANES, 9, 9])?,
                PyArray1::from_vec(py, e.mask).reshape([n, NUM_ACTIONS])?,
                PyArray1::from_vec(py, e.action),
                PyArray1::from_vec(py, e.side),
                PyArray1::from_vec(py, e.value),
                PyArray1::from_vec(py, e.foul),
                PyArray1::from_vec(py, e.game),
            ),
            e.outside_mask,
        ))
    }
}

#[pymodule]
fn tsuitate_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<RecordDataset>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("NUM_PLANES", NUM_PLANES)?;
    Ok(())
}
