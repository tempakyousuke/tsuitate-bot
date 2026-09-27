//! Python から n 局を並列に回す RL 環境（`tsuitate_rl.VecEnv`）。
//!
//! 1局ぶんの進行（相手戦略の自動着手・マスク・終局）は本体の `tsuitate_bot::rl::env::EnvGame`。
//! ここは並列化（rayon、GIL を外して回す）と numpy 変換だけを持つ薄いラッパー。
//! 使い方と罠（報酬の色は step の前に控える等）は rl-env/README.md。
//!
//! 終局した局は `step` の中で自動的に新しい局へ差し替わる（次の `observe` は新しい局の初手）。

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
    #[pyo3(signature = (n, opponent=None, seed=0))]
    fn new(py: Python<'_>, n: usize, opponent: Option<String>, seed: u64) -> PyResult<Self> {
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
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.games.len()
    }

    /// 各局の今の手番の (観測, マスク, 手番の色)
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
                    m.copy_from_slice(g.mask());
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
    /// 進めずに `ValueError` を返す（一部の局だけ進むと、終局した局の報酬が失われる）
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
        let mut actions = Vec::with_capacity(n);
        let mut errors = vec![];
        for (i, (&a, g)) in raw.iter().zip(&self.games).enumerate() {
            match usize::try_from(a) {
                Ok(a) => match g.check_action(a) {
                    Ok(()) => actions.push(a),
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
        } = self;
        let (rewards, done, internal) = py.detach(|| {
            let results: Vec<_> = games
                .par_iter_mut()
                .zip(actions.par_iter())
                .map(|(g, &a)| g.step(a))
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
            // 終局した局を記録し、新しい局へ差し替える（新しい局の構築も並列）
            let ended: Vec<usize> = (0..n).filter(|&i| done[i]).collect();
            for &i in &ended {
                let old = &games[i];
                finished.push(Finished {
                    game_no: old.game_no(),
                    learner: spec.learner(old.game_no()),
                    outcome: old.outcome().expect("done の局は終局している"),
                });
            }
            let fresh = build_fresh(spec, next_game_no, finished, ended.len());
            for (i, game) in ended.into_iter().zip(fresh) {
                games[i] = game;
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

#[pymodule]
fn tsuitate_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("NUM_PLANES", NUM_PLANES)?;
    Ok(())
}
