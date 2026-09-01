//! **戦力投資の限界効用の頻度・分類**（issue #40 P0-2 の粒子不要側）。
//!
//! 対局記録（`records/*.jsonl` / arena-records）の真実を再生し、bot 側の
//! 受理された各手について `marginal_work` の帯×しきい値横断表を作って、
//! - **link-only drop**: 紐は増えるが、需要帯（own_king / opp_king /
//!   backed_target / active_own_piece）への first/second が 0 の打ち
//! - **saturated promotion**: 非捕獲・非王手の成りで、正の利き増分の 70% 以上が
//!   neutral × redundant に入る手（70% は頻度記述用の閾値。P1 の hard gate ではない）
//! を相手別・手数帯別に数える。**粒子を回さないので一瞬で終わる**。
//! shadow 順位（score の由来）は粒子が要るので別経路
//! （`rank_probe` / `CandidateScore` の `promote_bias` / `drop_bias` 列）。
//!
//! 使い方:
//! `cargo run --release --bin investment_probe -- [--dump out.tsv] <records...>`
//!
//! - 引数はファイルでもディレクトリでもよい（ディレクトリは中の *.jsonl 全部）
//! - `--dump <path>`: 1行=1受理手の TSV も書く（両極端の手の抽出・レビュー用）
//!
//! 分類の定義は `src/marginal_work.rs`（事前登録）を参照。「正の link を得る」は
//! 働き係数で揺れる連続値でなく**紐の本数**（`defended_piece_count`）の増加で
//! 判定する（構造判定。runtime の `link` 列との突き合わせは shadow 側で行う）。

use std::collections::BTreeMap;
use std::io::Write;

use tsuitate_bot::marginal_work::{
    BAND_LABELS, Band, MarginalWorkBreakdown, N_BANDS, N_TRANSITIONS, TRANSITION_TAGS,
    attack_counts, attack_counts_after, band_map, breakdown, defended_piece_count,
};
use tsuitate_bot::board::{Promotion, promotion_choice};
use tsuitate_bot::protocol::Role;
use tsuitate_bot::shogi::{ShogiMove, parse_usi, piece_value, unpromote_role};
use tsuitate_bot::truth_replay::{for_each_decision_full, parse_bot_and_end, side_idx};

/// 手数帯（両者の受理手数 = plies で割る。analyze の終盤 90+ と同じ区切り）
const PHASES: usize = 3;
const PHASE_LABELS: [&str; PHASES] = ["0-49", "50-89", "90+"];

fn phase_of(plies: u32) -> usize {
    match plies {
        0..=49 => 0,
        50..=89 => 1,
        _ => 2,
    }
}

fn exchange_value(role: Role) -> f64 {
    (piece_value(role) + piece_value(unpromote_role(role))) / 2.0
}

#[derive(Default)]
struct Tally {
    games: u32,
    decisions: [u64; PHASES],
    drops: [u64; PHASES],
    link_only_drops: [u64; PHASES],
    promos_optional: [u64; PHASES],
    promos_forced: [u64; PHASES],
    saturated_promos: [u64; PHASES],
    /// 打ちの帯×遷移の合算
    drop_bd: MarginalWorkBreakdown,
    /// 任意成りの帯×遷移の合算（着手前 → 成りの後）
    promo_bd: MarginalWorkBreakdown,
    /// 任意成りの**不成双子比**（不成の後 → 成りの後 = 成ったことで増えた分）
    promo_twin_bd: MarginalWorkBreakdown,
    /// opp_king 帯の監査
    opp_king_fired: u64,
    opp_king_band_size_sum: u64,
    opp_king_cand_sum: u64,
}

fn bd_table(bd: &MarginalWorkBreakdown) -> String {
    let mut s = String::new();
    s.push_str("| 帯 | ");
    for t in TRANSITION_TAGS {
        s.push_str(&format!("{t}+ | "));
    }
    for t in TRANSITION_TAGS {
        s.push_str(&format!("{t}- | "));
    }
    s.push('\n');
    s.push_str(&format!("|---|{}\n", "---|".repeat(N_TRANSITIONS * 2)));
    for b in 0..N_BANDS {
        s.push_str(&format!("| {} | ", BAND_LABELS[b]));
        for t in 0..N_TRANSITIONS {
            s.push_str(&format!("{} | ", bd.gain[b][t]));
        }
        for t in 0..N_TRANSITIONS {
            s.push_str(&format!("{} | ", bd.loss[b][t]));
        }
        s.push('\n');
    }
    s
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut dump_path: Option<String> = None;
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        args.remove(i);
        if i < args.len() {
            dump_path = Some(args.remove(i));
        } else {
            eprintln!("--dump にはパスが要ります");
            std::process::exit(2);
        }
    }
    if args.is_empty() {
        eprintln!(
            "usage: investment_probe [--dump out.tsv] <records/*.jsonl | dir>..."
        );
        std::process::exit(2);
    }

    let mut files: Vec<std::path::PathBuf> = vec![];
    for a in &args {
        let p = std::path::Path::new(a);
        if p.is_dir() {
            let mut v: Vec<_> = std::fs::read_dir(p)
                .expect("ディレクトリを読めません")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p.to_path_buf());
        }
    }

    let mut dump = dump_path.map(|p| {
        let mut f = std::fs::File::create(&p).expect("dump ファイルを作れません");
        writeln!(
            f,
            "file\topponent\tplies\tmove_number\tusi\tkind\trole\texchange_value\tcaptured\tgives_check\tdemand_fs_gain\tneutral_first\tneutral_second\tneutral_redundant\ttotal_gain\ttotal_loss\tnr_frac\tlink_delta\tlink_only\tsaturated\topp_king_gated\topp_king_band"
        )
        .unwrap();
        (f, p)
    });

    let mut by_opp: BTreeMap<String, Tally> = BTreeMap::new();
    let mut broken = 0u32;
    let mut loaded = 0u32;

    for path in &files {
        let Ok(content) = std::fs::read_to_string(path) else {
            broken += 1;
            continue;
        };
        let Some((bot, end)) = parse_bot_and_end(&content) else {
            broken += 1;
            continue;
        };
        let opp_name = end.opponent.username.clone();
        let tally = by_opp.entry(opp_name.clone()).or_default();
        tally.games += 1;
        loaded += 1;

        // クロージャの借用を分けるため、決定ごとの行はいったん貯める
        struct Row {
            plies: u32,
            move_number: u32,
            usi: String,
            kind: &'static str,
            role: Role,
            captured: bool,
            gives_check: bool,
            bd: MarginalWorkBreakdown,
            twin_bd: Option<MarginalWorkBreakdown>,
            link_delta: i64,
            opp_king_gated: bool,
            opp_king_band: u32,
            opp_king_cands: u32,
        }
        let mut rows: Vec<Row> = vec![];

        let ok = for_each_decision_full(&end, |d| {
            if d.side != bot {
                return;
            }
            let mv_rec = &end.moves[d.decision_id as usize];
            let Some(mv) = parse_usi(&mv_rec.usi) else { return };
            let pieces = d.pos.pieces_of(bot);
            let log = &d.logs[side_idx(bot)];
            let you_in_check = d.pos.in_check(bot);
            let bm = band_map(&pieces, bot, log, d.pos.move_number(), you_in_check);
            let before = attack_counts(&pieces, bot);
            let after = attack_counts_after(&pieces, &mv, bot);
            let bd = breakdown(&bm, &before, &after);

            // 真実由来の事後記述（runtime 特徴量ではない）: 捕獲と王手宣言
            let (captured, gives_check) = {
                let mut p = d.pos.clone();
                let cap = p.play_unchecked(&mv);
                (cap.is_some(), p.in_check(bot.other()))
            };

            let (kind, role, twin_bd, link_delta) = match mv {
                ShogiMove::Drop { role, .. } => {
                    let link_delta = i64::from(defended_piece_count(
                        &tsuitate_bot::check_prep::pieces_after(&pieces, &mv),
                        bot,
                    )) - i64::from(defended_piece_count(&pieces, bot));
                    ("drop", role, None, link_delta)
                }
                ShogiMove::Board { from, to, promote } => {
                    let pre_role = pieces
                        .iter()
                        .find(|p| {
                            tsuitate_bot::board::parse_usi_square(&p.square) == Some(from)
                        })
                        .map(|p| p.role)
                        .unwrap_or(Role::King);
                    if promote {
                        let choice = promotion_choice(pre_role, from, to, bot);
                        let kind = if choice == Promotion::Forced {
                            "promo_forced"
                        } else {
                            "promo_optional"
                        };
                        // 不成双子（同じ from/to、promote=false）比: 成ったことで
                        // 増えた需要帯別の限界利きだけを取り出す
                        let twin = ShogiMove::Board { from, to, promote: false };
                        let after_twin = attack_counts_after(&pieces, &twin, bot);
                        let twin_bd = breakdown(&bm, &after_twin, &after);
                        (kind, pre_role, Some(twin_bd), 0)
                    } else {
                        ("board", pre_role, None, 0)
                    }
                }
            };
            rows.push(Row {
                plies: d.plies,
                move_number: d.pos.move_number(),
                usi: mv_rec.usi.clone(),
                kind,
                role,
                captured,
                gives_check,
                bd,
                twin_bd,
                link_delta,
                opp_king_gated: bm.opp_king_gated,
                opp_king_band: bm.opp_king_band_size,
                opp_king_cands: bm.opp_king_candidates,
            });
        });
        if !ok {
            broken += 1;
            tally.games -= 1;
            continue;
        }

        for r in rows {
            let ph = phase_of(r.plies);
            tally.decisions[ph] += 1;
            tally.opp_king_fired += u64::from(r.opp_king_gated);
            tally.opp_king_band_size_sum += u64::from(r.opp_king_band);
            tally.opp_king_cand_sum += u64::from(r.opp_king_cands);
            let mut link_only = false;
            let mut saturated = false;
            match r.kind {
                "drop" => {
                    tally.drops[ph] += 1;
                    tally.drop_bd.add(&r.bd);
                    link_only = r.link_delta > 0 && r.bd.demand_first_second_gain() == 0;
                    if link_only {
                        tally.link_only_drops[ph] += 1;
                    }
                }
                "promo_optional" | "promo_forced" => {
                    if r.kind == "promo_optional" {
                        tally.promos_optional[ph] += 1;
                        tally.promo_bd.add(&r.bd);
                        if let Some(t) = &r.twin_bd {
                            tally.promo_twin_bd.add(t);
                        }
                    } else {
                        tally.promos_forced[ph] += 1;
                    }
                    saturated = !r.captured
                        && !r.gives_check
                        && r.bd.neutral_redundant_frac().is_some_and(|f| f >= 0.7);
                    if saturated {
                        tally.saturated_promos[ph] += 1;
                    }
                }
                _ => {}
            }
            if let Some((f, _)) = dump.as_mut() {
                let nr_frac = r
                    .bd
                    .neutral_redundant_frac()
                    .map_or("".to_string(), |f| format!("{f:.3}"));
                let n = Band::Neutral as usize;
                writeln!(
                    f,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    path.display(),
                    opp_name,
                    r.plies,
                    r.move_number,
                    r.usi,
                    r.kind,
                    r.role,
                    exchange_value(r.role),
                    u8::from(r.captured),
                    u8::from(r.gives_check),
                    r.bd.demand_first_second_gain(),
                    r.bd.gain[n][0],
                    r.bd.gain[n][1],
                    r.bd.gain[n][2],
                    r.bd.total_gain(),
                    r.bd.total_loss(),
                    nr_frac,
                    r.link_delta,
                    u8::from(link_only),
                    u8::from(saturated),
                    u8::from(r.opp_king_gated),
                    r.opp_king_band,
                )
                .unwrap();
            }
        }
    }

    println!("## investment_probe（issue #40 P0-2 の頻度・分類、粒子不要）");
    println!();
    println!(
        "対象 {loaded} 局（読めなかった/壊れた記録 {broken}）。分類の定義は src/marginal_work.rs"
    );
    for (opp, t) in &by_opp {
        let d_total: u64 = t.decisions.iter().sum();
        let drops: u64 = t.drops.iter().sum();
        let lod: u64 = t.link_only_drops.iter().sum();
        let po: u64 = t.promos_optional.iter().sum();
        let pf: u64 = t.promos_forced.iter().sum();
        let sat: u64 = t.saturated_promos.iter().sum();
        println!();
        println!("### vs {opp}（{} 局・bot 決定点 {d_total}）", t.games);
        println!();
        println!("| 手数帯 | 決定点 | 打ち | link-only打ち | 任意成り | 強制成り | 飽和成り |");
        println!("|---|---|---|---|---|---|---|");
        for ph in 0..PHASES {
            println!(
                "| {} | {} | {} | {} | {} | {} | {} |",
                PHASE_LABELS[ph],
                t.decisions[ph],
                t.drops[ph],
                t.link_only_drops[ph],
                t.promos_optional[ph],
                t.promos_forced[ph],
                t.saturated_promos[ph],
            );
        }
        let pct = |num: u64, den: u64| -> String {
            if den == 0 {
                "-".into()
            } else {
                format!("{:.1}%", 100.0 * num as f64 / den as f64)
            }
        };
        println!();
        println!(
            "- link-only打ち: 全打ちの {}・{:.2} 回/局（発火3%未満なら「現象が稀」= P0-3 の中止側）",
            pct(lod, drops),
            lod as f64 / f64::from(t.games.max(1)),
        );
        println!(
            "- 飽和成り: 任意成り+強制成りの {}・{:.2} 回/局（70% 閾値は頻度記述用）",
            pct(sat, po + pf),
            sat as f64 / f64::from(t.games.max(1)),
        );
        println!(
            "- opp_king 帯: 発火 {}・発火時の帯サイズ平均 {:.1} マス・候補数の平均（全決定点）{:.1}（帯サイズの監査）",
            pct(t.opp_king_fired, d_total),
            t.opp_king_band_size_sum as f64 / t.opp_king_fired.max(1) as f64,
            t.opp_king_cand_sum as f64 / d_total.max(1) as f64,
        );
        println!();
        println!("帯×しきい値横断（打ち、+=gain/-=loss の本数合算）:");
        println!();
        print!("{}", bd_table(&t.drop_bd));
        println!();
        println!("帯×しきい値横断（任意成り、着手前→成り後）:");
        println!();
        print!("{}", bd_table(&t.promo_bd));
        println!();
        println!("帯×しきい値横断（任意成りの不成双子比 = 成ったことで増えた分）:");
        println!();
        print!("{}", bd_table(&t.promo_twin_bd));
    }
    if let Some((_, p)) = dump {
        println!();
        println!("1行=1受理手の TSV: {p}");
    }
}
