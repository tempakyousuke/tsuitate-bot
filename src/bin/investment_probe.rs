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
//! **選択時に実際についた `CandidateScore.link` そのもの**で判定する: 記録の
//! `chose.debug` に runtime が残す `link`（選択手の link 項）と `link_base`
//! （同じ決定・同じ粒子由来の敵玉重み表で測った着手前の link）の差を読む。
//! link の働き重み（既定 `link_work_w=1`）は粒子由来の `opp_king_w` を使うので
//! **オフラインでは再現できない**（PR #41 レビュー2巡目 — 本数近似も
//! `opp_king_w=None` の近似も、着手前後の差の符号ごと変わり得る）。
//! `link` / `link_base` の無い旧記録の打ちは link-only 分類から除外して
//! 本数を報告する（その記録では頻度判定を出せない）。

use std::collections::BTreeMap;
use std::io::Write;

use tsuitate_bot::marginal_work::{
    BAND_LABELS, Band, MarginalWorkBreakdown, N_BANDS, N_TRANSITIONS, TRANSITION_TAGS,
    attack_counts, attack_counts_after, band_map, breakdown,
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
    /// `chose.debug` に `link` / `link_base` が無い打ち（旧記録）。
    /// link-only 分類の分母から外し、本数だけ報告する
    link_na_drops: [u64; PHASES],
    promos_optional: [u64; PHASES],
    promos_forced: [u64; PHASES],
    /// 飽和成りは**任意/強制で分けて数える**（P1 の変更対象は任意成りだけなので、
    /// 発火3%門の分母も任意成り。強制成りは参考の別枠 — PR #41 レビュー）
    saturated_optional: [u64; PHASES],
    saturated_forced: [u64; PHASES],
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
        // 選択時の link / link_base（runtime が chose.debug に残す。issue #40）。
        // 同一手番の反則指し直しで chose が複数あるときは最後の行が受理手
        let mut chose_link: std::collections::HashMap<(u64, String), (f64, f64)> =
            std::collections::HashMap::new();
        for line in content.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            if v["type"] != "chose" {
                continue;
            }
            let (Some(mn), Some(usi)) = (v["move_number"].as_u64(), v["usi"].as_str()) else {
                continue;
            };
            if let (Some(l), Some(b)) =
                (v["debug"]["link"].as_f64(), v["debug"]["link_base"].as_f64())
            {
                chose_link.insert((mn, usi.to_string()), (l, b));
            }
        }
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
            /// 選択時に記録された実 link の増分（`chose.debug` の
            /// `link − link_base`。粒子由来の敵玉重み込み）。旧記録は None
            link_delta: Option<f64>,
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
                    // 選択時の実 link（粒子由来の敵玉重み込み）。オフラインでは
                    // 再現できないので、記録に無ければ None（分類から除外）
                    let link_delta = chose_link
                        .get(&(u64::from(d.pos.move_number()), mv_rec.usi.clone()))
                        .map(|(l, b)| l - b);
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
                        (kind, pre_role, Some(twin_bd), None)
                    } else {
                        ("board", pre_role, None, None)
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
                    match r.link_delta {
                        None => tally.link_na_drops[ph] += 1,
                        Some(d) => {
                            link_only = d > 0.0 && r.bd.demand_first_second_gain() == 0;
                            if link_only {
                                tally.link_only_drops[ph] += 1;
                            }
                        }
                    }
                }
                "promo_optional" | "promo_forced" => {
                    saturated = !r.captured
                        && !r.gives_check
                        && r.bd.neutral_redundant_frac().is_some_and(|f| f >= 0.7);
                    if r.kind == "promo_optional" {
                        tally.promos_optional[ph] += 1;
                        tally.promo_bd.add(&r.bd);
                        if let Some(t) = &r.twin_bd {
                            tally.promo_twin_bd.add(t);
                        }
                        if saturated {
                            tally.saturated_optional[ph] += 1;
                        }
                    } else {
                        tally.promos_forced[ph] += 1;
                        if saturated {
                            tally.saturated_forced[ph] += 1;
                        }
                    }
                }
                _ => {}
            }
            if let Some((f, _)) = dump.as_mut() {
                let nr_frac = r
                    .bd
                    .neutral_redundant_frac()
                    .map_or("".to_string(), |f| format!("{f:.3}"));
                let link_delta_s =
                    r.link_delta.map_or("".to_string(), |d| format!("{d:.4}"));
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
                    link_delta_s,
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
        let sat_o: u64 = t.saturated_optional.iter().sum();
        let sat_f: u64 = t.saturated_forced.iter().sum();
        println!();
        println!("### vs {opp}（{} 局・bot 決定点 {d_total}）", t.games);
        println!();
        println!(
            "| 手数帯 | 決定点 | 打ち | link-only打ち | 任意成り | 飽和(任意) | 強制成り | 飽和(強制) |"
        );
        println!("|---|---|---|---|---|---|---|---|");
        for ph in 0..PHASES {
            println!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                PHASE_LABELS[ph],
                t.decisions[ph],
                t.drops[ph],
                t.link_only_drops[ph],
                t.promos_optional[ph],
                t.saturated_optional[ph],
                t.promos_forced[ph],
                t.saturated_forced[ph],
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
        let link_na: u64 = t.link_na_drops.iter().sum();
        println!(
            "- link-only打ち: link 記録のある打ち {} 本中 {}・{:.2} 回/局（発火3%未満なら「現象が稀」= P0-3 の中止側）{}",
            drops - link_na,
            pct(lod, drops - link_na),
            lod as f64 / f64::from(t.games.max(1)),
            if link_na > 0 {
                format!(
                    "。**link/link_base の無い打ちが {link_na} 本**（chose.debug に link を残す commit より前の旧記録 = この記録では頻度判定を出せない）"
                )
            } else {
                String::new()
            },
        );
        println!(
            "- 飽和成り(任意): 任意成りの {}・{:.2} 回/局（**P0-3 の発火3%門の分母は任意成り** = P1 の変更対象。70% 閾値は頻度記述用）",
            pct(sat_o, po),
            sat_o as f64 / f64::from(t.games.max(1)),
        );
        println!(
            "- 飽和成り(強制): 強制成りの {}（参考。P1 の変更対象外なので門には数えない）",
            pct(sat_f, pf),
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
