// ブラウザ単体（Tauri の外）で UI を確認するための invoke モック。
// `http://localhost:1421/?mock=1` で開いたときだけ有効になる。
// フィクスチャは src-tauri の ignored テストで生成する（public/fixtures/ は gitignore 済み）:
//   FIXTURE_DIR=$(pwd)/public/fixtures cargo test --manifest-path src-tauri/Cargo.toml -- --ignored fixtures

export async function installMockIfRequested(): Promise<void> {
  if (!new URLSearchParams(location.search).has("mock")) return;
  if ("__TAURI_INTERNALS__" in window) return;
  const fixture = async (name: string) => {
    const res = await fetch(`/fixtures/${name}.json`);
    if (!res.ok) throw new Error(`fixture ${name} がありません（生成手順は mock.ts 冒頭）`);
    return res.json();
  };
  (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {
    invoke: async (cmd: string) => {
      if (cmd.startsWith("plugin:")) return null; // event listen / dialog open は何もしない
      switch (cmd) {
        case "engines":
          return ["estimator", "estimator_v10", "estimator_v9", "heuristic"];
        case "list_scenarios":
          return fixture("scenarios");
        case "load_kifu":
          return fixture("kakutori");
        case "eval_tally":
          return fixture("tally");
        case "eval_ranking":
          return fixture("ranking");
        case "cancel_eval":
          return;
        default:
          throw new Error(`mock 未対応のコマンド: ${cmd}`);
      }
    },
  };
  console.log("[mock] Tauri invoke をフィクスチャで置き換えました");
}
