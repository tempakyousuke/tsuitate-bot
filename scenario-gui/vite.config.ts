import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Tauri の devUrl と合わせる（tsuitate 本体の dev サーバー 5173 とは衝突させない）
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 1421,
    strictPort: true,
  },
});
