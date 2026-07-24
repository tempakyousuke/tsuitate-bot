import { mount } from "svelte";
import App from "./App.svelte";
import "./styles.css";
import { installMockIfRequested } from "./mock";

// トップレベル await はビルドターゲット（es2020）で使えないので then で繋ぐ
void installMockIfRequested().then(() => {
  mount(App, { target: document.getElementById("app")! });
});
