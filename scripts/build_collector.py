"""Build a self-contained offline collector HTML from stimuli (S7 experiment).

Reads ``stimuli.toml`` and emits ``collector.html``: a single file with no
dependencies and no network. A colleague opens it in a browser, enters an ID,
answers the incoming messages one by one (chat feel, one message at a time),
and at the end downloads ``answers_<ID>.json`` to send back.

Privacy: answers live only in the browser (localStorage) until the colleague
explicitly downloads them. Nothing is uploaded.

Run:
    uv run python scripts/build_collector.py
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

DEFAULT_DIR = Path("data/style_study")


def load_stimuli(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Прочитать стимулы и ярлык собеседника из TOML.

    Args:
        path: Путь к stimuli.toml.

    Returns:
        (interlocutor, stimuli) — ярлык собеседника и список стимулов.

    Raises:
        ValueError: Если стимулы пусты или у элемента нет id/text.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    interlocutor = str(data.get("interlocutor", "знакомый"))
    stimuli = []
    for item in data.get("stimulus", []):
        if "id" not in item or "text" not in item:
            raise ValueError(f"stimulus must have id and text: {item!r}")
        stimuli.append(
            {
                "id": str(item["id"]),
                "category": str(item.get("category", "")),
                "text": str(item["text"]),
            }
        )
    if not stimuli:
        raise ValueError("no stimuli found")
    return interlocutor, stimuli


_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Исследование: стиль общения</title>
<style>
  :root { --bg:#0f1115; --card:#1b1f27; --ink:#e8eaed; --muted:#9aa0a6;
          --accent:#4c8bf5; --in:#262b34; --out:#2b5278; --line:#2a2f38; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin:0; font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); overscroll-behavior-y: none; }
  .app { display:flex; flex-direction:column; min-height:100vh; min-height:100dvh;
         max-width:640px; margin:0 auto; }
  .scroll { flex:1; overflow-y:auto; padding:16px 16px 8px; }
  h1 { font-size:20px; margin:0 0 8px; }
  .muted { color:var(--muted); font-size:14px; }
  .card { background:var(--card); border-radius:14px; padding:18px; margin-top:16px; }
  input[type=text] { width:100%; padding:14px; border-radius:12px;
        border:1px solid var(--line); background:#12151b; color:var(--ink);
        font-size:16px; }
  button { border:0; border-radius:12px; padding:14px 18px; font-size:16px;
        background:var(--accent); color:#fff; cursor:pointer;
        touch-action:manipulation; min-height:48px; }
  button:disabled { opacity:.5; cursor:default; }
  button.ghost { background:var(--line); }
  button.block { width:100%; }
  textarea { width:100%; min-height:72px; max-height:40vh; padding:14px;
        border-radius:12px; border:1px solid var(--line); background:#12151b;
        color:var(--ink); font:16px/1.5 inherit; resize:none; }
  .bubble { max-width:82%; padding:11px 15px; border-radius:16px; margin:6px 0;
        white-space:pre-wrap; word-break:break-word; }
  .in { background:var(--in); border-bottom-left-radius:5px; }
  .row { display:flex; }
  .progress { height:6px; background:#232833; border-radius:3px; overflow:hidden; }
  .progress > div { height:100%; background:var(--accent); width:0;
        transition:width .2s ease; }
  .head { padding:14px 16px 10px; }
  .composer { position:sticky; bottom:0; background:var(--bg);
        padding:10px 16px calc(10px + env(safe-area-inset-bottom));
        border-top:1px solid var(--line); }
  .composer .hint { display:flex; justify-content:space-between; align-items:center;
        gap:10px; margin-top:10px; }
  .hidden { display:none !important; }
  ul { margin:8px 0 0 18px; padding:0; }
  li { margin:4px 0; }
  .stack { display:flex; flex-direction:column; gap:10px; margin-top:14px; }
  @media (min-width:560px) {
    .composer .hint { justify-content:flex-end; }
    .composer .hint .muted { margin-right:auto; }
  }
</style>
</head>
<body>
<div class="app">

  <div id="screen-start" class="scroll">
    <h1>Исследование стиля общения</h1>
    <p class="muted">Спасибо, что помогаете! Это займёт 10–15 минут. Можно с телефона.</p>
    <div class="card">
      <p><b>Что делать.</b> Вам будет приходить по одному сообщению от
      собеседника (__INTERLOCUTOR__). Отвечайте так, как обычно отвечаете в
      мессенджере.</p>
      <ul>
        <li>Пишите <b>свой обычный стиль</b> — не старайтесь «красиво».</li>
        <li><b>Не редактируйте и не переписывайте</b> — важен первый вариант.</li>
        <li>Одно-два предложения достаточно; можно эмодзи, строчные, как привыкли.</li>
        <li>Не пишите личных/конфиденциальных деталей — важен стиль, не факты.</li>
        <li>Ответы <b>остаются только в вашем браузере</b>; в конце скачайте файл и пришлите его.</li>
      </ul>
      <p class="muted" style="margin-top:12px">Введите ваш идентификатор (как договорились, напр. P1):</p>
      <input id="pid" type="text" placeholder="P1" autocomplete="off"
             autocapitalize="off" enterkeyhint="go">
      <div class="stack">
        <button id="btn-start" class="block">Начать</button>
        <span class="muted" id="resume-hint" style="text-align:center"></span>
      </div>
      <div id="diag" style="text-align:center;font-size:12px;opacity:.6;margin-top:10px"></div>
    </div>
  </div>

  <div id="screen-chat" class="hidden" style="display:flex;flex-direction:column;min-height:100dvh">
    <div class="head">
      <div class="progress"><div id="bar"></div></div>
      <p class="muted" id="counter" style="margin:8px 0 0"></p>
    </div>
    <div class="scroll">
      <div class="row"><div class="bubble in" id="incoming"></div></div>
    </div>
    <div class="composer">
      <textarea id="reply" placeholder="Ваш ответ…" enterkeyhint="send"
                autocapitalize="sentences" autocomplete="off"></textarea>
      <div class="stack" style="margin-top:10px">
        <button id="btn-send" class="block">Ответить</button>
        <span class="muted" style="text-align:center">Ctrl+Enter — отправить (на компьютере)</span>
      </div>
    </div>
  </div>

  <div id="screen-done" class="scroll hidden">
    <h1>Готово, спасибо!</h1>
    <div class="card">
      <p>Скачайте файл с ответами и пришлите его организатору.
      <b>.json</b> — основной; <b>.md</b> — просто для чтения.</p>
      <div class="stack">
        <button id="btn-download" class="block">Скачать ответы (.json)</button>
        <button id="btn-markdown" class="block ghost">Читаемая копия (.md)</button>
      </div>
    </div>
  </div>

</div>
<script>
const STIMULI = __STIMULI__;
const INTERLOCUTOR = __INTERLOCUTOR_JSON__;
const KEY = "style_study_v1";
const SCHEMA_VERSION = 1;
const LF = String.fromCharCode(10);

let state = load() || null;
let shownAt = null;

const $ = (id) => document.getElementById(id);
// Безопасное хранилище: на file:// (особенно Safari) localStorage может кидать
// SecurityError — тогда тихо работаем в памяти, лишь бы форма не падала.
const mem = {};
function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return mem[k] || null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { mem[k] = v; } }
function load() { try { return JSON.parse(lsGet(KEY)); } catch (e) { return null; } }
function save() { lsSet(KEY, JSON.stringify(state)); }

// Показываем реальную ошибку прямо на странице — иначе «кнопка не работает» слепая.
window.addEventListener("error", (e) => {
  const d = document.getElementById("diag");
  if (d) { d.textContent = "Ошибка: " + (e.message || e.error); d.style.color = "#ff6b6b"; }
});
function shuffle(a) { a = a.slice(); for (let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } return a; }
function byId(id) { return STIMULI.find(s => s.id === id); }

function start() {
  const pid = ($("pid").value || "").trim();
  if (!pid) { alert("Введите идентификатор (например, P1)"); return; }
  if (!state || state.participant !== pid) {
    state = { participant: pid, order: shuffle(STIMULI.map(s => s.id)), idx: 0, answers: [] };
    save();
  }
  showChat();
}

function showChat() {
  $("screen-start").classList.add("hidden");
  $("screen-done").classList.add("hidden");
  $("screen-chat").classList.remove("hidden");
  if (state.idx >= state.order.length) { showDone(); return; }
  const s = byId(state.order[state.idx]);
  $("incoming").textContent = s.text;
  $("reply").value = "";
  $("counter").textContent = `Сообщение ${state.idx + 1} из ${state.order.length}`;
  $("bar").style.width = (100 * state.idx / state.order.length) + "%";
  shownAt = Date.now();
  // Фокус не форсируем: на телефоне автофокус выдёргивает клавиатуру.
  if (!isTouch()) { $("reply").focus(); }
}

function isTouch() {
  return window.matchMedia && window.matchMedia("(hover: none)").matches;
}

function send() {
  const text = ($("reply").value || "").trim();
  if (!text) { $("reply").focus(); return; }
  const s = byId(state.order[state.idx]);
  state.answers.push({
    participant_id: state.participant,
    stimulus_id: s.id,
    category: s.category,
    stimulus: s.text,
    text: text,
    order: state.idx,
    latency_ms: Date.now() - shownAt,
    submitted_at: new Date().toISOString(),
  });
  state.idx += 1;
  save();
  if (state.idx >= state.order.length) { showDone(); } else { showChat(); }
}

function showDone() {
  $("screen-chat").classList.add("hidden");
  $("screen-start").classList.add("hidden");
  $("screen-done").classList.remove("hidden");
}

function saveBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportJson() {
  const payload = {
    schema_version: SCHEMA_VERSION,
    participant_id: state.participant,
    exported_at: new Date().toISOString(),
    answers: state.answers,
  };
  saveBlob(JSON.stringify(payload, null, 2),
           `answers_${state.participant}.json`, "application/json");
}

function exportMarkdown() {
  const out = [];
  out.push(`# Ответы участника ${state.participant}`);
  out.push("");
  out.push(`Всего ответов: ${state.answers.length}`);
  out.push("");
  for (const a of state.answers) {
    out.push(`## ${a.stimulus_id} — ${a.category}`);
    out.push("");
    out.push(`**Входящее:** ${a.stimulus}`);
    out.push("");
    out.push(`**Ответ:** ${a.text}`);
    out.push("");
  }
  saveBlob(out.join(LF), `answers_${state.participant}.md`, "text/markdown");
}

$("btn-start").addEventListener("click", start);
$("btn-send").addEventListener("click", send);
$("btn-download").addEventListener("click", exportJson);
$("btn-markdown").addEventListener("click", exportMarkdown);
$("reply").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});
$("pid").addEventListener("keydown", (e) => { if (e.key === "Enter") start(); });

// Подсказка о сохранённом прогрессе.
if (state && state.participant) {
  $("pid").value = state.participant;
  $("resume-hint").textContent = `Найден прогресс (${state.idx}/${state.order.length}) — можно продолжить.`;
}

// Индикатор, что скрипт реально выполнился. Если этой строки не видно —
// файл открыт не в браузере (Quick Look / превью VS Code), а не в Chrome/Safari.
$("diag").textContent = "форма активна";
</script>
</body>
</html>
"""


def build_html(interlocutor: str, stimuli: list[dict[str, str]]) -> str:
    """Собрать самодостаточный HTML с встроенными стимулами.

    Args:
        interlocutor: Ярлык собеседника (для инструкции).
        stimuli: Список стимулов.

    Returns:
        HTML-строка.
    """
    return (
        _TEMPLATE.replace("__STIMULI__", json.dumps(stimuli, ensure_ascii=False))
        .replace("__INTERLOCUTOR_JSON__", json.dumps(interlocutor, ensure_ascii=False))
        .replace("__INTERLOCUTOR__", interlocutor)
    )


def main() -> None:
    src = DEFAULT_DIR / "stimuli.toml"
    interlocutor, stimuli = load_stimuli(src)
    out = DEFAULT_DIR / "collector.html"
    out.write_text(build_html(interlocutor, stimuli), encoding="utf-8")
    print(f"stimuli: {len(stimuli)} | interlocutor: {interlocutor}")
    print(f"collector: {out}  (открыть в браузере, переслать коллеге)")


if __name__ == "__main__":
    main()
