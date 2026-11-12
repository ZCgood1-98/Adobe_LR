#!/usr/bin/env python3
"""vote_server.py — 500 个图对的本地人工投票网页（只用标准库，无需安装任何依赖）
================================================================================

盲测保证：
  · 页面上只有「图 A / 图 B」，不出现文件名、版本名；
  · 图片走不透明地址 /img/<序号>/a、/img/<序号>/b，哪张是原图只有服务端知道；
  · 浏览器拿不到任何能区分原图 / 修后图的字段（接口只回序号和你已投的 A/B）。
  A/B 位置与展示顺序由 build_pairs.py 用固定种子排好（原图在 A 位恰好 250 对）。

投票记录：每投一票立刻追加到 votes.jsonl（只追加、不改写）；同一图对重投以最后一票为准。
随时可以关掉，下次打开自动跳到第一个没投的图对。

通过 SSH 使用（服务只监听服务器本机 127.0.0.1，不对外网开放）：
  服务器上：  python 3_llm_eval/human500/vote_server.py
  自己电脑上：ssh -L 8765:localhost:8765 <用户名>@<服务器>
  浏览器打开：http://localhost:8765

投完后导出：
  python 3_llm_eval/human500/vote_server.py --export     # 写出 votes_final.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图对投票</title>
<style>
  :root { --bg:#3a3a3a; --panel:#2c2c2c; --fg:#e8e8e8; --dim:#a0a0a0; --line:#555; --pick:#f2c14e; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { background: var(--bg); color: var(--fg); display: flex; flex-direction: column;
         font: 15px/1.4 system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }
  header { background: var(--panel); padding: 8px 16px; display: flex; align-items: center; gap: 16px; }
  header .count { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .bar { flex: 1; height: 6px; background: #1c1c1c; border-radius: 3px; overflow: hidden; }
  .bar > div { height: 100%; width: 0; background: var(--pick); transition: width .15s; }
  main { flex: 1; min-height: 0; display: flex; gap: 10px; padding: 10px; }
  .side { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 8px; }
  .frame { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center;
           border: 3px solid transparent; border-radius: 4px; cursor: pointer; }
  .frame.picked { border-color: var(--pick); }
  .frame img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
  button { font: inherit; color: var(--fg); background: var(--panel); border: 1px solid var(--line);
           border-radius: 6px; padding: 10px 14px; cursor: pointer; }
  button:hover { border-color: var(--fg); }
  button:disabled { opacity: .4; cursor: default; }
  .pickbtn { font-size: 17px; font-weight: 600; }
  .pickbtn.picked { background: var(--pick); color: #222; border-color: var(--pick); }
  footer { background: var(--panel); padding: 8px 16px; display: flex; align-items: center; gap: 12px;
           color: var(--dim); flex-wrap: wrap; }
  footer .grow { flex: 1; }
  kbd { background: #1c1c1c; border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; color: var(--fg); }
  #done { display: none; margin: auto; text-align: center; max-width: 520px; padding: 24px; }
  #done h1 { font-size: 22px; }
  #msg { color: #ff9a8a; }
</style>
</head>
<body>
<header>
  <div class="count">第 <b id="pos">–</b> / <span id="total">–</span> 对</div>
  <div class="bar"><div id="fill"></div></div>
  <div class="count">已投 <b id="ndone">–</b></div>
</header>

<main id="stage">
  <div class="side">
    <div class="frame" id="frameA"><img id="imgA" alt="图 A"></div>
    <button class="pickbtn" id="btnA">图 A 更好看</button>
  </div>
  <div class="side">
    <div class="frame" id="frameB"><img id="imgB" alt="图 B"></div>
    <button class="pickbtn" id="btnB">图 B 更好看</button>
  </div>
</main>

<div id="done">
  <h1>500 对全部投完</h1>
  <p id="summary"></p>
  <p>在服务器上运行 <kbd>python 3_llm_eval/human500/vote_server.py --export</kbd> 导出结果。</p>
  <button id="review">回去检查 / 修改</button>
</div>

<footer>
  <button id="prev">上一对</button>
  <button id="next">下一对</button>
  <button id="todo">跳到未投的</button>
  <span class="grow"></span>
  <span id="msg"></span>
  <span>键盘：<kbd>A</kbd>/<kbd>←</kbd> 选图 A，<kbd>B</kbd>/<kbd>→</kbd> 选图 B，<kbd>Z</kbd> 上一对。点图片也可以选。</span>
</footer>

<script>
const $ = id => document.getElementById(id);
let total = 0, votes = {}, k = 0, shownAt = 0, busy = false;

async function api(path, body) {
  const r = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'},
                                      body: JSON.stringify(body)} : {});
  if (!r.ok) throw new Error(path + ' → ' + r.status);
  return r.json();
}
const nDone = () => Object.keys(votes).length;
function firstTodo(from = 0) {
  for (let i = 0; i < total; i++) { const j = (from + i) % total; if (!(j in votes)) return j; }
  return -1;
}
function preload(j) { if (j < total) { new Image().src = `/img/${j}/a`; new Image().src = `/img/${j}/b`; } }

function show(j) {
  k = Math.max(0, Math.min(total - 1, j));
  $('stage').style.display = 'flex'; $('done').style.display = 'none';
  $('imgA').src = `/img/${k}/a`; $('imgB').src = `/img/${k}/b`;
  const v = votes[k];
  for (const s of ['A', 'B']) {
    $('frame' + s).classList.toggle('picked', v === s);
    $('btn' + s).classList.toggle('picked', v === s);
  }
  $('pos').textContent = k + 1; $('total').textContent = total; $('ndone').textContent = nDone();
  $('fill').style.width = (100 * nDone() / total) + '%';
  $('prev').disabled = k === 0; $('next').disabled = k === total - 1;
  shownAt = performance.now();
  preload(k + 1); preload(k + 2);
}

async function showDone() {
  $('stage').style.display = 'none'; $('done').style.display = 'block';
  $('ndone').textContent = nDone(); $('fill').style.width = '100%';
  try {
    const s = await api('/api/summary');
    $('summary').textContent = `你选了修后图（version_0）${s.version_0} 对、原图 ${s.original} 对；` +
                               `选图 A 的有 ${s.pick_a} 对、图 B ${s.pick_b} 对。`;
  } catch (e) { $('summary').textContent = ''; }
}

async function vote(choice) {
  if (busy || $('stage').style.display === 'none') return;
  busy = true; $('msg').textContent = '';
  try {
    await api('/api/vote', {k, choice, rt_ms: Math.round(performance.now() - shownAt)});
    votes[k] = choice;
    if (nDone() === total) { await showDone(); }
    else { const n = firstTodo(k + 1); show(k + 1 < total && !((k + 1) in votes) ? k + 1 : n); }
  } catch (e) { $('msg').textContent = '保存失败，这一票没有记上：' + e.message; }
  busy = false;
}

$('btnA').onclick = $('frameA').onclick = () => vote('A');
$('btnB').onclick = $('frameB').onclick = () => vote('B');
$('prev').onclick = () => show(k - 1);
$('next').onclick = () => show(k + 1);
$('todo').onclick = () => { const n = firstTodo(k); if (n >= 0) show(n); else showDone(); };
$('review').onclick = () => show(k);
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === 'a' || key === 'arrowleft') { e.preventDefault(); vote('A'); }
  else if (key === 'b' || key === 'arrowright') { e.preventDefault(); vote('B'); }
  else if (key === 'z' || key === 'backspace') { e.preventDefault(); show(k - 1); }
});

(async () => {
  const s = await api('/api/state');
  total = s.total; votes = s.votes;
  const n = firstTodo(0);
  if (n < 0) { k = total - 1; $('total').textContent = total; $('pos').textContent = total; showDone(); }
  else show(n);
})();
</script>
</body>
</html>
"""


class Store:
    """图对清单 + 投票记录。votes.jsonl 只追加；内存里保留每对的最后一票。"""

    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.pairs = json.loads((data_dir / "pairs.json").read_text(encoding="utf-8"))["pairs"]
        self.pairs.sort(key=lambda p: p["k"])
        assert [p["k"] for p in self.pairs] == list(range(len(self.pairs)))
        self.votes_path = data_dir / "votes.jsonl"
        self.lock = threading.Lock()
        self.latest: dict[int, dict] = {}
        if self.votes_path.exists():
            for line in self.votes_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    v = json.loads(line)
                    self.latest[v["k"]] = v

    def image(self, k: int, side: str) -> Path:
        p = self.pairs[k]
        a_is = p["a_is"]
        which = a_is if side == "a" else ("version_0" if a_is == "original" else "original")
        return self.dir / p[which]["file"]

    def vote(self, k: int, choice: str, rt_ms) -> None:
        p = self.pairs[k]
        b_is = "version_0" if p["a_is"] == "original" else "original"
        rec = {
            "k": k, "pair_id": p["pair_id"], "sample_id": p["sample_id"],
            "a_is": p["a_is"], "choice": choice,
            "chosen": p["a_is"] if choice == "A" else b_is,
            "rt_ms": rt_ms if isinstance(rt_ms, (int, float)) else None,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self.lock:
            with open(self.votes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            self.latest[k] = rec

    def state(self) -> dict:
        with self.lock:
            return {"total": len(self.pairs), "votes": {str(k): v["choice"] for k, v in self.latest.items()}}

    def summary(self) -> dict | None:
        with self.lock:
            if len(self.latest) < len(self.pairs):
                return None  # 没投完不给汇总，免得影响后面的判断
            vs = list(self.latest.values())
        return {
            "version_0": sum(v["chosen"] == "version_0" for v in vs),
            "original": sum(v["chosen"] == "original" for v in vs),
            "pick_a": sum(v["choice"] == "A" for v in vs),
            "pick_b": sum(v["choice"] == "B" for v in vs),
        }

    def export(self) -> Path:
        out = self.dir / "votes_final.csv"
        aspect = {p["k"]: p["aspect_mismatch"] for p in self.pairs}
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["k", "sample_id", "pair_id", "a_is", "choice", "chosen", "rt_ms", "ts", "aspect_mismatch"])
            for k in sorted(self.latest):
                v = self.latest[k]
                w.writerow([k, v["sample_id"], v["pair_id"], v["a_is"], v["choice"], v["chosen"],
                            v["rt_ms"], v["ts"], aspect[k]])
        return out


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(store.state())
            if path == "/api/summary":
                s = store.summary()
                return self._json(s) if s else self._json({"error": "未投完"}, 409)
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "img" and parts[1].isdigit() and parts[2] in ("a", "b"):
                k = int(parts[1])
                if 0 <= k < len(store.pairs):
                    # 图片内容固定不变，允许浏览器缓存，翻回上一对时不必重新经 SSH 传
                    return self._send(200, store.image(k, parts[2]).read_bytes(), "image/jpeg",
                                      cache="private, max-age=86400")
            self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if self.path != "/api/vote":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                k, choice = body["k"], body["choice"]
                assert isinstance(k, int) and 0 <= k < len(store.pairs) and choice in ("A", "B")
            except Exception:
                return self._json({"error": "bad request"}, 400)
            store.vote(k, choice, body.get("rt_ms"))
            self._json({"ok": True})

        def log_message(self, *args):  # 不刷屏
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="500 个图对的本地人工投票网页")
    # 脚本若被放在数据目录里（旁边就有 pairs.json），默认用所在目录；否则用仓库里的默认位置
    here = Path(__file__).resolve().parent
    default_data = here if (here / "pairs.json").exists() else REPO_ROOT / "eval_results" / "human500"
    ap.add_argument("--data", type=Path, default=default_data, help="build_pairs.py 的输出目录")
    ap.add_argument("--host", default="127.0.0.1", help="默认只监听本机，经 SSH 端口转发访问")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--export", action="store_true", help="不启动服务，只把最终投票导出成 votes_final.csv")
    args = ap.parse_args()

    store = Store(args.data.resolve())
    if args.export:
        out = store.export()
        print(f"已导出 {len(store.latest)}/{len(store.pairs)} 对 → {out}")
        return

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"图对 {len(store.pairs)} 个，已投 {len(store.latest)} 个；投票记录：{store.votes_path}")
    print(f"服务已启动：http://{args.host}:{args.port}")
    print(f"在自己电脑上执行：ssh -L {args.port}:localhost:{args.port} <用户名>@<服务器>，"
          f"然后浏览器打开 http://localhost:{args.port}")
    print("按 Ctrl+C 停止（已投的票都已落盘）。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
