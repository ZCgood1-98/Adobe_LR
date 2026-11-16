#!/usr/bin/env python3
"""offline_pack.py — 把 500 图对投票做成可转发的离线包，并把对方导出的结果导回服务器
================================================================================

场景：让另一位同学标注，但对方连不上服务器。

  build   生成离线包 human500_vote_<名字>.zip：
            index.html   投票页（纯前端，双击即可打开，不联网）
            img/         1000 张图，文件名只有序号和 a/b
            README.txt   使用说明
          同时在服务器上留一份密钥 key.json（**不进离线包**），记录每个序号对应哪个图对、
          a/b 各是原图还是修后图。
  import  对方标完后点「导出结果」得到一个 json，传回服务器后用它生成
          votes_<名字>.jsonl / .csv，字段与本人投票的 votes.jsonl 相同。

盲测保证：离线包里没有图对 ID、没有 original / version_0 字样，哪张是原图只能靠服务器上的密钥还原。
展示顺序与 A/B 位置按标注者名字单独打乱（原图在 A 位恰好 250 对），与其他标注者互相独立。
图片与本人投票、模型评测用的是同一批文件（长边 1024、jpg95），原样复制。

用法（仓库根目录）：
  python 3_llm_eval/human500/offline_pack.py build --name D
  python 3_llm_eval/human500/offline_pack.py import --name D --file <对方发回的 json>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

README = """500 组图片投票 —— 使用说明

1. 先把压缩包完整解压，再双击解压出来的 index.html（不要在压缩包里直接打开，否则图片显示不出来）。
2. 每组显示两张图：图 A 和图 B。选你觉得更好看的那一张。
   键盘：A 或 ← 选图 A，B 或 → 选图 B，Z 回到上一组。也可以点图片或下面的按钮。
3. 进度自动保存在这个浏览器里，可以随时关掉，下次用同一个浏览器打开同一个 index.html 会接着来。
   中途请不要移动或重命名这个文件夹，也不要清除浏览器数据。
   保险起见，可以随时点右下角「导出结果」存一份；万一进度丢了，用「导入进度」选这个文件即可恢复。
4. 500 组全部选完后，点「导出结果」，把下载到的 json 文件发回即可。

全程不需要联网。
"""

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
  button, .filebtn { font: inherit; color: var(--fg); background: var(--panel); border: 1px solid var(--line);
           border-radius: 6px; padding: 10px 14px; cursor: pointer; }
  button:hover, .filebtn:hover { border-color: var(--fg); }
  button:disabled { opacity: .4; cursor: default; }
  .pickbtn { font-size: 17px; font-weight: 600; }
  .pickbtn.picked, button.primary { background: var(--pick); color: #222; border-color: var(--pick); }
  footer { background: var(--panel); padding: 8px 16px; display: flex; align-items: center; gap: 12px;
           color: var(--dim); flex-wrap: wrap; }
  footer .grow { flex: 1; }
  kbd { background: #1c1c1c; border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; color: var(--fg); }
  #done { display: none; margin: auto; text-align: center; max-width: 560px; padding: 24px; }
  #done h1 { font-size: 22px; }
  #msg { color: #ff9a8a; }
  #warn { display: none; background: #7a2e22; color: #fff; padding: 8px 16px; }
  input[type=file] { display: none; }
</style>
</head>
<body>
<div id="warn"></div>
<header>
  <div class="count">第 <b id="pos">–</b> / <span id="total">–</span> 组</div>
  <div class="bar"><div id="fill"></div></div>
  <div class="count">已选 <b id="ndone">–</b></div>
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
  <h1>全部选完了，谢谢！</h1>
  <p>请点下面的按钮导出结果，把下载到的 json 文件发回即可。</p>
  <p><button class="primary" id="export2">导出结果</button></p>
  <p><button id="review">回去检查 / 修改</button></p>
</div>

<footer>
  <button id="prev">上一组</button>
  <button id="next">下一组</button>
  <button id="todo">跳到没选的</button>
  <span class="grow"></span>
  <span id="msg"></span>
  <span>键盘：<kbd>A</kbd>/<kbd>←</kbd> 选图 A，<kbd>B</kbd>/<kbd>→</kbd> 选图 B，<kbd>Z</kbd> 上一组</span>
  <label class="filebtn">导入进度<input type="file" id="importFile" accept=".json,application/json"></label>
  <button id="export">导出结果</button>
</footer>

<script>
const PACK = "__PACK_ID__", TOTAL = __TOTAL__, KEY = "human500_" + PACK;
const $ = id => document.getElementById(id);
let votes = {}, k = 0, shownAt = 0, canStore = true;

function load() {
  try { votes = JSON.parse(localStorage.getItem(KEY) || "{}"); }
  catch (e) { canStore = false; votes = {}; }
}
function save() {
  if (!canStore) return;
  try { localStorage.setItem(KEY, JSON.stringify(votes)); }
  catch (e) { canStore = false; warnStore(); }
}
function warnStore() {
  $('warn').style.display = 'block';
  $('warn').textContent = '这个浏览器无法自动保存进度：关闭页面前请务必点右下角「导出结果」，下次用「导入进度」恢复。';
}
const pad = j => String(j + 1).padStart(3, '0');
const src = (j, s) => `img/${pad(j)}_${s}.jpg`;
const nDone = () => Object.keys(votes).length;
function firstTodo(from = 0) {
  for (let i = 0; i < TOTAL; i++) { const j = (from + i) % TOTAL; if (!(j in votes)) return j; }
  return -1;
}
function preload(j) { if (j < TOTAL) { new Image().src = src(j, 'a'); new Image().src = src(j, 'b'); } }

function show(j) {
  k = Math.max(0, Math.min(TOTAL - 1, j));
  $('stage').style.display = 'flex'; $('done').style.display = 'none';
  $('imgA').src = src(k, 'a'); $('imgB').src = src(k, 'b');
  const v = votes[k] && votes[k].c;
  for (const s of ['A', 'B']) {
    $('frame' + s).classList.toggle('picked', v === s);
    $('btn' + s).classList.toggle('picked', v === s);
  }
  $('pos').textContent = k + 1; $('total').textContent = TOTAL; $('ndone').textContent = nDone();
  $('fill').style.width = (100 * nDone() / TOTAL) + '%';
  $('prev').disabled = k === 0; $('next').disabled = k === TOTAL - 1;
  shownAt = performance.now();
  preload(k + 1); preload(k + 2);
}
function showDone() {
  $('stage').style.display = 'none'; $('done').style.display = 'block';
  $('ndone').textContent = nDone(); $('fill').style.width = '100%';
}
function vote(choice) {
  if ($('stage').style.display === 'none') return;
  votes[k] = {c: choice, rt: Math.round(performance.now() - shownAt), ts: new Date().toISOString()};
  save();
  if (nDone() === TOTAL) showDone();
  else show(k + 1 < TOTAL && !((k + 1) in votes) ? k + 1 : firstTodo(k + 1));
}
function exportVotes() {
  const n = nDone();
  if (n < TOTAL && !confirm(`还有 ${TOTAL - n} 组没选。现在导出的是未完成的结果，确定吗？`)) return;
  const out = {pack_id: PACK, total: TOTAL, n_done: n, exported_at: new Date().toISOString(),
               votes: Object.keys(votes).map(Number).sort((a, b) => a - b)
                        .map(j => ({k: j, choice: votes[j].c, rt_ms: votes[j].rt, ts: votes[j].ts}))};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = `votes_${PACK}_${n}of${TOTAL}.json`;
  document.body.appendChild(a); a.click(); a.remove();
}
function importVotes(file) {
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const d = JSON.parse(rd.result);
      if (d.pack_id !== PACK) throw new Error('这个文件不是这份投票包导出的');
      for (const v of d.votes) if (!(v.k in votes)) votes[v.k] = {c: v.choice, rt: v.rt_ms, ts: v.ts};
      save();
      const n = firstTodo(0); if (n < 0) showDone(); else show(n);
      $('msg').textContent = '';
    } catch (e) { $('msg').textContent = '导入失败：' + e.message; }
  };
  rd.readAsText(file);
}

$('btnA').onclick = $('frameA').onclick = () => vote('A');
$('btnB').onclick = $('frameB').onclick = () => vote('B');
$('prev').onclick = () => show(k - 1);
$('next').onclick = () => show(k + 1);
$('todo').onclick = () => { const n = firstTodo(k); if (n >= 0) show(n); else showDone(); };
$('review').onclick = () => show(k);
$('export').onclick = $('export2').onclick = exportVotes;
$('importFile').onchange = e => { if (e.target.files[0]) importVotes(e.target.files[0]); e.target.value = ''; };
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === 'a' || key === 'arrowleft') { e.preventDefault(); vote('A'); }
  else if (key === 'b' || key === 'arrowright') { e.preventDefault(); vote('B'); }
  else if (key === 'z' || key === 'backspace') { e.preventDefault(); show(k - 1); }
});

load();
if (!canStore) warnStore();
(() => { const n = firstTodo(0); if (n < 0) { k = TOTAL - 1; $('total').textContent = TOTAL; $('pos').textContent = TOTAL; showDone(); } else show(n); })();
</script>
</body>
</html>
"""


def paths(data: Path, name: str) -> dict[str, Path]:
    base = data / "offline" / name
    return {"base": base, "pack": base / f"human500_vote_{name}", "key": base / "key.json",
            "zip": base / f"human500_vote_{name}.zip"}


def build(data: Path, name: str, force: bool) -> None:
    P = paths(data, name)
    if P["key"].exists() and not force:
        raise SystemExit(f"{P['key']} 已存在：这个名字的离线包已经生成过。重新生成会换一套顺序，"
                         f"已发出去的旧包导出的结果将无法导入；确实要重做请加 --force。")
    pairs = sorted(json.loads((data / "pairs.json").read_text(encoding="utf-8"))["pairs"],
                   key=lambda p: p["sample_id"])
    n = len(pairs)
    # 按名字定种子：展示顺序与 A/B 位置和其他标注者互相独立，原图在 A 位恰好一半
    seed = int(hashlib.sha256(f"human500-offline-{name}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    order = list(range(n)); rng.shuffle(order)
    sides = ["original"] * (n // 2) + ["version_0"] * (n - n // 2); rng.shuffle(sides)
    pack_id = f"{name}-{hashlib.sha256(f'{seed}-{n}'.encode()).hexdigest()[:8]}"

    if P["pack"].exists():
        shutil.rmtree(P["pack"])
    (P["pack"] / "img").mkdir(parents=True)
    key = []
    for k, (idx, a_is) in enumerate(zip(order, sides)):
        p = pairs[idx]
        b_is = "version_0" if a_is == "original" else "original"
        shutil.copyfile(data / p[a_is]["file"], P["pack"] / "img" / f"{k + 1:03d}_a.jpg")
        shutil.copyfile(data / p[b_is]["file"], P["pack"] / "img" / f"{k + 1:03d}_b.jpg")
        key.append({"k": k, "sample_id": p["sample_id"], "pair_id": p["pair_id"], "a_is": a_is,
                    "aspect_mismatch": p["aspect_mismatch"]})
    html = PAGE.replace("__PACK_ID__", pack_id).replace("__TOTAL__", str(n))
    (P["pack"] / "index.html").write_text(html, encoding="utf-8")
    (P["pack"] / "README.txt").write_text(README, encoding="utf-8-sig")
    P["key"].write_text(json.dumps({"name": name, "pack_id": pack_id, "seed": seed, "pairs": key},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    # 打包：jpg 已是压缩格式，只存储不再压缩
    with zipfile.ZipFile(P["zip"], "w", zipfile.ZIP_STORED) as z:
        for f in sorted(P["pack"].rglob("*")):
            if f.is_file():
                z.write(f, f"{P['pack'].name}/{f.relative_to(P['pack'])}")
    size = P["zip"].stat().st_size / 1e6
    print(f"离线包：{P['zip']}（{size:.0f} MB，{2 * n} 张图）")
    print(f"包编号 {pack_id}；密钥留在服务器：{P['key']}（不要发给标注者）")


def import_votes(data: Path, name: str, file: Path) -> None:
    P = paths(data, name)
    key = json.loads(P["key"].read_text(encoding="utf-8"))
    exp = json.loads(file.read_text(encoding="utf-8-sig"))
    if exp.get("pack_id") != key["pack_id"]:
        raise SystemExit(f"包编号不匹配：文件是 {exp.get('pack_id')}，服务器上 {name} 的密钥是 {key['pack_id']}")
    by_k = {p["k"]: p for p in key["pairs"]}
    recs = []
    for v in exp["votes"]:
        p = by_k[v["k"]]
        assert v["choice"] in ("A", "B"), v
        b_is = "version_0" if p["a_is"] == "original" else "original"
        recs.append({"k": v["k"], "pair_id": p["pair_id"], "sample_id": p["sample_id"], "a_is": p["a_is"],
                     "choice": v["choice"], "chosen": p["a_is"] if v["choice"] == "A" else b_is,
                     "rt_ms": v.get("rt_ms"), "ts": v.get("ts"), "aspect_mismatch": p["aspect_mismatch"]})
    assert len({r["k"] for r in recs}) == len(recs), "导出文件里有重复的序号"
    out_j = data / f"votes_{name}.jsonl"
    out_j.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    with open(data / f"votes_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        cols = ["k", "sample_id", "pair_id", "a_is", "choice", "chosen", "rt_ms", "ts", "aspect_mismatch"]
        w.writerow(cols)
        for r in recs:
            w.writerow([r[c] for c in cols])
    n, total = len(recs), len(by_k)
    v0 = sum(r["chosen"] == "version_0" for r in recs)
    print(f"已导入 {n}/{total} 组（{'完整' if n == total else f'还缺 {total - n} 组'}）→ {out_j}、votes_{name}.csv")
    if n:
        print(f"选修后图 {v0}（{100 * v0 / n:.1f}%），选图 A {sum(r['choice'] == 'A' for r in recs)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="500 图对投票的离线包：生成与结果导入")
    # 脚本若被放在数据目录里（旁边就有 pairs.json），默认用所在目录
    here = Path(__file__).resolve().parent
    default_data = here if (here / "pairs.json").exists() else REPO_ROOT / "eval_results" / "human500"
    ap.add_argument("--data", type=Path, default=default_data)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="生成离线包")
    b.add_argument("--name", required=True, help="标注者代号，如 D")
    b.add_argument("--force", action="store_true", help="覆盖已有的同名离线包（会换一套顺序）")
    i = sub.add_parser("import", help="导入对方发回的结果")
    i.add_argument("--name", required=True)
    i.add_argument("--file", type=Path, required=True, help="对方点「导出结果」得到的 json")
    args = ap.parse_args()
    data = args.data.resolve()
    if args.cmd == "build":
        build(data, args.name, args.force)
    else:
        import_votes(data, args.name, args.file.resolve())


if __name__ == "__main__":
    main()
