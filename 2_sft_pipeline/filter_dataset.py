#!/usr/bin/env python3
"""
Web-based interactive filter for scraped_data_v4 dataset.

Workflow:
  1. Pick (or create) an output directory in the terminal.
  2. The script starts a local Flask server on http://localhost:5000.
     If running over SSH, forward the port from your laptop:
         ssh -L 5000:localhost:5000 user@host
     Or let VS Code's port forwarder handle it.
  3. Open the URL in your browser. For each group it shows
     original_captured_cropped.png + every version_*/main_image.png.
     Keyboard shortcuts (focus must be in the browser tab):
         y / space / enter   -> keep this group (copy to output)
         n / x / del         -> skip
         b                   -> undo last action
         q / esc             -> quit the server

Output layout mirrors the source:
    <out>/<split>/<group_id>/combined_details.json
    <out>/<split>/<group_id>/original_captured_cropped.png
    <out>/<split>/<group_id>/version_X/details.json
    <out>/<split>/<group_id>/version_X/main_image.png

Progress is written to <out>/_filter_log.tsv after every decision so you
can stop and resume any time.
"""

import os
import sys
import shutil
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort


SRC_ROOT = Path("/home/chen/Work/adobe_lr/scraped_data_v4")
PARENT = Path("/home/chen/Work/adobe_lr")
SPLITS = ("train", "test")
PORT = int(os.environ.get("FILTER_PORT", "5000"))


# ---------------------------------------------------------------- output dir

def choose_output_dir() -> Path:
    print("\n=== Choose output directory ===")
    dirs = sorted(
        p for p in PARENT.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p != SRC_ROOT
    )
    for i, d in enumerate(dirs):
        print(f"  [{i:>2}] {d.name}")
    print("  [ n] Create a new folder")
    print("  [ p] Type a custom absolute path")

    while True:
        choice = input("Selection: ").strip()
        if choice.lower() == "n":
            name = input(f"New folder name (under {PARENT}): ").strip()
            if not name:
                print("Empty name, try again.")
                continue
            out = PARENT / name
            out.mkdir(parents=True, exist_ok=True)
            return out
        if choice.lower() == "p":
            raw = input("Absolute path: ").strip()
            if not raw:
                continue
            out = Path(raw).expanduser().resolve()
            out.mkdir(parents=True, exist_ok=True)
            return out
        try:
            idx = int(choice)
            if 0 <= idx < len(dirs):
                return dirs[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


# ---------------------------------------------------------------- discovery

def collect_groups():
    groups = []
    for split in SPLITS:
        split_dir = SRC_ROOT / split
        if not split_dir.exists():
            continue
        for g in sorted(split_dir.iterdir()):
            if g.is_dir():
                groups.append((split, g))
    return groups


# ---------------------------------------------------------------- copy logic

def copy_group(src_group: Path, split: str, out_root: Path) -> None:
    dst = out_root / split / src_group.name
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("combined_details.json", "original_captured_cropped.png"):
        sp = src_group / fname
        if sp.exists():
            shutil.copy2(sp, dst / fname)
    for v in sorted(src_group.glob("version_*")):
        if not v.is_dir():
            continue
        dv = dst / v.name
        dv.mkdir(parents=True, exist_ok=True)
        for fname in ("details.json", "main_image.png"):
            sp = v / fname
            if sp.exists():
                shutil.copy2(sp, dv / fname)


def remove_group(out_root: Path, split: str, name: str) -> None:
    dst = out_root / split / name
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)


# ---------------------------------------------------------------- log helpers

def load_log(log_path: Path):
    seen = {}
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                key, action = line.split("\t", 1)
                seen[key] = action
    return seen


def append_log(log_path: Path, key: str, action: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{key}\t{action}\n")


def rewrite_log(log_path: Path, seen: dict) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for key, action in seen.items():
            f.write(f"{key}\t{action}\n")


# ---------------------------------------------------------------- state

class State:
    def __init__(self, groups, out_root: Path, log_path: Path):
        self.lock = threading.Lock()
        self.out_root = out_root
        self.log_path = log_path
        self.seen = load_log(log_path)
        self.remaining = [
            g for g in groups
            if f"{g[0]}/{g[1].name}" not in self.seen
        ]
        self.history = []   # list of (split, group_dir, action)
        self.idx = 0
        self.kept = sum(1 for v in self.seen.values() if v == "kept")
        self.skipped = sum(1 for v in self.seen.values() if v == "skipped")

        print(f"Total groups: {len(groups)}  "
              f"Already processed: {len(self.seen)} "
              f"(kept={self.kept}, skipped={self.skipped})  "
              f"Remaining: {len(self.remaining)}")

    def snapshot(self):
        with self.lock:
            if self.idx >= len(self.remaining):
                return {
                    "done": True,
                    "idx": self.idx,
                    "total": len(self.remaining),
                    "kept": self.kept,
                    "skipped": self.skipped,
                }
            split, gd = self.remaining[self.idx]
            images = []
            orig = gd / "original_captured_cropped.png"
            if orig.exists():
                images.append(("ORIGINAL (cropped)", str(orig)))
            for v in sorted(gd.glob("version_*")):
                mi = v / "main_image.png"
                if mi.exists():
                    images.append((v.name, str(mi)))
            return {
                "done": False,
                "split": split,
                "group": gd.name,
                "idx": self.idx,
                "total": len(self.remaining),
                "kept": self.kept,
                "skipped": self.skipped,
                "images": [
                    {"label": lab, "url": f"/img?path={p}"} for lab, p in images
                ],
            }

    def act(self, action: str):
        with self.lock:
            if action == "back":
                if not self.history:
                    return
                split, gd, prev = self.history.pop()
                key = f"{split}/{gd.name}"
                if prev == "kept":
                    remove_group(self.out_root, split, gd.name)
                    self.kept -= 1
                else:
                    self.skipped -= 1
                self.seen.pop(key, None)
                rewrite_log(self.log_path, self.seen)
                self.idx -= 1
                return
            if self.idx >= len(self.remaining):
                return
            split, gd = self.remaining[self.idx]
            key = f"{split}/{gd.name}"
            if action == "kept":
                try:
                    copy_group(gd, split, self.out_root)
                except Exception as e:
                    print(f"[ERROR] copy failed for {key}: {e}")
                    return
                self.kept += 1
            elif action == "skipped":
                self.skipped += 1
            else:
                return
            self.seen[key] = action
            append_log(self.log_path, key, action)
            self.history.append((split, gd, action))
            self.idx += 1


# ---------------------------------------------------------------- web app

app = Flask(__name__)
STATE: State = None  # type: ignore
ALLOWED_ROOT = SRC_ROOT.resolve()

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dataset Filter</title>
<style>
* { box-sizing: border-box; }
body { font-family: sans-serif; background:#1a1a1a; color:#ddd; margin:0; padding:10px; }
header { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
#title { font-size: 16px; }
#status { font-size: 14px; color:#9aa; }
#help .kbd { background:#333; padding:2px 6px; border-radius:3px; font-family: monospace; margin:0 2px; }
#grid { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; align-items:flex-start; }
.card { background:#222; padding:6px; border-radius:6px; flex:1 1 0; min-width: 220px; }
.card img { width: 100%; max-height: 80vh; object-fit: contain; display:block; background:#000; border-radius:4px; }
.card .label { text-align:center; padding-top:4px; font-size: 13px; color:#aaa; }
#done { font-size:22px; padding:40px; text-align:center; }
.flash { position:fixed; top:20px; right:20px; padding:8px 14px; border-radius:6px; opacity:0; transition: opacity .2s; pointer-events:none; }
.flash.show { opacity:1; }
.flash.keep { background:#2c6e3a; }
.flash.skip { background:#7a3535; }
.flash.back { background:#5a5a5a; }
</style>
</head>
<body>
<header>
  <div>
    <strong id="title">Loading...</strong>
    <span id="status"></span>
  </div>
  <div id="help">
    <span class="kbd">y/space</span> keep
    <span class="kbd">n</span> skip
    <span class="kbd">b</span> back
    <span class="kbd">q</span> quit
  </div>
</header>
<div id="grid"></div>
<div id="done" style="display:none"></div>
<div id="flash" class="flash"></div>
<script>
let lock = false;

async function refresh() {
  const r = await fetch('/state');
  render(await r.json());
}

function flash(kind, text) {
  const el = document.getElementById('flash');
  el.className = 'flash show ' + kind;
  el.textContent = text;
  setTimeout(() => { el.classList.remove('show'); }, 400);
}

async function act(action) {
  if (lock) return;
  lock = true;
  try {
    if (action === 'kept') flash('keep', 'KEEP');
    else if (action === 'skipped') flash('skip', 'SKIP');
    else if (action === 'back') flash('back', 'BACK');
    const r = await fetch('/act', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({action})
    });
    render(await r.json());
  } finally { lock = false; }
}

async function quit() {
  await fetch('/quit', {method:'POST'});
  document.body.innerHTML = '<div id="done">Server stopped. You can close this tab.</div>';
}

function render(s) {
  const grid = document.getElementById('grid');
  const done = document.getElementById('done');
  const title = document.getElementById('title');
  const status = document.getElementById('status');
  if (s.done) {
    grid.style.display = 'none';
    done.style.display = 'block';
    done.textContent = `All done! kept=${s.kept}  skipped=${s.skipped}`;
    title.textContent = 'Finished';
    status.textContent = '';
    return;
  }
  done.style.display = 'none';
  grid.style.display = 'flex';
  title.textContent = `[${s.idx+1}/${s.total}] ${s.split}/${s.group}`;
  status.textContent = `   kept=${s.kept}  skipped=${s.skipped}`;
  grid.innerHTML = '';
  for (const im of s.images) {
    const c = document.createElement('div'); c.className = 'card';
    const img = document.createElement('img'); img.src = im.url; img.loading = 'eager';
    const lab = document.createElement('div'); lab.className = 'label'; lab.textContent = im.label;
    c.appendChild(img); c.appendChild(lab); grid.appendChild(c);
  }
}

window.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  const k = (e.key || '').toLowerCase();
  if (k === 'y' || k === ' ' || k === 'enter') { e.preventDefault(); act('kept'); }
  else if (k === 'n' || k === 'x' || k === 'delete' || k === 'backspace') { e.preventDefault(); act('skipped'); }
  else if (k === 'b') { e.preventDefault(); act('back'); }
  else if (k === 'q' || k === 'escape') { e.preventDefault(); quit(); }
});

refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/state")
def state_endpoint():
    return jsonify(STATE.snapshot())


@app.route("/act", methods=["POST"])
def act_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    STATE.act(str(data.get("action", "")))
    return jsonify(STATE.snapshot())


@app.route("/img")
def img_endpoint():
    raw = request.args.get("path", "")
    if not raw:
        abort(404)
    p = Path(raw).resolve()
    try:
        p.relative_to(ALLOWED_ROOT)
    except ValueError:
        abort(403)
    if not p.exists() or not p.is_file():
        abort(404)
    return send_file(str(p))


@app.route("/quit", methods=["POST"])
def quit_endpoint():
    print("Quit requested from browser.")
    threading.Thread(target=lambda: (os._exit(0)), daemon=True).start()
    return jsonify({"ok": True})


# ---------------------------------------------------------------- main

def main():
    if not SRC_ROOT.exists():
        sys.exit(f"Source not found: {SRC_ROOT}")
    out = choose_output_dir()
    print(f"Output: {out}")
    groups = collect_groups()
    print(f"Found {len(groups)} groups in {SRC_ROOT}.")
    log_path = out / "_filter_log.tsv"

    global STATE
    STATE = State(groups, out, log_path)

    print()
    print(f"Server starting at http://localhost:{PORT}")
    print(f"If you are on a remote host over SSH, forward the port from your laptop:")
    print(f"    ssh -L {PORT}:localhost:{PORT} <user>@<host>")
    print("Then open http://localhost:%d in your browser." % PORT)
    print("(VS Code's Remote-SSH usually forwards this automatically.)")
    print()

    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
