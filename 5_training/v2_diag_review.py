#!/usr/bin/env python3
"""v2_diag_review.py — 诊断课试产复审页(逐组浏览,自包含 HTML)
================================================================

把 diag_d1_review.json 渲成一页 HTML:一次只展示一组(原图+终图 在上,
我们给的方向骨架 + 老师散文 在下),顶部「上一组/下一组/跳转/下一个被拒」翻页,
支持 ← → 键。所有数据 base64 内嵌,浏览器直接打开无需后端。

用法(仓库根目录,Chen 环境):
  python 5_training/v2_diag_review.py                # 生成 HTML 并自动起服务(默认)
  python 5_training/v2_diag_review.py --port 8080    # 端口被占时换一个
  python 5_training/v2_diag_review.py --no-serve     # 只写 HTML(想 scp 到本地直接双击时)
查看:脚本在服务器本机用 http.server 托管复审页并打印 http://localhost:<port>/…;
VS Code Remote-SSH 会自动转发该端口 → 终端里点链接即可;纯 SSH 则本地先跑一次 `ssh -L`。
(SSH 转发只能在本地发起,服务器端脚本代劳不了,故这里只负责起服务+给出转发命令。)
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))
from v2_diagnosis_gen import direction_skeleton  # noqa: E402


def img_b64(path: Path, disp: int) -> str:
    im = cv2.imread(str(path))
    if im is None:
        return ""
    s = disp / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)),
                        interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def find_paths(vid: str):
    key, vn = vid.rsplit("/v", 1)
    vd = REPO_ROOT / "fit_results" / key / f"version_{vn}"
    orig = vd / "original_cropped.png"
    if not orig.exists():
        orig = REPO_ROOT / "fit_results" / key / "original_captured_cropped.png"
    return orig, vd / "rendered.png", vd / "params.json"


CSS = """
*{box-sizing:border-box} body{margin:0;background:#0f1115;color:#e6e6e6;
 font:14px/1.6 system-ui,"Noto Sans CJK SC",sans-serif}
header{position:sticky;top:0;background:#1a1d24;border-bottom:1px solid #2a2f3a;
 padding:10px 18px;z-index:10;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:15px}
button{background:#232734;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:8px;
 padding:6px 14px;cursor:pointer;font-size:14px} button:hover{border-color:#4c8bf5}
button.pri{background:#4c8bf5;border-color:#4c8bf5;color:#fff;font-weight:600}
input{width:64px;background:#232734;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:5px}
.prog{color:#9aa4b2} .kbd{font:11px monospace;color:#9aa4b2;border:1px solid #2a2f3a;border-radius:4px;padding:0 4px}
.pill{padding:3px 12px;border-radius:20px;font-size:13px}
.pill.ok{background:#123d24;color:#4ade80} .pill.bad{background:#4a1520;color:#f87171}
main{max-width:1400px;margin:0 auto;padding:18px}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.vid{font:12px monospace;color:#9aa4b2} .tag{font-size:12px;padding:2px 8px;border-radius:6px;background:#232734;color:#9aa4b2}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
figure{margin:0} figcaption{font-size:12px;color:#9aa4b2;margin-bottom:4px}
img.cmp{width:100%;border-radius:8px;display:block;background:#000;cursor:zoom-in}
.cols{display:grid;grid-template-columns:1fr 1.4fr;gap:16px}
@media(max-width:800px){.cols{grid-template-columns:1fr} .pair{grid-template-columns:1fr}}
.box{background:#141821;border:1px solid #2a2f3a;border-radius:10px;padding:12px}
.box h4{margin:0 0 8px;font-size:12px;color:#4c8bf5;letter-spacing:.05em;text-transform:uppercase}
.skel{font:13px/1.8 monospace;color:#cbd5e1;white-space:pre-wrap}
.prose{white-space:pre-wrap;font-size:15px;line-height:1.75}
.viol{margin-top:10px;font-size:13px;color:#f87171}
.jsonl{font:11px monospace;color:#6b7280;margin-top:10px;word-break:break-all}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.93);display:none;align-items:center;
 justify-content:center;z-index:99;cursor:zoom-out} #zoom img{max-width:97vw;max-height:97vh}
"""

JS = r"""
let idx=0;
const z=document.getElementById('zoom'),zi=document.getElementById('zoomimg');
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function render(){
  const d=DATA[idx], v=document.getElementById('view');
  const badge=d.passed?'<span class="pill ok">通过</span>':'<span class="pill bad">被拒</span>';
  let viol='';
  if(d.violations&&d.violations.length){
    viol='<div class="viol">⚠ 违规: '+d.violations.map(x=>`${x.axis}: 期望 ${x.expected} / 老师 ${x.got} [${x.fatal?'致命·否决':(x.severity==='hard'?'hard·容忍':x.severity)}]`).join('; ')+'</div>';}
  v.innerHTML=`<div class="head">${badge}<span class="vid">${esc(d.vid)}</span>
      <span class="tag">${esc(d.category||'?')}</span><span class="tag">${d.bw?'黑白':'彩色'}</span></div>
    <div class="pair">
      <figure><figcaption>① 原图</figcaption>${d.o?`<img class="cmp" src="data:image/jpeg;base64,${d.o}">`:'<div class="box">原图缺失</div>'}</figure>
      <figure><figcaption>② 终图 (rendered)</figcaption>${d.r?`<img class="cmp" src="data:image/jpeg;base64,${d.r}">`:'<div class="box">终图缺失</div>'}</figure>
    </div>
    <div class="cols">
      <div class="box"><h4>我们给老师的方向骨架</h4><div class="skel">${esc(d.skel)}</div></div>
      <div class="box"><h4>老师写的诊断</h4><div class="prose">${esc(d.prose)}</div>${viol}
        <div class="jsonl">${esc(JSON.stringify(d.json||{}))}</div></div>
    </div>`;
  document.querySelectorAll('img.cmp').forEach(im=>im.onclick=()=>{zi.src=im.src;z.style.display='flex';});
  document.getElementById('prog').textContent=`${idx+1} / ${DATA.length}`;
  document.getElementById('jump').value=idx+1;
  window.scrollTo(0,0);
}
function go(d){idx=(idx+d+DATA.length)%DATA.length;render();}
function jumpTo(){let n=+document.getElementById('jump').value;if(n>=1&&n<=DATA.length){idx=n-1;render();}}
function nextRej(){for(let k=1;k<=DATA.length;k++){let j=(idx+k)%DATA.length;if(!DATA[j].passed){idx=j;render();return;}}alert('没有更多被拒样本');}
z.onclick=()=>z.style.display='none';
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
  if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1);});
render();
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """静默逐请求日志,保持终端干净。"""

    def log_message(self, *args):
        pass


def serve(html_path: Path, host: str, port: int):
    """在服务器本机起静态服务托管复审页;VS Code Remote-SSH 会自动把端口转发到本地,
    终端里点链接即可;纯 SSH 场景本地做一次 `ssh -L` 转发。Ctrl-C 停止。"""
    handler = functools.partial(_QuietHandler, directory=str(html_path.parent))
    for p in range(port, port + 20):                       # 端口被占则顺延
        try:
            httpd = http.server.ThreadingHTTPServer((host, p), handler)
            break
        except OSError:
            continue
    else:
        print(f"⚠ 端口 {port}–{port + 19} 都被占用,换 --port 再试")
        return
    url = f"http://localhost:{p}/{html_path.name}"
    bar = "═" * 64
    print(f"\n{bar}\n  复审页已上线 →  {url}\n"
          f"  · VS Code Remote-SSH:终端里 Ctrl/Cmd + 点上面链接即可(端口自动转发)\n"
          f"  · 纯 SSH:本地先跑  ssh -L {p}:localhost:{p} <user>@<server>  再点链接\n"
          f"  · Ctrl-C 停止服务\n{bar}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, default=REPO_ROOT / "sft_data_v2" / "diag_d1_review.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--disp", type=int, default=620, help="图片显示边长(逐组浏览可大一点)")
    ap.add_argument("--port", type=int, default=8000, help="本机静态服务端口(被占自动顺延)")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址(默认仅本机,配合 SSH/VSCode 端口转发)")
    ap.add_argument("--no-serve", action="store_true", help="只写 HTML 不起服务(想 scp 到本地时用)")
    args = ap.parse_args()
    out = args.out or args.review.with_suffix(".html")

    if not args.review.exists():
        sys.exit(f"找不到 {args.review}\n"
                 f"先跑 `python 5_training/v2_diagnosis_gen.py d1` 生成复审 JSON 再来。")

    review = json.loads(args.review.read_text())
    review.sort(key=lambda x: (x["passed"], x["vid"]))     # 被拒在前
    n_pass = sum(1 for x in review if x["passed"])

    data = []
    for x in review:
        orig, rend, pjson = find_paths(x["vid"])
        try:
            skel = direction_skeleton(json.loads(pjson.read_text()), x["bw"])
        except Exception:
            skel = "(参数读取失败)"
        data.append({
            "vid": x["vid"], "category": x.get("category"), "bw": x["bw"],
            "passed": x["passed"], "violations": x.get("violations", []),
            "skel": skel, "prose": x["prose"], "json": x.get("json", {}),
            "o": img_b64(orig, args.disp) if orig.exists() else "",
            "r": img_b64(rend, args.disp) if rend.exists() else "",
        })
    # 安全内嵌 JSON(防 </script> 截断)
    data_js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>诊断课复审 · {args.review.name}</title><style>{CSS}</style></head><body>
<header><b>诊断课复审</b>
  <span class="pill ok">通过 {n_pass}</span><span class="pill bad">被拒 {len(data)-n_pass}</span>
  <button onclick="go(-1)">← 上一组<span class="kbd">←</span></button>
  <button class="pri" onclick="go(1)">下一组 →<span class="kbd">→</span></button>
  <button onclick="nextRej()">下一个被拒</button>
  跳到 <input id="jump" type="number" min="1" onkeydown="if(event.key=='Enter')jumpTo()">
  <span class="prog" id="prog"></span>
  <span class="prog" style="font-size:12px">点图放大</span>
</header><main><div id="view"></div></main>
<div id="zoom"><img id="zoomimg"></div>
<script>const DATA={data_js};</script><script>{JS}</script></body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"写出 {out}  ({out.stat().st_size/1e6:.1f} MB, {len(data)} 组, 通过 {n_pass}/{len(data)})")

    if not args.no_serve:
        serve(out, args.host, args.port)


if __name__ == "__main__":
    main()
