#!/usr/bin/env python3
"""fit_results 人工标注工具（Flask Web 版）。

功能：
    逐组浏览 fit_results 下的拟合结果，对每一组：
      1) 选一个「内容分类」：动物/都市/风景/人物/食物/植物/其他（单选）
      2) 独立勾选「黑白」（色彩属性，与内容分类正交；用 fit_meta 的 bw_mode 智能预勾）
      3) 对该组下每个 version 选择「保留 / 舍弃」
      4) fit 失败（degenerate / param_l1==0 / n_active_params==0）的 version 会被
         自动高亮并默认预设为「舍弃」，方便一眼识别
    标注结果写入 JSON，支持断点续标；本工具【只读图片、只写标注文件，绝不删改任何原始数据】。

启动（在 A5000 服务器上）：
    python annotate_fit_results.py --split all --port 5000
本地访问（SSH 端口转发后用浏览器打开 http://localhost:5000）：
    ssh -L 5000:localhost:5000 <用户名>@<服务器地址>

命令行参数：
    --root    fit_results 根目录（默认脚本同级的 fit_results）
    --split   标注范围 all/train/test（默认 all）
    --out     标注结果 JSON 路径（默认脚本同级 fit_annotations.json）
    --host    监听地址（默认 127.0.0.1；如需局域网直连可设 0.0.0.0）
    --port    端口（默认 5000）
"""
import argparse
import json
import os
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

# ===== 常量 =====
CONTENT_CATEGORIES = ["动物", "都市", "风景", "人物", "食物", "植物", "其他"]  # 内容分类（单选；旧标注中的「抽象」「其它」已并入「其他」）
SPLITS = ("train", "test")
ORIG_NAME = "original_captured_cropped.png"  # 组内原图文件名

REPO_ROOT = Path(__file__).resolve().parent

# ===== 全局状态（启动时初始化）=====
app = Flask(__name__)
CFG = {}            # 运行配置：root / out / splits
GROUPS = []         # 组列表：[{"split","id","dir"(str),"versions":[0,1,..]}]
ANNO = {}           # 标注结果：key "split/id" -> {category,bw,versions,note,done,ts}
LOCK = threading.Lock()  # 保护 ANNO 读写与落盘


def gkey(split, gid):
    """组的唯一键。"""
    return f"{split}/{gid}"


def is_degenerate(meta):
    """判定某 version 是否为 fit 失败（参数塌成全 0）。

    三个信号在当前数据里完全一致，任一成立即视为失败，取并集更稳妥。
    """
    return (
        bool(meta.get("degenerate"))
        or meta.get("param_l1") == 0
        or meta.get("n_active_params") == 0
    )


def scan_groups(root, splits):
    """扫描 fit_results，收集所有含 version_* 的组（只列目录，不读 meta，启动快）。"""
    groups = []
    for split in splits:
        sdir = root / split
        if not sdir.is_dir():
            continue
        for gdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            versions = sorted(
                int(v.name.split("_")[1])
                for v in gdir.iterdir()
                if v.is_dir() and v.name.startswith("version_") and v.name.split("_")[1].isdigit()
            )
            if not versions:
                continue  # 没有 version 的组跳过
            groups.append({"split": split, "id": gdir.name, "dir": str(gdir), "versions": versions})
    return groups


def load_meta(group, v):
    """读取某 version 的 fit_meta.json（读不到返回 {}）。"""
    p = Path(group["dir"]) / f"version_{v}" / "fit_meta.json"
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


def normalize_anno(rec, versions_info):
    """把标注记录归一化为新 schema：每个 version -> {"decision","bw"}。

    兼容旧格式（versions 为 keep/discard 字符串、黑白是组级 rec["bw"]）：
      - 内容分类、每个 version 的保留/舍弃：原样保留（用户真实标注）
      - 黑白：改为逐 version 属性；旧的组级 bw 丢弃，用该 version 自己的 bw_mode 预勾
    """
    bwmode = {str(x["v"]): x["bw_mode"] for x in versions_info}
    out = {"category": rec.get("category"), "note": rec.get("note", ""),
           "done": bool(rec.get("done")), "discarded": bool(rec.get("discarded")),
           "ts": rec.get("ts"), "versions": {}}
    for k, v in (rec.get("versions") or {}).items():
        k = str(k)
        if isinstance(v, dict):  # 已是新格式
            out["versions"][k] = {"decision": "discard" if v.get("decision") == "discard" else "keep",
                                  "bw": bool(v.get("bw"))}
        else:  # 旧格式字符串：黑白改用 version 自身 bw_mode 预勾
            out["versions"][k] = {"decision": "discard" if v == "discard" else "keep",
                                  "bw": bool(bwmode.get(k, False))}
    return out


def save_anno():
    """原子写标注文件：先写临时文件再 os.replace，避免中途损坏。调用方需持有 LOCK。"""
    out = Path(CFG["out"])
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ANNO, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out)


def safe_media(split, gid, rel):
    """把请求的相对路径解析为 root 下的绝对文件路径，防止目录穿越。"""
    root = Path(CFG["root"]).resolve()
    target = (root / split / gid / rel).resolve()
    if root not in target.parents and target != root:
        abort(403)
    if not target.is_file():
        abort(404)
    return target


# ===== 路由 =====
@app.get("/")
def index():
    return INDEX_HTML


@app.get("/api/groups")
def api_groups():
    """返回精简的全部组列表 + 标注状态（供前端做进度、导航、跳转未标注）。"""
    out = []
    for i, g in enumerate(GROUPS):
        a = ANNO.get(gkey(g["split"], g["id"]))
        out.append({
            "idx": i,
            "split": g["split"],
            "id": g["id"],
            "n_versions": len(g["versions"]),
            "done": bool(a and a.get("done")),
            "discarded": bool(a and a.get("discarded")),
            "category": (a or {}).get("category"),
        })
    return jsonify({"groups": out, "categories": CONTENT_CATEGORIES})


@app.get("/api/group/<int:idx>")
def api_group(idx):
    """返回某组完整详情：各 version 的 meta / 图片 URL / 失败标志，以及已存标注或建议默认值。"""
    if idx < 0 or idx >= len(GROUPS):
        abort(404)
    g = GROUPS[idx]
    key = gkey(g["split"], g["id"])
    has_orig = (Path(g["dir"]) / ORIG_NAME).is_file()

    versions = []
    for v in g["versions"]:
        meta = load_meta(g, v)
        deg = is_degenerate(meta)
        base = f"/img/{g['split']}/{g['id']}/version_{v}"
        versions.append({
            "v": v,
            "degenerate": deg,
            "bw_mode": bool(meta.get("bw_mode")),
            "final_loss": meta.get("final_loss"),
            "param_l1": meta.get("param_l1"),
            "n_active_params": meta.get("n_active_params"),
            "compare_url": f"{base}/compare.png",
            "rendered_url": f"{base}/rendered.png",
        })

    saved = ANNO.get(key)
    if saved:
        anno = normalize_anno(saved, versions)  # 已标注：回填（兼容旧格式）
    else:
        # 未标注：给建议默认值（每个 version：黑白按自身 bw_mode 预勾；degenerate 预设舍弃）
        anno = {
            "category": None,
            "versions": {
                str(x["v"]): {"decision": "discard" if x["degenerate"] else "keep", "bw": x["bw_mode"]}
                for x in versions
            },
            "note": "",
            "done": False,
            "discarded": False,
        }

    return jsonify({
        "idx": idx,
        "split": g["split"],
        "id": g["id"],
        "has_orig": has_orig,
        "orig_url": f"/img/{g['split']}/{g['id']}/{ORIG_NAME}" if has_orig else None,
        "versions": versions,
        "anno": anno,
        "saved": bool(saved),
    })


@app.post("/api/group/<int:idx>")
def api_save(idx):
    """保存某组标注。"""
    if idx < 0 or idx >= len(GROUPS):
        abort(404)
    g = GROUPS[idx]
    key = gkey(g["split"], g["id"])
    body = request.get_json(force=True) or {}

    cat = body.get("category")
    if cat is not None and cat not in CONTENT_CATEGORIES:
        abort(400, "非法分类")
    versions = {}
    for k, v in (body.get("versions") or {}).items():
        if isinstance(v, dict):
            versions[str(k)] = {"decision": "discard" if v.get("decision") == "discard" else "keep",
                                "bw": bool(v.get("bw"))}
        else:  # 兼容纯字符串
            versions[str(k)] = {"decision": "discard" if v == "discard" else "keep", "bw": False}

    import time
    rec = {
        "category": cat,
        "versions": versions,
        "note": (body.get("note") or "").strip(),
        "done": bool(body.get("done", True)),
        "discarded": bool(body.get("discarded")),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with LOCK:
        ANNO[key] = rec
        save_anno()
        done = sum(1 for a in ANNO.values() if a.get("done"))
    return jsonify({"ok": True, "done_total": done, "grand_total": len(GROUPS)})


@app.get("/img/<split>/<gid>/<path:rel>")
def img(split, gid, rel):
    """图片服务（只读、带路径穿越防护）。"""
    f = safe_media(split, gid, rel)
    return send_file(f, max_age=3600)


# ===== 前端页面（原生 HTML/CSS/JS，无外部依赖，适配服务器无外网环境）=====
INDEX_HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fit_results 标注</title>
<style>
:root{--bg:#0f1115;--panel:#1a1d24;--line:#2a2f3a;--fg:#e6e6e6;--mut:#9aa4b2;--acc:#4c8bf5;--ok:#2ecc71;--bad:#e74c3c;--warn:#f1c40f}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,"Noto Sans CJK SC",sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header .title{font-weight:700}
.mut{color:var(--mut)}
.bar{flex:1;height:8px;background:var(--line);border-radius:6px;overflow:hidden;min-width:120px}
.bar>i{display:block;height:100%;background:var(--acc)}
button{background:#232734;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;font-size:14px}
button:hover{border-color:var(--acc)}
button.pri{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
input[type=number]{width:70px;background:#232734;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px}
main{padding:14px;max-width:1500px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.chip{padding:7px 14px;border-radius:20px;border:1px solid var(--line);background:#232734;cursor:pointer;user-select:none}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
.chip.bw.on{background:#ddd;color:#111;border-color:#ddd}
.vbw{padding:4px 10px;font-size:13px}
.kbd{font:12px monospace;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:4px}
.vlist{display:flex;flex-direction:column;gap:14px}
.vcard{border:1px solid var(--line);border-radius:10px;padding:10px;background:#141821}
.vcard.deg{border-color:var(--bad);box-shadow:0 0 0 1px var(--bad) inset}
.vhead{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.tag{font-size:12px;padding:2px 8px;border-radius:6px;background:#232734;color:var(--mut)}
.tag.bad{background:var(--bad);color:#fff;font-weight:700}
.tag.bw{background:#ddd;color:#111}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:6px 16px}
.seg button.keep.on{background:var(--ok);color:#0b1a10;font-weight:700}
.seg button.discard.on{background:var(--bad);color:#fff;font-weight:700}
img.cmp{width:100%;border-radius:8px;display:block;background:#000;cursor:zoom-in}
.cmplabel{display:flex;color:var(--mut);font-size:12px;margin-top:4px}
.cmplabel span{flex:1;text-align:center}
.orig{max-width:320px;border-radius:8px;border:1px solid var(--line)}
.miss{color:var(--warn)}
textarea{width:100%;background:#232734;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;min-height:44px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;z-index:99;cursor:zoom-out}
#zoom img{max-width:98vw;max-height:98vh}
.savest{font-size:13px}
.savest.dirty{color:var(--warn)}.savest.saved{color:var(--ok)}
button.dg.on{background:var(--bad);border-color:var(--bad);color:#fff;font-weight:700}
#groupcard.on{box-shadow:0 0 0 2px var(--bad) inset}
body.dg-on #cats,body.dg-on #versions{opacity:.4}
</style></head><body>
<header>
  <span class="title">fit_results 标注</span>
  <span class="bar"><i id="pbar"></i></span>
  <span class="mut" id="prog">…</span>
  <button onclick="go(-1)">← 上一组<span class="kbd">←</span></button>
  <button onclick="go(1)">下一组 →<span class="kbd">→</span></button>
  <button onclick="nextTodo()">下一个未标<span class="kbd">n</span></button>
  跳到 <input type="number" id="jump" min="1" onkeydown="if(event.key=='Enter')jumpTo()">
  <span class="savest" id="savest"></span>
</header>
<main>
  <div class="card" id="groupcard">
    <div class="row" style="justify-content:space-between">
      <div><b id="gid">…</b> <span class="mut" id="gmeta"></span></div>
      <div class="row">
        <span id="savedflag" class="mut"></span>
        <button id="dgbtn" class="dg" onclick="toggleDiscardGroup()">🗑 舍弃整组<span class="kbd">d</span></button>
      </div>
    </div>
    <div id="dgbanner" style="display:none;color:var(--bad);font-weight:700;margin-top:8px">⚠ 已标记「舍弃整组」——保存后整组视为全部舍弃（导出时整组剔除）</div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:8px"><b>内容分类</b><span class="mut">（整组单选，按数字键 1-7）</span></div>
    <div class="row" id="cats"></div>
  </div>

  <div class="card">
    <div class="row" style="justify-content:space-between;margin-bottom:6px">
      <b>各 version（保留/舍弃 · 黑白逐个勾选）</b>
      <span class="mut">对比图三联：<b>原图</b> · <b>我们的渲染</b> · <b>目标(参考)</b></span>
    </div>
    <div class="vlist" id="versions"></div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:6px"><b>原图</b> <span class="mut">original_captured_cropped.png</span></div>
    <div id="origbox"></div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:6px"><b>备注</b></div>
    <textarea id="note" placeholder="可选备注…"></textarea>
    <div class="row" style="margin-top:12px">
      <button class="pri" onclick="saveNext()">保存并下一个未标<span class="kbd">空格</span></button>
      <button onclick="save(false)">仅保存<span class="kbd">s</span></button>
    </div>
  </div>
</main>
<div id="zoom" onclick="this.style.display='none'"><img id="zoomimg"></div>
<script>
let GROUPS=[], CATS=[], idx=0, cur=null, dirty=false;

function setDirty(d){dirty=d;const e=document.getElementById('savest');
  e.textContent=d?'● 未保存':'✓ 已保存';e.className='savest '+(d?'dirty':'saved');}

async function boot(){
  const r=await (await fetch('/api/groups')).json();
  GROUPS=r.groups;CATS=r.categories;
  renderCats();
  // 默认从第一个未标注开始，没有则第 0 个
  let start=GROUPS.findIndex(g=>!g.done); if(start<0)start=0;
  await load(start);
  updateProg();
}
function renderCats(){
  const c=document.getElementById('cats');c.innerHTML='';
  CATS.forEach((name,i)=>{const s=document.createElement('span');s.className='chip';
    s.innerHTML=name+'<span class="kbd">'+(i+1)+'</span>';s.onclick=()=>pickCat(name);
    s.dataset.name=name;c.appendChild(s);});
}
function updateProg(){
  const done=GROUPS.filter(g=>g.done).length,tot=GROUPS.length;
  const dg=GROUPS.filter(g=>g.discarded).length;
  document.getElementById('pbar').style.width=(tot?done/tot*100:0)+'%';
  document.getElementById('prog').textContent=`${done}/${tot} 已标${dg?`（舍弃整组 ${dg}）`:''} · 当前 ${idx+1}`;
  document.getElementById('jump').max=tot;
}
async function load(i){
  if(i<0||i>=GROUPS.length)return;
  idx=i;
  const g=await (await fetch('/api/group/'+i)).json();cur=g;
  document.getElementById('gid').textContent=`[${g.split}] ${g.id}`;
  document.getElementById('gmeta').textContent=`第 ${i+1}/${GROUPS.length} 组 · ${g.versions.length} 个 version`;
  document.getElementById('savedflag').textContent=g.saved?'（此组已标注过）':'（尚未标注）';
  setDiscardGroup(!!g.anno.discarded);
  // 分类
  document.querySelectorAll('#cats .chip').forEach(s=>s.classList.toggle('on',s.dataset.name===g.anno.category));
  // versions（黑白改为逐 version 勾选）
  const vc=document.getElementById('versions');vc.innerHTML='';
  g.versions.forEach(v=>{
    const va=(g.anno.versions&&g.anno.versions[String(v.v)])||{decision:'keep',bw:v.bw_mode};
    const d=document.createElement('div');d.className='vcard'+(v.degenerate?' deg':'');
    const loss=(v.final_loss==null?'—':(+v.final_loss).toFixed(3));
    d.innerHTML=`<div class="vhead">
        <b>version_${v.v}</b>
        ${v.degenerate?'<span class="tag bad">FIT 失败 · 参数全 0</span>':''}
        ${v.bw_mode?'<span class="tag bw">拟合黑白模式</span>':''}
        <span class="tag">loss ${loss}</span>
        <span class="tag">活跃参数 ${v.n_active_params==null?'—':v.n_active_params}</span>
        <span class="chip bw vbw" data-v="${v.v}">⬜ 黑白</span>
        <span class="seg">
          <button class="keep" data-v="${v.v}" data-act="keep">保留</button>
          <button class="discard" data-v="${v.v}" data-act="discard">舍弃</button>
        </span>
      </div>
      <img class="cmp" loading="lazy" src="${v.compare_url}" onclick="zoom(this.src)"
           onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'miss',textContent:'compare.png 缺失'}))">
      <div class="cmplabel"><span>① 原图</span><span>② 我们的渲染</span><span>③ 目标(参考)</span></div>`;
    vc.appendChild(d);
    setSeg(v.v,va.decision);
    setVBW(v.v,va.bw);
  });
  vc.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>setSeg(+b.dataset.v,b.dataset.act,true));
  vc.querySelectorAll('.vbw').forEach(c=>c.onclick=()=>toggleVBW(+c.dataset.v));
  // 原图
  const ob=document.getElementById('origbox');
  ob.innerHTML=g.has_orig?`<img class="orig" src="${g.orig_url}" onclick="zoom(this.src)">`
    :'<span class="miss">⚠ 原图缺失（该组在源数据集中无对应原图）</span>';
  // 备注
  document.getElementById('note').value=g.anno.note||'';
  setDirty(false);updateProg();
}
function pickCat(name){document.querySelectorAll('#cats .chip').forEach(s=>s.classList.toggle('on',s.dataset.name===name));
  cur.anno.category=name;setDirty(true);}
function setDiscardGroup(on){if(cur)cur.anno.discarded=on;
  document.getElementById('dgbtn').classList.toggle('on',on);
  document.getElementById('groupcard').classList.toggle('on',on);
  document.getElementById('dgbanner').style.display=on?'block':'none';
  document.body.classList.toggle('dg-on',on);}
function toggleDiscardGroup(){setDiscardGroup(!cur.anno.discarded);setDirty(true);}
function vslot(v){cur.anno.versions=cur.anno.versions||{};
  cur.anno.versions[String(v)]=cur.anno.versions[String(v)]||{decision:'keep',bw:false};
  return cur.anno.versions[String(v)];}
function setSeg(v,act,mark){
  document.querySelectorAll(`.seg button[data-v="${v}"]`).forEach(b=>b.classList.toggle('on',b.dataset.act===act));
  if(cur)vslot(v).decision=act;
  if(mark)setDirty(true);
}
function setVBW(v,on){if(cur)vslot(v).bw=on;
  const c=document.querySelector(`.vbw[data-v="${v}"]`);
  if(c){c.classList.toggle('on',on);c.firstChild.textContent=(on?'✅':'⬜')+' 黑白';}}
function toggleVBW(v){setVBW(v,!vslot(v).bw);setDirty(true);}
function collect(){cur.anno.note=document.getElementById('note').value;return {
  category:cur.anno.category,versions:cur.anno.versions,note:cur.anno.note,discarded:!!cur.anno.discarded,done:true};}
async function save(){
  const r=await (await fetch('/api/group/'+idx,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(collect())})).json();
  GROUPS[idx].done=true;GROUPS[idx].category=cur.anno.category;GROUPS[idx].discarded=!!cur.anno.discarded;
  setDirty(false);document.getElementById('savedflag').textContent='（已保存）';updateProg();return r;}
async function saveNext(){await save();nextTodo();}
async function go(d){if(dirty)await save();load(idx+d);}
async function jumpTo(){const n=+document.getElementById('jump').value;if(n>=1&&n<=GROUPS.length){if(dirty)await save();load(n-1);}}
async function nextTodo(){if(dirty)await save();
  let n=GROUPS.findIndex((g,i)=>i>idx&&!g.done);
  if(n<0)n=GROUPS.findIndex(g=>!g.done);
  if(n<0){alert('全部标注完成！🎉');return;}load(n);}
function zoom(src){document.getElementById('zoomimg').src=src;document.getElementById('zoom').style.display='flex';}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(e.key>='1'&&e.key<='9'){const i=+e.key-1;if(i<CATS.length)pickCat(CATS[i]);}
  else if(e.key==='d')toggleDiscardGroup();
  else if(e.key===' '){e.preventDefault();saveNext();}
  else if(e.key==='s')save();
  else if(e.key==='n')nextTodo();
  else if(e.key==='ArrowLeft')go(-1);
  else if(e.key==='ArrowRight')go(1);
});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
boot();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="fit_results 人工标注工具（Flask Web）")
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "fit_results", help="fit_results 根目录")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all", help="标注范围")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "fit_annotations.json", help="标注结果 JSON")
    ap.add_argument("--only", type=Path, default=None,
                    help="只加载清单中的组（每行 <split>/<gid>，# 开头为注释）；复审重拟合结果用，"
                         "例如 sft_data_v2/refit_review_groups.txt")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址")
    ap.add_argument("--port", type=int, default=5000, help="端口")
    args = ap.parse_args()

    splits = SPLITS if args.split == "all" else (args.split,)
    CFG["root"] = str(args.root)
    CFG["out"] = str(args.out)

    global GROUPS, ANNO
    GROUPS = scan_groups(args.root, splits)
    if not GROUPS:
        raise SystemExit(f"未在 {args.root} 下找到任何含 version_* 的组")

    # --only 清单过滤（复审模式）：只保留清单里的组
    if args.only:
        keep = {ln.strip() for ln in args.only.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")}
        GROUPS = [g for g in GROUPS if gkey(g["split"], g["id"]) in keep]
        print(f"--only 清单 {args.only}：匹配到 {len(GROUPS)}/{len(keep)} 组")
        if not GROUPS:
            raise SystemExit("--only 清单没有匹配到任何组（检查行格式应为 <split>/<gid>）")

    # 断点续标：载入已有标注
    if args.out.is_file():
        ANNO = json.load(open(args.out, encoding="utf-8"))

    done = sum(1 for a in ANNO.values() if a.get("done"))
    print(f"共 {len(GROUPS)} 组（split={args.split}），已标注 {done} 组")
    print(f"标注文件：{args.out}")
    print(f"打开浏览器访问 http://{args.host}:{args.port}  （远程请先 SSH 端口转发）")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
