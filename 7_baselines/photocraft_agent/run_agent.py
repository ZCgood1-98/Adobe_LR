"""PhotoCraft 智能体修图基线：Claude Code / Codex 通过 PhotoCraft MCP 自动调参修图。

约束：修图决策只来自宿主智能体自己的模型（claude -p / codex exec），PhotoCraft 只负责确定性渲染，
不调用任何其他模型（禁止 Select Subject 等内置分割功能，运行后审计对话记录）。

每张图流程：
  1. 智能体 doc_open 原图 → 查看 → command_run filter.cameraRaw（一次给全参数）
  2. render_preview 看结果；要改就先 history.undo 回原图，再整组重下（不叠加），最多 MAX_ROUNDS 轮
  3. 最后一条消息：不超过 3 句理由 + 末行 {"params": {...}}（与 C 类格式一致）
  4. 本脚本用「原图 + 最终参数」经 photocraft-cli run 确定性重渲染 → 成品 jpg(q95)，
     所以成品可复现，且不受智能体中途叠加/误操作影响

提示词与 benchmark 对齐：审美目标 GOAL 与 B/C 类逐字一致，系统段照 C 类 SYS；保留看预览的反馈（C 类也有反馈）。

输出（与 run_a.py 一致）：runs/A/PhotoCraft-<提示词版本>_<模型>/<split>/<uuid>.jpg
旁路：.../_meta/<split>/<uuid>/{params.json, transcript.jsonl, audit.json, stderr.txt}；.../_meta/prompt.txt

用法：
  python run_agent.py --agent claude --model claude-haiku-5-5 --split test --n 10
  python run_agent.py --agent codex  --model gpt-6-luna      --split test --n 10
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = Path('/home/chen/Work/adobe_lr/7_baselines')
INPUTS = BASE / 'runs/A/_inputs'
MAX_ROUNDS = 5
# 智能体允许使用的 MCP 工具与引擎命令（审计依据）
ALLOWED_TOOLS = ['doc_open', 'command_run', 'render_preview', 'doc_inspect']
ALLOWED_CMDS = {'filter.cameraRaw', 'history.undo', 'history.redo', 'edit.undo', 'file.revert'}

# —— 与 benchmark 对齐 ——
# GOAL 逐字等于 7_baselines/smoke/run_bc.py 的 GOAL（= B_PROMPT 前两句 + 内容保持句），B/C/本基线三者审美目标一致。
# 系统段照搬 C 类 SYS 的结构，只把「自研渲染器」换成 PhotoCraft 的说明；参数说明按 measure_dirs.py 实测写。
# 与 C 类的唯一机制差异：反馈方式（C=九宫格选候选；本基线=自己看预览、可撤销重来）。
GOAL = ("Retouch this photograph like a professional photo editor working in Lightroom. "
        "Improve its overall aesthetic quality using only global photographic adjustments "
        "(exposure, contrast, highlights/shadows, white balance, color and saturation, clarity). "
        "Do NOT change the content, composition or framing.")
PROTO = 'p2'  # 提示词版本；改 PROMPT 必须改它（输出目录随之更换，避免新旧结果混用）

PROMPT = """You are a professional photo editor.
Goal: """ + GOAL + """
How you work: you do not edit pixels yourself. You choose values for the global parameters of the PhotoCraft RAW Develop filter (Camera Raw-style sliders applied to an already-rendered photo; it is not Lightroom, and the exact direction and strength of every parameter are documented below), and the filter applies them to the whole photo. No masks, no cropping. 0 or an omitted parameter means no change.

You work through the `photocraft` MCP tools:
1. Call doc_open with path "{name}", then render_preview (max_side 768) to see the original photo.
2. Apply ONE complete set of settings: command_run with id "filter.cameraRaw" and params = your settings.
3. After that command_run has returned (never in parallel with it), call render_preview (max_side 768) and judge the result against the goal by eye.
4. To try different settings, FIRST call command_run with id "history.undo" (the filter must never be stacked; undo returns to the original), then apply a complete new set and preview again. At most {rounds} filter.cameraRaw attempts in total. If no setting improves the photo, undo and finish with an empty params object.
5. Use only the commands filter.cameraRaw and history.undo; no other commands.

Controls of filter.cameraRaw (camelCase keys; measured behaviour of this filter):
- "exposure": -5..5 stops; +1 doubles linear light (mid-grey 128 -> 176), -1 halves it (128 -> 92).
- "temperature": -100..100 relative white balance; + warmer (more red, less blue), - cooler.
- "tint": -100..100; + more magenta (less green), - greener.
- "contrast": -100..100; pivots around mid-grey (mid-grey stays put, darker tones darker and lighter tones lighter for +).
- "highlights", "shadows": -100..100; - darkens / + brightens mainly the bright resp. dark tones (use -highlights to recover bright areas, +shadows to open dark areas).
- "whites", "blacks": -100..100; move the white / black end of the tone range (+whites brightens the upper range, -blacks deepens the lower range).
- "texture", "clarity", "dehaze": -100..100; local contrast at fine / medium scale and haze removal (+ crisper, - softer).
- "vibrance": -100..100; saturation boost weighted toward less-saturated colours, gentler on skin.
- "saturation": -100..100; uniform; -100 = fully greyscale, -50 = half saturation, +50 is already strong.
- "curveHighlights", "curveLights", "curveDarks", "curveShadows": -100..100 parametric tone curve regions (+ brightens that region).
- "hslHue", "hslSat", "hslLum": arrays of 8 numbers in -100..100 for [red, orange, yellow, green, aqua, blue, purple, magenta]. Hue + rotates each colour toward the next one in that list (red -> orange, green -> aqua, blue -> purple). Sat - desaturates that colour only; Lum - darkens it.
- "gradeShadows", "gradeMidtones", "gradeHighlights": {{"hue": 0..360, "sat": 0..100, "lum": -100..100}} colour tint added to that tonal range (hue in degrees: 0 red, 120 green, 220 blue).
- "vignetteAmount": -100..100; - darkens the corners, + lightens them.
- "sharpenAmount": 0..150.

Final answer: a short rationale (at most 3 sentences), then ONE JSON object on the last line with the settings of your last applied filter.cameraRaw call (or {{}} if you kept the original):
{{"params": {{...}}}}"""


def tag_of(model):
    """输出目录名带提示词版本，提示词一改就换目录。"""
    return f'PhotoCraft-{PROTO}_{model}'


def check_prompt(root):
    """目录里记录提示词全文；已有且不同 → 拒绝续跑（防止新旧提示词结果混在同一目录）。"""
    f = root / '_meta' / 'prompt.txt'
    if f.exists() and f.read_text() != PROMPT:
        sys.exit(f'{f} 与当前 PROMPT 不一致：改了提示词请同时改 PROTO')
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(PROMPT)


def sample_names(split, n, seed):
    names = sorted(p.name for p in (INPUTS / split).glob('*.png'))
    random.Random(seed).shuffle(names)
    return names[:n]


def build_cmd(agent, model, prompt, work, in_dir):
    """构造无头调用命令：只挂 photocraft 一个 MCP 服务器，读根=输入目录，写根=本图工作目录。"""
    server_args = ['mcp', '--automation-read-root', str(in_dir), '--automation-write-root', str(work)]
    if agent == 'claude':
        cfg = work / 'mcp.json'
        cfg.write_text(json.dumps({'mcpServers': {'photocraft': {'command': 'photocraft-cli', 'args': server_args}}}))
        return ['claude', '-p', prompt, '--model', model,
                '--mcp-config', str(cfg), '--strict-mcp-config',
                '--tools', '',  # 关掉内置工具（Bash/Read/Write 等），只剩 MCP
                '--allowedTools', ' '.join(f'mcp__photocraft__{t}' for t in ALLOWED_TOOLS),
                '--permission-mode', 'dontAsk',
                '--output-format', 'stream-json', '--verbose', '--no-session-persistence']
    # codex：只读沙箱、不审批；MCP 服务器经 -c 注入
    return ['codex', 'exec', '-m', model,
            '-c', 'model_reasoning_effort="medium"',
            '-c', 'mcp_servers.photocraft.command="photocraft-cli"',
            '-c', 'mcp_servers.photocraft.args=' + json.dumps(server_args),
            '-c', 'mcp_servers.photocraft.enabled_tools=' + json.dumps(ALLOWED_TOOLS),
            '-c', 'mcp_servers.photocraft.default_tools_approval_mode="approve"',
            '-c', 'approval_policy="never"',
            '-s', 'read-only', '--skip-git-repo-check', '--ephemeral', '--json',
            '-C', str(work), '-o', str(work / 'last_message.txt'), prompt]


def iter_json_objects(text):
    """从文本里找出所有顶层 JSON 对象（花括号配平）。"""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0:
                try:
                    yield json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass


def final_text(agent, stdout, work):
    if agent == 'codex':
        f = work / 'last_message.txt'
        return f.read_text() if f.exists() else ''
    for line in reversed(stdout.splitlines()):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get('type') == 'result':
            return ev.get('result') or ''
    return ''


def audit(agent, stdout):
    """从对话记录里统计 MCP 工具调用和引擎命令 id，标出越界调用。"""
    calls = []
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if agent == 'claude':
            for c in (ev.get('message') or {}).get('content') or []:
                if isinstance(c, dict) and c.get('type') == 'tool_use':
                    calls.append({'tool': c['name'].split('__')[-1], 'args': c.get('input', {})})
        else:
            item = ev.get('item') or {}
            if ev.get('type') == 'item.completed' and item.get('type') == 'mcp_tool_call':
                calls.append({'tool': item.get('tool'), 'args': item.get('arguments') or {},
                              'error': item.get('error')})
            elif ev.get('type') == 'item.completed' and item.get('type') == 'command_execution':
                calls.append({'tool': 'shell', 'args': {'command': item.get('command')}})
    cmds = [c['args'].get('id') for c in calls if c['tool'] == 'command_run']
    bad = [c for c in calls if c['tool'] not in ALLOWED_TOOLS] + \
          [{'tool': 'command_run', 'id': i} for i in cmds if i not in ALLOWED_CMDS]
    return {'n_calls': len(calls), 'tools': [c['tool'] for c in calls], 'commands': cmds,
            'n_filter': cmds.count('filter.cameraRaw'), 'n_preview': sum(c['tool'] == 'render_preview' for c in calls),
            'violations': bad}


def render(src, params, out_jpg, tmp_png):
    """原图 + 参数 → 成品（确定性重渲染），存 jpg q95。"""
    r = subprocess.run(['photocraft-cli', 'run', str(src), '--cmd', 'filter.cameraRaw',
                        '--params', json.dumps(params), '--out', str(tmp_png)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    Image.open(tmp_png).convert('RGB').save(out_jpg, quality=95)
    tmp_png.unlink()


# ---------- 额度监控 ----------
# 订阅额度/限流的报错特征（Claude Code 与 Codex 的措辞都覆盖；只在「没拿到参数」的失败调用上检查）
LIMIT_RE = re.compile(r'usage[ _-]?limit|rate[ _-]?limit|hit your (usage )?limit|limit (reached|exceeded)|'
                      r'quota|too many requests|\b429\b|overloaded|resets? (at|in)', re.I)


def error_text(agent, stdout, stderr):
    """只抽取报错相关文本（stderr + 错误事件），不看模型正文，避免误判。"""
    parts = [stderr or '']
    for line in (stdout or '').splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if agent == 'claude' and ev.get('type') == 'result' and ev.get('is_error'):
            parts.append(json.dumps(ev, ensure_ascii=False))
        elif agent == 'claude' and ev.get('type') == 'assistant' and ev.get('error'):
            parts.append(json.dumps(ev.get('error'), ensure_ascii=False))
        elif agent == 'codex' and ev.get('type') in ('error', 'turn.failed'):
            parts.append(json.dumps(ev, ensure_ascii=False))
    return '\n'.join(parts)


class LimitGate:
    """全局暂停闸门：任一线程撞到额度 → 所有线程暂停；首次等 limit_wait（默认 5h），
    醒来后若仍限流则每 retry_wait（默认 30min）重试一次；有一次成功就复位。"""

    def __init__(self, limit_wait, retry_wait, status_file):
        self.lock = threading.Lock()
        self.until = 0.0
        self.after_wake = False   # 刚从暂停中醒来、尚未成功过
        self.limit_wait, self.retry_wait = limit_wait, retry_wait
        self.status_file = status_file
        self.done = self.failed = self.trips = self.fail_streak = 0
        self.total = 0
        self.write('running')

    def write(self, state, note=''):
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.write_text(json.dumps({
            'state': state, 'note': note, 'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'resume_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.until)) if self.until > time.time() else None,
            'done': self.done, 'failed': self.failed, 'total': self.total, 'limit_trips': self.trips,
        }, ensure_ascii=False, indent=1))

    def wait(self):
        while True:
            with self.lock:
                left = self.until - time.time()
                if left <= 0:
                    return
            time.sleep(min(60, left))

    def trip(self, who, msg):
        with self.lock:
            now = time.time()
            if now < self.until:          # 别的线程已经触发过暂停
                return
            wait = self.retry_wait if self.after_wake else self.limit_wait
            self.until = now + wait
            self.after_wake = True
            self.trips += 1
            resume = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.until))
            print(f'[额度] {time.strftime("%H:%M:%S")} {who} 撞到额度：{msg[:160]!r} → 暂停 {wait / 3600:.2f}h，'
                  f'{resume} 恢复', flush=True)
            self.write('paused', msg[:300])

    def success(self, ok):
        with self.lock:
            self.after_wake = self.after_wake and not ok
            self.fail_streak = 0 if ok else self.fail_streak + 1
            self.done += ok
            self.failed += (not ok)
            self.write('running')


GATE = None


def run_one(a, name):
    tag = tag_of(a.model)
    uuid = Path(name).stem
    out_jpg = BASE / 'runs/A' / tag / a.split / f'{uuid}.jpg'
    meta = BASE / 'runs/A' / tag / '_meta' / a.split / uuid
    if out_jpg.exists() and not a.force:
        return {'name': uuid, 'status': 'skip'}
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    in_dir = INPUTS / a.split
    prompt = PROMPT.format(name=name, rounds=MAX_ROUNDS)
    while True:  # 撞额度就等闸门放行后重试同一张
        GATE.wait()
        work = meta / 'work'
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        t0 = time.time()
        try:
            r = subprocess.run(build_cmd(a.agent, a.model, prompt, work, in_dir), capture_output=True,
                               text=True, timeout=a.timeout, cwd=work, stdin=subprocess.DEVNULL)
            stdout, stderr, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            stdout, stderr, rc = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), 'TIMEOUT', -1
        dt = time.time() - t0
        (meta / 'transcript.jsonl').write_text(stdout)
        (meta / 'stderr.txt').write_text(stderr)
        # 取最后一个带 params 的 JSON 作为最终参数
        objs = [o for o in iter_json_objects(final_text(a.agent, stdout, work)) if isinstance(o.get('params'), dict)]
        if a.fake_limit_once and not getattr(a, '_faked', False):  # 测试用：第一次调用假装撞额度
            a._faked, objs, stderr = True, [], 'FAKE: You have hit your usage limit'
        if not objs:
            err = error_text(a.agent, stdout, stderr)
            m = LIMIT_RE.search(err)
            if m:
                GATE.trip(uuid, err[max(0, m.start() - 80):m.end() + 80])
                continue
            # 保险：报错措辞没被正则覆盖时，连续 3 张非零退出且无参数也按撞额度处理
            if rc != 0 and GATE.fail_streak >= 2:
                GATE.trip(uuid, f'连续失败兜底(rc={rc}): ' + err[-200:])
                continue
        break
    au = audit(a.agent, stdout)
    au.update({'rc': rc, 'seconds': round(dt, 1)})
    status = 'ok'
    if not objs:
        status = 'no_params'
    else:
        params = objs[-1]['params']
        # 理由 = 最终回答里 JSON 之前的文字（与 C 类「理由 + 末行 JSON」格式一致）
        ft = final_text(a.agent, stdout, work)
        rationale = ft[:ft.rfind('{"params"')].strip() if '{"params"' in ft else ft.strip()
        (meta / 'params.json').write_text(json.dumps({'params': params, 'rationale': rationale},
                                                     ensure_ascii=False, indent=1))
        try:
            render(in_dir / name, params, out_jpg, meta / 'render.png')
        except Exception as e:  # 参数不合法等
            status = f'render_fail: {e}'
    au['status'] = status
    (meta / 'audit.json').write_text(json.dumps(au, ensure_ascii=False, indent=1))
    GATE.success(status == 'ok')
    return {'name': uuid, 'status': status, 'sec': round(dt), 'filters': au['n_filter'],
            'previews': au['n_preview'], 'violations': len(au['violations'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', choices=['claude', 'codex'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--names', nargs='*', help='指定文件名（覆盖抽样）')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit-wait', type=float, default=5 * 3600, help='撞额度后首次暂停秒数（默认 5 小时）')
    ap.add_argument('--retry-wait', type=float, default=1800, help='醒来仍限流时的重试间隔秒数（默认 30 分钟）')
    ap.add_argument('--fake-limit-once', action='store_true', help='测试：第一次调用假装撞额度')
    a = ap.parse_args()
    names = a.names or sample_names(a.split, a.n, a.seed)
    global GATE
    root = BASE / 'runs/A' / tag_of(a.model)
    check_prompt(root)
    GATE = LimitGate(a.limit_wait, a.retry_wait, root / f'_status_{a.split}.json')
    GATE.total = len(names)
    print(f'{a.agent}/{a.model} {a.split} n={len(names)}  状态文件 {GATE.status_file}', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(lambda n: run_one(a, n), names):
            print(json.dumps(res, ensure_ascii=False), flush=True)
    GATE.write('finished')


if __name__ == '__main__':
    main()
