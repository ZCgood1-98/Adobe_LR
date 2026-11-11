"""PhotoCraft 智能体修图基线：Claude Code / Codex 通过 PhotoCraft MCP 自动调参修图。

约束：修图决策只来自宿主智能体自己的模型（claude -p / codex exec），PhotoCraft 只负责确定性渲染，
不调用任何其他模型（禁止 Select Subject 等内置分割功能，运行后审计对话记录）。

每张图流程：
  1. 智能体 doc_open 原图 → 查看 → command_run filter.cameraRaw（一次给全参数）
  2. render_preview 看结果；要改就先 history.undo 回原图，再整组重下（不叠加），最多 MAX_ROUNDS 轮
  3. 最后一条消息输出 {"params": {...}, "rationale": "..."}
  4. 本脚本用「原图 + 最终参数」经 photocraft-cli run 确定性重渲染 → 成品 jpg(q95)，
     所以成品可复现，且不受智能体中途叠加/误操作影响

输出（与 run_a.py 一致）：runs/A/PhotoCraft_<模型>/<split>/<uuid>.jpg
旁路：runs/A/PhotoCraft_<模型>/_meta/<split>/<uuid>/{params.json, transcript.jsonl, audit.json, stderr.txt}

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

PROMPT = """You are a professional photo retoucher. Use the `photocraft` MCP tools to globally enhance ONE photo so it looks like a polished, natural edit by an expert (good exposure, pleasing white balance, balanced contrast and color). Keep it realistic; do not apply heavy stylization.

Workflow (follow exactly):
1. Call doc_open with path "{name}". Then call render_preview (max_side 768) to look at the original.
2. Decide a full set of RAW Develop settings and apply them in ONE call:
   command_run with id "filter.cameraRaw" and params = your settings.
3. After that command_run has returned (never in the same turn / never in parallel), call render_preview (max_side 768) to inspect the result.
4. If it needs changes: FIRST call command_run with id "history.undo" (so the filter is never stacked), then apply a complete new settings set with filter.cameraRaw, then preview again. At most {rounds} filter.cameraRaw attempts in total.
5. Stop when satisfied. Only use the commands filter.cameraRaw and history.undo. Do not use any other command (no selections, masks, auto or AI features).

filter.cameraRaw params (Adobe Camera Raw slider units, camelCase, omit = 0):
- exposure: -5..5 (stops)
- temperature, tint: -100..100 (relative white balance; + = warmer / more magenta)
- contrast, highlights, shadows, whites, blacks: -100..100
- texture, clarity, dehaze: -100..100
- vibrance, saturation: -100..100
- curveHighlights, curveLights, curveDarks, curveShadows: -100..100 (parametric tone curve)
- hslHue, hslSat, hslLum: arrays of 8 numbers in -100..100 for [red, orange, yellow, green, aqua, blue, purple, magenta]
- gradeShadows, gradeMidtones, gradeHighlights: objects {{"hue": 0..360, "sat": 0..100, "lum": -100..100}}
- vignetteAmount: -100..100
- sharpenAmount: 0..150

Your FINAL message must end with exactly one JSON object on its own line, containing the settings of your last applied filter.cameraRaw call:
{{"params": {{...}}, "rationale": "<one sentence>"}}
"""


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
    tag = f'PhotoCraft_{a.model}'
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
        (meta / 'params.json').write_text(json.dumps(objs[-1], ensure_ascii=False, indent=1))
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
    GATE = LimitGate(a.limit_wait, a.retry_wait, BASE / 'runs/A' / f'PhotoCraft_{a.model}' / f'_status_{a.split}.json')
    GATE.total = len(names)
    print(f'{a.agent}/{a.model} {a.split} n={len(names)}  状态文件 {GATE.status_file}', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(lambda n: run_one(a, n), names):
            print(json.dumps(res, ensure_ascii=False), flush=True)
    GATE.write('finished')


if __name__ == '__main__':
    main()
