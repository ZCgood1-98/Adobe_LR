"""PhotoCraft MCP 冒烟测试：不经过模型，手动走一遍 JSON-RPC，确认智能体闭环要用的工具都能用。

用法：python mcp_smoke.py <输入目录> <输出目录> <图片文件名>
"""
import json
import subprocess
import sys

IN, OUT, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
p = subprocess.Popen(
    ["photocraft-cli", "mcp", "--automation-read-root", IN, "--automation-write-root", OUT],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
_id = 0


def call(method, params=None):
    global _id
    _id += 1
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": _id, "method": method, "params": params or {}}) + "\n")
    p.stdin.flush()
    while True:
        r = json.loads(p.stdout.readline())
        if r.get("id") == _id:
            return r


def tool(name, args):
    r = call("tools/call", {"name": name, "arguments": args})
    res = r.get("result", {})
    # 文本截断显示，图片只显示类型与 base64 长度
    brief = [(c["type"], c.get("text", "")[:300] if c["type"] == "text" else (c.get("mimeType"), len(c.get("data", ""))))
             for c in res.get("content", [])]
    print(f"[{name}] isError={res.get('isError')} {brief if res else r.get('error')}")
    return res


call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "0"}})
p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
p.stdin.flush()
tools = call("tools/list")["result"]["tools"]
print("tools:", [t["name"] for t in tools])

tool("doc_open", {"path": NAME})
tool("command_run", {"id": "filter.cameraRaw", "params": {"exposure": 1.0}})
tool("render_preview", {"max_side": 1024})
tool("command_run", {"id": "history.undo"})
tool("doc_inspect", {})
tool("doc_export", {"path": "mcp_smoke_out.png"})
p.terminate()
