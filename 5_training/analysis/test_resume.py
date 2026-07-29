#!/usr/bin/env python3
"""离线验证 d2 断点续跑:mock 掉 call_one(不打 API),测四条路径。"""
import sys, json
sys.path.insert(0, "5_training")
import v2_diagnosis_gen as G

OUT = G.REPO_ROOT / "sft_data_v2"
files = {n: OUT / n for n in ["diag_d2_checkpoint.jsonl", "diag_d2_checkpoint.sig",
                              "task_v2_diag_train.json", "diag_gen_report.json"]}
backup = {n: (p.read_bytes() if p.exists() else None) for n, p in files.items()}

calls = {"n": 0}
def fake_call(job):                       # 替代真实 API 调用
    calls["n"] += 1
    keys = ["exposure", "wb", "tint", "contrast", "highlights", "shadows",
            "saturation", "vignette", "dehaze"]
    j = {k: "keep" for k in keys}; j["bw"] = job["bw"]
    return {"vid": f"{job['key']}/v{job['vn']}", "ok": True, "passed": True,
            "prose": "MOCK PROSE", "json": j, "violations": [], "category": job["category"],
            "bw": job["bw"], "raw_len": 4, "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}
G.call_one = fake_call

jobs = G.get_jobs(0)[:50]
seed = jobs[:25]

def run(argv):
    calls["n"] = 0
    old = sys.argv
    sys.argv = ["v2_diagnosis_gen.py"] + argv
    try:
        G.main()
    finally:
        sys.argv = old

def ck_lines():
    p = files["diag_d2_checkpoint.jsonl"]
    return len([l for l in p.read_text().splitlines() if l.strip()]) if p.exists() else 0
def n_samples():
    p = files["task_v2_diag_train.json"]
    return len(json.loads(p.read_text())) if p.exists() else 0

ok = True
try:
    print("=== Test A: 部分续跑(种子 25/50,应只补 25) ===")
    files["diag_d2_checkpoint.jsonl"].write_text(
        "".join(json.dumps(fake_call(j), ensure_ascii=False) + "\n" for j in seed))
    files["diag_d2_checkpoint.sig"].write_text(G._run_sig())
    run(["d2", "--limit", "50", "--workers", "4"])
    print(f"  本次 mock 调用={calls['n']}(期望25) checkpoint={ck_lines()}(期望50) 样本={n_samples()}(期望50)")
    assert calls["n"] == 25 and ck_lines() == 50 and n_samples() == 50
    print("  ✓ 通过")

    print("=== Test B: 幂等重跑(已全完成,应 0 调用) ===")
    run(["d2", "--limit", "50", "--workers", "4"])
    print(f"  本次 mock 调用={calls['n']}(期望0) checkpoint={ck_lines()}(期望50) 样本={n_samples()}(期望50)")
    assert calls["n"] == 0 and ck_lines() == 50 and n_samples() == 50
    print("  ✓ 通过")

    print("=== Test C: 签名护栏(prompt变了,应拒绝续跑) ===")
    files["diag_d2_checkpoint.sig"].write_text("DEADBEEF00")
    hit = False
    try:
        run(["d2", "--limit", "50"])
    except SystemExit as e:
        hit = True
        print("  SystemExit:", str(e)[:70].replace("\n", " "))
    assert hit
    print("  ✓ 通过(正确拒绝)")

    print("=== Test D: --fresh 清 checkpoint 从头跑(应 50 调用) ===")
    run(["d2", "--limit", "50", "--workers", "4", "--fresh"])
    print(f"  本次 mock 调用={calls['n']}(期望50) checkpoint={ck_lines()}(期望50)")
    assert calls["n"] == 50 and ck_lines() == 50
    print("  ✓ 通过")
except AssertionError:
    ok = False
    import traceback; traceback.print_exc()
finally:
    for n, p in files.items():            # 恢复/删除测试产物
        if backup[n] is None:
            if p.exists():
                p.unlink()
        else:
            p.write_bytes(backup[n])
    print("清理完成:测试产物已删、原有文件已恢复")

print("\n" + ("全部通过 ✅" if ok else "有失败 ❌"))
sys.exit(0 if ok else 1)
