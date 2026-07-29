import random
from collections import Counter

SIZES = {"taste": 29896, "mech": 31044, "diag": 7429}

# ---------- 当前实现(阶段配比) ----------
def cur(sizes, total, seed=42):
    rng = random.Random(seed)
    pools = {c: list(range(n)) for c, n in sizes.items()}
    for p in pools.values(): rng.shuffle(p)
    ptr = {c: 0 for c in sizes}
    def draw(c):
        p = pools[c]
        if ptr[c] >= len(p): rng.shuffle(p); ptr[c] = 0
        i = p[ptr[c]]; ptr[c] += 1; return (c, i)
    seq = []
    for k in range(total):
        ph = k / total; r = rng.random()
        if ph < 0.20: c = "mech"
        elif ph < 0.90: c = "taste" if r < 0.44 else ("mech" if r < 0.88 else "diag")
        else: c = "taste" if r < 0.57 else ("mech" if r < 0.85 else "diag")
        seq.append(draw(c))
    return seq

# ---------- 新实现(span 位置,精确覆盖) ----------
def new(draws, sizes, seed=42, spans=None):
    rng = random.Random(seed)
    spans = spans or {"mech": (0.0, 0.55), "taste": (0.10, 1.0), "diag": (0.15, 1.0)}
    items = []
    for c, nd in draws.items():
        n = sizes[c]; lo, hi = spans.get(c, (0.0, 1.0))
        order = []
        while len(order) < nd:
            perm = list(range(n)); rng.shuffle(perm); order += perm
        for idx in order[:nd]:
            items.append((lo + (hi - lo) * rng.random(), c, idx))
    items.sort(key=lambda t: t[0])
    return [(c, i) for _, c, i in items]

def passes(seq, sizes):
    cnt = Counter(c for c, _ in seq)
    return {c: round(cnt[c] / sizes[c], 2) for c in sizes}, dict(cnt)

def deciles(seq):
    n = len(seq); print("    十分位课程占比(taste/mech/diag):")
    for d in range(10):
        chunk = seq[int(n*d/10):int(n*(d+1)/10)]
        cc = Counter(c for c, _ in chunk); tot = len(chunk)
        print(f"      [{d*10:>2}-{d*10+10:>3}%] "
              f"t={cc['taste']/tot*100:4.0f}%  m={cc['mech']/tot*100:4.0f}%  d={cc['diag']/tot*100:4.0f}%")

print("="*60)
print("当前默认 (epochs=1, total=sum(sizes)):")
s = cur(SIZES, sum(SIZES.values()))
p, c = passes(s, SIZES); print(f"  遍数 {p}   抽取 {c}")

print("="*60)
print("新版 span,目标 品味2/机制1/诊断2:")
draws = {"taste": 2*SIZES["taste"], "mech": 1*SIZES["mech"], "diag": 2*SIZES["diag"]}
s2 = new(draws, SIZES)
p2, c2 = passes(s2, SIZES); print(f"  遍数 {p2}   抽取 {c2}   总步 {len(s2)}")
deciles(s2)

print("="*60)
print("冒烟 (每课抽 min(24,size)),确认三课都走到:")
sm = new({c: min(24, n) for c, n in SIZES.items()}, SIZES)
print(f"  {passes(sm, SIZES)[1]}  共 {len(sm)} 条")
