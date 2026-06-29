# tests/test_xqcpp_parity.py
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cchess, xqcpp

def legal_py(b):
    return [m.uci() for m in b.legal_moves]

def legal_cpp(b):
    return xqcpp.legal_moves_uci(
        b.pawns, b.rooks, b.knights, b.bishops, b.advisors, b.kings, b.cannons,
        b.occupied_co[cchess.RED], b.occupied_co[cchess.BLACK], b.turn)

def random_boards(n, max_depth=60, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = cchess.Board()
        for _ in range(rng.randint(0, max_depth)):
            lm = list(b.legal_moves)
            if not lm: break
            b.push(rng.choice(lm))
            if b.is_game_over(): break
        out.append(b)
    return out

boards = random_boards(3000)

# 1) 集合 + 顺序都要一致
mism = 0
for b in boards:
    py, cpp = legal_py(b), legal_cpp(b)
    if py != cpp:                       # 先比顺序
        if sorted(py) == sorted(cpp):
            print("⚠️ 顺序不同但集合相同:", b.fen())
        else:
            mism += 1
            if mism <= 5:
                print("❌ 集合不同:", b.fen())
                print("   py - cpp:", set(py)-set(cpp))
                print("   cpp - py:", set(cpp)-set(py))
print(f"集合不一致局面数: {mism} / {len(boards)}")
assert mism == 0, "C++ 生成器与库不一致，先别上线"

# 2) 微基准
def bench(fn, label, repeat=5):
    best=1e9
    for _ in range(repeat):
        t=time.perf_counter()
        for b in boards: fn(b)
        best=min(best, time.perf_counter()-t)
    print(f"  {label:<12} {best*1000:8.2f} ms  ({best/len(boards)*1e6:.1f} us/局)")
    return best
print("微基准（3000 局面全合法着）:")
tp = bench(legal_py,  "python")
tc = bench(legal_cpp, "c++")
print(f"加速比: {tp/tc:.1f}x")