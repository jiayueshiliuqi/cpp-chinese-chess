# tests/test_outcome_parity.py
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cchess, xqcpp


def fast_outcome(b):
    """与 cchess.Board.outcome() 等价，但用 C++ 一次生成 + in_check。
    返回 (is_over, winner, reason)。检查顺序严格复刻 cchess.outcome()。"""
    moves, in_check = xqcpp.gen_legal_and_in_check(
        b.pawns, b.rooks, b.knights, b.bishops, b.advisors, b.kings, b.cannons,
        b.occupied_co[cchess.RED], b.occupied_co[cchess.BLACK], b.turn)
    T = cchess.Termination
    if in_check and not moves:
        return (True, (not b.turn), str(T.CHECKMATE))
    if b.is_insufficient_material():
        return (True, None, str(T.INSUFFICIENT_MATERIAL))
    if not moves:
        return (True, (not b.turn), str(T.STALEMATE))
    if b.is_perpetual_check():
        return (True, b.turn, str(T.PERPETUAL_CHECK))
    if b.is_fourfold_repetition():
        return (True, None, str(T.FOURFOLD_REPETITION))
    if b.is_sixty_moves():
        return (True, None, str(T.SIXTY_MOVES))
    return (False, None, None)


def ref_outcome(b):
    o = b.outcome()
    if o is None:
        return (False, None, None)
    return (True, o.winner, str(o.termination))


rng = random.Random(0)
mism = 0
checked = 0
for _ in range(5000):
    b = cchess.Board()
    for _ in range(rng.randint(0, 120)):  # 走深一点，触发重复/长将/六十回合
        exp = ref_outcome(b)
        got = fast_outcome(b)
        checked += 1
        if exp != got:
            mism += 1
            if mism <= 10:
                print("❌", b.fen())
                print("   ref :", exp)
                print("   fast:", got)
        if exp[0]:
            break
        lm = list(b.legal_moves)
        if not lm:
            break
        b.push(rng.choice(lm))
print(f"终局判定不一致: {mism} / {checked}")
assert mism == 0, "fast_outcome 与库不一致，先别接入"

# 微基准：result/outcome 这条路
def make_boards(n, max_depth=80, seed=1):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        b = cchess.Board()
        for _ in range(r.randint(0, max_depth)):
            lm = list(b.legal_moves)
            if not lm or b.outcome() is not None:
                break
            b.push(r.choice(lm))
        out.append(b)
    return out

boards = make_boards(3000)

def bench(fn, label, repeat=5):
    best = 1e9
    for _ in range(repeat):
        t = time.perf_counter()
        for b in boards:
            fn(b)
        best = min(best, time.perf_counter() - t)
    print(f"  {label:<8} {best*1000:8.2f} ms  ({best/len(boards)*1e6:.1f} us/局)")
    return best

print("微基准（3000 局面 outcome 判定）:")
tr = bench(ref_outcome,  "python")
tf = bench(fast_outcome, "c++")
print(f"加速比: {tr/tf:.1f}x")