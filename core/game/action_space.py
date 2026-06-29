"""
预生成全部"几何上可能"的走法，作为固定动作空间。
合法性在每一步用 board_wrapper 过滤（mask）。
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# 棋盘大小
ROWS, COLS = 10, 9

Move = Tuple[int, int, int, int]  # (r1, c1, r2, c2)


def _is_in_palace(r: int, c: int, red_side: bool) -> bool:
    if c not in (3, 4, 5):
        return False
    return (r in (0, 1, 2)) if red_side else (r in (7, 8, 9))


def generate_all_moves() -> List[Move]:
    moves = set()
    for r1 in range(ROWS):
        for c1 in range(COLS):
            for r2 in range(ROWS):
                for c2 in range(COLS):
                    if (r1, c1) == (r2, c2):
                        continue
                    dr, dc = r2 - r1, c2 - c1
                    adr, adc = abs(dr), abs(dc)

                    # 车/炮/将/兵：同行或同列
                    if r1 == r2 or c1 == c2:
                        moves.add((r1, c1, r2, c2))
                        continue
                    # 马：日字
                    if (adr, adc) in ((1, 2), (2, 1)):
                        moves.add((r1, c1, r2, c2))
                        continue
                    # 象：田字（不过河）
                    if (adr, adc) == (2, 2):
                        if (r1 < 5 and r2 < 5) or (r1 >= 5 and r2 >= 5):
                            moves.add((r1, c1, r2, c2))
                        continue
                    # 士：1 格斜线（在九宫内）
                    if (adr, adc) == (1, 1):
                        if _is_in_palace(r1, c1, red_side=True) and _is_in_palace(r2, c2, red_side=True):
                            moves.add((r1, c1, r2, c2))
                        elif _is_in_palace(r1, c1, red_side=False) and _is_in_palace(r2, c2, red_side=False):
                            moves.add((r1, c1, r2, c2))
                        continue
    return sorted(moves)


def move_to_uci(mv: Move) -> str:
    r1, c1, r2, c2 = mv
    return f"{chr(ord('a') + c1)}{r1}{chr(ord('a') + c2)}{r2}"


def uci_to_move(uci: str) -> Move:
    c1 = ord(uci[0]) - ord('a')
    r1 = int(uci[1])
    c2 = ord(uci[2]) - ord('a')
    r2 = int(uci[3])
    return (r1, c1, r2, c2)


# === 全局单例（启动时构建一次） ===
ALL_MOVES: List[Move] = generate_all_moves()
ALL_UCIS: List[str] = [move_to_uci(m) for m in ALL_MOVES]
NUM_ACTIONS: int = len(ALL_MOVES)
UCI_TO_IDX: Dict[str, int] = {u: i for i, u in enumerate(ALL_UCIS)}
IDX_TO_UCI: Dict[int, str] = {i: u for i, u in enumerate(ALL_UCIS)}


def legal_action_mask(legal_ucis: List[str]) -> np.ndarray:
    """返回 shape=(NUM_ACTIONS,) 的 0/1 mask。"""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for u in legal_ucis:
        idx = UCI_TO_IDX.get(u)
        if idx is not None:
            mask[idx] = 1.0
    return mask