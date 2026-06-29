"""
FEN ↔ 15通道张量。
关键：始终从"当前走子方"的视角编码。
  - 红方走时：直接编码
  - 黑方走时：棋盘旋转 180°，并交换红黑通道
这样网络只需要学一种视角，效果显著。
"""
from __future__ import annotations
import numpy as np

# 我方 7 通道 + 对方 7 通道 + 1 通道当前走子方常量
NUM_PLANES = 15
ROWS, COLS = 10, 9

# FEN 字符 → 通道索引（红方大写，黑方小写）
PIECE_TO_PLANE = {
    'K': 0, 'A': 1, 'B': 2, 'N': 3, 'R': 4, 'C': 5, 'P': 6,
    'k': 0, 'a': 1, 'b': 2, 'n': 3, 'r': 4, 'c': 5, 'p': 6,
}


def encode_fen(fen: str) -> np.ndarray:
    """FEN → (15, 10, 9) float32。只遍历实际棋子，不扫空格。"""
    parts = fen.split()
    board_str = parts[0]
    is_red_turn = (parts[1] == 'w')  # 'w'=红

    planes = np.zeros((NUM_PLANES, ROWS, COLS), dtype=np.float32)

    rows = board_str.split('/')
    assert len(rows) == ROWS, f"bad fen rows: {fen}"
    for i, row in enumerate(rows):
        rank = ROWS - 1 - i  # FEN 顶行是 rank 9
        col = 0
        for ch in row:
            if ch.isdigit():
                col += int(ch)
                continue
            piece_idx = PIECE_TO_PLANE[ch]
            is_red_piece = ch.isupper()
            # 我方 0..6 / 对方 7..13
            plane = piece_idx if (is_red_piece == is_red_turn) else piece_idx + 7
            # 黑方走时旋转坐标
            if is_red_turn:
                rr, cc = rank, col
            else:
                rr, cc = ROWS - 1 - rank, COLS - 1 - col
            planes[plane, rr, cc] = 1.0
            col += 1

    planes[14, :, :] = 1.0
    return planes


def flip_uci(uci: str) -> str:
    """把(从黑方视角输出的)走法翻转回真实棋盘坐标。"""
    c1 = ord(uci[0]) - ord('a')
    r1 = int(uci[1])
    c2 = ord(uci[2]) - ord('a')
    r2 = int(uci[3])
    r1, c1, r2, c2 = ROWS - 1 - r1, COLS - 1 - c1, ROWS - 1 - r2, COLS - 1 - c2
    return f"{chr(ord('a') + c1)}{r1}{chr(ord('a') + c2)}{r2}"