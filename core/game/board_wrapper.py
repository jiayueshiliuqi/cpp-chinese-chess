"""
对 cchess 库的薄封装。所有上层模块只通过这个类接触规则引擎。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import cchess

# ---- C++ 加速开关 ----
# 想完全退回纯 cchess：把 _USE_XQCPP 设为 False（或删掉 xqcpp.so）。
# 接口与产出完全不变，只是慢一些。
_USE_XQCPP = True
try:
    import xqcpp
    _HAS_XQCPP = True
    print('导入cpp成功')
except ImportError:
    print('导入cpp失败')
    _HAS_XQCPP = False
_ACCEL = _USE_XQCPP and _HAS_XQCPP


RED, BLACK = True, False


@dataclass
class GameResult:
    winner: Optional[bool]  # True=红胜, False=黑胜, None=和
    reason: str


class XiangqiBoard:
    """统一的中国象棋棋盘接口。"""

    def __init__(self, fen: Optional[str] = None):
        self._board = cchess.Board(fen) if fen else cchess.Board()
        self._legal_cache = None

    # ---------- 状态查询 ----------
    @property
    def turn(self) -> bool:
        return self._board.turn

    def fen(self) -> str:
        return self._board.fen()

    def legal_moves_uci(self) -> List[str]:
        if self._legal_cache is None:
            if _ACCEL:
                b = self._board
                self._legal_cache = xqcpp.legal_moves_uci(
                    b.pawns, b.rooks, b.knights, b.bishops,
                    b.advisors, b.kings, b.cannons,
                    b.occupied_co[cchess.RED], b.occupied_co[cchess.BLACK],
                    b.turn)
            else:
                self._legal_cache = [m.uci() for m in self._board.legal_moves]
        return self._legal_cache

    def is_legal_uci(self, uci: str) -> bool:
        try:
            return self._board.is_legal(cchess.Move.from_uci(uci))
        except Exception:
            return False

    # ---------- 走子 ----------
    def push_uci(self, uci: str) -> None:
        self._board.push(cchess.Move.from_uci(uci))
        self._legal_cache = None

    def pop(self) -> None:
        self._board.pop()
        self._legal_cache = None

    def copy(self) -> "XiangqiBoard":
        new = XiangqiBoard.__new__(XiangqiBoard)
        new._board = self._board.copy()
        new._legal_cache = None
        return new

    # ---------- 终局 ----------
    def _rule_outcome(self) -> Optional[GameResult]:
        """与 cchess.Board.outcome() 等价（不含 max_moves），返回 GameResult 或 None。"""
        b = self._board
        if _ACCEL:
            moves, in_check = xqcpp.gen_legal_and_in_check(
                b.pawns, b.rooks, b.knights, b.bishops,
                b.advisors, b.kings, b.cannons,
                b.occupied_co[cchess.RED], b.occupied_co[cchess.BLACK],
                b.turn)
            self._legal_cache = moves  # 顺手填缓存
            T = cchess.Termination
            if in_check and not moves:
                return GameResult(not b.turn, str(T.CHECKMATE))
            if b.is_insufficient_material():
                return GameResult(None, str(T.INSUFFICIENT_MATERIAL))
            if not moves:
                return GameResult(not b.turn, str(T.STALEMATE))
            if b.is_perpetual_check():
                return GameResult(b.turn, str(T.PERPETUAL_CHECK))
            if b.is_fourfold_repetition():
                return GameResult(None, str(T.FOURFOLD_REPETITION))
            if b.is_sixty_moves():
                return GameResult(None, str(T.SIXTY_MOVES))
            return None
        else:
            outcome = b.outcome()
            if outcome is None:
                return None
            return GameResult(outcome.winner, str(outcome.termination))

    def is_game_over(self, max_moves: Optional[int] = None) -> bool:
        if self._rule_outcome() is not None:
            return True
        if max_moves is not None and len(self._board.move_stack) >= max_moves:
            return True
        return False

    def result(self, max_moves: Optional[int] = None) -> Optional[GameResult]:
        res = self._rule_outcome()
        if res is not None:
            return res
        if max_moves is not None and len(self._board.move_stack) >= max_moves:
            return GameResult(winner=None, reason="max_moves_draw")
        return None

    def num_moves_played(self) -> int:
        return len(self._board.move_stack)

    # ---------- 调试 ----------
    def __repr__(self) -> str:
        return self._board.unicode(axes=True, axes_type=1)