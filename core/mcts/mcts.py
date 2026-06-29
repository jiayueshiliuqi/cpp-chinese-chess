"""
批量化 PUCT MCTS。
search_batch() 同时对 M 个局面跑搜索，每轮模拟把 M 个叶子凑成一个
batch 做一次前向，GPU 利用率显著提升。
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
import torch

from core.game.board_wrapper import XiangqiBoard
from core.game.encoder import encode_fen, flip_uci
from core.game.action_space import (
    NUM_ACTIONS, IDX_TO_UCI, UCI_TO_IDX, legal_action_mask,
)


class Node:
    __slots__ = ("P", "N", "W", "children", "is_expanded")

    def __init__(self, prior: float = 0.0):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.children: Dict[int, "Node"] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, model, device: str = "cuda",
                 num_simulations: int = 200, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps

    # ---------- 批量 NN 评估 ------
    @torch.inference_mode()
    def _evaluate_batch(self, boards: List[XiangqiBoard]):
        """对一批局面前向，返回 (priors_list, values[np.ndarray])。
        masked-softmax 在 GPU 上一次做完，数学等价于原来的
        softmax(全部) * mask / sum(legal)，已对拍验证 Δ=0。"""
        self.model.eval()
        xs = np.stack([encode_fen(b.fen()) for b in boards]).astype(np.float32)

        if str(self.device).startswith("cuda"):
            x_t = torch.from_numpy(xs).pin_memory().to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_t, value_t = self.model(x_t)        # forward → logits
        else:
            x_t = torch.from_numpy(xs).to(self.device)
            logits_t, value_t = self.model(x_t)

        logits_t = logits_t.float()
        B = len(boards)

        rows, cols = [], []
        for i, b in enumerate(boards):
            legal_real = b.legal_moves_uci()
            legal_view = legal_real if b.turn else [flip_uci(u) for u in legal_real]
            for u in legal_view:
                j = UCI_TO_IDX.get(u)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        mask = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=self.device)
        if rows:
            mask[rows, cols] = True

        neg = torch.finfo(logits_t.dtype).min
        masked = torch.where(mask, logits_t, torch.full_like(logits_t, neg))
        probs_t = torch.softmax(masked, dim=-1)

        probs = probs_t.cpu().numpy()
        values = value_t.float().cpu().numpy().reshape(-1)

        priors_list = []
        for i in range(B):
            p = probs[i]
            s = p.sum()
            if s < 1e-8 or not np.isfinite(s):
                m = mask[i].float().cpu().numpy()
                ms = m.sum()
                p = m / (ms if ms > 0 else 1.0)
            priors_list.append(p.astype(np.float32))
        return priors_list, values

    # --------- 选择 / 展开 / 回传 ----------
    def _select(self, node: Node) -> int:
        # 父节点的总访问数 = 所有子节点访问数之和 = node.N，
        # 直接用 node.N 省掉每次 sum 遍历。
        # +1 保证根节点首次选择时探索项 U 不为 0。
        sqrt_total = math.sqrt(node.N + 1)
        best_a, best_score = -1, -1e9
        for a, child in node.children.items():
            # child.Q 是“子节点走子方(对手)视角”，父节点要对手最差，故用 -child.Q
            u = self.c_puct * child.P * sqrt_total / (1 + child.N)
            score = -child.Q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _expand(self, node: Node, priors: np.ndarray) -> None:
        # 象棋每局面合法着只有 30~40 个，priors 里绝大多数是 0，
        # 只遍历非零项，避免每次扫全长 2238 的数组。
        for a in np.nonzero(priors)[0]:
            node.children[int(a)] = Node(prior=float(priors[a]))
        node.is_expanded = True

    @staticmethod
    def _backup(path: List[Node], leaf_value: float) -> None:
        """leaf_value 是 path[-1] 走子方视角，向上每层翻转符号。"""
        v = leaf_value
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v

    @staticmethod
    def _terminal_value(res, mover_is_red,
                        draw_value=0.0, max_moves_draw_value=0.0,
                        repetition_draw_value=0.0):
        if res.winner is None:
            r = str(res.reason)
            if "max_moves" in r.lower():
                return float(max_moves_draw_value)
            if "REPETITION" in r.upper():
                return float(repetition_draw_value)
            return float(draw_value)
        return 1.0 if res.winner == mover_is_red else -1.0

    @staticmethod
    def _undo(board: XiangqiBoard, n: int) -> None:
        for _ in range(n):
            board.pop()

    def _apply_root_noise(self, priors: np.ndarray) -> np.ndarray:
        legal_idx = np.where(priors > 0)[0]
        if len(legal_idx) == 0:
            return priors
        noise = np.random.dirichlet([self.alpha] * len(legal_idx))
        priors = priors.copy()
        priors[legal_idx] = (1 - self.eps) * priors[legal_idx] + self.eps * noise
        return priors

    # ---------- 批量搜索（核心） ------
    def search_batch(
            self,
            boards: List[XiangqiBoard],
            add_root_noise: bool = True,
            max_moves: int | None = None,
            draw_value: float = 0.0,
            max_moves_draw_value: float = 0.0,
            repetition_draw_value: float = 0.0
    ) -> List[np.ndarray]:
        n = len(boards)
        roots = [Node() for _ in range(n)]

        # 1) 初始：一次 batch 评估并展开所有 root
        priors_list, _ = self._evaluate_batch(boards)
        for i in range(n):
            priors = priors_list[i]
            if add_root_noise:
                priors = self._apply_root_noise(priors)
            self._expand(roots[i], priors)

        # 2) num_simulations 轮，每轮每局选一个叶子，凑 batch 一起评估
        for _ in range(self.num_simulations):
            pending_boards: List[XiangqiBoard] = []
            pending = []  # (game_idx, path, n_push)

            for i in range(n):
                board = boards[i]
                node = roots[i]
                path = [node]
                n_push = 0

                # 沿 PUCT 一直下降到「未展开节点」。途中经过的都是已展开节点，
                # 已展开 ⟹ 非终局，所以下降途中【不调用 result()】。
                while node.is_expanded and node.children:
                    a = self._select(node)
                    uci_view = IDX_TO_UCI[a]
                    uci_real = uci_view if board.turn else flip_uci(uci_view)
                    board.push_uci(uci_real)
                    n_push += 1
                    node = node.children[a]
                    path.append(node)

                # 到这里 node 是叶子（未展开，或无子节点）。只在此处判一次终局。
                res = board.result(max_moves=max_moves)
                if res is not None:
                    v = self._terminal_value(
                        res,
                        mover_is_red=board.turn,
                        draw_value=draw_value,
                        max_moves_draw_value=max_moves_draw_value,
                        repetition_draw_value=repetition_draw_value
                    )
                    self._backup(path, v)
                    self._undo(board, n_push)
                    continue

                # 非终局叶子：记下来批量评估
                pending_boards.append(board)
                pending.append((i, path, n_push))

            if pending_boards:
                priors_list, values = self._evaluate_batch(pending_boards)
                for k, (i, path, n_push) in enumerate(pending):
                    self._expand(path[-1], priors_list[k])
                    self._backup(path, float(values[k]))
                    self._undo(boards[i], n_push)

        # 3) 收集每局的访问分布
        out = []
        for i in range(n):
            visits = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for a, child in roots[i].children.items():
                visits[a] = child.N
            out.append(visits)
        return out

    # ---------- 单局接口（think()/对弈脚本用，内部走批量） ----------
    def search(
            self,
            board: XiangqiBoard,
            add_root_noise: bool = True,
            max_moves: int | None = None,
            draw_value: float = 0.0,
            max_moves_draw_value: float = 0.0,
            repetition_draw_value: float = 0.0,
    ) -> np.ndarray:
        return self.search_batch(
            [board],
            add_root_noise=add_root_noise,
            max_moves=max_moves,
            draw_value=draw_value,
            max_moves_draw_value=max_moves_draw_value,
            repetition_draw_value=repetition_draw_value,
        )[0]

    @staticmethod
    def visits_to_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
        total = float(visits.sum())
        if total <= 0 or not np.isfinite(total):
            # 异常访问数：返回全 0，由调用方负责兜底（不要在这里瞎猜动作）
            return np.zeros_like(visits, dtype=np.float32)
        if temperature <= 1e-2:
            policy = np.zeros_like(visits, dtype=np.float32)
            policy[int(visits.argmax())] = 1.0
            return policy
        v = visits.astype(np.float64) / total
        v = v ** (1.0 / temperature)
        s = v.sum()
        if s <= 0 or not np.isfinite(s):
            return (visits.astype(np.float64) / total).astype(np.float32)
        return (v / s).astype(np.float32)