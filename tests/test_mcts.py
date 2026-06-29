"""
对拍并测速：同一套 MCTS，在两种规则引擎后端下对比。
  - 后端 A：纯 cchess（board_wrapper._ACCEL = False）—— 原版
  - 后端 B：C++ 加速 xqcpp（board_wrapper._ACCEL = True）—— 优化版

  对拍逻辑：同一个模型、同一批局面、add_root_noise=False（确定性可比），
  切换后端跑两遍，比较 visits 分布是否 bit 级一致。
  因为只换了底层规则引擎（合法着生成 + 终局判定），MCTS 算法本身没动，
  visits 必须完全相等；不相等说明 C++ 后端与 cchess 有语义差异。

用法：
  python tests/test_mcts.py
退出码：一致返回 0，不一致返回 1（可用于 CI）。
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
torch.set_num_threads(1)

from core.network.model import ChessNet
from core.game.action_space import NUM_ACTIONS
from core.mcts.mcts import MCTS
import core.game.board_wrapper as bw
from core.game.board_wrapper import XiangqiBoard

# ----------------- 可调参数 -----------------
PARALLEL_BOARDS = 16     # 同时搜索的局面数
NUM_SIMULATIONS = 100    # 每个局面的模拟次数（测试用，调小点）
MAX_MOVES = 200
TIMING_REPEATS = 3       # 测速重复次数，取最优
SEED = 1234
# --------------------------------------------


def set_backend(use_cpp: bool) -> bool:
    """切换 board_wrapper 的规则引擎后端。返回实际是否启用了 C++。"""
    bw._ACCEL = bool(use_cpp) and bw._HAS_XQCPP
    return bw._ACCEL


def load_config():
    try:
        import yaml
        cfg_path = ROOT / "config.yaml"
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] 读取 config.yaml 失败，用默认参数：{e}")
    return None


def build_model(cfg, device):
    if cfg and "network" in cfg:
        nc = cfg["network"]
        model = ChessNet(
            in_channels=nc.get("in_channels", 15),
            channels=nc.get("channels", 128),
            num_blocks=nc.get("num_blocks", 10),
            num_actions=nc.get("num_actions", NUM_ACTIONS),
        )
    else:
        model = ChessNet(in_channels=15, channels=128,
                         num_blocks=10, num_actions=NUM_ACTIONS)
    if cfg:
        ckpt = Path(cfg.get("pipeline", {}).get("checkpoint_dir", "checkpoints")) / "best_latest.pt"
        if ckpt.exists():
            try:
                blob = torch.load(ckpt, map_location=device, weights_only=False)
                state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
                model.load_state_dict(state)
                print(f"[info] 已加载权重 {ckpt}")
            except Exception as e:
                print(f"[warn] 加载权重失败，用随机权重：{e}")
    return model.to(device).eval()


def make_positions(n, max_random_plies=20):
    """从开局出发，各走若干随机步，生成多样化的测试局面。
    固定用纯 cchess 后端生成，保证无论 xqcpp 是否可用，局面都可复现。"""
    prev = bw._ACCEL
    set_backend(False)
    try:
        rng = np.random.default_rng(SEED)
        boards = []
        for _ in range(n):
            b = XiangqiBoard()
            plies = int(rng.integers(0, max_random_plies + 1))
            for _ in range(plies):
                legal = b.legal_moves_uci()
                if not legal or b.is_game_over(max_moves=MAX_MOVES):
                    break
                b.push_uci(legal[int(rng.integers(len(legal)))])
            boards.append(b)
        return boards
    finally:
        bw._ACCEL = prev


def clone(boards):
    return [b.copy() for b in boards]


def run_search(mcts, boards):
    # add_root_noise=False -> 搜索完全确定，可对拍
    return mcts.search_batch(
        boards, add_root_noise=False, max_moves=MAX_MOVES,
        draw_value=0.0, max_moves_draw_value=-0.1, repetition_draw_value=-0.05,
    )


def timeit(mcts, base_boards, use_cpp):
    best = float("inf")
    for _ in range(TIMING_REPEATS):
        set_backend(use_cpp)
        boards = clone(base_boards)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_search(mcts, boards)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def compare_visits(a_list, b_list):
    """返回 (是否完全一致, 最大绝对差, argmax 不一致的局面数)。"""
    exact = True
    max_abs = 0.0
    argmax_mismatch = 0
    for a, b in zip(a_list, b_list):
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        max_abs = max(max_abs, d)
        if not np.array_equal(a, b):
            exact = False
        if a.sum() > 0 and b.sum() > 0 and int(a.argmax()) != int(b.argmax()):
            argmax_mismatch += 1
    return exact, max_abs, argmax_mismatch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    print("=" * 64)
    print(f"device={device}  parallel_boards={PARALLEL_BOARDS}  "
          f"num_simulations={NUM_SIMULATIONS}")
    print(f"xqcpp 可用: {bw._HAS_XQCPP}")
    print("=" * 64)

    if not bw._HAS_XQCPP:
        print("❌ 未找到 xqcpp 模块，无法对比 C++ 后端。")
        print("   请先编译 xqcpp 并确保可 import，再运行本测试。")
        sys.exit(1)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = build_model(cfg, device)
    mcts = MCTS(model, device=device, num_simulations=NUM_SIMULATIONS)

    base_boards = make_positions(PARALLEL_BOARDS)

    # ---- 预热（CUDA kernel / cudnn autotune），不计入计时 ----
    print("预热中 ...")
    set_backend(False)
    run_search(mcts, clone(base_boards))
    set_backend(True)
    run_search(mcts, clone(base_boards))

    # ---- 一致性对拍（确定性搜索）----
    set_backend(False)
    v_py = run_search(mcts, clone(base_boards))
    set_backend(True)
    v_cpp = run_search(mcts, clone(base_boards))

    exact, md, am = compare_visits(v_py, v_cpp)

    # ---- 测速 ----
    t_py = timeit(mcts, base_boards, use_cpp=False)
    t_cpp = timeit(mcts, base_boards, use_cpp=True)

    def line(name, t, base):
        per_board = t / PARALLEL_BOARDS * 1000
        speed = base / t if t > 0 else float("nan")
        print(f"  {name:<24} {t*1000:8.1f} ms/批  "
              f"{per_board:7.2f} ms/局  加速比 {speed:5.2f}x")

    print("\n----------------- 速度 -----------------")
    line("原版 (纯 cchess)", t_py, t_py)
    line("C++ 加速 (xqcpp)", t_cpp, t_py)

    print("\n----------------- 一致性 -----------------")
    print(f"  C++ vs 原版:  完全一致={exact}  "
          f"最大visits差={md:.0f}  argmax不一致局面={am}")

    print("\n----------------- 结论 -----------------")
    ok = exact
    if exact:
        print("  ✅ C++ 后端与纯 cchess bit 级一致，可放心启用 _USE_XQCPP。")
    else:
        print("  ❌ C++ 后端与原版不一致！MCTS 产出被改变了。")
        print("     重点查 xqcpp 的合法着生成 / 终局判定（先跑 test_xqcpp_parity 和 test_outcome_parity）。")

    print(f"\n  整体 MCTS 加速: {t_py / t_cpp:.2f}x")
    print("  注: 此处加速包含 NN 前向等无法压缩的部分，所以会低于着法生成的单点 84x。")
    print("      看火焰图里 cchess 合法着生成 / outcome 占比应明显塌下去。")
    print("=" * 64)

    # 恢复默认（按 _USE_XQCPP 的意图）
    set_backend(bw._USE_XQCPP)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()