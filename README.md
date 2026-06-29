# Cpp-Chinese-Chess
`Cpp-Chinese-Chess` 是一个中国象棋项目，基于 [Python-Chinese-Chess](https://github.com/windshadow233/python-chinese-chess) 进行改写与加速。
## 项目简介
本项目的主要目标是通过引入 C++ 扩展来提升中国象棋引擎的运行效率与执行速度。
目前项目并不是完全由 C++ 实现的：
- `xqcpp.cpp` 是核心的 C++ 加速代码
- `core/` 目录下主要是 Python 代码，用于封装棋盘接口、动作空间、编码、MCTS 和模型等逻辑
- `tests/` 目录下是测试代码，用于验证结果正确性以及比较优化前后的性能差异
也就是说，本项目目前是一个 **以 Python 为主体、使用 C++ 对关键路径进行加速** 的中国象棋库。
如果你在使用过程中发现任何错误、兼容性问题或实现上的不足，欢迎提出 issue 或反馈建议。
## 测试说明
`tests` 文件夹中的文件主要用于：
- 验证 C++ 实现与原始 Python 实现的结果是否一致
- 测试合法走法生成、终局判定等核心逻辑的正确性
- 对比优化前后的性能差异
当前测试脚本包括：
- `tests/test_xqcpp_parity.py`：测试合法走法生成结果是否一致，并比较性能
- `tests/test_outcome_parity.py`：测试终局判定结果是否一致，并比较性能
- `tests/test_mcts.py`：测试在不同规则后端下 MCTS 搜索结果是否一致，并比较整体速度
运行测试前，你可以先安装原版 Python 库，再执行对应测试脚本。
仓库中包含一份已编译好的 Python 3.11 对应扩展文件；如果你的 Python 版本或平台环境不同，请自行重新编译。

## 项目结构

```text
.
├── xqcpp.cpp          # C++ 加速核心
├── core/              # Python 主要逻辑
├── tests/             # 正确性与性能测试
└── README.md
```