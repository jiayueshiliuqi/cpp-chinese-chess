import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(x + y, inplace=True)


class ChessNet(nn.Module):
    def __init__(self, in_channels: int = 15, channels: int = 256,
                 num_blocks: int = 20, num_actions: int = 2238):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(4 * 10 * 9, num_actions)

        # value head
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.v_fc = nn.Sequential(
            nn.Linear(8 * 10 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        h = self.res(self.stem(x))
        p = self.p_fc(self.p_conv(h).flatten(1))   # logits, (B, A)
        v = self.v_fc(self.v_conv(h).flatten(1)).squeeze(-1)  # (B,)
        return p, v

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """推理用：返回 softmax 后的概率与 value。"""
        self.eval()
        logits, v = self.forward(x)
        return F.softmax(logits, dim=-1), v

    @torch.no_grad()
    def predict_logits(self, x):
        self.eval()
        logits, v = self.forward(x)
        return logits, v  # 不做 softmax
