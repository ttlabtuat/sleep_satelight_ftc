# ssft_pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """
    - input:  (B, T, D)
    - output: (B, T, D)
    """
    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.q = nn.Linear(depth, depth, bias=False)
        self.k = nn.Linear(depth, depth, bias=False)
        self.v = nn.Linear(depth, depth, bias=False)
        self.out = nn.Linear(depth, depth, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        logits = torch.matmul(q, k.transpose(-2, -1))  # (B, T, T)

        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)  # (B, T, D)
        out = self.out(out)
        return out

class SAHopBlock(nn.Module):
    """
    Self-Attention + Residual + Linear + BN + ReLU + Dropout + AvgPool1D(4)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_skip_proj: bool):
        super().__init__()
        self.attn = SimpleSelfAttention(in_dim)
        self.use_skip_proj = use_skip_proj
        if use_skip_proj:
            self.skip_proj = nn.Linear(in_dim, in_dim)
        else:
            self.skip_proj = None

        self.bn1 = nn.BatchNorm1d(in_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.proj = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.pool = nn.AvgPool1d(kernel_size=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C=in_dim)
        return: (B, T', C=out_dim)  # T' = floor(T/4)
        """
        if self.use_skip_proj and self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x

        # Self-attention
        y = self.attn(x)  # (B, T, C)

        # BN over channel dim (C)
        y_bn = self.bn1(y.permute(0, 2, 1)).permute(0, 2, 1)
        y_bn = self.dropout1(y_bn)

        # Residual add
        y = skip + y_bn

        # Dense to next_dim
        y = self.proj(y)  # (B, T, out_dim)
        y = self.bn2(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.relu(y)
        y = self.dropout2(y)

        y = self.pool(y.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T', out_dim)

        return y

class StreamBlock(nn.Module):
    """
    Shared block for both temporal and frequency streams.

    input:  (B, n_channels, L, 1)  # same shape convention as Keras (NCHW here)
    output: features after the final attention hop
    """
    def __init__(
        self,
        segment_length_div: int,  # 2 or 4 （time: 2, freq: 4）
        n_channels: int,
        fact_conv_size: int,
        n_fact_conv: int,
        D: int,
        dropout_rate: float,
        n_hop: int,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_fact_conv,
            kernel_size=(1, fact_conv_size),
            padding=(0, fact_conv_size // 2),
            bias=False,
        )

        self.depthwise = nn.Conv2d(
            in_channels=n_fact_conv,
            out_channels=n_fact_conv * D,
            kernel_size=(n_channels, 1),
            groups=n_fact_conv,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(n_fact_conv * D)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.pool2d = nn.AvgPool2d(kernel_size=(1, 4))

        # Self-Attention Hops
        blocks = []
        dim = n_fact_conv * D 
        for hop in range(n_hop):
            curr_dim = dim
            next_dim = n_fact_conv * D * (hop + 2)  
            use_skip_proj = hop > 0
            blocks.append(SAHopBlock(curr_dim, next_dim, dropout_rate, use_skip_proj))
            dim = next_dim
        self.hops = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_channels, L, 1)
        """
        x = self.conv(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # reshape before pooling
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, T, C)

        x = x.permute(0, 2, 1)       # (B, C, T)
        x = F.avg_pool1d(x, kernel_size=4)
        x = x.permute(0, 2, 1)       # (B, T/4, C)

        # Self-Attention hops
        for block in self.hops:
            x = block(x)

        return x


class SleepSatelightPT(nn.Module):
    """
    PyTorch version of SleepSatelight.
    The final dense layers produce a 5-class output.
    Use extract_feature() to obtain the 300-D intermediate representation.
    """
    def __init__(
        self,
        segment_length: int = 1500,
        n_channels: int = 1,
        fact_conv_size: int = 250,
        n_fact_conv: int = 16,
        D: int = 2,
        dropout_rate: float = 0.2,
        n_hop: int = 3,
    ):
        super().__init__()

        self.time_stream = StreamBlock(
            segment_length_div=2,
            n_channels=n_channels,
            fact_conv_size=fact_conv_size,
            n_fact_conv=n_fact_conv,
            D=D,
            dropout_rate=dropout_rate,
            n_hop=n_hop,
        )

        self.freq_stream = StreamBlock(
            segment_length_div=4,
            n_channels=n_channels,
            fact_conv_size=fact_conv_size,
            n_fact_conv=n_fact_conv,
            D=D,
            dropout_rate=dropout_rate,
            n_hop=n_hop,
        )

        self.fc1 = nn.LazyLinear(300)
        self.fc2 = nn.Linear(300, 5)

    def forward(self, input_t: torch.Tensor, input_f: torch.Tensor) -> torch.Tensor:
        """
        input_t: (B, n_channels, L_t, 1)  # Kerasの input_t と同形
        input_f: (B, n_channels, L_f, 1)
        """
        ft = self.time_stream(input_t)   # (B, Tt, Ct)
        ff = self.freq_stream(input_f)   # (B, Tf, Cf)

        # Flatten
        ft = ft.reshape(ft.size(0), -1)
        ff = ff.reshape(ff.size(0), -1)
        x = torch.cat([ft, ff], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def extract_feature(self, input_t, input_f):
        self.eval()
        with torch.no_grad():
            ft = self.time_stream(input_t)   # (B, Tt, Ct)
            ff = self.freq_stream(input_f)   # (B, Tf, Cf)
            # flatten as in forward
            ft = ft.reshape(ft.size(0), -1)
            ff = ff.reshape(ff.size(0), -1)
            x = torch.cat([ft, ff], dim=1)   # (B, full_dim)
            x = self.fc1(x) #(B, 300)
            return x




def build_model_pt(
    segment_length: int = 1500,
    n_channels: int = 1,
    fact_conv_size: int = 250,
    n_fact_conv: int = 16,
    D: int = 2,
    dropout_rate: float = 0.2,
    n_hop: int = 3,
) -> SleepSatelightPT:

    model = SleepSatelightPT(
        segment_length=segment_length,
        n_channels=n_channels,
        fact_conv_size=fact_conv_size,
        n_fact_conv=n_fact_conv,
        D=D,
        dropout_rate=dropout_rate,
        n_hop=n_hop,
    )
    return model
