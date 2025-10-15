# fkan.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_taylor_powers(x, order):
    """
    规范化幂 z^k / k!，z 已约束到 [-1,1]（或接近）
    返回 (B, In, order)
    """
    if order <= 0:
        return None
    B, In = x.shape
    k = torch.arange(order, device=x.device).view(1, 1, -1)  # 0..order-1
    fac = torch.lgamma(k + 1).exp()  # k!
    z_exp = x.unsqueeze(-1)  # (B, In, 1)
    P = (z_exp ** k) / fac  # (B, In, order)
    return P


def jacobi_polynomials_stable(x, degree, a=1.0, b=1.0):
    """
    简化稳定递推 + 阶别缩放（1/sqrt(n+1)）抑制高阶振幅
    返回 (B, In, degree+1)
    """
    if degree < 0:
        return None
    B, In = x.shape
    J = torch.zeros(B, In, degree + 1, device=x.device, dtype=x.dtype)
    J[:, :, 0] = 1.0
    if degree >= 1:
        J[:, :, 1] = 0.5 * ((2 * (a + 1)) * x + (a - b))
        J[:, :, 1] = J[:, :, 1] / math.sqrt(2.0)
    result = J.clone()

    for n in range(2, degree + 1):
        k = n - 1
        # 参考标准递推，做了数值夹紧与缩放
        A1 = 2 * k + a + b
        A2 = 2 * (k + 1) * (k + a + b + 1) * (A1 + 1)
        A3 = (A1 + 1) * ((A1 + 2) * A1 * x + a * a - b * b)
        A4 = 2 * (k + a) * (k + b) * (A1 + 2)
        Jnm1 = result[:, :, n - 1]
        Jnm2 = result[:, :, n - 2]
        Jn = ((A3 / A2) * Jnm1 - (A4 / A2) * Jnm2)
        # 使用索引更新而非原地操作
        normalized_Jn = Jn / math.sqrt(n + 1.0)
        result = torch.cat([
            result[:, :, :n],
            normalized_Jn.unsqueeze(-1),
            result[:, :, n+1:]
        ], dim=-1)
    return result


def compute_chebyshev_T(x: torch.Tensor, degree: int, project: bool = False, norm: str = "deg"):
    """
    Chebyshev T_n 基，返回 (B, In, degree+1)
    norm: "none" | "deg"  (deg: 每阶除以 sqrt(n+1) 抑制高阶振幅)
    """
    if degree < 0:
        return None
    if project:
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        x = x.clamp(min=-1e6, max=1e6)

    T0 = torch.ones_like(x)
    if degree == 0:
        T = torch.stack([T0], dim=-1)
    else:
        T1 = x
        terms = [T0, T1]
        for k in range(1, degree):
            T_next = 2.0 * x * terms[-1] - terms[-2]
            terms.append(T_next)
        T = torch.stack(terms, dim=-1)  # (B, In, degree+1)

    if norm == "deg":
        n = torch.arange(0, degree + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        T = T / (n + 1.0).sqrt()  # 每阶÷√(n+1)
    return T


def compute_fourier_feats(x: torch.Tensor, num_frequencies: int, normalize: bool = True):
    """
    Fourier 基：cos(kx), sin(kx), k=1..K；返回 (B, In, 2K)
    normalize=True 时统一除以 sqrt(2K) 使各维总体能量一致
    """
    if num_frequencies <= 0:
        return None
    k = torch.arange(1, num_frequencies + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
    xk = x.unsqueeze(-1) * k                      # (B, In, K)
    Phi = torch.cat([torch.cos(xk), torch.sin(xk)], dim=-1)  # (B, In, 2K)
    if normalize:
        Phi = Phi / (2.0 * num_frequencies) ** 0.5
    return Phi


def compute_wavelet_bases(x: torch.Tensor,
                          scales: torch.Tensor,
                          shifts: torch.Tensor,
                          wavelet_type: str = 'mexican_hat',
                          norm: str = "none"):
    """
    小波基：返回 (B,In,Cw)
    norm: "none" | "channels" | "batch"
      - channels: ÷√Cw，固定按通道数归一
      - batch:    按 batch 统计每通道标准差做标准化（更稳，但多一点开销）
    """
    B, In = x.shape
    In2, Cw = scales.shape
    assert In == In2, "wavelet scales/shifts shape mismatch"
    a = F.softplus(scales) + 1e-6
    b = shifts
    u = (x.unsqueeze(-1) - b.unsqueeze(0)) / a.unsqueeze(0)  # (B,In,Cw)

    if wavelet_type == 'mexican_hat':
        Phi = (u ** 2 - 1.0) * torch.exp(-0.5 * u ** 2)
    elif wavelet_type == 'morlet':
        omega0 = 5.0
        Phi = torch.exp(-0.5 * u ** 2) * torch.cos(omega0 * u)
    elif wavelet_type == 'dog':
        Phi = -u * torch.exp(-0.5 * u ** 2)
    else:
        raise ValueError(f"Unsupported wavelet_type: {wavelet_type}")

    if norm == "channels":
        Phi = Phi / (float(Cw) ** 0.5)
    elif norm == "batch":
        # per-channel标准化：对(B,In)两维求均值/方差
        mu = Phi.mean(dim=(0,1), keepdim=True)
        sigma = Phi.std(dim=(0,1), keepdim=True).clamp_min(1e-6)
        Phi = (Phi - mu) / sigma
    return Phi.contiguous()



def compute_b_splines(x: torch.Tensor, grid_size: int, order: int = 3):
    """
    计算 KAN 用的一元 B-样条基函数（Cox–de Boor 递推）。
    仅接受 grid_size(int) 与 order，结点在每个维度的 [min, max] 上均匀生成并 clamped。

    参数：
        x:         (B, In)
        grid_size: 区间数 G（>0）
        order:     样条阶数 p（>=0），如 3 表示三次样条
    返回：
        bases: (B, In, G + p)  —— 每个维度对应的样条基函数值
    """
    assert x.dim() == 2, "x 必须是形状 (B, In)"
    B, In = x.shape
    G, p = int(grid_size), int(order)
    assert G > 0 and p >= 0, "grid_size>0 且 order>=0"

    device, dtype = x.device, x.dtype

    # 1) 为每个输入维生成 clamped 均匀结点：长度 = G + 2p + 1
    xmin = x.min(dim=0).values
    xmax = x.max(dim=0).values
    span = xmax - xmin
    # 避免零跨度导致退化
    pad_mask = span < 1e-8
    xmin = torch.where(pad_mask, xmin - 0.5, xmin)
    xmax = torch.where(pad_mask, xmax + 0.5, xmax)

    n_knots = G + 2 * p + 1
    grid = x.new_empty((In, n_knots))
    for i in range(In):
        internal = torch.linspace(xmin[i], xmax[i], steps=G + 1, device=device, dtype=dtype)
        grid[i] = torch.cat([internal[:1].repeat(p), internal, internal[-1:].repeat(p)], dim=0)  # clamped

    # 2) 零阶基函数 N_{i,0}(x)
    x_exp = x.unsqueeze(-1)  # (B, In, 1)
    bases = ((x_exp >= grid[:, :-1]) & (x_exp < grid[:, 1:])).to(dtype)  # (B, In, G + 2p)

    # 边界修正：x 恰好等于最后结点 -> 归入最后一个基函数
    last_knot = grid[:, -1]  # (In,)
    mask_last = (x == last_knot.unsqueeze(0)).unsqueeze(-1)  # (B, In, 1)
    if mask_last.any():
        bases = bases.clone()
        bases[mask_last.expand_as(bases)] = 0.0
        bases[..., -1] = torch.where(mask_last.squeeze(-1), torch.ones_like(bases[..., -1]), bases[..., -1])

    # 3) Cox–de Boor 递推：从 1 到 p
    for r in range(1, p + 1):
        # 左项
        left_num = x_exp - grid[:, :-(r + 1)]                              # (B, In, G + 2p - r)
        left_den = (grid[:, r:-1] - grid[:, :-(r + 1)]).clamp_min(1e-12)   # (In, G + 2p - r)
        left = (left_num / left_den.unsqueeze(0)) * bases[..., :-1]

        # 右项
        right_num = grid[:, r + 1:] - x_exp
        right_den = (grid[:, r + 1:] - grid[:, 1:(-r)]).clamp_min(1e-12)
        right = (right_num / right_den.unsqueeze(0)) * bases[..., 1:]

        bases = left + right  # (B, In, G + 2p - r)

    # 4) 输出 (B, In, G + p)
    return bases.contiguous()


class FKANLinear(nn.Module):
    """
    Fusion KAN Linear with selectable function families.
    支持: 'bspline' (默认), 'taylor', 'jacobi', 'cheby', 'fourier', 'wavelet'
    """
    def __init__(
        self,
        in_features,
        out_features,
        # 选择要启用的函数族
        families=("bspline", "taylor", "jacobi", "cheby", "fourier", "wavelet"),
        # B-spline
        bs_grid=8, bs_order=3,
        # Taylor
        taylor_order=4,
        # Jacobi
        jacobi_degree=4, jacobi_a=1.0, jacobi_b=1.0,
        # Chebyshev
        cheby_degree=4, cheby_project=True,
        # Fourier
        fourier_frequencies=8,
        # Wavelet
        wavelet_type="mexican_hat", wavelet_channels=4,
        # 融合方式
        mix_mode="static",        # 'static' 或 'input'
        mix_temperature=2.0,
        # 其他
        use_bias=True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # families 规范化并记下开关
        fams = tuple(f.lower() for f in families)
        assert len(fams) > 0, "至少选择一个函数族"
        self.use_bspline = "bspline" in fams
        self.use_taylor  = "taylor"  in fams and taylor_order > 0
        self.use_jacobi  = "jacobi"  in fams and jacobi_degree >= 0
        self.use_cheby   = "cheby"   in fams and cheby_degree  >= 0
        self.use_fourier = "fourier" in fams and fourier_frequencies > 0
        self.use_wavelet = "wavelet" in fams and wavelet_channels   > 0

        # 保存各自超参
        self.bs_grid, self.bs_order = bs_grid, bs_order
        self.taylor_order = taylor_order
        self.jacobi_degree, self.jacobi_a, self.jacobi_b = jacobi_degree, jacobi_a, jacobi_b
        self.cheby_degree, self.cheby_project = cheby_degree, cheby_project
        self.fourier_frequencies = fourier_frequencies
        self.wavelet_type, self.wavelet_channels = wavelet_type, wavelet_channels

        # Base 线性
        self.base = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=use_bias))
        nn.init.kaiming_uniform_(self.base.weight_v, a=math.sqrt(5))
        if use_bias:
            nn.init.zeros_(self.base.bias)

        # === 为启用的每个 family 准备系数 & 增益 ===
        fam_count = 0
        if self.use_bspline:
            self.bspline_coef = nn.Parameter(torch.zeros(out_features, in_features, bs_grid + bs_order))
            self.gain_bspline = nn.Parameter(torch.tensor(-10.0))   # softplus≈0 起步
            fam_count += 1

        if self.use_taylor:
            self.taylor_coef = nn.Parameter(torch.zeros(out_features, in_features, taylor_order))
            self.gain_taylor = nn.Parameter(torch.tensor(-10.0))
            fam_count += 1

        if self.use_jacobi:
            self.jacobi_coef = nn.Parameter(torch.zeros(out_features, in_features, jacobi_degree + 1))
            self.gain_jacobi = nn.Parameter(torch.tensor(-10.0))
            fam_count += 1

        if self.use_cheby:
            self.cheby_coef = nn.Parameter(torch.zeros(out_features, in_features, cheby_degree + 1))
            self.gain_cheby = nn.Parameter(torch.tensor(-10.0))
            fam_count += 1

        if self.use_fourier:
            # 拼成 2K 通道：cos/sin
            self.fourier_coef = nn.Parameter(torch.zeros(out_features, in_features, 2 * fourier_frequencies))
            self.gain_fourier = nn.Parameter(torch.tensor(-10.0))
            fam_count += 1

        if self.use_wavelet:
            # 每维多通道基础形状；系数按 (Out,In,Cw)
            self.wavelet_coef = nn.Parameter(torch.zeros(out_features, in_features, wavelet_channels))
            # 每维多通道的可学习尺度/平移（不依赖 Out）
            self.wavelet_scale_logit = nn.Parameter(torch.zeros(in_features, wavelet_channels))
            self.wavelet_shift       = nn.Parameter(torch.zeros(in_features, wavelet_channels))
            self.gain_wavelet = nn.Parameter(torch.tensor(-10.0))
            fam_count += 1

        # 非线性总增益 beta（per-layer），初始≈0；Base vs Nonlinear 门控 alpha（per-layer），初始≈0.95
        self.beta_logit  = nn.Parameter(torch.tensor(-10.0))
        self.alpha_logit = nn.Parameter(torch.tensor(2.94))

        # 融合权重 π（Out, In, F）
        self.mix_mode = mix_mode
        self.mix_temperature = mix_temperature
        self.mix_logits = nn.Parameter(torch.zeros(out_features, in_features, fam_count))
        if mix_mode == "input":
            self.mix_ax = nn.Parameter(torch.zeros(out_features, in_features, fam_count))
            self.mix_bx = nn.Parameter(torch.zeros(out_features, in_features, fam_count))

    def _softplus(self, x):  # ≥0 增益
        return F.softplus(x, beta=1.0)

    def _mix_weights(self, x):
        temp = max(self.mix_temperature, 1e-6)
        if self.mix_mode == "static":
            return F.softmax(self.mix_logits / temp, dim=-1)
        else:
            xi = x.unsqueeze(0).unsqueeze(-1)  # (1,B,In,1)
            logits = self.mix_logits.unsqueeze(1) + self.mix_ax.unsqueeze(1) * xi + self.mix_bx.unsqueeze(1)
            return F.softmax(logits / temp, dim=-1).mean(dim=1)  # (Out,In,F)

    def forward(self, x):
        orig = x
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))

        base_out = self.base(x)

        fam_list, fam_coefs, fam_gains = [], [], []

        # B-spline
        if self.use_bspline:
            B = compute_b_splines(x, self.bs_grid, self.bs_order)                    # (B,In,Cb)
            fam_list.append(B); fam_coefs.append(self.bspline_coef); fam_gains.append(self._softplus(self.gain_bspline))

        # Taylor
        if self.use_taylor:
            T = compute_taylor_powers(x, self.taylor_order)                           # (B,In,Ct)
            fam_list.append(T); fam_coefs.append(self.taylor_coef); fam_gains.append(self._softplus(self.gain_taylor))

        # Jacobi
        if self.use_jacobi:
            J = jacobi_polynomials_stable(x, self.jacobi_degree, self.jacobi_a, self.jacobi_b)  # (B,In,Cj)
            fam_list.append(J); fam_coefs.append(self.jacobi_coef); fam_gains.append(self._softplus(self.gain_jacobi))

        # Chebyshev
        if self.use_cheby:
            Cb = compute_chebyshev_T(x, self.cheby_degree, project=True)             # (B,In,Cc)
            fam_list.append(Cb); fam_coefs.append(self.cheby_coef); fam_gains.append(self._softplus(self.gain_cheby))

        # Fourier
        if self.use_fourier:
            Fz = compute_fourier_feats(x, self.fourier_frequencies)                  # (B,In,2K)
            fam_list.append(Fz); fam_coefs.append(self.fourier_coef); fam_gains.append(self._softplus(self.gain_fourier))

        # Wavelet
        if self.use_wavelet:
            W = compute_wavelet_bases(x, self.wavelet_scale_logit, self.wavelet_shift, self.wavelet_type)  # (B,In,Cw)
            fam_list.append(W); fam_coefs.append(self.wavelet_coef); fam_gains.append(self._softplus(self.gain_wavelet))

        # 逐 family 边贡献：(B, Out, In)
        contribs = []
        for Phi, coef, g in zip(fam_list, fam_coefs, fam_gains):
            c = torch.einsum("bic,oic->boi", Phi, coef) * g
            contribs.append(c)

        # 融合权重 π（Out, In, F）
        pi = self._mix_weights(x)
        Cstack = torch.stack(contribs, dim=-1)                   # (B,Out,In,F)
        fused_edge = (Cstack * pi.unsqueeze(0)).sum(dim=-1)      # (B,Out,In)

        nonlinear = fused_edge.sum(dim=-1)                       # (B,Out)
        beta = torch.sigmoid(self.beta_logit)
        alpha = torch.sigmoid(self.alpha_logit)
        y = alpha * base_out + (1 - alpha) * (beta * nonlinear)
        return y.view(*orig.shape[:-1], self.out_features)


# ========== 简单封装成网络 ==========
class FKAN(nn.Module):
    def __init__(self, layers_hidden, **configs):
        super().__init__()
        self.layers = nn.ModuleList([
            FKANLinear(d_in, d_out, **configs)
            for d_in, d_out in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def test_cf_kan():
    torch.autograd.set_detect_anomaly(True)
    layers_hidden = [784, 256, 128, 10]
    model = FKAN(
        layers_hidden=layers_hidden,
    )
    print("F-KAN 模型创建成功!")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

    batch_size = 32
    x = torch.randn(batch_size, layers_hidden[0])
    print(f"输入数据形状: {x.shape}")

    model.train()
    output = model(x)
    print(f"输出数据形状: {output.shape}")

    target = torch.randint(0, 10, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"损失值: {loss.item():.4f}")
    loss.backward()
    print("反向传播执行成功!")


if __name__ == "__main__":
    test_cf_kan()