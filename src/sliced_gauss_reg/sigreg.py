"""SIGReg: Sketch Isotropic Gaussian Regularizer.

Implements the Epps-Pulley test statistic as a differentiable loss for
pushing embedding distributions toward N(0, I), with three variants for
how |φ_θ(t)|² is estimated from the empirical characteristic function.
"""

import torch


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!).

    The Epps-Pulley test statistic is a weighted L2 distance between the
    empirical characteristic function (ECF) and the standard normal CF.
    The plug-in estimator |φ̂(t)|² = (Re φ̂)² + (Im φ̂)² has a positive bias
    of (1 − |φ_θ(t)|²) / n that vanishes only as n → ∞. ``bias_mode``
    selects how |φ_θ(t)|² is estimated:

    - ``"biased"`` (default): plug-in estimator. Matches the original
      SIGReg behavior; introduces a 1/n positive offset to the loss.
    - ``"ustat"``: U-statistic debiasing,
      |φ_θ|² ≈ (n/(n-1))·|φ̂|² − 1/(n-1).
      Unbiased; differs from biased by an n/(n-1) rescaling on the
      off-diagonal pair sum.
    - ``"split"``: sample-split estimator,
      |φ_θ|² ≈ Re(φ̂_A · φ̂_B*) where A, B are disjoint halves of the
      batch. Unbiased; uses ~half the pairs of the U-stat estimator
      (higher variance) but each sample's gradient depends only on the
      *other* half. The linear cross-term −2 Re(φ̂)·φ_N is computed on
      the full batch (still unbiased, no data wasted).

    See experiments/synthetic/FAIRNESS.md for the broader experimental
    context. The bias is only meaningful at small n; with n ≳ 10³ the
    three variants are essentially identical.
    """

    def __init__(self, knots=17, num_proj=1024, bias_mode: str = "biased"):
        super().__init__()
        if bias_mode not in ("biased", "ustat", "split"):
            raise ValueError(
                f"bias_mode must be 'biased', 'ustat', or 'split'; got {bias_mode!r}")
        self.num_proj = num_proj
        self.bias_mode = bias_mode
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        n = proj.size(-2)

        if self.bias_mode == "split":
            if n < 2:
                raise ValueError(f"sample-split SIGReg needs n>=2, got n={n}")
            n_half = n // 2
            proj_A = proj.narrow(-2, 0, n_half)
            proj_B = proj.narrow(-2, n_half, n_half)
            xA = (proj_A @ A).unsqueeze(-1) * self.t
            xB = (proj_B @ A).unsqueeze(-1) * self.t
            cA = xA.cos().mean(-3)
            sA = xA.sin().mean(-3)
            cB = xB.cos().mean(-3)
            sB = xB.sin().mean(-3)
            # |φ_θ(t)|² ≈ Re(φ̂_A · φ̂_B*) (unbiased; uses A×B pairs only)
            sq_norm = cA * cB + sA * sB
            # Linear term uses average c̄ = (cA+cB)/2 = full-batch ECF (no waste)
            c_full = 0.5 * (cA + cB)
            err = sq_norm - 2.0 * c_full * self.phi + self.phi.square()
            statistic = (err @ self.weights) * n
        else:
            # compute the epps-pulley statistic
            x_t = (proj @ A).unsqueeze(-1) * self.t
            c_bar = x_t.cos().mean(-3)
            s_bar = x_t.sin().mean(-3)
            if self.bias_mode == "ustat":
                if n < 2:
                    raise ValueError(f"U-stat SIGReg needs n>=2, got n={n}")
                # |φ_θ|² ≈ (n/(n-1))(c̄² + s̄²) − 1/(n-1)
                sq_norm = (n / (n - 1)) * (c_bar.square() + s_bar.square()) - 1.0 / (n - 1)
                err = sq_norm - 2.0 * c_bar * self.phi + self.phi.square()
            else:  # "biased"
                err = (c_bar - self.phi).square() + s_bar.square()
            statistic = (err @ self.weights) * n

        return statistic.mean()  # average over projections and time
