# Shared physics evaluation module

All 13 methods are evaluated with **the same** physics functions for
apples-to-apples comparison. Strip-down version of what the working
repo's `physics_metrics.py` provided — only the metrics that appear in
the paper's tables and figures.

| File | Exports |
|---|---|
| `metrics.py` | `epsilon_strain`, `enstrophy`, `inertial_slope`, `relative_divergence` |
| `spectra.py` | `energy_spectrum` — shell-binned radial `E(k)` |
| `reconstruction_metrics.py` | `rel_l1`, `rel_l2`, `mae`, `rmse`, `psnr_db` |

## Definitions

| Quantity | Formula | Implementation note |
|---|---|---|
| `ε` (dissipation rate) | `2 ν ⟨S_ij S_ij⟩`, `ν = 1.85 × 10⁻⁴` | strain-rate tensor via central differences; reported as `ε_ratio = ε_recon / ε_DNS` |
| `Ω` (enstrophy) | `½ ⟨\|ω\|²⟩`, `ω = ∇ × u` | curl via central differences; reported as `Ω_ratio` |
| `E(k)` | shell-sum of `½ \|û\|²` at integer wavenumber `k = 0..511` | FFT'd on the GPU; matches `scipy.fft` to 3 decimal places |
| `β` (inertial slope) | least-squares fit of `log E(k)` vs `log k` on `k ∈ [4, 40]` | DNS reference at 1024³ ≈ −1.50 |
| relative divergence | `std(∇·u) / std(u)` | central differences |
| rel L1 / L2 | `‖u_r − u_g‖_{1,2} / ‖u_g‖_{1,2}` | voxel-wise across stacked u/v/w/p |
| PSNR | `10 log₁₀ (peak² / MSE)`, peak = 2.0 in normalized `[−1, 1]` space | dB |

Three bugs that were found and fixed during paper preparation are
documented in `physics/metrics.py` as comments next to the relevant
code blocks:

1. `E_k` was previously shell-averaged (divided by shell counts) →
   spurious −2 added to slope. Fixed: shell SUM.
2. `ik = 2j·π` was wrong for a `(2π)³` periodic domain. Fixed:
   `ik = 1j · k_integer`.
3. The `torch.meshgrid(indexing="ij")` axis convention swapped the
   channel-to-axis pairing for divergence. Fixed: explicit per-channel
   k assignment.

These fixes are baked in; the open-source release does not regress.

## What's NOT here (deliberately excluded — not used in the paper)

- Compensated spectrum `k^{5/3} E(k)`
- By-band energy fractions (inertial / mid / dissipation)
- Length scales `L`, `η`, `λ_T`
- Taylor-microscale Reynolds number `Re_λ`
- Content vector `CV` (256-dim per-frame statistic)
- NS residual `R_i = ∂_t u + u·∇u + ∇p − ν ∇²u`
- Frame-pair drifts (momentum, KE, enstrophy, divergence drift)
- Peak vorticity, `p_rms`

These exist in the working repo (`scripts/compute_spectral_metrics.py`)
but are excluded here because they don't feed any table or figure in
the paper.
