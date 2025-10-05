import torch
import math
from thesis import Environment,Agent


C0 = 299792458.0  # speed of light [m/s]


def _range_and_unit(a_pos: torch.Tensor, b_pos: torch.Tensor):
    """
    Returns:
      R:  distance |b-a|  [N,]
      u:  unit vector (b-a)/|b-a|  [N,2]
    """
    diff = b_pos - a_pos           # [N,2]
    R = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
    u = diff / R.unsqueeze(-1)
    return R, u


def _to_grid_indices(tau, nu, tau_max, nu_max, M, N):
    """
    Map continuous (tau, nu) to fractional grid indices (m_f, n_f).
    Delay in [0, tau_max], Doppler in [-nu_max, +nu_max].
    """
    dtau = tau_max / M
    dnu  = 2 * nu_max / N  # symmetric around 0

    m_f = tau / dtau                  # in [0, M]
    n_f = (nu + nu_max) / dnu         # in [0, N]

    return m_f, n_f, dtau, dnu


def _sinc_kernel(frac, halfw=1):
    """
    Simple 1D sinc-like spreading kernel around nearest bin.
    frac: fractional offset in bins (value - round(value))
    halfw: neighborhood half-width (1 -> 3 taps, 2 -> 5 taps)
    Returns taps and offsets.
    """
    # neighbor offsets
    offs = torch.arange(-halfw, halfw+1, device=frac.device)
    # fractional distances to neighbors
    x = offs.unsqueeze(-1) - frac.unsqueeze(0)  # [K, P]
    # sinc window (avoid explicit pi scale differences; this is a soft kernel)
    eps = 1e-9
    k = torch.sin(math.pi * x) / ((math.pi * x) + eps)
    # normalize across neighbors
    k = k / (k.sum(dim=0, keepdim=True) + eps)
    return offs, k


def build_dd_response(env,
                      fc: float,
                      M: int, N: int,
                      tau_max: float,
                      nu_max: float,
                      snr_db: float = None,
                      range_loss: bool = True,
                      spread_halfw: int = 1,
                      complex_dtype=torch.complex64):
    """
    Build a complex DD response H[m, n] from env (TxList, RxList, ObjList).

    Args
    ----
    env: Environment holding .TxList/.RxList/.ObjList with .pos, .vel (2D torch)
         and optional .rcs (float) field per agent (defaults to 1.0 if missing).
    fc:  carrier frequency [Hz]
    M,N: DD grid sizes (delay x doppler)
    tau_max: maximum modeled delay [s] (delay axis spans 0..tau_max)
    nu_max:  maximum modeled doppler [Hz] (doppler axis spans -nu_max..+nu_max)
    snr_db:  if not None, adds AWGN at this SNR (per complex sample)
    range_loss: if True, include simple 1/R^2 amplitude decay
    spread_halfw: fractional-bin spread half-width for 2D kernel (1 -> 3x3)
    complex_dtype: output dtype

    Returns
    -------
    H: complex tensor [M, N]
    paths: dict with tensors of tau, nu, amp for debugging/inspection
    """
    device = env.RxList[0].pos.device if env.RxList else (
             env.TxList[0].pos.device if env.TxList else
             env.ObjList[0].pos.device)

    H = torch.zeros((M, N), dtype=complex_dtype, device=device)

    # Gather current positions/velocities into stacks (one row per agent)
    def stack_list(lst, attr):
        return torch.stack([getattr(a, attr) for a in lst]) if lst else torch.empty((0,2), device=device)

    def get_rcs(lst):
        if not lst:
            return torch.empty((0,), device=device)
        vals = []
        for a in lst:
            vals.append(torch.tensor(getattr(a, "rcs", 1.0), device=device, dtype=torch.float32))
        return torch.stack(vals)

    TxP = stack_list(env.TxList, "pos")   # [T,2]
    TxV = stack_list(env.TxList, "vel")   # [T,2]
    RxP = stack_list(env.RxList, "pos")   # [R,2]
    RxV = stack_list(env.RxList, "vel")   # [R,2]
    ObP = stack_list(env.ObjList, "pos")  # [K,2]
    ObV = stack_list(env.ObjList, "vel")  # [K,2]
    ObRCS = get_rcs(env.ObjList)          # [K]

    # If any list is empty, nothing to do
    if TxP.numel() == 0 or RxP.numel() == 0 or ObP.numel() == 0:
        return H, {"tau": torch.empty(0, device=device),
                   "nu": torch.empty(0, device=device),
                   "amp": torch.empty(0, device=device)}

    # Build all combinations T x K x R
    T, K, R = TxP.shape[0], ObP.shape[0], RxP.shape[0]

    # Expand to broadcast shapes [T,K,R,2]
    TxP_b = TxP[:, None, None, :]  # [T,1,1,2]
    TxV_b = TxV[:, None, None, :]
    ObP_b = ObP[None, :, None, :]  # [1,K,1,2]
    ObV_b = ObV[None, :, None, :]
    RxP_b = RxP[None, None, :, :]  # [1,1,R,2]
    RxV_b = RxV[None, None, :, :]

    # Ranges and unit vectors
    R_to, u_to = _range_and_unit(TxP_b, ObP_b)   # Tx->Obj
    R_or, u_or = _range_and_unit(ObP_b, RxP_b)   # Obj->Rx

    # Delay
    tau = (R_to + R_or) / C0  # [T,K,R]

    # Bistatic Doppler
    # d/dt |b-a| = u_ab · (v_b - v_a)
    term1 = (u_to * (ObV_b - TxV_b)).sum(dim=-1)  # [T,K,R]
    term2 = (u_or * (RxV_b - ObV_b)).sum(dim=-1)  # [T,K,R]
    nu = (fc / C0) * (term1 + term2)              # [T,K,R]

    # Amplitude model (very simple): object RCS and optional range loss
    amp = torch.ones_like(tau, dtype=torch.float32, device=device)
    amp = amp * ObRCS[None, :, None]  # broadcast RCS over T and R
    if range_loss:
        amp = amp / (R_to * R_or + 1.0)  # crude 1/(R1*R2) decay to avoid div-by-zero

    # Flatten paths to a list of scatter points
    tau_f = tau.reshape(-1)  # [P]
    nu_f  = nu.reshape(-1)   # [P]
    amp_f = amp.reshape(-1)  # [P]

    # Map to fractional grid indices
    m_f, n_f, dtau, dnu = _to_grid_indices(tau_f, nu_f, tau_max, nu_max, M, N)

    # Keep only points inside grid
    mask = (m_f >= 0) & (m_f <= (M-1)) & (n_f >= 0) & (n_f <= (N-1))
    m_f = m_f[mask]; n_f = n_f[mask]; amp_f = amp_f[mask]

    # Integer centers and fractional parts
    m0 = torch.round(m_f)
    n0 = torch.round(n_f)
    frac_m = m_f - m0
    frac_n = n_f - n0

    # Neighborhood spreads (1D kernels)
    offs_m, km = _sinc_kernel(frac_m, halfw=spread_halfw)  # [Km, P]
    offs_n, kn = _sinc_kernel(frac_n, halfw=spread_halfw)  # [Kn, P]

    # 2D separable spreading: for each path, distribute energy over (2h+1)x(2h+1)
    Km = offs_m.numel(); Kn = offs_n.numel()
    P  = m0.numel()

    for im in range(Km):
        for in_ in range(Kn):
            w = (km[im, :] * kn[in_, :]) * amp_f  # [P]
            mi = (m0 + offs_m[im]).long()
            ni = (n0 + offs_n[in_]).long()
            # mask valid neighbor bins
            valid = (mi >= 0) & (mi < M) & (ni >= 0) & (ni < N)
            if valid.any():
                H.index_put_((mi[valid], ni[valid]), w[valid].to(H.dtype), accumulate=True)

    # Add complex AWGN to reach desired SNR if requested
    if snr_db is not None:
        sig_pow = (H.real.pow(2).mean() + H.imag.pow(2).mean()).clamp_min(1e-12)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_pow = sig_pow / snr_lin
        noise = torch.sqrt(noise_pow/2) * (
            torch.randn_like(H.real) + 1j*torch.randn_like(H.real)
        ).to(H.dtype)
        H = H + noise

    paths = {"tau": tau_f, "nu": nu_f, "amp": amp_f, "dtau": torch.tensor(dtau), "dnu": torch.tensor(dnu)}
    return H, paths
# Assume env already exists and holds Agent lists with .pos, .vel (and optional .rcs)
fc = 28e9            # 28 GHz (mmWave example)
M, N = 128, 128
tau_max = 5e-6       # 5 microseconds delay span  (~1500 m total path)
nu_max  = 3e3        # +/- 3 kHz Doppler span


# assuming you’ve already defined Agents as before
RxList = [Agent([0, 0], [0, 0])]
TxList = [Agent([100, 0], [0, 0])]
ObjList = [Agent([50, 30], [0, 0])]
env = Environment(M=64, N=64, Snr=10, RxList=RxList, TxList=TxList, ObjList=ObjList)

H, paths = build_dd_response(env, fc, M, N, tau_max, nu_max, snr_db=20)
H

# H is your complex DD response on an (M x N) grid (delay x doppler)
# 'paths' holds the continuous tau/nu list for inspection
