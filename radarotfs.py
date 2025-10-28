import torch

# =========================
# Device & dtype
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex64
FDTYPE = torch.float32

# =========================
# Utilities & math helpers
# =========================

def range2time(R, c=3e8):
    return 2.0 * R / c  # two-way delay (s)

def speed2dop(v, lam):
    # baseband Doppler (Hz) for radial speed v and wavelength lam (two-way already if you pass 2*v)
    return 2.0 * v / lam

def awgn(x, snr_db, measured=True, gen=None):
    """
    Add complex AWGN to a torch tensor x (complex).
    Works with all PyTorch versions.
    """
    x = x.to(CDTYPE)
    Px = (x.abs() ** 2).mean() if measured else torch.tensor(1.0, device=x.device, dtype=FDTYPE)
    snr_lin = 10.0 ** (snr_db / 10.0)
    N0 = Px / snr_lin
    sigma = torch.sqrt(N0 / 2.0)
    n_real = torch.randn_like(x.real)
    n_imag = torch.randn_like(x.real)
    n = sigma * (n_real + 1j * n_imag)
    return x + n.to(CDTYPE)

# =========================
# QAM (unit average power)
# =========================

def qam_constellation(M, device=device, dtype=CDTYPE):
    m_side = int(M ** 0.5)
    assert m_side * m_side == M, "M must be a perfect square (4,16,64,...)"
    vals = torch.arange(-(m_side - 1), m_side, 2, device=device, dtype=FDTYPE)
    xv = vals.repeat(m_side)
    yv = vals.repeat_interleave(m_side)
    const = (xv + 1j * yv).to(dtype)
    const = const / torch.sqrt((const.abs() ** 2).mean())
    return const

def qam_mod(bits, M, device=device):
    k = int(torch.log2(torch.tensor(M, device=device, dtype=FDTYPE)).item())
    assert bits.shape[1] == k
    idx = torch.zeros(bits.shape[0], device=device, dtype=torch.long)
    for b in range(k):
        idx = (idx << 1) | bits[:, b].to(torch.long)
    return qam_constellation(M, device=device)[idx]

# =========================
# OTFS core ops (rect pulses)
# =========================

def ISFFT(X_dd):
    M, N = X_dd.shape
    A = torch.fft.fft(X_dd, dim=0) / torch.sqrt(torch.tensor(M, device=X_dd.device, dtype=FDTYPE))
    X_tf = torch.fft.ifft(A, dim=1) * torch.sqrt(torch.tensor(N, device=X_dd.device, dtype=FDTYPE))
    return X_tf

def SFFT(X_tf):
    M, N = X_tf.shape
    A = torch.fft.ifft(X_tf, dim=0) * torch.sqrt(torch.tensor(M, device=X_tf.device, dtype=FDTYPE))
    X_dd = torch.fft.fft(A, dim=1) / torch.sqrt(torch.tensor(N, device=X_tf.device, dtype=FDTYPE))
    return X_dd

def heisenberg_modulate(X_tf):
    M, N = X_tf.shape
    x_time_cols = torch.fft.ifft(X_tf, dim=0) * torch.sqrt(torch.tensor(M, device=X_tf.device, dtype=FDTYPE))
    return x_time_cols.reshape(M * N)

def wigner_like_TF(rx, M, N):
    Yt = rx.reshape(M, N)
    return torch.fft.fft(Yt, dim=0) / torch.sqrt(torch.tensor(M, device=Yt.device, dtype=FDTYPE))

# =========================
# Channel / OTFS responses (FIXED)
# =========================

def apply_delay_doppler(s, M, N, delay_sec, doppler_hz, T, deltaf):
    s = s.to(CDTYPE)

    # TF (per column)
    S_tf = torch.fft.fft(s.reshape(M, N), dim=0) / torch.sqrt(torch.tensor(M, device=s.device, dtype=FDTYPE))
    k = torch.arange(M, device=s.device, dtype=FDTYPE).view(M, 1)

    # fractional delay via per-subcarrier phase
    phase_delay = torch.exp(-1j * 2.0 * torch.pi * k * deltaf * delay_sec)  # (M,1)
    S_tf_del = phase_delay * S_tf

    # back to time (MxN)
    s_time = torch.fft.ifft(S_tf_del, dim=0) * torch.sqrt(torch.tensor(M, device=s.device, dtype=FDTYPE))

    # integer delay: roll **rows** of the MxN matrix, not the flattened vector
    Ts = T / M
    l_int = int(torch.round(torch.tensor(delay_sec / Ts, device=s.device, dtype=FDTYPE)).item()) % M
    s_time = torch.roll(s_time, shifts=l_int, dims=0)

    # serialize column-wise
    st = s_time.reshape(M * N)

    # Doppler across time samples (spacing Ts)
    n = torch.arange(M * N, device=s.device, dtype=FDTYPE)
    doppler_phase = torch.exp(1j * 2.0 * torch.pi * doppler_hz * n * Ts)
    return st * doppler_phase

def OTFS_output(Xdd, T, delay, doppler, deltaf):
    M, N = Xdd.shape
    Xtf = ISFFT(Xdd)
    s = heisenberg_modulate(Xtf)
    y = apply_delay_doppler(s, M, N, delay, doppler, T, deltaf)
    Ytf = wigner_like_TF(y, M, N)
    Ydd = SFFT(Ytf)
    return Ydd.reshape(M * N)

def OTFS_approximatedOutput(Xdd, T, delay, doppler, deltaf):
    """
    Grid-aligned model (coarse): use round, and wrap modulo (M,N).
    """
    M, N = Xdd.shape
    Ts = T / M
    fd_bin = deltaf / N
    lt = int(torch.round(torch.tensor(delay / Ts, device=Xdd.device, dtype=FDTYPE)).item()) % M
    kn = int(torch.round(torch.tensor(doppler / fd_bin, device=Xdd.device, dtype=FDTYPE)).item()) % N
    Ydd = torch.roll(Xdd, shifts=(lt, kn), dims=(0, 1))
    return Ydd.reshape(M * N)

# =========================
# Golden-section 2D search (clean)
# =========================

def golden_section_2d(Xdd, T, deltaf, ydd_vec, mi, ni, N, K=60):
    phi = (5.0 ** 0.5 - 1.0) / 2.0
    a1, b1 = float(mi - 2), float(mi)
    a2, b2 = float(ni - N/2 - 2), float(ni - N/2)
    M = Xdd.shape[0]
    Ts = T / M
    fd_bin = deltaf / N

    def score(x_bin, y_bin):
        dly = x_bin * Ts
        dop = y_bin * fd_bin
        ydd = OTFS_output(Xdd, T, dly, dop, deltaf)
        inner = torch.vdot(ydd, ydd_vec) if hasattr(torch, "vdot") else (ydd.conj() * ydd_vec).sum()
        return (inner.abs() ** 2).real

    for _ in range(K):
        I1, I2 = b1 - a1, b2 - a2
        x1 = a1 + (1.0 - phi) * I1
        x2 = a1 + phi * I1
        y1 = a2 + (1.0 - phi) * I2
        y2 = a2 + phi * I2

        f11 = score(x1, y1)
        f12 = score(x1, y2)
        f21 = score(x2, y1)
        f22 = score(x2, y2)
        fmax = torch.tensor([f11, f12, f21, f22]).argmax().item()
        if   fmax == 0: b1, b2 = x2, y2
        elif fmax == 1: b1, a2 = x2, y1
        elif fmax == 2: a1, b2 = x1, y2
        else:           a1, a2 = x1, y1

    est_delay   = ((a1 + b1) / 2.0) * Ts
    est_doppler = ((a2 + b2) / 2.0) * fd_bin
    return est_delay, est_doppler

# =========================
# Main demo
# =========================

def main():
    torch.manual_seed(0)

    # Waveform
    M = 256
    N = 16
    modSize = 4
    deltaf = 10e3 * (2**4)  # 160 kHz
    T = 1.0 / deltaf        # symbol duration
    cpSize = M // 4
    cpDuration = (cpSize / M) * T

    # Channel / radar
    c0 = 3e8
    fc = 30e9
    lam = c0 / fc

    targetDistance = 30.0           # m
    targetVelocity = 72.0 / 3.6     # m/s
    targetDelay    = range2time(targetDistance, c0)                      # s
    targetDoppler = speed2dop(targetVelocity, lam)  # Hz, correct two-way Doppler
    targetCoeff    = torch.exp(1j * 2.0 * torch.pi * torch.rand((), device=device)).to(CDTYPE)
    SNRdB = -10.0  # try -5.0 for faster convergence

    # Transmitter (DD -> TF -> time)
    k = int(torch.log2(torch.tensor(modSize, dtype=FDTYPE)).item())
    bits = torch.randint(0, 2, (M * N, k), device=device)
    Xdd = qam_mod(bits, modSize, device=device).reshape(M, N).to(CDTYPE)
    Xtf = ISFFT(Xdd)
    tx  = heisenberg_modulate(Xtf)

    # Channel (corrected)
    y_ch = targetCoeff * apply_delay_doppler(tx, M, N, targetDelay, targetDoppler, T, deltaf)
    y = awgn(y_ch, SNRdB, measured=True)

    # Receiver
    Ytf = wigner_like_TF(y, M, N)
    Ydd = SFFT(Ytf)
    ydd_vec = Ydd.reshape(M * N)

    # Phase I: coarse grid (rounded, wrapped)
    Ts = T / M
    fd_bin = deltaf / N
    delays_bins  = torch.arange(M, device=device, dtype=FDTYPE)            # 0..M-1
    doppler_bins = torch.arange(-N//2, N//2, device=device, dtype=FDTYPE)  # symmetric
    profile = torch.zeros((M, N), device=device, dtype=FDTYPE)

    for mi in range(M):
        d = delays_bins[mi] * Ts
        for nj in range(N):
            fD = doppler_bins[nj] * fd_bin
            ydd_p = OTFS_approximatedOutput(Xdd, T, float(d.item()), float(fD.item()), deltaf)  # (M*N,)
            inner = torch.vdot(ydd_p, ydd_vec) if hasattr(torch, "vdot") else (ydd_p.conj() * ydd_vec).sum()
            profile[mi, nj] = (inner.abs() ** 2).real

    flat = profile.argmax()
    mi, ni = torch.unravel_index(flat, profile.shape)

    # Phase II: golden-section refinement around (mi, ni)
    est_delay, est_dopp = golden_section_2d(Xdd, T, deltaf, ydd_vec, int(mi), int(ni), N, K=60)

    # Convert to range & velocity
    est_range    = est_delay * c0 / 2.0
    est_velocity = (est_dopp * lam) / 2.0  # v = (fD * λ)/2

    # LS estimate for complex alpha
    Hp = OTFS_output(Xdd, T, est_delay, est_dopp, deltaf)
    denom = (Hp.conj() * Hp).sum()
    alpha_hat = (Hp.conj() * ydd_vec).sum() / denom if denom.abs() > 0 else torch.tensor(0.0, device=device, dtype=CDTYPE)

    print(f"Estimated range:    {est_range:.3f} m   (true {targetDistance:.3f})")
    print(f"Estimated velocity: {est_velocity:.3f} m/s (true {targetVelocity:.3f})")
    print(f"Estimated alpha:    {alpha_hat.real:.3f} + {alpha_hat.imag:.3f}j")

if __name__ == "__main__":
    main()
