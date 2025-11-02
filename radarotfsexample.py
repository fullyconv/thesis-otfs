# %%
import torch, math
from torch.fft import ifft, fft

# %%

def bi2de(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: shape [num_symbols, b], dtype=int64/uint8 {0,1}
    Interprets column 0 as LSB (MATLAB bi2de default: 'right-msb').
    """
    b = bits.shape[1]
    weights = (2 ** torch.arange(b, device=bits.device, dtype=bits.dtype))
    return (bits * weights).sum(dim=1)

def qammod(sym_idx: torch.Tensor, M: int) -> torch.Tensor:
    """
    MATLAB-like qammod(sym_idx, M, 'gray', 'UnitAveragePower', true)
    Works for square QAM: M = 4, 16, 64, ...
    """
    assert int(math.sqrt(M))**2 == M, "M must be a perfect square for square QAM."
    L = int(math.sqrt(M))               # PAM order per axis
    m = int(math.log2(L))               # bits per axis

    # Split natural binary index into I/Q natural indices
    i_nat = sym_idx % L
    q_nat = sym_idx // L

    # Natural -> Gray per axis (MATLAB qammod uses Gray by default)
    i_gray = i_nat ^ (i_nat >> 1)
    q_gray = q_nat ^ (q_nat >> 1)

    # Map Gray indices to PAM levels: -(L-1), -(L-3), ..., +(L-1)
    # level = 2*k - (L-1)
    I = 2 * i_gray.to(torch.int64) - (L - 1)
    Q = 2 * q_gray.to(torch.int64) - (L - 1)

    I = I.to(torch.float32)
    Q = Q.to(torch.float32)

    # Form complex symbols (note +j*Q; MATLAB uses +j for imag)
    x = torch.complex(I, Q)

    # Unit average power scaling: Es(avg) = 2/3 * (M - 1) for this lattice
    scale = math.sqrt(1.0 / (2.0/3.0 * (M - 1)))
    return x * scale




def ISFFT(ddSignal: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """
    Inverse Symplectic Finite Fourier Transform (Delay-Doppler → Time-Frequency)
    MATLAB: tfSignal = fft(ifft(ddSignal.').') * sqrt(N) / sqrt(M)
    """
    tfSignal = torch.fft.fft(torch.fft.ifft(ddSignal.T, dim=0), dim=1).T
    tfSignal = tfSignal * (N ** 0.5) / (M ** 0.5)
    return tfSignal


def WignerTransform(rxSignal: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """
    Wigner Transform (Time-Domain → Time-Frequency)
    MATLAB: tfSignal = reshape(rxSignal, M, N); tfSignal = fft(tfSignal) / sqrt(M)
    """
    tfSignal = rxSignal.view(M, N)
    tfSignal = torch.fft.fft(tfSignal, dim=0) / (M ** 0.5)
    return tfSignal


def SFFT(tfSignal: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """
    Symplectic Finite Fourier Transform (Time-Frequency → Delay-Doppler)
    MATLAB: ddSignal = ifft(fft(tfSignal.').') / sqrt(N) * sqrt(M)
    """
    ddSignal = torch.fft.ifft(torch.fft.fft(tfSignal.T, dim=0), dim=1).T
    ddSignal = ddSignal / (N ** 0.5) * (M ** 0.5)
    return ddSignal

def awgn(x: torch.Tensor, SNRdB: float, measured: bool = False) -> torch.Tensor:
    """
    Add complex Additive White Gaussian Noise (AWGN) to tensor x
    to achieve the specified SNR (in dB), matching MATLAB awgn().
    
    Args:
        x: input signal (real or complex tensor)
        SNRdB: target signal-to-noise ratio in dB
        measured: if True, use measured power of x;
                  if False, assume signal power = 1
    
    Returns:
        x_noisy: tensor with AWGN added
    """
    # Compute signal power
    if measured:
        signal_power = torch.mean(torch.abs(x)**2)
    else:
        signal_power = 1.0

    # Compute noise power for given SNR
    SNR_linear = 10 ** (SNRdB / 10)
    noise_power = signal_power / SNR_linear

    # Generate complex noise
    noise = (torch.randn_like(x) + 1j * torch.randn_like(x)) * math.sqrt(noise_power / 2)

    # Add noise
    return x + noise


def OTFS_approximatedOutput(Xdd: torch.Tensor, T: float, delay: float, Doppler: float, deltaf: float) -> torch.Tensor:
    M, N = Xdd.shape
    lt = math.ceil(delay / (T / M))
    deltaf = 1.0 / T
    kn = math.ceil(Doppler / (deltaf / N))
    Ydd = torch.roll(Xdd, shifts=(lt, kn), dims=(0, 1))
    return Ydd.T.contiguous().view(-1)  # <-- 1-D

# --- add this next to your other helpers ---
def OTFS_output(Xdd: torch.Tensor, T: float, delay: float, Doppler: float) -> torch.Tensor:
    """
    PyTorch version of your MATLAB OTFS_output:
      - fractional delay via diagonal phasor in TF then IFFT
      - fractional Doppler via time-domain complex exponential
      - returns a 1-D vector (column-major flatten) to ease inner products
    """
    M, N = Xdd.shape
    lt = math.ceil(delay / (T / M))         # integer delay shift used in circshift
    deltaf = 1.0 / T

    # ISFFT: Delay-Doppler -> TF
    Xtf = ISFFT(Xdd, M, N)                  # shape [M, N], complex

    # Apply fractional delay in TF: diag(exp(-j2π k Δf delay)) * Xtf
    k = torch.arange(M, device=Xdd.device, dtype=torch.float32)
    ph_delay = torch.exp(-1j * 2 * math.pi * k * deltaf * float(delay))  # [M]
    A = (ph_delay[:, None] * Xtf)            # broadcast over N

    # IFFT across subcarriers (rows), scale by sqrt(M)
    B = torch.fft.ifft(A, dim=0) * math.sqrt(M)  # [M, N]

    # circshift rows by -lt, then vectorize column-major and shift by +lt
    B_shift = torch.roll(B, shifts=-lt, dims=0)  # [M, N]
    vec = B_shift.T.contiguous().view(-1)        # (MN,) column-major
    vec = torch.roll(vec, shifts=lt, dims=0)     # (MN,)

    # Apply fractional Doppler in time domain: exp(j2π Doppler n T/M)
    n = torch.arange(M * N, device=Xdd.device, dtype=torch.float32)
    doppler_ph = torch.exp(1j * 2 * math.pi * float(Doppler) * n * (T / M))
    rt = doppler_ph * vec                         # (MN,)

    # Reshape back to [M, N] consistent with MATLAB reshape(rt,M,N)
    Rt = rt.view(N, M).T                          # [M, N]

    # Ydd = fft(Rt.').' / sqrt(N)
    Ydd = torch.fft.fft(Rt.T, dim=0).T / math.sqrt(N)  # [M, N]

    # Return column-major vector (1-D) for easy vdot
    return Ydd.T.contiguous().view(-1)            # (MN,)



# %%
#% Waveform parameters
M = 256 #% subcarrier number
N = 16 #% symbol number
modSize = 4 #% modulation size
deltaf = 15e3 * 2**4 #% subcarrier spacing
T = 1 / deltaf #% symbol duration
cpSize = M / 4
cpDuration = cpSize / M * T

# %%
#% Channel parameters
c0 = 3e8# % light of speed
fc = 30e9# % carrier frequency
targetDistance = 10
targetDelay = 2*targetDistance/c0
targetVelocity = 72 / 3.6
targetDoppler = (2 * targetVelocity)/ (c0 / fc)
targetCoefficient = torch.exp(1j * 2 * torch.pi * torch.rand(1))
SNRdB = 10
maximumSensingRange = c0 * cpDuration / 2


# %%
# % OTFS ISAC transmitter
dataBits = torch.randint(0, 2, (M * N, int(math.log2(modSize))))
dataDe = bi2de(dataBits)
dataDe = torch.reshape(dataDe, (M, N))
data = qammod(dataDe, modSize)
ddSignal = data
tfSignal = ISFFT(ddSignal, M, N)
txFrame = torch.fft.ifft(tfSignal,dim=0) * math.sqrt(M)
txSignal = torch.flatten(txFrame)


# %%
# % Channel realization
alpha = targetCoefficient
delay = targetDelay
doppler = targetDoppler
tfSignal = fft(torch.reshape(txSignal, (M, N))) / math.sqrt(M)
txSignal_delay = torch.zeros(M * N, 1)
l_tau = math.ceil(delay / (T / M))
txSignal_delay= torch.roll(
    (
        torch.fft.ifft(
            torch.diag(torch.exp(-1j * 2 * math.pi * torch.arange(M) * deltaf * delay))
            @ tfSignal,
            dim=0
        ) * math.sqrt(M)
    )
    .roll(-l_tau, dims=0)
    .T.contiguous()
    .view(-1, 1),
    shifts=l_tau,
    dims=0
)
dopplerEffect = torch.exp(1j * 2 * math.pi * doppler * torch.arange(M*N) * T / M)
rxSignal = alpha * dopplerEffect * txSignal_delay
rxSignal = torch.sum(rxSignal, dim=1)
# rxSignal = awgn(x=rxSignal,SNRdB=SNRdB,measured=False)


# %%
# % Sensing receiver
Ytf = WignerTransform(rxSignal, M, N)
Ydd = SFFT(Ytf, M, N)
Xdd = ddSignal

# %%
# % Two-phase sensing estimation algorithm


ydd = torch.flatten(Ydd)
K = 60
# % phase 
delayList = torch.arange(M) * T / M
DopplerList = (torch.arange(-N/2, N/2) * deltaf / N)

Mdelays = len(delayList)
Ndopplers = len(DopplerList)

profile = torch.zeros((Mdelays, Ndopplers), dtype=torch.float32)

for m in range(Mdelays):
    for n in range(Ndopplers):
        ydd_p = OTFS_approximatedOutput(Xdd, T, delayList[m], DopplerList[n],deltaf)
        # inner product ydd_pᴴ ydd → conj transpose * vector
        value = torch.abs(torch.vdot(ydd_p, ydd)) ** 2
        profile[m, n] = value.real

# profile: tensor of shape [M, N]
val, idx = torch.max(profile.view(-1), dim=0)          # flat (row-major) index
M, Np = profile.shape
mi0 = (idx // Np).item()                               # 0-based row
ni0 = (idx %  Np).item()                               # 0-based col

# Convert to MATLAB-like 1-based for your formulas
mi = mi0 + 1
ni = ni0 + 1

phi = (math.sqrt(5) - 1) / 2.0
a1, b1 = mi - 2, mi
a2, b2 = ni - N/2 - 2, ni - N/2
for k in range(K):
    I1 = b1 - a1; I2 = b2 - a2
    x1 = a1 + (1 - phi) * I1; x2 = a1 + phi * I1
    y1 = a2 + (1 - phi) * I2; y2 = a2 + phi * I2
    ydd_11 = OTFS_output(Xdd, T, x1 * T / M, y1 * deltaf / N)
    ydd_12 = OTFS_output(Xdd, T, x1 * T / M, y2 * deltaf / N)
    ydd_21 = OTFS_output(Xdd, T, x2 * T / M, y1 * deltaf / N)
    ydd_22 = OTFS_output(Xdd, T, x2 * T / M, y2 * deltaf / N)
# Inside the loop, after computing ydd_11, ydd_12, ydd_21, ydd_22:
    ydd_11 = ydd_11.to(torch.complex64).view(-1)
    ydd_12 = ydd_12.to(torch.complex64).view(-1)
    ydd_21 = ydd_21.to(torch.complex64).view(-1)
    ydd_22 = ydd_22.to(torch.complex64).view(-1)

    f11 = torch.abs(torch.vdot(ydd_11, ydd)) ** 2
    f12 = torch.abs(torch.vdot(ydd_12, ydd)) ** 2
    f21 = torch.abs(torch.vdot(ydd_21, ydd)) ** 2
    f22 = torch.abs(torch.vdot(ydd_22, ydd)) ** 2
    vals = torch.stack([f11, f12, f21, f22])
    idx  = int(torch.argmax(vals))
    if   idx == 0:  b1 = x2; b2 = y2
    elif idx == 1:  b1 = x2; a2 = y1
    elif idx == 2:  a1 = x1; b2 = y2
    else:           a1 = x1; a2 = y1

# ---- Estimates from the final rectangle center ----
estimatedDelay    = ((a1 + b1) / 2.0) * (T / M)          # seconds
estimatedDoppler  = ((a2 + b2) / 2.0) * (deltaf / N)     # Hz

estimatedRange    = float(estimatedDelay * c0 / 2.0)     # meters
estimatedVelocity = float(estimatedDoppler * c0 / (2.0 * fc))  # m/s

# ---- Build the steering vector (atom) at the estimate ----
Hp = OTFS_output(Xdd, T, float(estimatedDelay), float(estimatedDoppler))  # (MN,) complex

# ---- Least-squares estimate of complex amplitude alpha ----
# ydd is your received vectorized DD grid; make sure it's 1-D complex
ydd = torch.flatten(Ydd.T).to(torch.complex64)  # column-major like MATLAB (:)

num = torch.vdot(Hp, ydd)                       # Hp^H y
den = torch.vdot(Hp, Hp) + 1e-12                # Hp^H Hp  (regularized)
estimatedAlpha = num / den

# ---- Report ----
print(f"Estimated range:    {estimatedRange:.3f} m")
print(f"Estimated velocity: {estimatedVelocity:.3f} m/s")
print(f"Estimated alpha:    {estimatedAlpha.real:.3f} + {estimatedAlpha.imag:.3f}j")







