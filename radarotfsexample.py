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


def OTFS_approximatedOutput(Xdd: torch.Tensor, T: float, delay: float, Doppler: float,deltaf: float) -> torch.Tensor:
    """
    PyTorch equivalent of MATLAB OTFS_approximatedOutput
    (integer delay–Doppler shift approximation).
    
    Args:
        Xdd: delay–Doppler input grid (M x N) complex tensor
        T: frame duration (s)
        delay: target delay (s)
        Doppler: target Doppler shift (Hz)
    Returns:
        ydd: flattened column-major vector (M*N x 1)
    """
    M, N = Xdd.shape
    lt = math.ceil(delay / (T / M))       # delay index shift
    deltaf = 1.0 / T                      # subcarrier spacing
    kn = math.ceil(Doppler / (deltaf / N))# Doppler index shift

    # circshift in both dimensions
    Ydd = torch.roll(Xdd, shifts=(lt, kn), dims=(0, 1))

    # column-major flattening (MATLAB : operator)
    ydd = Ydd.T.contiguous().view(-1, 1)
    return ydd



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
targetDistance = 30
targetDelay = 2*targetDistance/c0
targetVelocity = 72 / 3.6
targetDoppler = (2 * targetVelocity)/ (c0 / fc)
targetCoefficient = torch.exp(1j * 2 * torch.pi * torch.rand(1))
SNRdB = -10
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
rxSignal = awgn(x=rxSignal,SNRdB=SNRdB,measured=False)


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
    I1 = b1 - a1; I2 = b2 - a2;
    x1 = a1 + (1 - phi) * I1; x2 = a1 + phi * I1;
    y1 = a2 + (1 - phi) * I2; y2 = a2 + phi * I2;
    ydd_11 = OTFS_output(Xdd, T, x1 * T / M, y1 * deltaf / N);
    ydd_12 = OTFS_output(Xdd, T, x1 * T / M, y2 * deltaf / N);
    ydd_21 = OTFS_output(Xdd, T, x2 * T / M, y1 * deltaf / N);
    ydd_22 = OTFS_output(Xdd, T, x2 * T / M, y2 * deltaf / N);
    f11 = abs(ydd_11.T * ydd)^2;
    f12 = abs(ydd_12.T * ydd)^2;
    f21 = abs(ydd_21.T * ydd)^2;
    f22 = abs(ydd_22.T * ydd)^2;
    idx,_ = torch.max([f11, f12, f21, f22]);
    match idx:
        case 1:
              b1 = x2; b2 = y2
        case 2:
              b1 = x2; a2 = y1
        case 3:
              a1 = x1; b2 = y2
        case 4:
              a1 = x1; a2 = y1

# estimatedDelay = (a1 + b1) / 2 * T / M;
# estimatedDoppler = (a2 + b2) / 2 * deltaf / N;
# estimatedRange = estimatedDelay * c0 / 2;
# estimatedVelocity = estimatedDoppler * c0 / fc / 2;
# Hp = OTFS_output(Xdd, T, estimatedDelay, estimatedDoppler);
# estimatedAlpha = (Hp' * Hp) \ (Hp' * ydd);

# % Display sensing estimation result
# sensingResult = ['The estimated target range is ', num2str(estimatedRange), ' m.'];
# sensingResult2 = ['The estimated target velocity is ', num2str(estimatedVelocity), ' m/s.'];
# disp(sensingResult);
# disp(sensingResult2);






