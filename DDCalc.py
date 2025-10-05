# Re-run MIMO OTFS with your added target (600,100, 10,-5) and improve visibility:
# - Increase M to cover larger delays
# - Use log-magnitude color scale
# - Annotate detected peaks

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Sequence

C0 = 299_792_458.0
TWO_PI = 2.0 * np.pi

@dataclass
class Agent2D:
    pos: np.ndarray
    vel: np.ndarray
    @staticmethod
    def from_xy_vxvy(x, y, vx, vy):
        return Agent2D(np.array([x, y], float), np.array([vx, vy], float))

@dataclass
class Array2D:
    platform: Agent2D
    elem_offsets: np.ndarray
    @property
    def n_elem(self): return self.elem_offsets.shape[0]

def build_ula(cx, cy, vx, vy, num, spacing, axis="x"):
    plat = Agent2D.from_xy_vxvy(cx, cy, vx, vy)
    idx = np.arange(num) - (num-1)/2.0
    if axis == "x":
        offs = np.stack([idx*spacing, np.zeros_like(idx)], axis=1)
    else:
        offs = np.stack([np.zeros_like(idx), idx*spacing], axis=1)
    return Array2D(plat, offs.astype(float))

@dataclass
class OTFSParams:
    fc: float; B: float; T: float; M: int; N: int

@dataclass
class Target2D:
    pos: np.ndarray; vel: np.ndarray; rcs: float=1.0
    @staticmethod
    def from_xy_vxvy(x,y,vx,vy,rcs=1.0):
        return Target2D(np.array([x,y],float), np.array([vx,vy],float), float(rcs))

def dd_for_link(tx_pos, tx_vel, rx_pos, rx_vel, tgt: Target2D, fc: float):
    lam = C0 / fc
    r_tx = tgt.pos - tx_pos
    r_rx = rx_pos - tgt.pos
    R_tx = np.linalg.norm(r_tx) + 1e-15
    R_rx = np.linalg.norm(r_rx) + 1e-15
    u_tx = r_tx / R_tx; u_rx = r_rx / R_rx
    tau = (R_tx + R_rx) / C0
    dL_dt = u_tx @ (tgt.vel - tx_vel) + u_rx @ (rx_vel - tgt.vel)
    nu = dL_dt / lam
    amp = np.sqrt(tgt.rcs) / (R_tx * R_rx)
    return tau, nu, amp, u_tx, u_rx

def mimo_otfs_dd(tx_array: Array2D, rx_array: Array2D, targets, otfs: OTFSParams):
    lam = C0 / otfs.fc
    d_tau = 1/otfs.B; d_nu = 1/otfs.T
    H = np.zeros((otfs.M, otfs.N, rx_array.n_elem, tx_array.n_elem), dtype=np.complex128)
    tap_list = []
    for tgt in targets:
        tau, nu, amp, u_tx, u_rx = dd_for_link(tx_array.platform.pos, tx_array.platform.vel,
                                               rx_array.platform.pos, rx_array.platform.vel,
                                               tgt, otfs.fc)
        common = amp * np.exp(-1j*TWO_PI*otfs.fc*tau)
        m = int(np.round(tau/d_tau)) % otfs.M
        n = int(np.round(nu /d_nu )) % otfs.N
        tx_ph = np.exp(-1j * TWO_PI/lam * (tx_array.elem_offsets @ u_tx))
        rx_ph = np.exp(-1j * TWO_PI/lam * (rx_array.elem_offsets @ u_rx))
        for ir in range(rx_array.n_elem):
            for it in range(tx_array.n_elem):
                H[m,n,ir,it] += common * rx_ph[ir] * tx_ph[it]
        tap_list.append((tau, nu, amp, m, n))
    return tap_list, H

# ---------- Parameters (expanded delay span) ----------
fc = 28e9
B  = 100e6
T  = 5e-3
M, N = 1024, 256   # M increased to extend delay axis (~10.23 µs)
lam = C0/fc
otfs = OTFSParams(fc=fc, B=B, T=T, M=M, N=N)

tx_arr = build_ula(0.0, 0.0, 0.0, 0.0, num=8, spacing=lam/2, axis="x")
rx_arr = build_ula(5.0, 0.0, 0.0, 0.0, num=8, spacing=lam/2, axis="x")

# ---------- Targets with your added one ----------
targets = [
    Target2D.from_xy_vxvy(1200,   0,  25,  0, rcs=1.0),
    Target2D.from_xy_vxvy( 900, 200,  10, -5, rcs=0.7),
    Target2D.from_xy_vxvy( 700,-150, -20,  4, rcs=0.5),
    Target2D.from_xy_vxvy( 600, 100,  10, -5, rcs=0.8),  # <- added
    Target2D.from_xy_vxvy( 600, 100,  10, -5, rcs=0.8),  # <- added

]

taps, H = mimo_otfs_dd(tx_arr, rx_arr, targets, otfs)
H_mag = np.linalg.norm(H, axis=(2,3))

# Print tap summaries to verify unique bins
print("Taps (τ in µs, ν in Hz) and their grid bins (m,n):")
for i,(tau,nu,amp,m,n) in enumerate(taps):
    print(f" #{i}: τ={tau*1e6:6.3f} µs, ν={nu:8.2f} Hz, m={m}, n={n}")

# ---------- Plot with log scale for visibility ----------
d_tau = 1/B; d_nu = 1/T
extent = [0, (M-1)*d_tau*1e6, 0, (N-1)*d_nu]

plt.figure(figsize=(8,4.6))
plt.imshow(20*np.log10(H_mag.T/np.max(H_mag)+1e-12), origin="lower", aspect="auto", extent=extent)
plt.xlabel("Delay τ (µs)"); plt.ylabel("Doppler ν (Hz)")
plt.title("MIMO OTFS DD (log magnitude, dB re. max)")
cbar = plt.colorbar()
cbar.set_label("dB")
plt.tight_layout()
plt.show()
