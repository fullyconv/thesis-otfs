import torch
import matplotlib.pyplot as plt

def plot_ray_ellipse_intersection_torch(F1, F2, a, ray_angle_deg, device="cpu"):
    """
    Plot the intersection of a ray starting at focus F1 with the ellipse
    defined by foci F1, F2 and semi-major axis a.

    All math is done in PyTorch.

    Parameters
    ----------
    F1 : (2,) list/tuple/tensor
        First focus [x1, y1], and ray origin.
    F2 : (2,) list/tuple/tensor
        Second focus [x2, y2].
    a : float
        Semi-major axis length (must satisfy 2a > ||F2 - F1||).
    ray_angle_deg : float
        Ray direction angle in degrees (global frame).
    device : str
        "cpu" or "cuda" (if available).
    """
    dtype = torch.float64
    F1 = torch.as_tensor(F1, dtype=dtype, device=device)
    F2 = torch.as_tensor(F2, dtype=dtype, device=device)
    a  = torch.as_tensor(a,  dtype=dtype, device=device)

    # basic validity check
    focal_dist = torch.linalg.vector_norm(F2 - F1)
    if (2*a <= focal_dist).item():
        raise ValueError("Invalid ellipse: need 2*a > ||F2 - F1||.")

    # ray direction (unit)
    th = torch.deg2rad(torch.tensor(ray_angle_deg, dtype=dtype, device=device))
    u = torch.stack((torch.cos(th), torch.sin(th)))
    u = u / torch.linalg.vector_norm(u)

    # ellipse geometry from foci + a
    C = 0.5 * (F1 + F2)                         # center
    d = F2 - F1
    c = 0.5 * torch.linalg.vector_norm(d)       # focal distance
    b = torch.sqrt(a*a - c*c)                   # semi-minor
    theta = torch.atan2(d[1], d[0])             # orientation

    # rotation (global -> local)
    cth, sth = torch.cos(theta), torch.sin(theta)
    R  = torch.stack((torch.stack((cth, -sth)), torch.stack((sth, cth))))  # 2x2
    Rt = R.T

    # line in local coords: p'(t) = p0' + t * u'
    p0_local = Rt @ (F1 - C)
    u_local  = Rt @ u

    # quadratic for (x'/a)^2 + (y'/b)^2 = 1
    A = (u_local[0]**2)/(a*a) + (u_local[1]**2)/(b*b)
    B = 2*((p0_local[0]*u_local[0])/(a*a) + (p0_local[1]*u_local[1])/(b*b))
    C0 = (p0_local[0]**2)/(a*a) + (p0_local[1]**2)/(b*b) - 1.0

    D = B*B - 4*A*C0
    if (D < 0).item():
        print("No real intersection between the ray and ellipse.")
        _plot_scene_torch(F1, F2, a, C, b, theta, u, None)
        return None

    sqrtD = torch.sqrt(torch.clamp(D, min=0.0))
    t1 = (-B - sqrtD) / (2*A)
    t2 = (-B + sqrtD) / (2*A)
    t_all = torch.stack((t1, t2))
    t_all, _ = torch.sort(t_all)

    # forward ray t >= 0 → choose nearest
    mask = t_all >= 0
    if not torch.any(mask):
        print("Ray does not hit the ellipse in the forward direction.")
        _plot_scene_torch(F1, F2, a, C, b, theta, u, None)
        return None

    t = t_all[mask][0]
    P_int = F1 + t * u

    _plot_scene_torch(F1, F2, a, C, b, theta, u, P_int)
    print(f"Intersection point: {P_int.cpu().numpy()}   (ray distance t = {float(t):.6f})")
    return P_int


def _plot_scene_torch(F1, F2, a, C, b, theta, u, P_int):
    """helper: draw ellipse, foci, ray, and optional intersection (torch math, matplotlib plot)."""
    dtype = F1.dtype; device = F1.device

    # ellipse param points in local
    t_plot = torch.linspace(0, 2*torch.pi, 400, dtype=dtype, device=device)
    x_loc = a * torch.cos(t_plot)
    y_loc = b * torch.sin(t_plot)

    cth, sth = torch.cos(theta), torch.sin(theta)
    R = torch.stack((torch.stack((cth, -sth)), torch.stack((sth, cth))))  # 2x2
    XY = R @ torch.vstack((x_loc, y_loc))
    x = XY[0, :] + C[0]
    y = XY[1, :] + C[1]

    # convert to numpy for plotting
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    F1_np = F1.detach().cpu().numpy()
    F2_np = F2.detach().cpu().numpy()

    plt.figure()
    plt.plot(x_np, y_np, 'b-', lw=1.7, label='Ellipse')
    plt.plot([F1_np[0]], [F1_np[1]], 'ro', label='F1 (focus)')
    plt.plot([F2_np[0]], [F2_np[1]], 'ro', label='F2 (focus)')

    # draw the ray from F1
    L = float(4 * a.detach().cpu())
    ray_end = (F1 + L * u).detach().cpu().numpy()
    plt.plot([F1_np[0], ray_end[0]], [F1_np[1], ray_end[1]], 'k--', label='Ray from F1')

    # intersection point
    if P_int is not None:
        Pint_np = P_int.detach().cpu().numpy()
        plt.plot([Pint_np[0]], [Pint_np[1]], 'gs', ms=8, label='Intersection')

    plt.axis('equal'); plt.grid(True, linestyle=':')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Ray–Ellipse Intersection (arbitrary foci, PyTorch)')
    plt.legend(loc='best')
    plt.show()


# -------- example usage --------
if __name__ == "__main__":
    F1 = [1.5, 2.0]
    F2 = [9.0, -1.0]
    a = 6.5
    ray_angle_deg = 40.0
    plot_ray_ellipse_intersection_torch(F1, F2, a, ray_angle_deg,device="cuda")
