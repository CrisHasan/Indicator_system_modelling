import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm 
from scipy.optimize import minimize_scalar
################################################
#set the default font for all figures
plt.rcParams['font.family'] = 'helvet'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.usetex'] = True
# Global text size for figures
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
baseline_linewidth = 2.0
################################################
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/')  
###############################################
# ============================================================
# Model functions
# ============================================================
def g_fertiliser(F: float, zeta: float, D0: float) -> float:
    """g(F) = F^2 - 2*zeta*F + D0"""
    return F * F - 2.0 * zeta * F + D0


def alpha_logistic(S: float, alpha_max: float, rho: float, S_T: float) -> float:
    z = np.clip(-rho * (S - S_T), -160, 160)
    return alpha_max / (1.0 + np.exp(z))


def h_yield(F: float, xi: float, psi: float) -> float:
    """
    h(F) = max(0, (-F^2 + xi*F) * exp(-psi*F))
    """
    val = (-F * F + xi * F) * np.exp(-psi * F)
    return float(max(0.0, val))


def phi_matrix(n_lands: int, n_years: int, tau: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 1 - tau, size=(n_lands, n_years))
    return np.round(phi, 2)


def low_intervals_from_phase_row(phase_row: np.ndarray, tau: float) -> np.ndarray:
    """
    Return absolute time intervals [start, end) where B is low for one parcel.
    """
    n_years = len(phase_row)
    intervals = np.zeros((n_years, 2), dtype=float)

    for j in range(n_years):
        start = j + float(phase_row[j])
        intervals[j, 0] = start
        intervals[j, 1] = start + tau

    return intervals


def B_continuous_from_intervals(
    t: float,
    low_intervals: np.ndarray,
    high: float = 0.9,
    low: float = 0.1,
) -> float:
    """
    Evaluate B(t) using low intervals [start, end).
    """
    starts = low_intervals[:, 0]
    ends = low_intervals[:, 1]

    idx = int(np.searchsorted(starts, t, side="right") - 1)

    if idx >= 0 and t < ends[idx]:
        return low

    return high


def set_initial_soil_quality(
    n_lands: int,
    S0_min: float,
    S0_max: float,
    random: bool = True,
    seed: int = 42,
) -> np.ndarray:
    if random:
        rng = np.random.default_rng(seed)
        return rng.uniform(S0_min, S0_max, size=n_lands)

    return np.linspace(S0_min, S0_max, n_lands)


def soil_ode_single(
    t: float,
    S: np.ndarray,
    deg_params: dict,
    recovery_params: dict,
    low_intervals: np.ndarray,
    high: float,
    low: float,
) -> np.ndarray:
    S_val = float(S[0])

    alpha = alpha_logistic(S_val, **recovery_params)
    g_F = g_fertiliser(**deg_params)

    B_t = B_continuous_from_intervals(
        t,
        low_intervals,
        high=high,
        low=low,
    )

    D_t = g_F * B_t
    net = alpha - D_t

    dSdt = net * S_val * (1.0 - S_val)

    return np.array([dSdt], dtype=float)


def run_single_S0(S0):
    sol = solve_ivp(
        soil_ode_single,
        method="RK45",
        t_span=(0.0, T_max),
        y0=[S0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.02,
        args=(
            deg_params,
            recovery_params,
            low_intervals,
            high,
            low,
        ),
    )

    return sol.y[0]

def auc_for_S0(S0):
    # Compute the integral of |S(t) - S0| over time for a given initial soil quality S0
    S0 = float(S0)
    S = run_single_S0(S0)
    return np.trapz(np.abs(S - S[0]), t_eval)

# ============================================================
# Parameters
# ============================================================
tau = 0.3
high = 0.9
low = 0.1
F = 1.25
zeta = 1.0
D0 = 1.2
alpha_max = 0.25
rho = 50.0
S_T = 0.2
n_lands = 60
xi = 10.0
psi = 0.5
Total_land_area = 550_000.0
S0_min, S0_max = 0.05, 0.95
deg_params = {
    "F": F,
    "zeta": zeta,
    "D0": D0,
}
recovery_params = {
    "alpha_max": alpha_max,
    "rho": rho,
    "S_T": S_T,
}

# ============================================================
# Sweep over initial soil qualities to estimate Scrit
# ============================================================
T_max = 50.0
dt = 0.01
t_eval = np.arange(0.0, T_max + dt, dt)
# Use one fixed phase pattern for all S0 values
# This isolates the effect of initial condition S0
n_years = int(np.ceil(T_max)) + 1
phi = phi_matrix(
    n_lands=n_lands,
    n_years=n_years,
    tau=tau,
    seed=42,
)
low_intervals = low_intervals_from_phase_row(phi[0], tau)

# ============================================================
# Range of initial soil qualities to sweep over
# ============================================================
eps = 1e-6
range_Scrit = 0.217615 + eps, 0.217615 - eps
S0_grid = np.linspace(range_Scrit[0], range_Scrit[1], 200)
# ============================================================
# Optimise to find S_crit
# ============================================================
result = minimize_scalar(
    auc_for_S0,
    bounds=(min(S0_grid), max(S0_grid)),
    method="bounded",
    options={"xatol": 1e-18, "maxiter": 500},
)
Scrit_est = float(result.x)
soil_opt = run_single_S0(Scrit_est)
print(f"Optimised S_crit = {Scrit_est:.14f}")
print(f"Minimum AUC = {result.fun:.14f}")
# ============================================================
# Integrate the ODE for each S0 and compute AUC and final soil quality
# ============================================================
auc_values = []
final_values = []
all_soils = []
new_eps = 1e-7
new_S0_grid = np.linspace(Scrit_est - new_eps, Scrit_est + new_eps, 200)
for S0 in tqdm(new_S0_grid):
    S = run_single_S0(S0)
    auc = np.trapz(np.abs(S - S[0]), t_eval)
    auc_values.append(auc)
    final_values.append(S[-1])
    all_soils.append(S)

auc_values = np.array(auc_values)
final_values = np.array(final_values)
all_soils = np.array(all_soils)

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(4, 2.5), dpi=180)
plt.plot(new_S0_grid, auc_values, linewidth=1.5, color="blue", label=r"Objective function")
plt.axvline(x=Scrit_est, color="crimson", linestyle="--", linewidth=1.2, label=r"Estimated $S_{\mathrm{crit}}$")
plt.xlabel(r"Initial soil quality, $S_0$")
plt.ylabel(r"$\int_0^{T} |S(t) - S_0| dt$")
plt.xlim(S0_grid[0], S0_grid[-1])
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 2.5), dpi=180)
for soil in all_soils:
    plt.plot(t_eval, soil, alpha=0.35, color="grey")

plt.axhline(y=Scrit_est, color="red", linestyle="--", linewidth=1.5, label=r"Estimated $S_{\mathrm{crit}}$")
plt.plot(t_eval, soil_opt, color="green", linewidth=2.0, label=r"Optimised solution")
plt.xlabel(r"Time, $t$")
plt.ylabel(r"Soil quality, $S(t)$")
plt.title(r"Solutions within $S_{\mathrm{crit}}$ range")
plt.legend()
plt.tight_layout()
plt.show()