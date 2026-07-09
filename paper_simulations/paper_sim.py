import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict
from scipy.integrate import solve_ivp
################################################
#set the default font for all figures
font_families = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#plt.rcParams['font.family'] = 'serif'
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
def g_fertiliser(F: float, zeta: float, D0: float) -> float:
    """g(F) = F^2 - 2*zeta*F + D0"""
    F = float(F)
    zeta = float(zeta)
    D0 = float(D0)
    return F * F - 2.0 * zeta * F + D0

def low_intervals_from_phase_row(phase_row: np.ndarray, tau: float) -> np.ndarray:
    """Return absolute time intervals [start, end) where B is low for one parcel."""
    T = len(phase_row)
    intervals = np.zeros((T, 2), dtype=float)
    for j in range(T):
        start = j + float(phase_row[j])
        end = start + float(tau)
        intervals[j, 0] = start
        intervals[j, 1] = end
    return intervals

def B_continuous_from_intervals(
    t: float,
    low_intervals: np.ndarray,
    high: float = 0.9,
    low: float = 0.1,
) -> float:
    """Evaluate B(t) using low intervals [start, end)."""
    t_val = float(t)
    starts = low_intervals[:, 0]
    ends = low_intervals[:, 1]
    idx = int(np.searchsorted(starts, t_val, side="right") - 1) # index of the last start time that is <= t_val
    if idx >= 0 and t_val < ends[idx]:
        return low
    return high

def alpha_logistic(S: float, alpha_max: float, rho: float, S_T: float) -> float:
    z = np.clip(-rho * (S - S_T), -160, 160)
    return alpha_max / (1.0 + np.exp(z))

def h_yield(F: float, xi: float, psi: float) -> float:
    """
    h(F) = max(0, (-F^2 + xi*F) * exp(-psi*F))
    """
    F = float(F)
    xi = float(xi)
    psi = float(psi)
    val = (-F * F + xi * F) * np.exp(-psi * F)
    return float(max(0.0, val))

def phi_matrix(N: int, T: int, tau: float) -> np.ndarray:
    phi = np.random.uniform(0, 1 - tau, size=(N, T))
    phi = np.round(phi, 2)   # approximate to two decimal places
    return phi

def set_initial_soil_quality(n_lands: int, S0_min: float, S0_max: float, random: bool = True) -> np.ndarray:
    if random:
        S0 = np.random.uniform(S0_min, S0_max, size=n_lands)
    else:
        S0 = np.linspace(S0_min, S0_max, n_lands)
    return S0

# ------------------------------------------------------------
# Core solver
# ------------------------------------------------------------

def _soil_ode_single(
    t: float,
    S: np.ndarray,
    g_fertiliser,
    deg_params: Dict,
    alpha_logistic,
    recovery_params: Dict,
    low_intervals: np.ndarray,
    high: float = 0.9,
    low: float = 0.1
) -> np.ndarray:
    S_val = float(S[0])
    alpha = float(alpha_logistic(S_val, **recovery_params))
    g_F = float(g_fertiliser(**deg_params))
    B_t = B_continuous_from_intervals(t, low_intervals, high=high, low=low)
    # Degradation active only during cultivated phase
    D_t = g_F * B_t
    net = alpha - D_t

    return np.array([net * S_val * (1.0 - S_val)], dtype=float)

###############################################
# ---- Input parameters ----
tau = float(0.3)
high = float(0.9)
low = float(0.1)
F = float(1.25)
zeta = float(1.0)
D0 = float(1.2)
alpha_max = float(0.25)
rho = float(50.0)
S_T = float(0.2)
xi = float(10.0)
psi = float(0.5)
Total_land_area = float(550_000.0)
[S0_min, S0_max] = [0.05, 0.95]
deg_params = {"F": F, "zeta": zeta, "D0": D0}
recovery_params = {"alpha_max": alpha_max, "rho": rho, "S_T": S_T}
###############################################
T_max = float(50.0)
n_lands = int(60)
phi = phi_matrix(n_lands, int(T_max), tau)
dt = float(0.001)
n_steps = int(np.ceil(T_max / dt))
t_eval = np.linspace(0.0, float(T_max), n_steps + 1)
soils = np.zeros((n_lands, t_eval.size))
random_initial_conditions_flag = False
initial_soil_quality = set_initial_soil_quality(n_lands, S0_min, S0_max, random=random_initial_conditions_flag)
low_intervals_collection = [low_intervals_from_phase_row(phi[i], tau) for i in range(n_lands)]
# ---- land loops ----
for land_ind in range(n_lands):
    S0 = initial_soil_quality[land_ind]
    low_intervals = low_intervals_collection[land_ind]
    sol = solve_ivp(
    _soil_ode_single,
    method="RK45",
    t_span=(0.0, T_max),
    y0=[S0],
    t_eval=t_eval,
    #max_step=min_window / 10.0,
    rtol=1e-8,
    atol=1e-10,
    args=(g_fertiliser, deg_params, alpha_logistic, recovery_params, low_intervals, high, low),
    )
    S_i = sol.y[0]
    soils[land_ind, :] = S_i


# plot the soil quality index for all lands
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.21, color='#DC243C', linestyle='--', linewidth=1.5, label=r'$S_{crit}$')
#plt.plot(t_eval, soils[0, :], linewidth=1.0, color='grey', alpha=0.5, label='Individual')
plt.plot(t_eval, soils[0, :], linewidth=1.0, color='grey', alpha=0.5)
for land_ind in range(1, n_lands):
    plt.plot(t_eval, soils[land_ind, :], linewidth=1.0, color='grey', alpha=0.5)    
#plt.plot(t_eval, np.mean(soils, axis=0), linewidth=baseline_linewidth, color='blue', label='Mean')
#plt.plot(t_eval, np.median(soils, axis=0), linewidth=baseline_linewidth, color='red', label='Median')
plt.xlabel("Time [yr]")
plt.ylabel("Soil Quality Index")
plt.xlim(0, T_max)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("soil_initial_conditions_ensemble.pdf", bbox_inches='tight')
plt.savefig("soil_initial_conditions_ensemble.eps", bbox_inches='tight')
plt.show()
###############################
# Production series for all lands
production_series = np.zeros((n_lands, n_steps + 1))
h_F = h_yield(F, xi, psi)
individual_land_size = Total_land_area / n_lands
for t_ind, t in enumerate(t_eval):
    low_intervals = low_intervals_collection[0]
    B_t = B_continuous_from_intervals(t, low_intervals, high, low)
    S_t = soils[:, t_ind]
    production_series[:, t_ind] = h_F * B_t * S_t * individual_land_size

# Moving average of production for all lands
production_moving_avg = np.zeros_like(production_series)
window_size = int(1.0 / dt)  # 1 year window
for land_ind in range(n_lands):
    for t_ind in range(n_steps + 1):
        start_ind = max(0, t_ind - window_size + 1)
        production_moving_avg[land_ind, t_ind] = np.mean(production_series[land_ind, start_ind:t_ind + 1])

# Plot the moving average of production for all lands
plt.figure(figsize=(6, 2.5), dpi=180)
plt.plot(t_eval, np.sum(production_moving_avg, axis=0), linewidth=baseline_linewidth, color='red', label='Total Production (Moving Avg)')
plt.xlabel("Time [yr]")
plt.ylabel("Production (Moving Avg)")
plt.xlim(0, T_max)
plt.legend()
plt.show() 

plt.figure(figsize=(6, 2.5), dpi=180)
for land_ind in range(n_lands):
    plt.plot(t_eval, production_moving_avg[land_ind, :], linewidth=1.5, color='grey', alpha=0.5)
plt.xlabel("Time [yr]")
plt.ylabel("Production (Moving Avg)")
plt.xlim(0, T_max)
plt.legend()
plt.show() 


# Plot the production series for all lands
plt.figure(figsize=(6, 2.5), dpi=180)
for land_ind in range(n_lands):
    plt.plot(t_eval, production_series[land_ind, :], linewidth=1.5, color='grey', alpha=0.5)
#plt.plot(t_eval, np.mean(production_series, axis=0), linewidth=baseline_linewidth, color='blue', label='Mean Production')
plt.plot(t_eval, np.sum(production_series, axis=0), linewidth=baseline_linewidth, color='red', label='Total Production')
plt.xlabel("Time [yr]")
plt.ylabel("Production")
plt.xlim(0, T_max)
plt.legend()
plt.show()  
###############################
B_matrix = np.zeros((n_lands, t_eval.size))
for land_ind in range(n_lands):
    low_intervals = low_intervals_collection[land_ind]
    for t_ind, t in enumerate(t_eval):
        B_matrix[land_ind, t_ind] = B_continuous_from_intervals(t, low_intervals, high=high, low=low)   




