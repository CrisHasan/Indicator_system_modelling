from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import os
from scipy.optimize import brentq
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from auto_loaders import load_time_S_blocks, parse_time_S_blocks
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
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/critical_threshold_plot/')  
###############################################
def g_fertiliser(F: float, zeta: float, D0: float) -> float:
    """g(F) = F^2 - 2*zeta*F + D0"""
    F = float(F)
    zeta = float(zeta)
    D0 = float(D0)
    return F * F - 2.0 * zeta * F + D0

def alpha_logistic(S: float, alpha_max: float, rho: float, S_T: float) -> float:
    z = np.clip(-rho * (S - S_T), -160, 160)
    return alpha_max / (1.0 + np.exp(z))

def dS_dt_RHS(S: float, Dt: float, alpha_max: float, rho: float, S_T: float) -> float:
    """Evaluate the RHS of dS/dt in terms of S and D(t)."""
    alpha_S = alpha_logistic(S, alpha_max, rho, S_T)
    rhs = (alpha_S - Dt) * S * (1.0 - S)
    return rhs

def analytical_solution(B_con: float, alpha_max: float, rho: float, S_T: float, F: float, zeta: float, D0: float) -> float:
    """Compute the analytical solution of dS/dt for constant B(t) = B_con."""
    D_con = g_fertiliser(F, zeta, D0) * B_con
    S_eq = -(1.0/rho) * (np.log(alpha_max - D_con) - np.log(D_con)) + S_T
    return S_eq

def load_F_S0_from_auto_b(path):
    F_vals = []
    S0_vals = []
    in_table = False
    with open(path, "r") as f:
        for line in f:
            # Find start of numeric table
            if not in_table:
                if "PT" in line and "TY" in line and "LAB" in line:
                    in_table = True
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            # Data rows usually start with branch number (e.g., 1, 2, ...)
            # Metadata rows start with 0 and are before table anyway.
            try:
                _ = int(parts[0])  # BR
                _ = int(parts[1])  # PT
            except ValueError:
                continue

            try:
                F_vals.append(float(parts[4]))   # F column
                S0_vals.append(float(parts[8]))  # S0 column
            except ValueError:
                continue

    return np.array(F_vals), np.array(S0_vals)


###############################################
# ---- Input parameters ----
high = 0.9
low = 0.1
F = 1.25
zeta = 1.0
D0 = 1.2
alpha_max = 0.25
rho = 50.0
S_T = 0.2
[S_min, S_max] = [0.0, 1.0]
Dt_min = g_fertiliser(F, zeta, D0) * low
Dt_max = g_fertiliser(F, zeta, D0) * high
###############################################
# Array of S values
S_values = np.linspace(S_min, S_max, 101)
Dt_values = np.linspace(Dt_min, Dt_max, 101)
# Create a meshgrid of S and Dt values
S_grid, Dt_grid = np.meshgrid(S_values, Dt_values)
# Evaluate the RHS of dS/dt for each combination of S and Dt
dS_dt_values = dS_dt_RHS(S_grid, Dt_grid, alpha_max, rho, S_T)

# Evaluate dS/dt for Dt_min and Dt_max
dS_dt_min = dS_dt_RHS(S_values, Dt_min, alpha_max, rho, S_T) # \tau = 1
dS_dt_max = dS_dt_RHS(S_values, Dt_max, alpha_max, rho, S_T) # \tau = 0
# Find the root of dS/dt (0<S<1) for Dt_min and Dt_max
root_min = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_min, alpha_max, rho, S_T))
root_max = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_max, alpha_max, rho, S_T))

# Plot DS/dt vs S for different Dt_min and Dt_max values
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axvspan(root_min, root_max, color='lightgrey', label=r'Critical region', alpha=0.5)
plt.axvline(x=root_min, color='gray', linestyle='--', label = r'$C^-$', linewidth=1.0)
plt.axvline(x=root_max, color='gray', linestyle='--', label = r'$C^+$', linewidth=1.0)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
plt.axvline(x=S_T, color="darkorange", linestyle='--', linewidth=1.5)
plt.plot(S_values, dS_dt_min, color='deepskyblue', linewidth=1.5)
plt.plot(S_values, dS_dt_max, color='royalblue', linewidth=1.5)
plt.text(S_T - 0.072, 0.04, r'$S_T$', color='darkorange')
plt.text(0.47, 0.048, r'$\tau = 1$', color='deepskyblue') 
plt.text(0.7, 0.005, r'$\tau = 0$', color='royalblue') 
plt.xlabel(r'$S(t)$')
plt.ylabel(r'$\frac{dS}{dt}$')
plt.xlim(S_min, S_max)
plt.ylim(dS_dt_values.min() - 0.002, dS_dt_values.max() + 0.002)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("dS_dt_vs_S_plot.eps", bbox_inches='tight')
plt.savefig("dS_dt_vs_S_plot.pdf", bbox_inches='tight')
plt.show()
######################################
# Find bifurcation points of the autonomous system
bifurcation_tau0_lower = brentq(lambda F: g_fertiliser(F, zeta, D0) - alpha_max/high, 0.0, 1.0)
bifurcation_tau0_upper = brentq(lambda F: g_fertiliser(F, zeta, D0) - alpha_max/high, 1.0, 2.0)
bifurcation_tau1_upper = brentq(lambda F: g_fertiliser(F, zeta, D0) - alpha_max/low, 1.0, 3.0)
# Loop over values of F and find the critical region for each F
F_values_tau0 = np.linspace(bifurcation_tau0_lower, bifurcation_tau0_upper, 280)
critical_tau0 = np.zeros(len(F_values_tau0))
analytical_tau0 = np.zeros(len(F_values_tau0))
for i, F in enumerate(F_values_tau0):  
    Dt_max = g_fertiliser(F, zeta, D0) * high
    # Find the S-value root of dS/dt (0<S<1)
    root_max = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_max, alpha_max, rho, S_T))
    critical_tau0[i] = root_max
    # This is just to check that the root found above agrees the analytical solution
    # And it does!
    analytical_tau0[i] = analytical_solution(high, alpha_max, rho, S_T, F, zeta, D0)

# Correct the rounding error by changing the first and last values to 1.0, which is the correct value at the two bifurcation points
critical_tau0[0] = 1.0
critical_tau0[-1] = 1.0
analytical_tau0[0] = 1.0
analytical_tau0[-1] = 1.0

# Do the same for tau = 1, i.e. find the critical region for each F
F_values_tau1 = np.linspace(0, bifurcation_tau1_upper - 1e-15, 280) # remove 1e-15 correct the rounding error
critical_tau1 = np.zeros(len(F_values_tau1))
analytical_tau1 = np.zeros(len(F_values_tau1))
for i, F in enumerate(F_values_tau1):  
    Dt_min = g_fertiliser(F, zeta, D0) * low
    root_min = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_min, alpha_max, rho, S_T))
    critical_tau1[i] = root_min
    analytical_tau1[i] = analytical_solution(low, alpha_max, rho, S_T, F, zeta, D0)

critical_tau1[-1] = 1.0
analytical_tau1[-1] = 1.0

# Plot bifurcation of the autonomous system for tau = 0
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_lower, bifurcation_tau0_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot(F_values_tau0, analytical_tau0, color='royalblue', linewidth=baseline_linewidth, linestyle='--')
plt.plot(bifurcation_tau0_lower, 1.0, 'o', color='royalblue', markersize=4)
plt.plot(bifurcation_tau0_upper, 1.0, 'o', color='royalblue', markersize=4)
plt.text(0.53, 0.41, r'$S^m$', color='royalblue')
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("tau_0_bifurcation.eps", bbox_inches='tight')
plt.savefig("tau_0_bifurcation.pdf", bbox_inches='tight')
plt.show()

# Plot bifurcation of the autonomous system for tau = 1
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, bifurcation_tau1_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau1_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)  
plt.plot(F_values_tau1, analytical_tau1, color='deepskyblue', linewidth=baseline_linewidth, linestyle='--')
plt.plot(bifurcation_tau1_upper, 1.0, 'o', color='deepskyblue', markersize=4)
plt.text(1.48, 0.21, r'$S^m$', color='deepskyblue')
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("tau_1_bifurcation.eps", bbox_inches='tight')
plt.savefig("tau_1_bifurcation.pdf", bbox_inches='tight')
plt.show()

# Plot the combined bifurcation diagram for both tau = 0 and tau = 1 in 3D
plt.figure(figsize=(4, 2.5), dpi=180)
ax = plt.axes(projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2,1,1,1]))
ax.plot3D([high, high],[0, 3], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
ax.plot3D([high, high],[0, 3], [0, 0], color='black', linestyle='-', linewidth=baseline_linewidth)
ax.plot3D([high]*len(F_values_tau0), F_values_tau0, analytical_tau0, color='royalblue', linewidth=baseline_linewidth, linestyle='--')
ax.plot3D([low, low],[0, 3], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
ax.plot3D([low, low],[0, 3], [0, 0], color='black', linestyle='-', linewidth=baseline_linewidth)
ax.plot3D([low]*len(F_values_tau1), F_values_tau1, analytical_tau1, color='deepskyblue', linewidth=baseline_linewidth, linestyle='--')
ax.plot3D([low, high],[2, 2], [0.55, 0.55], color='red', linestyle='-', linewidth=baseline_linewidth)
ax.plot3D([low, high],[2, 2], [0.45, 0.45], color='red', linestyle='-', linewidth=baseline_linewidth)

ax.quiver(low, 2, 0.55, 0.4, 0, 0, color='red', linewidth=baseline_linewidth, arrow_length_ratio=0.3)
ax.quiver(low, 2, 0.55, 0.50, 0, 0, color='red', linewidth=baseline_linewidth, arrow_length_ratio=0.25)
ax.quiver(low, 2, 0.55, 0.6, 0, 0, color='red', linewidth=baseline_linewidth, arrow_length_ratio=0.2)
ax.quiver(high, 2, 0.45, -0.4, 0, 0, color='red', linewidth=baseline_linewidth, arrow_length_ratio=0.3)
ax.quiver(high, 2, 0.45, -0.50, 0, 0, color='red', linewidth=baseline_linewidth, arrow_length_ratio=0.25)
ax.quiver(high, 2, 0.45, -0.6, 0, 0, color='red', linewidth=baseline_linewidth, arrow_length_ratio=0.2)
plt.xlim(low, high)
plt.ylim(0, 3)
ax.set_zlim(0,1)
ax.set_xticks([0.1,0.5,0.9], labels=[])
ax.set_yticks([0,1.5,3], labels=[])
ax.set_zticks([0, 0.5, 1], labels=[])
ax.grid(False)
ax.view_init(elev=8, azim=13, roll=0)
ax.set_box_aspect([2, 2, 2])  # aspect ratio is 1:1:0.5
plt.tight_layout()
plt.subplots_adjust(left=-0.05)
plt.savefig("bifurcation_combined_taus_3D.pdf", bbox_inches='tight')
plt.savefig("bifurcation_combined_taus_3D.eps", bbox_inches='tight')
plt.show()

# Plot the combined bifurcation diagram for both tau = 0 and tau = 1
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.fill_between(F_values_tau0, analytical_tau0, 1.0, color='yellowgreen')
plt.fill_between(F_values_tau1, analytical_tau1, 0.0, color='palevioletred')
plt.fill_betweenx([0, 1], bifurcation_tau1_upper, 3, color='palevioletred')
plt.plot([0, bifurcation_tau0_lower], [1, 1], color='gray', linestyle='-.', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_lower, bifurcation_tau0_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_upper, bifurcation_tau1_upper], [1, 1], color='gray', linestyle='-.', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau1_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)

plt.plot(F_values_tau0, analytical_tau0, color='royalblue', linewidth=baseline_linewidth, linestyle='--', label=r'$\tau = 0$') 
plt.plot(F_values_tau1, analytical_tau1, color='deepskyblue', linewidth=baseline_linewidth, linestyle='--', label=r'$\tau = 1$') 
plt.plot(bifurcation_tau0_lower, 1.0, 'o', color='royalblue', markersize=4)
plt.plot(bifurcation_tau0_upper, 1.0, 'o', color='royalblue', markersize=4)
plt.plot(bifurcation_tau1_upper, 1.0, 'o', color='deepskyblue', markersize=4)
plt.text(0.37, 0.41, r'$S^m_{high}$', color='royalblue') 
plt.text(2.2, 0.41, r'$S^m_{low}$', color='deepskyblue') 
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10) 
plt.text(0.9, 0.81, r'No', color='black', fontsize=10)
plt.text(0.8, 0.71, r'tipping', color='black', fontsize=10)
plt.text(1.7, 0.71, r'Partial', color='black', fontsize=10)
plt.text(1.69, 0.61, r'tipping', color='black', fontsize=10)
plt.text(2.2, 0.09, r'Tipping', color='black', fontsize=10)

plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("bifurcation_combined_taus.pdf", bbox_inches='tight')
plt.savefig("bifurcation_combined_taus.eps", bbox_inches='tight')
plt.show() 


##################################################
# Import and plot the AUTO bifurcation diagrams
auto_file_path = '/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/AUTO/soil_auto/'
F_tau_10, S0_tau_10 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_10')
F_tau_20, S0_tau_20 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_20')
F_tau_30, S0_tau_30 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_30')
F_tau_40, S0_tau_40 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_40')
F_tau_50, S0_tau_50 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_50')
F_tau_60, S0_tau_60 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_60')
F_tau_70, S0_tau_70 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_70')
F_tau_80, S0_tau_80 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_80')
F_tau_90, S0_tau_90 = load_F_S0_from_auto_b(auto_file_path + '/b.tau_curve_90')

# plot the AUTO bifurcation diagram
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.fill_between(F_values_tau0, analytical_tau0, 1.0, color='yellowgreen')
plt.fill_between(F_values_tau1, analytical_tau1, 0.0, color='palevioletred')
plt.fill_betweenx([0, 1], bifurcation_tau1_upper, 3, color='palevioletred')
plt.plot([0, bifurcation_tau0_lower], [1, 1], color='gray', linestyle='-.', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_lower, bifurcation_tau0_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_upper, bifurcation_tau1_upper], [1, 1], color='gray', linestyle='-.', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau1_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)

plt.plot(F_tau_30, S0_tau_30, color='darkred', linewidth=baseline_linewidth, label=r'$\tau = 30$')
plt.plot(F_values_tau0, analytical_tau0, color='royalblue', linewidth=baseline_linewidth, linestyle='--', label=r'$\tau = 0$')
plt.plot(F_values_tau1, analytical_tau1, color='deepskyblue', linewidth=baseline_linewidth, linestyle='--', label=r'$\tau = 1$')
plt.plot(bifurcation_tau0_lower, 1.0, 'o', color='royalblue', markersize=4)
plt.plot(bifurcation_tau0_upper, 1.0, 'o', color='royalblue', markersize=4)
plt.plot(bifurcation_tau1_upper, 1.0, 'o', color='deepskyblue', markersize=4)
plt.plot(F_tau_30[0], S0_tau_30[0], 'o', color='darkred', markersize=4)
plt.plot(F_tau_30[-1], S0_tau_30[-1], 'o', color='darkred', markersize=4)
plt.text(0.75, 0.41, r'$S^m_{high}$', color='royalblue')
plt.text(2.2, 0.41, r'$S^m_{low}$', color='deepskyblue')
plt.text(1.47, 0.41, r'$S_{crit}(F,0.3)$', color='darkred', rotation=90)
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')

plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("bifurcation_combined_taus_with_AUTO.pdf", bbox_inches='tight')
plt.savefig("bifurcation_combined_taus_with_AUTO.eps", bbox_inches='tight')
plt.show()

# Plot the only AUTO bifurcation diagram for tau = 0.3
plt.figure(figsize=(4, 2.5), dpi=180)
plt.fill_betweenx([0, 1], 0, 3, color='palevioletred')
plt.fill_between(F_tau_30, S0_tau_30, 1.0, color='yellowgreen')
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)
plt.plot([F_tau_30[-1], F_tau_30[0]], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot(F_tau_30, S0_tau_30, color='darkred', linewidth=baseline_linewidth, label=r'$\tau = 0.3$')
plt.plot(F_tau_30[0], S0_tau_30[0], 'o', color='darkred', markersize=4)
plt.plot(F_tau_30[-1], S0_tau_30[-1], 'o', color='darkred', markersize=4)

plt.text(1.25, 0.41, r'$S_{crit}(F,0.3)$', color='darkred', rotation=90)
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)
plt.text(0.9, 0.81, r'No', color='black', fontsize=10)
plt.text(0.8, 0.71, r'tipping', color='black', fontsize=10)
plt.text(1.8, 0.5, r'Tipping', color='black', fontsize=10)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("bifurcation_AUTO_tau_30.pdf", bbox_inches='tight')
plt.savefig("bifurcation_AUTO_tau_30.eps", bbox_inches='tight')
plt.show()


# Plot the bifurcation diagram for various taus
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, bifurcation_tau0_lower], [1, 1], color='gray', linestyle='-.', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_lower, bifurcation_tau0_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau0_upper, bifurcation_tau1_upper], [1, 1], color='gray', linestyle='-.', linewidth=baseline_linewidth)
plt.plot([bifurcation_tau1_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)
plt.plot(F_tau_30, S0_tau_30, color='darkred', linewidth=baseline_linewidth , label=r'$\tau = 0.3$')
plt.plot(F_tau_70, S0_tau_70, color='crimson', linewidth=baseline_linewidth , label=r'$\tau = 0.7$')
#plt.plot(F_tau_80, S0_tau_80, color='chocolate', linewidth=baseline_linewidth , label=r'$\tau = 0.8$')
plt.plot(F_tau_90, S0_tau_90, color='chocolate', linewidth=baseline_linewidth , label=r'$\tau = 0.9$')

plt.plot(F_tau_30[0], S0_tau_30[0], 'o', color='darkred', markersize=4)
plt.plot(F_tau_30[-1], S0_tau_30[-1], 'o', color='darkred', markersize=4)
plt.plot(F_tau_70[0], S0_tau_70[0], 'o', color='crimson', markersize=4)
plt.plot(F_tau_70[-1], S0_tau_70[-1], 'o', color='crimson', markersize=4)
plt.plot(F_tau_90[0], S0_tau_90[0], 'o', color='chocolate', markersize=4)
plt.plot(F_tau_90[-1], S0_tau_90[-1], 'o', color='chocolate', markersize=4)

plt.text(1.25, 0.41, r'$S_{crit}(F,0.3)$', color='darkred', rotation=90)
plt.text(1.55, 0.41, r'$S_{crit}(F,0.7)$', color='crimson', rotation=90)
plt.text(1.92, 0.41, r'$S_{crit}(F,0.9)$', color='chocolate', rotation=90)
plt.text(0.08, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("bifurcation_AUTO_tau_combined.pdf", bbox_inches='tight')
plt.savefig("bifurcation_AUTO_tau_combined.eps", bbox_inches='tight')
plt.show()  

# Load all the AUTO bifurcation diagrams and store them in one dictionary 
auto_bifurcation_data = {}
for tau in range(1, 100, 1):
    if not os.path.exists(auto_file_path + f'/b.tau_curve_{tau}'):
        print(f"File for tau={tau} does not exist. Skipping.")
        continue
    F_tau, S0_tau = load_F_S0_from_auto_b(auto_file_path + f'/b.tau_curve_{tau}')
    auto_bifurcation_data[tau] = (F_tau, S0_tau)

# Load the last tau curve (tau=0.999) and add it to the dictionary
F_tau, S0_tau = load_F_S0_from_auto_b(auto_file_path + f'/b.tau_curve_{999}')
auto_bifurcation_data[100] = (F_tau, S0_tau)
# Plot 10 tau curves
plt.figure(figsize=(4, 2.5), dpi=180)
for tau in range(10-2, 110-2, 10):
    if tau in auto_bifurcation_data:
        F_tau, S0_tau = auto_bifurcation_data[tau]
        plt.plot(F_tau, S0_tau, label=rf'$\tau = {tau/100:.2f}$')
plt.xlabel(r'$F$')
plt.ylabel(r'$S_0$')
plt.xlim(0, 3)
plt.ylim(0.0, 1.05)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
####################################
all_F = []
all_tau = []
all_S0 = []
min_curve = []
max_curve = []
for tau, (F_tau, S0_tau) in auto_bifurcation_data.items():
    all_F.extend(F_tau)
    all_tau.extend([tau / 100.0] * len(F_tau))
    all_S0.extend(S0_tau)
    min_curve.append((np.min(F_tau), tau / 100.0))
    max_curve.append((np.max(F_tau), tau / 100.0))

all_F = np.array(all_F)
all_tau = np.array(all_tau)
all_S0 = np.array(all_S0)
min_curve = np.array(min_curve)
max_curve = np.array(max_curve)

xi_all = np.linspace(0, 3, 300)
yi_all = np.linspace(0.0, 1.0, 150)
Xg_all, Yg_all = np.meshgrid(xi_all, yi_all)
Zg_all = griddata((all_F, all_S0), all_tau, (Xg_all, Yg_all), method='linear')

cmap_color = plt.get_cmap('PiYG_r')  # PiYG_r cool spring_r afmhot_r copper RdGy_r
# Plot contour values for all taus
plt.figure(figsize=(4, 2.5), dpi=180)
contour_all = plt.contourf(Xg_all, Yg_all, Zg_all, levels=100, cmap=cmap_color)
#plt.plot(F_tau_50, S0_tau_50, color='gray', linewidth=1.0, linestyle='--')
#plt.plot(F_tau_70, S0_tau_70, color='gray', linewidth=1.0, linestyle='--')
#plt.plot(F_tau_90, S0_tau_90, color='gray', linewidth=1.0, linestyle='--')

plt.fill_between(F_values_tau0, analytical_tau0, 1.0, color='yellowgreen')
plt.fill_between(F_values_tau1, analytical_tau1, 0.0, color='palevioletred')
plt.fill_betweenx([0, 1], bifurcation_tau1_upper, 3, color='palevioletred')
plt.plot([0,3], [0, 0], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0,3], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot(F_values_tau0, analytical_tau0, color='black', linewidth=baseline_linewidth, linestyle='-', label=r'$\tau = 0$')
plt.plot(F_values_tau1, analytical_tau1, color='black', linewidth=baseline_linewidth, linestyle='-', label=r'$\tau = 1$')
plt.text(0.85, 0.81, r'No', color='black', fontsize=10)
plt.text(0.75, 0.71, r'tipping', color='black', fontsize=10)
plt.text(1.45, 0.61, r'Partial', color='black', fontsize=10)
plt.text(1.40, 0.51, r'tipping', color='black', fontsize=10)
plt.text(2.2, 0.09, r'Tipping', color='black', fontsize=10)
plt.xlim(0, 3)
plt.ylim(0.0, 1.0)

# Not that this may take a wee while to render due to the high resolution of the contour plot
cbar = plt.colorbar(contour_all, label=r'Resting Period, $\tau$')
cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.xlabel(r'Fertiliser Use, $F$')
plt.ylabel(r'Critical Threshold, $S_{crit}(F,\tau)$')
plt.tight_layout()
plt.savefig("bifurcation_AUTO_tau_combined_contour.eps", bbox_inches='tight')
plt.savefig("bifurcation_AUTO_tau_combined_contour.pdf", bbox_inches='tight')
plt.show()

# For the paper
cbar = plt.colorbar(contour_all, label=r'Resting Period [yr]')
cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.xlabel(r'Fertiliser Use [M/(ha$\cdot$yr)]')
plt.ylabel(r'Soil Critical Threshold ($S_{crit}$)')
plt.text(0.02, 0.93, r'(b)', transform=plt.gca().transAxes, fontsize=10)
plt.tight_layout()
plt.savefig("tipping_diagram.eps", bbox_inches='tight')
plt.show()
####################################
# Scatter plot in the (F, tau) plane with color map representing S0
plt.figure(figsize=(4, 2.5), dpi=180)
scatter = plt.scatter(all_F, all_tau, c=all_S0, cmap='PiYG_r', s=1)
#scatter.set_norm(LogNorm()) 
cbar = plt.colorbar(scatter, label=r'Critical Threshold, $S_{crit}(F,\tau)$')
cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.plot(min_curve[:, 0], min_curve[:, 1], color='black', linewidth=1.5, linestyle='--', label='F_min boundary')
plt.plot(max_curve[:, 0], max_curve[:, 1], color='black', linewidth=1.5, linestyle='--', label='F_max boundary')
plt.xlabel(r'Fertiliser Use, $F$')
plt.ylabel(r'Resting Period, $\tau$')
plt.xlim(0, 3)
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.show()
####################################
# Plot contour in a different projection: (x,y,z) -> (F, tau, S0)
xi_new = np.linspace(0, 3, 300)
yi_new = np.linspace(0.0, 1.0, 300)
Xg_new, Yg_new = np.meshgrid(xi_new, yi_new)
Zg_new = griddata((all_F, all_tau), all_S0, (Xg_new, Yg_new), method='linear')
plt.figure(figsize=(4, 2.5), dpi=180)
contour_all_new = plt.contourf(Xg_new, Yg_new, Zg_new, levels=100, cmap=cmap_color)
cbar = plt.colorbar(contour_all_new, label=r'Critical Threshold, $S_{crit}(F,\tau)$')
#plt.contour(Xg_all, Yg_all, Zg_all, levels=[0.2, 0.25,0.3,0.35], colors='black', linewidths=0.5)
cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.fill_betweenx(max_curve[:, 1], max_curve[:, 0], x2=3, color='palevioletred')
plt.fill_between(min_curve[:, 0], min_curve[:, 1], 0.0, color='palevioletred')
plt.plot(min_curve[:, 0], min_curve[:, 1], color='black', linewidth=1.5)
plt.plot(max_curve[:, 0], max_curve[:, 1], color='black', linewidth=1.5)
plt.xlabel(r'Fertiliser Use, $F$')
plt.ylabel(r'Resting Period, $\tau$')
plt.xlim(0, 3)
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.show()
####################################
t_values, S_nearby_values = load_time_S_blocks(auto_file_path + 's.nearby_all')
t_values = t_values * 50  # Scale time values by 50
# Plot trajectories
plt.figure(figsize=(4, 2.5), dpi=180)
for i in range(S_nearby_values.shape[1]):
    plt.plot(t_values[:, i], S_nearby_values[:, i], color='silver', linewidth=1.5)

plt.plot(t_values[:, 9], S_nearby_values[:, 9], color='green', linewidth=2.0)
plt.text(25.8, 0.23, r'$\Gamma_{crit}(t)$', color='green', fontsize=10)
plt.xlabel(r'$t$')
plt.ylabel(r'$S$')
plt.xlim(t_values.min(), t_values.max())
plt.ylim(S_nearby_values.min(), S_nearby_values.max())
plt.tight_layout()
plt.savefig("AUTO_nearby_solution.eps", bbox_inches='tight')
plt.show()
####################################
# plot critical trajectory 
S_crit_est = 0.5 * (S_nearby_values[:, 9].max() + S_nearby_values[:, 9].min())
plt.figure(figsize=(4, 2.5), dpi=180)
plt.plot(t_values[:, 9], S_nearby_values[:, 9], color='green', linewidth=2.0, label=r'$\Gamma_{crit}(t)$')
plt.plot(t_values[:, 9], S_crit_est * np.ones_like(t_values[:, 9]), color='gray', linewidth=2.0, label=r'$S_{crit}$')
plt.xlabel(r'$t$')
plt.ylabel(r'$S$')
plt.xlim(t_values.min(), t_values.max())
plt.ylim(S_nearby_values[:, 9].min() - 0.004, S_nearby_values[:, 9].max() + 0.001)
plt.legend()
plt.tight_layout()
plt.savefig("AUTO_critical_solution.eps", bbox_inches='tight')
plt.show()
####################################
T_val, S0_val = load_F_S0_from_auto_b(auto_file_path + '/b.extended_crit_all')
# Plot relation between T and S0
plt.figure(figsize=(4, 2.5), dpi=180)
#plt.plot(T_val, S_nearby_values[0, 10]*np.ones_like(T_val), color='black', linewidth=baseline_linewidth)
plt.plot(T_val, np.mean(S0_val)*np.ones_like(T_val), color='silver', linewidth=baseline_linewidth)
plt.scatter(T_val, S0_val, color='crimson', s=3)
plt.xlabel(r'$T$')
plt.ylabel(r'$S_{crit}(T)$')
plt.xlim(T_val.min(), T_val.max())
plt.ylim(S0_val.min(), S0_val.max())
plt.tight_layout()
#plt.savefig("AUTO_extended_crit_all.eps", bbox_inches='tight')
plt.show()

####################################
# This just to plot the middle equilibrium solution for the autonomous system
F_test = bifurcation_tau1_upper + 0.0000000000000002
# plot log(alpha_max - g(F)*0.1) against F
F_values = np.linspace(2.51657, F_test, 10000000)
log_values = np.zeros(len(F_values))
for i, F in enumerate(F_values):
    if g_fertiliser(F, zeta, D0) * low >= alpha_max:
        log_values[i] = np.nan
    else:
        log_values[i] = -(np.log(alpha_max - g_fertiliser(F, zeta, D0) * low) - np.log(g_fertiliser(F, zeta, D0) * low))/rho + S_T

plt.figure(figsize=(4, 2.5), dpi=180)
plt.plot(F_values, log_values, color='blue', linewidth=baseline_linewidth)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(2.51657, F_test)
plt.tight_layout()
plt.show()